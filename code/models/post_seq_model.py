import torch
from torch import nn
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence
from tqdm import tqdm
import numpy as np
import pandas as pd

from typing import List, Dict, Iterable
from .text_encoder import TextEncoderBuilder
import utils
import os
from os.path import join
from operator import itemgetter

class PostSequenceModel(nn.Module):
    def __init__(self, device: torch.device, n_classes: int, n_features: int, conf: dict, has_cls_enc=False):
        """
        Args:
            device (torch.device): Computation device (CPU or GPU)
            n_classes (int): Number of output classes
            n_features (int): Number of input features. Each features can be either a skalar value or text, which is further encoded.
            conf (dict): Model configuration
            has_cls_enc (bool, optional): Whether to use pre-computed CLS token encodings
        """
        if conf["psm"]["type"] not in ["RNN", "LSTM", "GRU"]:
            raise Exception(f"psm_type must be one of ['RNN', 'LSTM', 'GRU'], but found {conf['psm']['type']}")
        if conf["psm"]["output"] not in ["last_state", "mean", "max", "mean+max"]:
            raise Exception(f"psm_output must be one of ['last_state', 'mean', 'max', 'mean+max'], but found {conf['psm']['output']}")

        super().__init__()
        self.device = device
        self._psm_output = conf["psm"]["output"]
        
        te_builder = TextEncoderBuilder(available_models=["roberta-base"])
        self._has_cls_enc = has_cls_enc
        if self._has_cls_enc:
            self._text_encoders = nn.ModuleDict({
                "post|text_cls_enc": te_builder.build_cls2doc_encoder("roberta-base", device, conf),
                "user|description_cls_enc": te_builder.build_cls2doc_encoder("roberta-base", device, conf),
                "art_cont|title_cls_enc": te_builder.build_cls2doc_encoder("roberta-base", device, conf),
                "art_cont|text_cls_enc": te_builder.build_cls2doc_encoder("roberta-base", device, conf),
                "art_cont|meta_description_cls_enc": te_builder.build_cls2doc_encoder("roberta-base", device, conf)
            })
        else:
            self._text_encoders = nn.ModuleDict({
                "post|text": te_builder.build_text_encoder("roberta-base", device),
                "user|description": te_builder.build_text_encoder("roberta-base", device),
                "art_cont|title": te_builder.build_text_encoder("roberta-base", device),
                "art_cont|text": te_builder.build_text_encoder("roberta-base", device),
                "art_cont|meta_description": te_builder.build_text_encoder("roberta-base", device)
            })
        
        self._meta_embedder = nn.Sequential(
            nn.Linear(
                in_features=n_features - len(self._text_encoders),
                out_features=n_features - len(self._text_encoders)
            ),
            nn.Tanh()
        ).to(device)

        if conf["psm"]["type"] == "RNN":
            recurrent_module = nn.RNN
        elif conf["psm"]["type"] == "LSTM":
            recurrent_module = nn.LSTM
        elif conf["psm"]["type"] == "GRU":
            recurrent_module = nn.GRU
        
        input_size = n_features + \
            sum([m.dim for m in list(self._text_encoders.values())]) - len(self._text_encoders)
        self._post_seq_encoder = recurrent_module(
            input_size=input_size,
            hidden_size=conf["psm"]["hidden_size"],
            batch_first=True
        ).to(device)
        self._classifier = nn.Sequential(
            nn.Linear(
                in_features=conf["psm"]["hidden_size"] if self._psm_output != "mean+max" else 2 * conf["psm"]["hidden_size"],
                out_features=n_classes
            ),
            nn.Softmax(dim=1)
        ).to(device)

    def forward(self, post_sequences_batch: Iterable[Dict[str, List]], pbar=None, te_batch_size=None, lm_batch_size=None, pooler_batch_size=None) -> torch.Tensor:
        """
        Args:
            post_sequences_batch (Iterable[Dict[str, List]]): len(post_sequences_batch): batch_size. 
                Batch of sequences, each sequence contains multiple posts referring to an article.
                A sequence is represented as a dict with lists of values per feature.
            pbar (tqdm, optional): Progressbar showing sentence progress. Defaults to None.
            te_batch_size (int, optional): Batch size for encoding documents. If None, encode all documents at once. Defaults to None.
            lm_batch_size (int, optional): Batch size for encoding sentences in language model. If None, encode all sentences of all documents at once. Defaults to None.
            pooler_batch_size (int, optional): Batch size for processing cls token encodings. If None, process all cls tokens of the document. Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, n_classes]
        """
        # TODO: Instead of processing each batch one by one, take all at once
        batch = [self._seq2tensor(seq, pbar, te_batch_size, lm_batch_size, pooler_batch_size) for seq in post_sequences_batch]
        # output: PackedSequence
        output, _ = self._post_seq_encoder(pack_sequence(batch, enforce_sorted=False))
        # padded_hidden_states.shape: [batch_size, max(seq_lens), hidden_size]
        # seq_lens[0] == batch[0].shape[0], ...
        hidden_states_batch, seq_lens = pad_packed_sequence(output, batch_first=True)
        # recurrent_outputs.shape: [batch_size, (1|2) * hidden_size]
        recurrent_outputs = self._aggregate_hidden_states(hidden_states_batch, seq_lens)        
        # pred.shape: [batch_size, n_classes]
        pred = self._classifier(recurrent_outputs)
        return pred

    def _seq2tensor(self, post_sequence: Dict[str, List], pbar=None, te_batch_size=None, lm_batch_size=None, pooler_batch_size=None) -> torch.Tensor:
        """Encodes one post sequence as input for post sequence model.

        Args:
            post_sequence (Dict[str, List]): Post sequence for one news source. Contains all features (feature is key, list is values). Each list has length seq_length.
            pbar (tqdm, optional): Progressbar showing sentence progress. Defaults to None.
            te_batch_size (int, optional): Batch size for encoding documents. If None, encode all documents at once. Defaults to None.
            lm_batch_size ([type], optional): Batch size for encoding sentences in language model. If None, encode all sentences of all documents at once. Defaults to None.
            pooler_batch_size (int, optional): Batch size for processing cls token encodings. If None, process all cls tokens of the document. Defaults to None.

        Returns:
            torch.Tensor: Post sequence encoding of shape [seq_length, self._post_seq_encoder.input_size]
        """
        # text_encoding.shape: [seq_length, sum of text encoder dims]
        text_encoding = self._encode_text(post_sequence, pbar, te_batch_size, lm_batch_size, pooler_batch_size)
        # meta_embeddings.shape: [seq_length, n_metadata_variables]
        meta_embeddings = self._embed_metadata(post_sequence)
        return torch.cat([text_encoding, meta_embeddings], dim=1)

    def _encode_text(self, post_sequence: Dict[str, List], pbar=None, te_batch_size=None, lm_batch_size=None, pooler_batch_size=None):
        """Encode the text variables of the sequence

        Args:
            post_sequence (Dict[str, List]): Post sequence for one news source. Contains all features (feature is key, list is values). Each list has length seq_length.
            pbar (tqdm, optional): Progressbar showing sentence progress. Defaults to None.
            te_batch_size (int, optional): Batch size for encoding documents. If None, encode all documents at once. Defaults to None.
            lm_batch_size ([type], optional): Batch size for encoding sentences in language model. If None, encode all sentences of all documents at once. Defaults to None.
            pooler_batch_size (int, optional): Batch size for processing cls token encodings. If None, process all cls tokens of the document. Defaults to None.

        Returns:
            torch.Tensor: Tensor of text encodings, with shape [seq_len, sum(text_encoder.dim)]
        """
        encodings = []
        for key, text_encoder in sorted(self._text_encoders.items(), key=itemgetter(0)):
            id_col = key.split("|")[0] + "|id"
            unique_ids = list(set(post_sequence[id_col]))
            unique_docs = [post_sequence[key][post_sequence[id_col].index(id)] for id in unique_ids]
            te_batch_size = te_batch_size or len(unique_docs)
            doc_batches = [unique_docs[i:(i + te_batch_size)] for i in range(0, len(unique_docs), te_batch_size)]
            
            if self._has_cls_enc:
                doc_encodings = torch.cat([text_encoder(docs, pooler_batch_size) for docs in doc_batches])
            else:
                doc_encodings = torch.cat([text_encoder(docs, pbar, lm_batch_size, pooler_batch_size) for docs in doc_batches])
            # doc_indices has for each doc the index for doc_encodings
            doc_indices = [unique_ids.index(id) for id in post_sequence[id_col]]
            # Select for each doc the respective row from doc_encodings, some rows are repeatedly accessed
            encodings.append(doc_encodings[doc_indices])

        return torch.cat(encodings, dim=1)

    def _embed_metadata(self, post_sequence):
        """Embedd the remaining variables (besides text variables)

        Args:
            post_sequence (Dict[str, List]): Post sequence for one news source. Contains all features (feature is key, list is values). Each list has length seq_length.
            
        Returns:
            torch.Tensor: Tensor of text encodings, with shape [seq_len, n_features - sum(text_encoder.dim)]
        """
        raw_values = []
        for key, values in sorted(post_sequence.items(), key=itemgetter(0)):
            if key not in self._text_encoders and not key.endswith("|id"):
                raw_values.append(torch.as_tensor(values, dtype=torch.float32, device=self.device))

        # raw_values.shape: [seq_length, n_metadata_variables]
        raw_values = torch.stack(raw_values, dim=1)
        return self._meta_embedder(raw_values)

    def _aggregate_hidden_states(self, hidden_states_batch, seq_lens):
        """Aggregates the hidden states from each input of the sequence

        Args:
            hidden_states_batch (torch.Tensor): Padded hidden states of shape [batch_size, max(seq_lens), hidden_size]
            seq_lens (torch.Tensor): Tensor of sequence lengths of each hidden state sequence in the batch. Has dimension 1
        """
        if self._psm_output == "last_state":
            # return.shape: [batch_size, hidden_size]
            return self._last_hidden_state(hidden_states_batch, seq_lens)
        elif self._psm_output == "mean":
            # return.shape: [batch_size, hidden_size]
            return self._mean_hidden_state(hidden_states_batch, seq_lens)
        elif self._psm_output == "max":
            # return.shape: [batch_size, hidden_size]
            return self._max_hidden_state(hidden_states_batch, seq_lens)
        elif self._psm_output == "mean+max":
            # return.shape: [batch_size, 2*hidden_size]
            return torch.cat([
                    self._mean_hidden_state(hidden_states_batch, seq_lens), 
                    self._max_hidden_state(hidden_states_batch, seq_lens)
                ], dim=1)

    def _last_hidden_state(self, hidden_states_batch, seq_lens):
        """Gets the last hidden state from a padded tensor of hidden states

        Args:
            hidden_states_batch (torch.Tensor): Padded hidden states of shape [batch_size, max(seq_lens), hidden_size]
            seq_lens (torch.Tensor): Tensor of sequence lengths of each hidden state sequence in the batch. Has dimension 1

        Returns:
            torch.Tensor: Last hidden states extracted from hidden_states_batch, of shape [batch_size, hidden_size]
        """
        return torch.stack([hidden_states[seq_len - 1] 
            for hidden_states, seq_len in zip(hidden_states_batch, seq_lens)], dim=0)

    def _mean_hidden_state(self, hidden_states_batch, seq_lens):
        """Same as _last_hidden_state, but average hidden state
        """
        return torch.stack([hidden_states[:seq_len].mean(dim=0) 
            for hidden_states, seq_len in zip(hidden_states_batch, seq_lens)], dim=0)

    def _max_hidden_state(self, hidden_states_batch, seq_lens):
        """Same as _last_hidden_state, but max hidden state
        """
        return torch.stack([hidden_states[:seq_len].max(dim=0)[0] 
            for hidden_states, seq_len in zip(hidden_states_batch, seq_lens)], dim=0)
