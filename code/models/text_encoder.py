import torch
from torch import nn
from torch.nn.utils.rnn import pack_sequence, PackedSequence
from transformers import AutoTokenizer, AutoModel, PreTrainedTokenizerBase, PreTrainedModel
from transformers.tokenization_utils_base import BatchEncoding
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import numpy as np
import pandas as pd

import os
from os.path import join
from typing import List, Dict, Tuple
from collections import defaultdict
import utils
from copy import deepcopy
from itertools import islice

class Text2ClsEncoder(nn.Module):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, lang_model: PreTrainedModel):
        """
        Args:
            tokenizer (PreTrainedTokenizerBase): Text tokenizer.
            lang_model (PreTrainedModel): Language model for encoding sentences.
        """
        super().__init__()
        
        self._tokenizer = tokenizer
        self._lang_model = lang_model
        
        with torch.no_grad():
            self._empty_doc_cls_encoding = self._lang_model(**self._tokenizer("", return_tensors="pt").to(self._lang_model.device))[0][:, :1].detach().clone()
    
    def forward(self, docs: List[str], pbar: tqdm, sent_batch_size=None, lm_batch_size=None, save_dir=None) -> List[torch.Tensor]:
        """Get encodings for CLS token (first token in sequence).

        Args:
            docs (List[str]): len(docs): Post sequence length. List of docs in a single post sequence.
            pbar (tqdm): Progressbar showing sentence progress
            sent_batch_size (int, optional): Number of sentences to tokenize. If None, tokenize all sentences. Defaults to None.
            lm_batch_size (int, optional): Batch size for the language model. Needed for preventing out-of-memory error. If None, do not batch. Defaults to None.
            save_dir (str, optional): Directory, where each sentence batch is stored. If None, do not store intermediate results. Defaults to None.

        Returns:
            List[torch.Tensor]: CLS token encoding for each document, same length as docs. Returned list at index 0 has tensor of shape [n_sentences0, ...], 
                where n_sentences0 is the number of sentences of docs[0].
        """
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        docs_df = pd.DataFrame()
        docs_df["sentences"] = [sent_tokenize(doc) for doc in docs]
        docs_df = docs_df.reset_index().rename(columns={"index": "doc_id"})
        sents_df = docs_df.explode("sentences", ignore_index=True).fillna({"sentences": ""})
        sents_df = sents_df.reset_index().rename(columns={"index": "sent_id"})
        
        sents_to_encode = sents_df[sents_df["sentences"].str.strip() != ""]
        if sents_to_encode.shape[0] == 0:
            return [self._empty_doc_cls_encoding.clone() for _ in docs]
        else:
            sents_to_encode["sequence_len_approx"] = sents_to_encode["sentences"].apply(lambda s: len(s.split()))
            sents_to_encode = sents_to_encode.sort_values(by="sequence_len_approx", ascending=False)
            it = iter(range(sents_to_encode.shape[0]))
            sent_batch_size = sent_batch_size or sents_to_encode.shape[0]
            cls_tok_encodings = []
            i = -1
            while True:
                i += 1
                i_batch = list(islice(it, sent_batch_size))
                if len(i_batch) == 0:
                    break

                sents_batch = sents_to_encode.iloc[i_batch]
                encodings_tensor = self._tokenize_and_encode_cls_token(sents_batch, pbar, lm_batch_size)
                cls_tok_encodings += encodings_tensor.chunk(encodings_tensor.shape[0])

                if save_dir is not None:
                    id2cls_enc = dict(zip(sents_batch["sent_id"].tolist(), cls_tok_encodings))
                    torch.save(id2cls_enc, join(save_dir, f"{str(i).zfill(3)}.pt"))
                    # Do not store all encodings
                    cls_tok_encodings = []

            empty_sents = sents_df[sents_df["sentences"].str.strip() == ""]
            empty_sents["cls_tok_encoding"] = [self._empty_doc_cls_encoding.clone() for _ in range(empty_sents.shape[0])]
            
            if save_dir is not None:
                id2cls_enc = dict(zip(empty_sents["sent_id"].tolist(), empty_sents["cls_tok_encoding"].tolist()))
                torch.save(id2cls_enc, join(save_dir, f"empty-sents.pt"))
                return
            
            sents_to_encode["cls_tok_encoding"] = cls_tok_encodings
            
            encodings_df = pd.concat([empty_sents[["doc_id", "sent_id", "cls_tok_encoding"]], sents_to_encode[["doc_id", "sent_id", "cls_tok_encoding"]]])
            tensor_series = sents_df.merge(encodings_df, how="left", on=["doc_id", "sent_id"]) \
                .sort_values(by=["doc_id", "sent_id"]) \
                .groupby("doc_id")["cls_tok_encoding"].apply(lambda t: torch.cat(t.tolist(), dim=0)) \
                .sort_index()
            return tensor_series.tolist()

    def _tokenize_and_encode_cls_token(self, sents_df, pbar, lm_batch_size=None) -> torch.Tensor:
        """Tokenizes sentences and compute their CLS token encoding.

        Args:
            sents_df (pd.DataFrame): Dataframe containing one sentence per row
            pbar (tqdm): Progressbar showing sentence progress
            lm_batch_size (int, optional): Batch size for the language model. Needed for preventing out-of-memory error. If None, do not batch. Defaults to None.
            
        Returns:
            torch.Tensor: Encoding of CLS token for each sentence. Has shape [sents_df.shape[0], 1, self._lang_model.hidden_size].
        """
        sent_embeddings = self._tokenizer(sents_df["sentences"].tolist(), padding="longest", truncation=True, return_tensors="pt")
        sent_embeddings_li = zip(*[sent_embeddings[key].chunk(sent_embeddings[key].shape[0]) for key in sent_embeddings.keys()])
        sent_embeddings_li = [{key: tensor for key, tensor in zip(sent_embeddings.keys(), tensors)} for tensors in sent_embeddings_li]

        sents_df["sent_embedding"] = sent_embeddings_li
        sents_df["sequence_len"] = [e["attention_mask"].sum().item() for e in sent_embeddings_li]
        sents_df = sents_df.sort_values(by="sequence_len", ascending=False)
        
        lm_batch_size = lm_batch_size or len(sent_embeddings_li)
        cls_tok_encodings = []
        sent_embedding_batches = self._concat_and_rebatch_embeddings(sent_embeddings_li, lm_batch_size)
        for sent_embeddings in sent_embedding_batches:
            with torch.no_grad():
                # Detach and clone encoding of first token to prevent RAM leak
                # last_hidden_state.shape: [lm_batch_size, max_position_embeddings, hidden_size]
                cls_tok_encodings.append(self._lang_model(**sent_embeddings)[0][:, :1].detach().clone())
                pbar.update(sent_embeddings["input_ids"].shape[0])
        
        return torch.cat(cls_tok_encodings, dim=0)

    def _concat_and_rebatch_embeddings(self, embeddings: List[BatchEncoding], lm_batch_size: int) -> List[Dict[str, torch.tensor]]:
        """Concatenates document embeddings, s.t. it can be handeled as a batch of sentences.

        Args:
            embeddings (List[BatchEncoding]): len(embeddings): number of documents. Contains for each document its encoded sentences as a batch.
            lm_batch_size (int): Batch size for the language model. Needed for preventing out-of-memory error.

        Returns:
            Dict[str, torch.tensor]: Dictionary with the same keys as BatchEncoding. Each tensor contains the encoding of all sentences over all documents.
        """
        concat_dict = defaultdict(list)
        for embedding in embeddings:
            for key, tensor in embedding.items():
                concat_dict[key].append(tensor.to(self._lang_model.device))

        for key, tensors in concat_dict.items():
            concat_dict[key] = torch.cat(tensors, dim=0)
        sequence_lengths = concat_dict["attention_mask"].sum(dim=1).tolist()

        sent_batch = []
        total_sentences = len(sequence_lengths)
        for i in range(0, total_sentences, lm_batch_size):
            max_sequence_len = max(sequence_lengths[i:(i + lm_batch_size)])
            sent_batch.append(
                {key: tensor[i:(i + lm_batch_size), :max_sequence_len] for key, tensor in concat_dict.items()}
            )

        return sent_batch


class Cls2DocEncoder(nn.Module):
    def __init__(self, pooler: nn.Module, conf: dict):
        """
        Args:
            pooler (nn.Module): Pooling layer, that takes encoding of the first token (CLS) as input.
            conf (dict): Model configuration
        """
        super().__init__()
        
        self._pooler = pooler
        self.dim = conf["te"]["embedding_size"] or self._pooler.dense.out_features
        self._te_type = conf["te"]["type"]

        if self._te_type == "mean":
            self._doc_encoder = None
        elif self._te_type == "LSTM":
            self._doc_encoder = nn.LSTM(
                input_size=self._pooler.dense.out_features,
                hidden_size=self.dim,
                num_layers=1,
                batch_first=True
            )
        else:
            raise Exception(f"te_type must be one of ['mean', 'LSTM'], but found {self._te_type}")

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, docs_cls_tok_encodings: List[torch.Tensor], pooler_batch_size=None) -> torch.Tensor:
        """Encodes a sequence of CLS token encodings to a single tensor.

        Args:
            docs_cls_tok_encodings (List[torch.Tensor]): CLS token encodings of each document (len(docs_cls_tok_encodings) == number documents). 
                Shape of tensor at index 0 is [n_sents0, ...], where n_sents0 is the number of sentences of the first document, etc.
            pooler_batch_size (int, optional): Batch size for processing cls token encodings. If None, process all cls tokens of the document. Defaults to None.

        Returns:
            torch.Tensor: Encoding of the input documents. Shape is [number documents, self.dim].
        """
        docs_cls_tok_encodings = [t.to(self.device) for t in docs_cls_tok_encodings]
        n_sentences_list = [t.shape[0] for t in docs_cls_tok_encodings]
        # cls_tok_encodings.shape: [sum(n_sentences_list), 1, hidden_size]
        cls_tok_encodings = torch.cat(docs_cls_tok_encodings, dim=0)
        cls_tok_encoding_batches = torch.tensor_split(cls_tok_encodings, torch.as_tensor(range(pooler_batch_size, cls_tok_encodings.shape[0], pooler_batch_size))) \
            if pooler_batch_size and pooler_batch_size < cls_tok_encodings.shape[0] else [cls_tok_encodings]

        sent_encodings = []
        for cls_tok_encodings_batch in cls_tok_encoding_batches:
            # sent_encodings[-1].shape: [pooler_batch_size, hidden_size]
            sent_encodings.append(self._pooler(cls_tok_encodings_batch))       
        # sent_encodings.shape: [sum(n_sentences_list), hidden_size]
        sent_encodings = torch.cat(sent_encodings, dim=0)
        
        docs_batch = self._reshape_to_docs_batch(sent_encodings, n_sentences_list)
        if self._te_type == "mean":
            # states_mean[0].shape: [hidden_size,]
            states_mean = [doc.mean(dim=0) for doc in docs_batch]
            # return.shape: [batch_size, hidden_size]
            return torch.stack(states_mean, dim=0)
        else:
            # Each document has different number of sentences, pack tensors with different sequence length (n_sentences) into one batch
            docs_batch_packed = pack_sequence(docs_batch, enforce_sorted=False)
            _, (hn, _) = self._doc_encoder(docs_batch_packed)
            # hn.shape: [num_layers, batch_size, self.dim]
            # hn contains final state for each layer. Only keep state of last layer
            return hn[-1]

    def _reshape_to_docs_batch(self, sent_encodings: torch.Tensor, n_sentences_list: List[int]) -> Tuple[torch.Tensor]:
        """Splits a tensor of sentence encodings into a batch of sentence encoding sequences.

        Args:
            sent_encodings (torch.Tensor): Encodings of all sentences, with shape [sum(n_sentences_list), ...].
            n_sentences_list (List[int]): len(n_sentences_list): Number of documents. List containing the number of sentences of each document. Empty documents have 1 dummy sentence.

        Returns:
            Tuple[torch.Tensor]: Sliced sent_encodings, tensor on index 0 has shape [n_sentences_list[0], ...] and so on. 
        """
        split_indices = np.cumsum(n_sentences_list).tolist()[:-1]
        return torch.tensor_split(sent_encodings, split_indices)

class TextEncoder(nn.Module):
    def __init__(self, text2cls_encder: Text2ClsEncoder, cls2doc_encoder: Cls2DocEncoder):
        super().__init__()
        self._text2cls_encoder = text2cls_encder
        self._cls2doc_encoder = cls2doc_encoder
        self.dim = self._cls2doc_encoder.dim

    def forward(self, docs: List[str], pbar: tqdm, sent_batch_size=None, lm_batch_size=None, pooler_batch_size=None) -> torch.Tensor:
        """Concatenates application of both submodules. First compute CLS token encoding of each sentence in a document. Second encode sequence of CLS token encodings, s.t. each document is encoded as single tensor.

        Args:
            docs (List[str]): len(docs): Post sequence length. List of docs in a single post sequence.
            pbar (tqdm): Progressbar showing sentence progress
            sent_batch_size (int, optional): Number of sentences to tokenize. If None, tokenize all sentences. Defaults to None.
            lm_batch_size (int, optional): Batch size for the language model. Needed for preventing out-of-memory error. If None, do not batch. Defaults to None.
            pooler_batch_size (int, optional): Batch size for processing cls token encodings. If None, process all cls tokens of the document. Defaults to None.

        Returns:
            torch.Tensor: Encoding of the input documents. Shape is [number documents, self.dim].
        """
        docs_cls_tok_encodings = self._text2cls_encoder(docs, pbar, sent_batch_size, lm_batch_size)
        return self._cls2doc_encoder(docs_cls_tok_encodings, pooler_batch_size)


class TextEncoderBuilder:
    def __init__(self, available_models):
        """Loads pretrained models once and build TextEncoder objects with it

        Args:
            available_models (List[str]): Model names to load
        """
        self._tokenizers = {}
        self._lang_models = {}
        self._poolers = {}
        for model_name in available_models:
            self._tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name)
            lang_model = AutoModel.from_pretrained(model_name)
            self._poolers[model_name] = lang_model.pooler
            lang_model.pooler = None
            self._lang_models[model_name] = lang_model.eval()

    def build_text_encoder(self, lang_model_name, device, conf):
        return TextEncoder(
            self.build_text2cls_encoder(lang_model_name, device), 
            self.build_cls2doc_encoder(lang_model_name, device, conf)
        )

    def build_text2cls_encoder(self, lang_model_name, device) -> Text2ClsEncoder:
        """Build a Text2ClsEncoder using single instance of language model

        Args:
            lang_model_name (str): Name of pretrained language model
            device (torch.device): Computation device (CPU or GPU).

        Returns:
            Text2ClsEncoder: Text2ClsEncoder
        """
        tokenizer = self._tokenizers[lang_model_name]
        lang_model = self._lang_models[lang_model_name].to(device)
        return Text2ClsEncoder(tokenizer, lang_model).to(device)

    def build_cls2doc_encoder(self, lang_model_name: str, device, conf: dict) -> Cls2DocEncoder:
        """Build a Text2ClsEncoder using cloned pooler layer of pretrained language models

        Args:
            lang_model_name (str): Name of pretrained language model
            device (torch.device): Computation device (CPU or GPU).
            conf: Model configuration

        Returns:
            Cls2DocEncoder: Cls2DocEncoder
        """
        pooler = deepcopy(self._poolers[lang_model_name]).to(device)    # TODO review wether weights are copied properly
        return Cls2DocEncoder(pooler, conf).to(device)