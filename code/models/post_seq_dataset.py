import torch
from torch.utils.data import Dataset
from nltk.tokenize import sent_tokenize
import pandas as pd
from tqdm import tqdm

from typing import Tuple, List, Dict
import os
from os.path import join
import json
import gzip

class PostSequenceDataset(Dataset):
    def __init__(self, gt_labels, label_col, shared_articles_tw, shared_articles_fb, art_conts_dir, tweets_dir, fb_posts_dir, tw_users_dir, fb_acc_dir, has_cls_enc=False):
        self._shared_articles_tw = shared_articles_tw
        self._shared_articles_fb = shared_articles_fb
        self.gt_labels = gt_labels.sort_values(by="gt|url", ignore_index=True)
        self.label_col = label_col
        
        self._has_cls_enc = has_cls_enc
        self._lazy_loaders = {
            "art_conts": LazyLoader(art_conts_dir, "pkl", compression="gzip"),
            "tweets": LazyLoader(tweets_dir, "pkl", compression="gzip", dtype={"id": str, "author_id": str}),
            "fb_posts": LazyLoader(fb_posts_dir, "pkl", compression="gzip", dtype={"id": str, "account.id": str}),
            "tw_users": LazyLoader(tw_users_dir, "pkl", compression="gzip", dtype={"id": str}),
            "fb_acc": LazyLoader(fb_acc_dir, "pkl", compression="gzip", dtype={"id": str})
        }

    def get_labels(self) -> pd.Series:
        return self.gt_labels[self.label_col]

    def get_classes(self) -> List[str]:
        return sorted(self.gt_labels[self.label_col.replace("_enc", "")].unique())

    def __len__(self) -> int:
        return self.gt_labels.shape[0]

    def get_by_url(self, url) -> Tuple[Dict, Dict]:
        idx = self.gt_labels.index[self.gt_labels["gt|url"] == url][0]
        return self[idx]

    def __getitem__(self, index) -> Tuple[Dict, Dict]:
        """Reads all necessary files, which belong to a news source specified by the index.

        Args:
            index (int): Index position of self.gt_labels

        Returns:
            Tuple[Dict, Dict]: Tuple representing X, y. 
                X contains the actual data, y the ground truth label.
                X is dict, representing a sequence of posts, where each key of the dict is one feature. 
        """
        gt = self.gt_labels.iloc[index]
        posts = self._get_posts_shared_url(gt["gt|url"])
        articles = self._lazy_loaders["art_conts"].load(gt["gt|url"]) \
            .add_prefix("art_cont|")
        merged = posts.merge(articles, left_on="shared_art|article_id", right_on="art_cont|id")
        
        merged[["post|n_comments", "post|n_likes", "post|n_shares", "user|n_followers"]] = \
            merged[["post|n_comments", "post|n_likes", "post|n_shares", "user|n_followers"]].fillna(0)
        merged["user|verified"] = merged["user|verified"].fillna(False)

        merged["post|created_at"] = pd.to_datetime(merged["post|created_at"], utc=True)
        merged = merged.sort_values(by="post|created_at")
        col_postfix = "_cls_enc" if self._has_cls_enc else ""
        text_dtype = object if self._has_cls_enc else str
        coltypes = {
            'post|id': str, 'post|n_comments': int, 'post|n_likes': int, 'post|n_shares': int, f'post|text{col_postfix}': text_dtype, 
            'user|id': str, 'user|verified': bool, f'user|description{col_postfix}': text_dtype, 'user|n_followers': int, 
            'art_cont|id': str, f'art_cont|meta_description{col_postfix}': text_dtype, f'art_cont|text{col_postfix}': text_dtype, f'art_cont|title{col_postfix}': text_dtype
        }
        merged = merged[coltypes.keys()].astype(coltypes)

        post_sequence = merged.to_dict(orient="list")
        labels = gt.to_dict()

        return post_sequence, labels

    def _get_posts_shared_url(self, url):
        tweets = self._get_tweets_shared_url(url)
        fb_posts = self._get_fb_posts_shared_url(url)

        col_postfix = "_cls_enc" if self._has_cls_enc else ""
        select_cols = ["post|id", "post|created_at", "post|n_comments", "post|n_likes", "post|n_shares", f"post|text{col_postfix}"] \
            + ["shared_art|article_id", "shared_art|queried_url"] \
            + ["user|id", "user|verified", f"user|description{col_postfix}", "user|n_followers"]
        tweets = tweets[select_cols]
        fb_posts = fb_posts[select_cols]
        
        return pd.concat([tweets, fb_posts], ignore_index=True)

    def _get_tweets_shared_url(self, url):
        col_postfix = "_cls_enc" if self._has_cls_enc else ""
        tweets = self._lazy_loaders["tweets"].load(url) \
            .add_prefix("tweet|") \
            .merge(self._shared_articles_tw, left_on="tweet|id", right_on="shared_art|tweet_id") \
            .rename(columns={
                "tweet|id": "post|id",
                "tweet|created_at": "post|created_at",
                "tweet|public_metrics.reply_count": "post|n_comments",
                "tweet|public_metrics.like_count": "post|n_likes",
                f"tweet|text{col_postfix}": f"post|text{col_postfix}"
            })
        tweets["post|n_shares"] = tweets[["tweet|public_metrics.quote_count", "tweet|public_metrics.retweet_count"]].sum(axis=1)
        
        tw_users = self._lazy_loaders["tw_users"].load(tweets["tweet|author_id"].unique().tolist()) \
            .add_prefix("tw_user|")
        
        return tweets \
            .merge(tw_users, left_on="tweet|author_id", right_on="tw_user|id") \
            .rename(columns={
                "tw_user|id": "user|id",
                "tw_user|verified": "user|verified",
                f"tw_user|description{col_postfix}": f"user|description{col_postfix}",
                "tw_user|public_metrics.followers_count": "user|n_followers"
            })

    def _get_fb_posts_shared_url(self, url):
        col_postfix = "_cls_enc" if self._has_cls_enc else ""
        fb_posts = self._lazy_loaders["fb_posts"].load(url) \
            .add_prefix("fb_post|") \
            .merge(self._shared_articles_fb, left_on="fb_post|id", right_on="shared_art|post_id") \
            .rename(columns={
                "fb_post|id": "post|id",
                "fb_post|date": "post|created_at",
                "fb_post|statistics.actual.commentCount": "post|n_comments",
                "fb_post|statistics.actual.shareCount": "post|n_shares",
                f"fb_post|message{col_postfix}": f"post|text{col_postfix}"
            })
        fb_posts["post|n_likes"] = fb_posts[
                [f"fb_post|statistics.actual.{e}Count" for e in ["angry", "care", "haha", "like", "love", "sad", "thankful", "wow"]]
            ].sum(axis=1)

        fb_accounts = self._lazy_loaders["fb_acc"].load(fb_posts["fb_post|account.id"].unique().tolist()) \
            .add_prefix("fb_acc|")
        
        return fb_posts \
            .merge(fb_accounts, left_on="fb_post|account.id", right_on="fb_acc|id") \
            .rename(columns={
                "fb_acc|id": "user|id",
                "fb_acc|verified": "user|verified",
                f"fb_acc|pageDescription{col_postfix}": f"user|description{col_postfix}",
                "fb_acc|subscriberCount": "user|n_followers"
            })

class LazyLoader():
    def __init__(self, dir: str, filetype: str, compression=None, dtype=None):
        """[summary]

        Args:
            dir (str): Directory of partitions
            filetype (str): Filetype of partitions
            compression (str, optional): Compression method used by pandas. Defaults to None.
            dtype (optional): Dtype argument used by pandas. Defaults to None.
        """
        self._dir = dir
        with open(join(dir, "index.json")) as f:
            self._idx = json.load(f)
        
        self._filetype = filetype
        self._compression = compression
        self._dtype = dtype
        
        self._idx_col = list(self._idx.keys())[0].split("=")[0]
        self._columns = self.load(list(self._idx.keys())[0].split("=")[-1]).columns

    def load(self, keys, verbose=False) -> pd.DataFrame:
        """Reads partitions from disk which are indexed by given keys.

        Args:
            keys (str | list): Single key or list of keys, which are the index values
            verbose (bool, optional): Whether to log read files. Defaults to False.

        Raises:
            Exception: If filetype is unknown

        Returns:
            pd.DataFrame: DataFrame containing only rows having the given keys.
        """
        if type(keys) != list:
            keys = [keys]
        
        filepaths = set([join(self._dir, self._idx[f"{self._idx_col}={key}"]) for key in keys if f"{self._idx_col}={key}" in self._idx])
        if len(filepaths) == 0:
            return pd.DataFrame(columns=self._columns)
        
        if verbose:
            print(f"Read {self._filetype} files", self._dir, flush=True)
            filepaths = tqdm(filepaths)
        if self._filetype in ["json", "ndjson"]:
            dfs = [pd.read_json(filepath, lines=self._filetype == "ndjson", compression=self._compression, dtype=self._dtype) 
                    for filepath in filepaths]
            df = pd.concat(dfs, ignore_index=True)
        elif self._filetype == "pkl":
            dfs = [pd.read_pickle(filepath, compression=self._compression) for filepath in filepaths]
            df = pd.concat(dfs, ignore_index=True)
            if self._dtype is not None:
                df = df.astype(self._dtype)
        else:
            raise Exception(f"Unknown filetype '{self._filetype}'")

        return df[df[self._idx_col].isin(keys)]