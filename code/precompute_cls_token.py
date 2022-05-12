# %%
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import nltk
from nltk.tokenize import sent_tokenize

import utils
import os
from os.path import join
from glob import glob
from models import TextEncoderBuilder

pd.set_option("display.max_columns", None)
nltk.download("punkt")

# %% [markdown]
# # Load data

# %%
final_dir = "/data/misinformation-domains/final"
# final_dir = ".." + final_dir

n_sent_arts = pd.read_csv(join(final_dir, "sentence_counts/articles.csv"))
n_sent_tw = pd.read_csv(join(final_dir, "sentence_counts/tweets.csv"))
n_sent_fb_posts = pd.read_csv(join(final_dir, "sentence_counts/fb_posts.csv"))
n_sent_tw_user = pd.read_csv(join(final_dir, "sentence_counts/tw_users.csv"))
n_sent_fb_accs = pd.read_csv(join(final_dir, "sentence_counts/fb_accs.csv"))

final_articles = utils.read_partitions_as_pd(join(final_dir, "with-text/articles"), "part-*.json.gzip", filetype="json", compression="gzip", dtype=False)
final_tweets = utils.read_partitions_as_pd(join(final_dir, "with-text/tweets"), "part-*.json.gzip", filetype="json", compression="gzip", dtype=False)
final_fb_posts = utils.read_partitions_as_pd(join(final_dir, "with-text/fb_posts"), "part-*.json.gzip", filetype="json", compression="gzip", dtype=False)

final_tw_users = utils.read_partitions_as_pd(join(final_dir, "with-text/tw_users"), "part-*.json.gzip", filetype="json", compression="gzip", dtype=False)
final_fb_accs = utils.read_partitions_as_pd(join(final_dir, "with-text/fb_accs"), "part-*.json.gzip", filetype="json", compression="gzip", dtype=False)

# %% [markdown]
# # Precompute CLS Token encodings
# Each Sentence from each text document is pre encoded

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
te_builder = TextEncoderBuilder(available_models=["roberta-base"])
roberta = te_builder.build_text2cls_encoder("roberta-base", device)

# %%
total_sentences = n_sent_arts[["title", "text", "meta_description"]].sum().sum() \
    + n_sent_tw["text"].sum() + n_sent_fb_posts["message"].sum() \
    + n_sent_tw_user["description"].sum() + n_sent_fb_accs["pageDescription"].sum()

with tqdm(total=total_sentences, desc="Sentence Encoding") as pbar:
    for col in ["title", "text", "meta_description"]:
        roberta(final_articles[col].fillna("").tolist(), pbar, sent_batch_size=512 * 20, lm_batch_size=512, save_dir=join(final_dir, "cls_encodings", col))
    
    for df, doc_col, filename in [(final_tweets, "text", "tw_text"), (final_fb_posts, "message", "fb_post_text")] \
        + [(final_tw_users, "description", "tw_user_text"), (final_fb_accs, "pageDescription", "fb_user_text")]:
        
        roberta(df[doc_col].fillna("").tolist(), pbar, sent_batch_size=512 * 20, lm_batch_size=512, save_dir=join(final_dir, "cls_encodings", filename))

# %%
def merge_sents2docs(sents_dir, df, doc_col, filename):
    docs_df = pd.DataFrame()
    docs_df["sentences"] = [sent_tokenize(doc) for doc in tqdm(df[doc_col].fillna("").tolist(), desc="Tokenize")]
    docs_df = docs_df.reset_index().rename(columns={"index": "doc_id"})
    sents_df = docs_df.explode("sentences", ignore_index=True).fillna({"sentences": ""})
    sents_df = sents_df.reset_index().rename(columns={"index": "sent_id"})

    dfs = []
    for filepath in tqdm(sorted(glob(join(final_dir, "cls_encodings", sents_dir, "*.pt"))), desc="Read parts"):
        part = torch.load(filepath, map_location=device)
        dfs.append(pd.DataFrame(part.items(), columns=["sent_id", "cls_tok_encoding"]))
    dfs = pd.concat(dfs)

    tensor_series = sents_df[["doc_id", "sent_id"]].merge(dfs, how="left", on="sent_id") \
        .sort_values(by=["doc_id", "sent_id"]) \
        .groupby("doc_id")["cls_tok_encoding"].apply(lambda t: torch.cat(t.tolist(), dim=0)) \
        .sort_index()
    cls_encodings = tensor_series.tolist()
    id2cls_enc = dict(zip(df["id"].tolist(), cls_encodings))
    torch.save(id2cls_enc, join(final_dir, f"cls_encodings/{filename}"))

merge_sents2docs("title", final_articles, "title", "art_title.pt")
merge_sents2docs("text", final_articles, "text", "art_text.pt")
merge_sents2docs("meta_description", final_articles, "meta_description", "art_meta_description.pt")
merge_sents2docs("tw_text", final_tweets, "text", "tw_text.pt")
merge_sents2docs("fb_post_text", final_fb_posts, "message", "fb_post_text.pt")
merge_sents2docs("tw_user_text", final_tw_users, "description", "tw_user_text.pt")
merge_sents2docs("fb_user_text", final_fb_accs, "pageDescription", "fb_user_text.pt")

# %% [markdown]
# ## Replace text columns with encodings

# %%
art_title_enc = torch.load(join(final_dir, "cls_encodings/art_title.pt"), map_location=device)
art_text_enc = torch.load(join(final_dir, "cls_encodings/art_text.pt"), map_location=device)
art_meta_description_enc = torch.load(join(final_dir, "cls_encodings/art_meta_description.pt"), map_location=device)

final_articles_cls = final_articles.merge(pd.DataFrame(art_title_enc.items(), columns=["id", "title_cls_enc"]), on="id", how="left") \
    .merge(pd.DataFrame(art_text_enc.items(), columns=["id", "text_cls_enc"]), on="id", how="left") \
    .merge(pd.DataFrame(art_meta_description_enc.items(), columns=["id", "meta_description_cls_enc"]), on="id", how="left") \
    .drop(columns=["title", "text", "meta_description"])
utils.write_pd_to_chunks(final_articles_cls, join(final_dir, "with-cls/articles"), filetype="pkl")

for df, filename, col, subdir in [(final_tweets, "tw_text.pt", "text", "tweets"), 
    (final_fb_posts, "fb_post_text.pt", "message", "fb_posts"), 
    (final_tw_users, "tw_user_text.pt", "description", "tw_users"), 
    (final_fb_accs, "fb_user_text.pt", "pageDescription", "fb_accs")]:

    cls_encoding = torch.load(join(final_dir, f"cls_encodings/{filename}"), map_location=device)
    df_cls = df.merge(pd.DataFrame(cls_encoding.items(), columns=["id", f"{col}_cls_enc"]), on="id", how="left") \
        .drop(columns=[col])
    idx_col = "queried_url" if "queried_url" in df_cls.columns else "id"
    chunk_size = None if idx_col == "queried_url" else 100
    utils.write_pd_to_chunks(df_cls, join(final_dir, f"with-cls/{subdir}"), filetype="pkl", idx_col=idx_col, chunk_size=chunk_size)
