# %%
import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize
import torch
from tqdm.notebook import tqdm

import utils
import os
from os.path import join
from glob import glob
from models import TextEncoderBuilder

pd.set_option("display.max_columns", None)

# %%
data_dir = "/data/misinformation-domains/"
art_conts_dir = join(data_dir, "articles/all/content")
shared_articles_tw_path = join(data_dir, "twitter/preproc/shared-articles-in-tweet.csv")
shared_articles_fb_path = join(data_dir, "ct/preproc/shared-articles-in-post.csv")
tweets_dir = join(data_dir, "twitter/preproc/tweets")
fb_posts_dir = join(data_dir, "ct/preproc/posts")
tw_users_dir = join(data_dir, "twitter/preproc/users")
fb_acc_dir = join(data_dir, "ct/preproc/accounts")

final_dir = join(data_dir, "final")

# %% [markdown]
# # Filter websites
# Keep only webistes with minimum 5 articles, where each contains text and each was shared in min. 5 posts

# %%
all_articles = utils.read_partitions_as_pd(art_conts_dir, "part-*.json.gzip", filetype="json", compression="gzip", dtype=False)
# Keep only articles having title, text, and meta_description
articles_with_content = all_articles[
        (all_articles[["text", "title", "meta_description"]] != "").all(axis=1)
    ][["id", "queried_url", "url"]]

twitter_urls = pd.read_csv(join(data_dir, "twitter/preproc/shared-url-in-tweet.csv.gzip"), compression="gzip", dtype=str)
ct_urls = utils.read_partitions_as_pd(join(data_dir, "ct/preproc/shared-url-in-post"), "*.csv", dtype=str)

articles_tw = articles_with_content.merge(twitter_urls, left_on="url", right_on="response_url", suffixes=("", "_y"))[["id", "queried_url", "tweet_id"]]
articles_ct = articles_with_content.merge(ct_urls, left_on="url", right_on="response_url", suffixes=("", "_y"))[["id", "queried_url", "post_id"]]

tw_post_counts = articles_tw.groupby(["queried_url", "id"]).nunique() \
    .rename(columns={"tweet_id": "n_posts"})["n_posts"]
ct_post_counts = articles_ct.groupby(["queried_url", "id"]).nunique() \
    .rename(columns={"post_id": "n_posts"})["n_posts"]

post_count = tw_post_counts.add(ct_post_counts, fill_value=0).astype(int)
# Only keep articles with min. 5 shares on social media
post_count = post_count[post_count >= 5]
article_count = post_count.reset_index() \
    .groupby("queried_url").nunique() \
    .rename(columns={"id": "n_articles"})["n_articles"]
# Only keep websites with min. 5 articles, each having min. 5 posts
article_count = article_count[article_count >= 5]
articles_keep = post_count.loc[article_count.index].reset_index()[["queried_url", "id"]]

shared_articles_tw = articles_tw.merge(articles_keep).rename(columns={"id": "article_id"}) \
    .drop_duplicates()
shared_articles_ct = articles_ct.merge(articles_keep).rename(columns={"id": "article_id"}) \
    .drop_duplicates()

# %% [markdown]
# # Final articles

# %%
article_ids = set(shared_articles_tw["article_id"].unique().tolist() + shared_articles_ct["article_id"].unique().tolist())
final_articles = all_articles[all_articles["id"].isin(article_ids)] \
    [["queried_url", "id", "publish_date", "title", "text", "meta_description"]]

# %% [markdown]
# # Final Twitter data

# %%
tweet_ids = shared_articles_tw["tweet_id"].unique()
all_tweets = utils.read_partitions_as_pd(tweets_dir, "part-*.json.gzip", filetype="json", compression="gzip", dtype=False)
final_tweets = all_tweets[all_tweets["id"].isin(tweet_ids)] \
    [["queried_url", "id", "created_at", "text", "author_id", "public_metrics.reply_count", "public_metrics.like_count", "public_metrics.quote_count", "public_metrics.retweet_count"]]

tw_user_ids = final_tweets["author_id"].unique()
all_tw_users = utils.read_partitions_as_pd(tw_users_dir, "part-*.json.gzip", filetype="json", compression="gzip", dtype=False)
final_tw_users = all_tw_users[all_tw_users["id"].isin(tw_user_ids)] \
    [["id", "verified", "description", "created_at", "public_metrics.followers_count"]]

# %% [markdown]
# # Final Facebook data

# %%
post_ids = shared_articles_ct["post_id"].unique()
all_posts = utils.read_partitions_as_pd(fb_posts_dir, "part-*.ndjson", filetype="ndjson", dtype=False)
final_fb_posts = all_posts[all_posts["id"].isin(post_ids)] \
    [["queried_url", "id", "date", "message", "account.id"] 
        + [f"statistics.actual.{e}Count" for e in ["comment", "share", "angry", "care", "haha", "like", "love", "sad", "thankful", "wow"]]]

fb_acc_ids = final_fb_posts["account.id"].unique()
all_fb_accs = utils.read_partitions_as_pd(fb_acc_dir, "part-*.ndjson", filetype="ndjson", dtype=False)
final_fb_accs = all_fb_accs[all_fb_accs["id"].isin(fb_acc_ids)] \
    [["id", "verified", "pageDescription", "subscriberCount", "pageCreatedDate"]]

# %% [markdown]
# # Update shared articles DF by posting user

# %%
shared_articles_tw = shared_articles_tw.merge(final_tweets[["id", "author_id"]].drop_duplicates(), how="left", left_on="tweet_id", right_on="id").drop(columns="id")
shared_articles_ct = shared_articles_ct.merge(final_fb_posts[["id", "account.id"]].drop_duplicates(), how="left", left_on="post_id", right_on="id").drop(columns="id")

# %% [markdown]
# # Count sentences per object

# %%
def count_sentences(df, cols):
    df = df[["id"] + cols].drop_duplicates(subset=["id"])
    df[cols] = df[cols].fillna("")
    return df.apply(lambda row: pd.Series({col: (len(sent_tokenize(row[col])) if row[col].strip() != "" else 0) if col != "id" else row[col] 
        for col in ["id"] + cols}), axis=1)

n_sent_arts = count_sentences(final_articles, ["title", "text", "meta_description"])
n_sent_tw = count_sentences(final_tweets, ["text"])
n_sent_fb_posts = count_sentences(final_fb_posts, ["message"])
n_sent_tw_user = count_sentences(final_tw_users, ["description"])
n_sent_fb_accs = count_sentences(final_fb_accs, ["pageDescription"])

# %% [markdown]
# # Write final datasets

# %%
shared_articles_tw.to_csv(join(final_dir, "shared-articles-in-tweet.csv"), index=False)
shared_articles_ct.to_csv(join(final_dir, "shared-articles-in-fb-post.csv"), index=False)

# with-text
utils.write_pd_to_chunks(final_articles, join(final_dir, "with-text/articles"))
utils.write_pd_to_chunks(final_tweets, join(final_dir, "with-text/tweets"))
utils.write_pd_to_chunks(final_fb_posts, join(final_dir, "with-text/fb_posts"))

utils.write_pd_to_chunks(final_tw_users, join(final_dir, "with-text/tw_users"), idx_col="id", chunk_size=100)
utils.write_pd_to_chunks(final_fb_accs, join(final_dir, "with-text/fb_accs"), idx_col="id", chunk_size=100)

# Sentence counts
n_sent_arts.to_csv(join(final_dir, "sentence_counts/articles.csv"), index=False)
n_sent_tw.to_csv(join(final_dir, "sentence_counts/tweets.csv"), index=False)
n_sent_fb_posts.to_csv(join(final_dir, "sentence_counts/fb_posts.csv"), index=False)
n_sent_tw_user.to_csv(join(final_dir, "sentence_counts/tw_users.csv"), index=False)
n_sent_fb_accs.to_csv(join(final_dir, "sentence_counts/fb_accs.csv"), index=False)
