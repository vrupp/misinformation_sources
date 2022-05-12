# %%
import utils

import os
from os.path import join
import json
from flatten_json import flatten
import gzip
import pandas as pd
import re
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map
import random

pd.set_option("display.max_colwidth", None)
tweets_dir = "/data/misinformation-domains/twitter"
preproc_dir = join(tweets_dir, "preproc")

# %%
logs = []
with gzip.open(join(tweets_dir, "tweets.ndjson.gzip"), "rt") as f:
    for line in f:
        logs.append(json.loads(line))

# %%
def extract_queried_urls(query):
    return re.findall(r'url:"([^"]+)"', query)

def flatten_tweet(resp_tweet, root_keys_to_ignore):
    flat_tweet = flatten(resp_tweet, ".", root_keys_to_ignore=root_keys_to_ignore)
    for key in root_keys_to_ignore:
        if key in resp_tweet:
            flat_tweet[key] = resp_tweet[key]

    return flat_tweet

def match_response_urls(response_urls, tweet):
    matches = []
    for resp_url in response_urls:
        for q_url in tweet["queried_urls"]:
            if q_url.lower() in resp_url.lower() and q_url != resp_url.lower():
                # only include articles (queried url not identical to the response url)
                matches.append({"tweet_id": tweet["id"], "queried_url": q_url, "response_url": resp_url})
                break
    return matches

# %%
flattened_tweets = []
context_annotations = []
entities = []
users = []
user_ids = set()
places = []
place_ids = set()
keys_to_ignore = ["context_annotations", "entities"]
for log in tqdm(logs):
    queried_urls = extract_queried_urls(log["request"]["query"])

    for resp_tweet in log["response"]["data"]:
        flat_tweet = flatten_tweet(resp_tweet, keys_to_ignore)
        flat_tweet["queried_urls"] = queried_urls
    
        context_annotations.append(flat_tweet.pop("context_annotations", []))
        entities.append(flat_tweet.pop("entities", []))
        flattened_tweets.append(flat_tweet)

    for user in log["response"]["includes"].get("users", []):
        if user["id"] not in user_ids:
            users.append(flatten(user, "."))
            user_ids.add(user["id"])
    for place in log["response"]["includes"].get("places", []):
        if place["id"] not in place_ids:
            places.append(place)
            place_ids.add(place["id"])

# %%
expanded_urls = []
for entity in tqdm(entities):
    expanded_urls += [url["expanded_url"] for url in entity["urls"]]
expanded_urls = sorted(list(set(expanded_urls)))

random.seed(19)
random.shuffle(expanded_urls)

# %%
with open("params/proxy.json") as f:
    proxies = json.load(f)

status_codes = thread_map(utils.check_url_status, expanded_urls, (proxies for _ in expanded_urls), max_workers=os.cpu_count())
status_codes = pd.DataFrame(status_codes).rename(columns={"url": "expanded_url"})
status_codes.to_csv(join(preproc_dir, "url-status-codes.csv"), index=False)

# %%
status_codes = pd.read_csv(join(preproc_dir, "url-status-codes.csv")).set_index("expanded_url")
url_matches = []
for flat_tweet, entity in tqdm(zip(flattened_tweets, entities), total=len(entities)):
    response_urls = []
    for url in entity["urls"]:
        response_url = status_codes.loc[url["expanded_url"]]["response_url"]
        response_urls.append(response_url)
    
    url_matches_temp = match_response_urls(response_urls, flat_tweet)
    url_matches += url_matches_temp
    flat_tweet["queried_urls"] = list(set([match["queried_url"] for match in url_matches_temp]))

url_matches = pd.DataFrame(url_matches)
url_matches = url_matches[-url_matches.duplicated(keep="first")]
url_matches = url_matches.merge(status_codes.reset_index(), on="response_url", how="left") \
    .drop(columns=["expanded_url"])
assert url_matches["status_code"].isna().any() == False

# %%
tweets_df = pd.DataFrame(flattened_tweets) \
    .explode("queried_urls") \
    .rename(columns={"queried_urls": "queried_url"}) \
    .drop_duplicates(subset=["id", "queried_url"]) \
    .dropna(subset=["queried_url"])

users_df = pd.DataFrame(users)

# %%
url_matches.to_csv(join(preproc_dir, "shared-url-in-tweet.csv.gzip"), index=False, compression="gzip")
utils.write_pd_to_chunks(tweets_df, join(preproc_dir, "tweets"), idx_col="queried_url")

with gzip.open(join(preproc_dir, "context-annotations.json.gzip"), "wt") as f:
    json.dump(context_annotations, f)

with gzip.open(join(preproc_dir, "entities.json.gzip"), "wt") as f:
    json.dump(entities, f)

utils.write_pd_to_chunks(users_df, join(preproc_dir, "users"), idx_col="id", n_chunks=4096)

with open(join(preproc_dir, "places.json"), "w") as f:
    json.dump(places, f)
