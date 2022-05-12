# %%
import twitter_functions as twf
import json
import gzip
import os
from os.path import join
from tqdm import tqdm
from datetime import datetime, timedelta
from copy import deepcopy
import time

# %%
data_dir = '../data'
tweet_dst = join(data_dir, "tweet-comments", "tweets")
counts_dst = join(data_dir, "tweet-comments", "counts")
for dir in [tweet_dst, counts_dst]:
    if not os.path.exists(dir):
        os.makedirs(dir)

params_path = "params/search_tweets.json"

MAX_RESULTS = 100
ACADEMIC = True
with open("api-keys/jana.json") as f:
    api_keys = json.load(f)

config = {
    "bearer_token": api_keys["bearer-token"]
}

# %%
logs = []
with gzip.open(join(data_dir, "tweets.ndjson.gzip")) as f:
    for line in f:
        logs.append(json.loads(line.decode("utf-8")))
print(len(logs))
logs[0]

# %%
conversation_ids = []
for log in logs:
    if "data" in log["response"]:
        # Include only conversation ids, which have comments
        conversation_ids += [t["conversation_id"] for t in log["response"]["data"] 
            if t["public_metrics"]["reply_count"] + t["public_metrics"]["quote_count"] > 0]

conversation_ids = list(set(conversation_ids))
print(len(conversation_ids))
conversation_ids[:10]

# %%
with open(params_path) as f:
    default_params = json.load(f)

def build_query(default_query, conversation_ids):
    conv_id_clause = " OR ".join([f"conversation_id:{c_id}" for c_id in conversation_ids])
    return f"{default_query} ({conv_id_clause})"

def split_conv_id_ranges(conversation_ids, default_query):
    conv_id_ranges = []
    i = 0
    for j in range(len(conversation_ids)):
        query = build_query(default_query, conversation_ids[i:j+1])
        if len(query) > 1024:
            conv_id_ranges.append(conversation_ids[i:j])
            i = j

    return conv_id_ranges

conv_id_ranges = split_conv_id_ranges(conversation_ids, default_params["query"])
print(len(conv_id_ranges))
conv_id_ranges[0]

# %%
def build_tweet_params(conversation_ids, all=True):
    tweet_params = deepcopy(default_params)
    tweet_params["query"] = build_query(default_params["query"], conversation_ids)
    if all:
        tweet_params["start_time"] = datetime(2021, 4, 1).strftime(twf.DATE_FORMAT)
        tweet_params["end_time"] = datetime(2021, 10, 1).strftime(twf.DATE_FORMAT)
        tweet_params["max_results"] = MAX_RESULTS
    else:
        # query only few tweets for testing reasons
        end_time = (datetime.now() - timedelta(days=1))
        tweet_params["end_time"] = end_time.strftime(twf.DATE_FORMAT)
        tweet_params["start_time"] = (end_time - timedelta(days=5)).strftime(twf.DATE_FORMAT)
        tweet_params["max_results"] = 10

    return tweet_params

def build_count_params(conversation_ids, all=True):
    count_params = {
        "granularity": "day",
        "query": build_query(default_params["query"], conversation_ids)
    }
    if all:
        count_params["end_time"] = datetime(2021, 10, 1).strftime(twf.DATE_FORMAT)
        count_params["start_time"] = datetime(2021, 4, 1).strftime(twf.DATE_FORMAT)
    else:
        count_params["end_time"] = (datetime.now() - timedelta(days=1)).strftime(twf.DATE_FORMAT)

    return count_params

# %%
for i, conv_ids in enumerate(tqdm(conv_id_ranges)):
    config["params"] = build_count_params(conv_ids, all=ACADEMIC)
    write_path = join(counts_dst, f"{i}.json")
    config["write_path"] = write_path

    twf.validate_config(config)
    responses = []
    next_token = None
    while True:
        response, _ = twf.count_tweets(config, next_token, all=ACADEMIC)
        responses.append(response)
        if 'meta' in response and 'next_token' in response['meta']:
            next_token = response['meta']['next_token']
        else:
            break

    log = {"request": config["params"], "responses": responses}
    with open(write_path, "w") as f:
        json.dump(log, f)

# %%
total_count = 0
for filename in os.listdir(counts_dst):
    with open(join(counts_dst, filename)) as f:
        log = json.load(f)
    for resp in log["responses"]:
        total_count += resp["meta"]["total_tweet_count"]
total_count

# %%
for i, conv_ids in enumerate(tqdm(conv_id_ranges)):
    config["params"] = build_tweet_params(conv_ids, all=ACADEMIC)
    success_path = join(tweet_dst, "success", f"{i}.json.gzip")
    error_path = join(tweet_dst, "error", f"{i}.json")
    config["write_path"] = success_path

    twf.validate_config(config)
    response, error = twf.search_tweets(config, all=ACADEMIC)
    
    log = {"request": config["params"], "response": response}
    if error:
        with open(error_path, "w") as f:
            json.dump(log, f)
    else:
        with gzip.open(success_path, "w") as f:
            f.write(json.dumps(log).encode("utf-8"))

    # Full archive search only allows 1 request / 1 second
    if ACADEMIC:
        time.sleep(1)
