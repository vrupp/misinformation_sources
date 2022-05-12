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
data_dir = '../resources/twitter'
tweet_dst = join(data_dir, 'tweets')
params_path = "params/search_tweets.json"

ACADEMIC = True
with open("api-keys/jana.json") as f:
    api_keys = json.load(f)

config = {
    "bearer_token": api_keys["bearer-token"]
}

# %%
MAX_RESULTS = 100
with open(join(data_dir, f"queries-{MAX_RESULTS}.json")) as f:
    queries = json.load(f)
with open(params_path) as f:
    default_params = json.load(f)

def build_tweet_params(query, all=True):
    tweet_params = deepcopy(default_params)
    url_clause = " OR ".join([f'url:"{url}"' for url in query["urls"]])
    tweet_params["query"] = f"{tweet_params['query']} ({url_clause})"

    if all:
        tweet_params["start_time"] = query["start"]
        tweet_params["end_time"] = query["end"]
        tweet_params["max_results"] = MAX_RESULTS
    else:
        # query only few tweets for testing reasons
        twitter_dateformat = "%Y-%m-%dT%H:%M:%SZ"
        end_time = (datetime.now() - timedelta(days=1))
        tweet_params["end_time"] = end_time.strftime(twitter_dateformat)
        tweet_params["start_time"] = (end_time - timedelta(days=5)).strftime(twitter_dateformat)
        tweet_params["max_results"] = 10
    return tweet_params

# %%
for i, query in enumerate(tqdm(queries)):
    config["params"] = build_tweet_params(query, all=ACADEMIC)
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

# %%
# Collect all json dumps in a single ndjson file (newline delimited json)
with gzip.open(join(data_dir, "tweets.ndjson.gzip"), "w") as f_out:
    for i in range(len(queries)):
        with gzip.open(join(tweet_dst, "success", f"{i}.json.gzip")) as f_in:
            log_bytes = f_in.read()
        
        f_out.write(log_bytes + "\n".encode("utf-8"))
