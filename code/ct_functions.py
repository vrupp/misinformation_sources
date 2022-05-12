import pandas as pd
import requests
import json
import gzip
from tqdm import tqdm
import os
import time
from datetime import datetime, timedelta
from os.path import join
from copy import deepcopy

BASE_URL = "https://api.crowdtangle.com"
DATE_QUERY_FORMAT = "%Y-%m-%dT%H:%M:%S"
_next_quota_reset = datetime.min

def get_posts(params, save_dir, max_queries=None):
    params["count"] = 100
    _query_endpoint("/posts", params, save_dir, max_queries=max_queries)

def get_links(links, params, save_dir, max_queries=None, max_queries_per_link=None):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    params["count"] = 1000
    total_queries = 0
    for i, link in enumerate(tqdm(links)):
        params["link"] = link
        n_queries = _query_endpoint("/links", params, 
            save_path=join(save_dir, f"link{i}.ndjson.gzip"), 
            max_queries=max_queries_per_link or max_queries - total_queries)

        total_queries += n_queries
        if max_queries is not None and total_queries >= max_queries:
            break
        
def _query_endpoint(endpoint, params, save_path, max_queries=None):
    global _next_quota_reset

    n_queries = 0
    params = deepcopy(params)
    
    end_date = datetime.strptime(params["endDate"], DATE_QUERY_FORMAT)
    start_date = datetime.strptime(params["startDate"], DATE_QUERY_FORMAT)
    
    while start_date < end_date:
        if max_queries and n_queries >= max_queries:
            break

        params["endDate"] = end_date.strftime(DATE_QUERY_FORMAT)
        now = datetime.now()
        if now >= _next_quota_reset:
            _next_quota_reset = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
        resp = requests.get(BASE_URL + endpoint, params=params)
        
        if resp.status_code == 429:
            seconds = (_next_quota_reset - datetime.now()).total_seconds()
            if seconds > 0:
                print(f"Sleep {seconds} seconds...", flush=True)
                time.sleep(seconds)
            
            _next_quota_reset += timedelta(minutes=1)
            # Request again
            continue

        n_queries += 1
        result = resp.json().get("result")
        log = {"request": params, "response": result}
        if resp.status_code != 200 or not result:
            log["error"] = resp.json()
        with gzip.open(save_path, "a") as f:
            s = json.dumps(log) + "\n"
            f.write(s.encode("utf-8"))
        
        if resp.status_code == 200 and result and "nextPage" in result["pagination"]:
            # paginate with start_date/end_date since offset parameter has a max value
            end_date = datetime.strptime(result["posts"][-1]["date"], "%Y-%m-%d %H:%M:%S")    # date has no "T" separator
        else:
            break

    return n_queries
