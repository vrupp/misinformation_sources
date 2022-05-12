import pandas as pd
import requests
from tqdm import tqdm
import torch

import re
from glob import glob
import os
from os.path import join
import json
import math
import gc

def clean_url(url):
    # reformat entries that have the domain after a general name in parantheses
    enclosed_parantheses = r".+\s\(([^\(\)]+)\)"
    if re.match(enclosed_parantheses, url):
        findings = re.findall(enclosed_parantheses, url)
        if len(findings) > 0:
            url = findings[-1]
    # trailing "/" and spaces
    url = url.strip('/').strip()
    # transform all domains to lowercase
    url = url.lower()
    # remove any white spaces
    url = url.replace(' ', '')
    # if present: remove the protocol
    if url.startswith(("http", "https")):
        url = url.split('//')[1]
    # remove "www." 
    url = url.replace('www.', '')
    return url

def check_url_status(url, proxies=None):
    def try_request(url, method, timeout=10, proxies=None):
        try:
            headers = {
                "User-Agent": "Firefox/93.0"
            }
            resp = requests.request(method, url, headers=headers, timeout=timeout, proxies=proxies)
            return resp, resp.status_code
        except:
            return None, 404

    if re.match(r"https?://.+", url, re.IGNORECASE):
        full_urls = [url]
    else:
        full_urls = [protocol + url for protocol in ["https://", "http://"]]
    
    for full_url in full_urls:
        resp, code = try_request(full_url, method="HEAD", proxies=proxies)
        if code in [301, 302]:
            # Handle redirects
            resp, code = try_request(full_url, method="GET", timeout=20, proxies=proxies)
        
        if resp:
            # Break after successful response
            break

    if resp is None:
        response_url = full_url
    else:
        response_url = resp.url

    return {"url": url, "response_url": response_url, "status_code": code}

def read_partitions_as_pd(dir, filepattern, filetype="csv", compression=None, dtype=None):
    if filetype not in ["csv", "json", "ndjson", "pkl"]:
        raise Exception(f"filetype must be one of ['csv', 'json', 'ndjson', 'pkl'], but found {filetype}")
    
    print(f"Read partitions in {dir}", flush=True)
    partitions = []
    for filepath in tqdm(glob(join(dir, filepattern))):
        if filetype == "csv":
            partitions.append(pd.read_csv(filepath, compression=compression, dtype=dtype))
        elif filetype in ["json", "ndjson"]:
            partitions.append(pd.read_json(filepath, lines=(filetype == "ndjson"), compression=compression, dtype=dtype))
        elif filetype == "pkl":
            partitions.append(pd.read_pickle(filepath, compression=compression))
    return pd.concat(partitions, ignore_index=True)

def create_idx_for_spark_partitions(dir, idx_col="queried_url"):
    index = {}
    for filename in tqdm(sorted(os.listdir(dir))):
        if filename.endswith(".ndjson"):
            with open(join(dir, filename)) as f:
                df_part = pd.DataFrame([json.loads(line) for line in f])
            for idx in df_part[idx_col].unique():
                key = f"{idx_col}={idx}"
                if key in index:
                    raise Exception(f"{key} appears in two partitions")
                index[key] = filename
    with open(join(dir, "index.json"), "w") as f:
        json.dump(index, f)

def write_pd_to_chunks(df, dir, filetype="json", idx_col="queried_url", n_chunks=None, chunk_size=None):
    if n_chunks and chunk_size:
        raise Exception("Define either n_chunks or chunk_size, but both given")
    if filetype not in ["json", "pkl"]:
        raise Exception(f"filetype must be one of ['json', 'pkl'], but found {filetype}")
    if not os.path.isdir(dir):
        os.makedirs(dir)

    values = sorted(df[idx_col].unique().tolist())
    chunks = []
    if n_chunks:
        chunk_size = math.ceil(len(values) / n_chunks)
    elif chunk_size:
        n_chunks = math.ceil(len(values) / chunk_size)
    else:
        n_chunks = len(values)
        chunk_size = 1
    
    for i in range(0, len(values), chunk_size):
        chunks.append(values[i:(i + chunk_size)])
    index = {}
    for i, chunk in enumerate(chunks):
        for idx in chunk:
            index[f"{idx_col}={idx}"] = f"part-{str(i).zfill(len(str(n_chunks)))}.{filetype}.gzip"
    print(f"Write partitions to {dir}, indexed by '{idx_col}'", flush=True)
    with open(join(dir, "index.json"), "w") as f:
        json.dump(index, f)
    for i, chunk in enumerate(tqdm(chunks)):
        df_part = df[df[idx_col].isin(chunk)]
        if filetype == "json":
            df_part.to_json(join(dir, f"part-{str(i).zfill(len(str(n_chunks)))}.{filetype}.gzip"), 
                orient="records", compression="gzip")
        elif filetype == "pkl":
            df_part.to_pickle(join(dir, f"part-{str(i).zfill(len(str(n_chunks)))}.{filetype}.gzip"), 
                compression="gzip")
            
# CUDA Debugging
def get_gc_tensors():
    """Get tensors allocated by garbage collector

    Returns:
        List[Tuple]: List of tuples with tensor size, type, and whether it requires gradients
    """
    li = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                li.append((obj.size(), obj.dtype, obj.requires_grad))
        except:
            pass
    return li

def gc_diff(gc_objs):
    """Compute difference (added and removed objects) from garbage collector logs
    """
    diff = []
    for i, last_elem in enumerate(tqdm(gc_objs)):
        if i + 1 < len(gc_objs):
            elem = gc_objs[i + 1]
            removed = [(last_elem["gc"].count(obj) - elem["gc"].count(obj), obj) for obj in set(last_elem["gc"]) if last_elem["gc"].count(obj) - elem["gc"].count(obj) > 0]
            added = [(elem["gc"].count(obj) - last_elem["gc"].count(obj), obj) for obj in set(elem["gc"]) if elem["gc"].count(obj) - last_elem["gc"].count(obj)]
            diff.append(dict({k: v for k, v in elem.items() if k != "gc"}, removed=removed, added=added))

    return diff

def get_used_ram():
    if torch.cuda.is_available():
        used_memory = torch.cuda.memory_allocated()
    else:
        _, used_memory, _ = map(int, os.popen('free -t -b').readlines()[-1].split()[1:])
    return round(used_memory / 2 ** 30, 4)