# %%
import utils
import json
import os
from os.path import join
from glob import glob
import random

from tqdm import trange
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

# %%
ct_dir = "/data/misinformation-domains/ct"
expanded_urls = []
for filepath in sorted(glob(join(ct_dir, "expanded-urls/part-*"))):
    with open(filepath) as f:
        for line in f:
            expanded_urls.append(line.strip())
len(expanded_urls)

# %%
random.seed(19)
random.shuffle(expanded_urls)

# %%
with open("params/proxy.json") as f:
    proxies = json.load(f)

n_cores = os.cpu_count()
partition_size = 100 * n_cores
for i in trange(0, len(expanded_urls), partition_size):
    url_partition = expanded_urls[i:(i + partition_size)]
    with ThreadPoolExecutor(max_workers=n_cores) as e:
        status_codes = list(e.map(utils.check_url_status, url_partition, (proxies for _ in url_partition)))
    status_codes = pd.DataFrame(status_codes).rename(columns={"url": "expanded_url"})
    status_codes.to_csv(join(ct_dir, f"preproc/url-status-codes-{i // partition_size}.csv.gzip"), index=False, compression="gzip")
