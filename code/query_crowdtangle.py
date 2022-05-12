# %%
from requests.api import request
import utils
import ct_functions as ctf

import json
import gzip
import os
from os.path import join
import pandas as pd
import requests
from datetime import datetime
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map

# %%
domains = pd.read_csv("../resources/domains/clean/domain_list_clean.csv")
domain_list = domains["url"].unique()

links_src = "../resources/ct-links"

# %%
check_status_codes = False

if check_status_codes:
    status_codes = thread_map(utils.check_url_status, domain_list, max_workers=os.cpu_count() * 4)
    status_codes = pd.DataFrame(status_codes)
    status_codes.to_csv("../resources/domains/clean/domain_status_codes.csv", index=False)

# %%
status_codes = pd.read_csv("../resources/domains/clean/domain_status_codes.csv")
print(status_codes.shape, status_codes.query("status_code == 200").shape)
status_codes.head()

# %%
# Query each URL once per month, max. 1000 responses
links = status_codes.query("status_code == 200")["url"].tolist()
for i in range(6):
    end_date = datetime(2021, 10 - i, 1)
    start_date = datetime(2021, 9 - i, 1)
    params = {
        "token": "9J4LRiuCxEO51I296LzpYByxieGg2heIyZVUYR7I",
        "endDate": end_date.strftime(ctf.DATE_QUERY_FORMAT),
        "startDate": start_date.strftime(ctf.DATE_QUERY_FORMAT),
        "sortBy": "date",
        "includeSummary": "true"
    }
    ctf.get_links(links, params, join(links_src, end_date.strftime("%Y-%m-%d")), max_queries_per_link=1)
