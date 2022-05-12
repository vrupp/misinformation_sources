# %%
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from tqdm.contrib.concurrent import process_map, thread_map
from multiprocessing.managers import SyncManager
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from newspaper import Article
from flatten_json import flatten

import utils
import os
from os.path import join
import json
import gzip
from glob import glob
import itertools

pd.set_option("display.max_columns", None)

# %% [markdown]
# # Load shared URLs

# %%
data_dir = "/data/misinformation-domains"
articles_dir = join(data_dir, "articles/all")

# %%
twitter_urls = pd.read_csv(join(data_dir, "twitter/preproc/shared-url-in-tweet.csv.gzip"), compression="gzip")
print(twitter_urls.shape)
twitter_urls.head()

# %%
ct_urls = utils.read_partitions_as_pd(join(data_dir, "ct/preproc/shared-url-in-post"), "*.csv")
print(ct_urls.shape)
ct_urls.head()

# %%
all_urls = pd.concat([twitter_urls.drop(columns=["tweet_id"]), ct_urls.drop(columns=["post_id"])], ignore_index=True) \
    .drop_duplicates(subset=["response_url", "status_code"]) \
    .query("status_code == 200")
unique_urls = all_urls["response_url"].sort_values().unique()

np.random.seed(19)
np.random.shuffle(unique_urls)
unique_urls.shape

# %% [markdown]
# # Download HTML

# %%
def download_article(url, proxies=None):
    try:
        article = Article(url, request_timeout=20, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:94.0) Gecko/20100101 Firefox/94.0"}, proxies=proxies)
        article.download()
        return article.html, False
    except Exception as e:
        return e, True

with open("params/proxy.json") as f:
    proxies = json.load(f)

n_workers = os.cpu_count() * 2
partition_size = 50 * n_workers

# %%
if False:
    print("Start download", flush=True)
    for i in trange(0, len(unique_urls), partition_size):
        start, end = i, i + partition_size
        url_partition = unique_urls[start:end]
        html_downloads = thread_map(download_article, url_partition, (proxies for _ in url_partition), max_workers=n_workers)
        htmls = [{"article_id": j, "url": url, "html": str(html), "error": error} for j, url, (html, error) in zip(range(start, end), url_partition, html_downloads)]
        with gzip.open(join(articles_dir, f"html/part-{i // partition_size}.ndjson.gzip"), "wt") as f:
            for html in htmls:
                f.write(json.dumps(html) + "\n")

# %% [markdown]
# # Parse HTML to article

# %%
def parse_line(html_dump, queried_urls):
    article = Article(html_dump["url"], fetch_images=False)
    article.download(input_html=html_dump["html"])

    article.parse()
    article_content = {
        "id": html_dump["article_id"],
        "queried_url": queried_urls[html_dump["url"]],
        "source_url": article.source_url,
        "url": html_dump["url"],
        "title": article.title,
        "top_img": article.top_img,
        "meta_img": article.meta_img,
        "text": article.text,
        "meta_keywords": ",".join(article.meta_keywords),
        "tags": ",".join(article.tags),
        "authors": ",".join(article.authors),
        "publish_date": article.publish_date,
        "meta_description": article.meta_description,
        "meta_lang": article.meta_lang,
        "meta_favicon": article.meta_favicon,
        "canonical_link": article.canonical_link
    }
    images = [{"article_id": html_dump["article_id"], "image_url": image_url} for image_url in article.images]
    movies = [{"article_id": html_dump["article_id"], "movie_url": movie_url} for movie_url in article.movies]
    meta_data = dict(flatten(article.meta_data, separator="."), article_id=html_dump["article_id"])
    return article_content, images, movies, meta_data

def parse_html_dump(filepath, queried_urls, articles_dir, chunk_id, pbar, lock=None):
    article_chunk = []
    with gzip.open(filepath, "rt") as f:
        for i, line in enumerate(f):
            html_dump = json.loads(line)
            if html_dump["html"] != "":
                try:
                    article_chunk.append(parse_line(html_dump, queried_urls))
                except Exception as e:
                    print("[ERROR]", f"file {filepath}", f"line {i}", e, flush=True)

            if lock:
                with lock:
                    pbar.update(1)
            else:
                pbar.update(1)
    
    dump_article(article_chunk, articles_dir, chunk_id)

def dump_article(articles_complex, articles_dir, i):
    article_contents_df = pd.DataFrame([a[0] for a in articles_complex])

    images_df = pd.DataFrame(flatten_list([a[1] for a in articles_complex]))
    movies_df = pd.DataFrame(flatten_list([a[2] for a in articles_complex]))
    meta_data_df = pd.DataFrame([a[3] for a in articles_complex])

    for subdir in ["content", "images", "movies", "meta_data"]:
        os.makedirs(join(articles_dir, subdir, "temp"), exist_ok=True)
    article_contents_df.to_json(join(articles_dir, "content/temp", f"part-{str(i).zfill(3)}.json.gzip"), orient="records", compression="gzip")
    images_df.to_json(join(articles_dir, "images/temp", f"part-{str(i).zfill(3)}.json.gzip"), orient="records", compression="gzip")
    movies_df.to_json(join(articles_dir, "movies/temp", f"part-{str(i).zfill(3)}.json.gzip"), orient="records", compression="gzip")
    meta_data_df.to_json(join(articles_dir, "meta_data/temp", f"part-{str(i).zfill(3)}.json.gzip"), orient="records", compression="gzip")

def flatten_list(li):
    flat_list = []
    for e in li:
        flat_list += e
    return flat_list

queried_urls = all_urls.set_index("response_url").to_dict(orient="index")
queried_urls = {key: value["queried_url"] for key, value in queried_urls.items()}

# %%
print("Start parsing", flush=True)
with ProcessPoolExecutor(max_workers=190) as e:
    SyncManager.register('tqdm', tqdm)
    m = SyncManager()
    m.start()
    pbar = m.tqdm(total=len(unique_urls), desc="URLs")
    lock = m.Lock()
    futures = []
    for i, filepath in enumerate(sorted(glob(join(articles_dir, "html/part-*.ndjson.gzip")))):
        futures.append(e.submit(parse_html_dump, filepath, queried_urls, articles_dir, i, pbar, lock))
    
    status = []
    for f in tqdm(as_completed(futures), total=len(futures), desc="Futures"):
        status.append(f.done())

    m.shutdown()
    assert all(status)

# %% [markdown]
# # Reindex article dumps

# %%
article_contents_df = utils.read_partitions_as_pd(join(articles_dir, "content/temp"), "*.json.gzip", filetype="json", compression="gzip")
# %%
utils.write_pd_to_chunks(article_contents_df, join(articles_dir, "content"))
