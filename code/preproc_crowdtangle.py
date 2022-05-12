# %%
import utils

from pyspark import SparkConf
from pyspark.sql import SparkSession
import pyspark.sql.functions as f
import pandas as pd
import re
from tqdm import tqdm, trange
from tqdm.contrib.concurrent import thread_map

import os
from os.path import join
from glob import glob
import json
from flatten_json import flatten
from copy import deepcopy
import gzip
import dateutil

pd.set_option("display.max_columns", None)

# %%
os.environ["PYSPARK_PYTHON"] = "./venv/.venv-36/bin/python"
conf = SparkConf() \
    .set("spark.executor.instances", 19) \
    .set("spark.executor.memory", "32g") \
    .set("spark.driver.memory", "32g") \
    .set("spark.yarn.appMasterEnv.PYSPARK_PYTHON", "./venv/.venv-36/bin/python") \
    .set("spark.yarn.dist.archives", "hdfs:///user/e01525090/venv-36.zip#venv")
spark = SparkSession.builder \
    .master("yarn") \
    .appName("Preprocess Crowdtangle") \
    .config(conf=conf).getOrCreate()
sc = spark.sparkContext

sc.addPyFile("utils.py")
import utils

# %%
def gzip2json(rdd_pair):
    # rdd_pair is a key-value pair: filename -> value
    ndjson_str = gzip.decompress(rdd_pair[1]).decode("utf-8")
    return [json.loads(s) for s in ndjson_str.split("\n") if s != ""]

logs = None
months = ["05", "06", "07", "08", "09", "10"]
for month in months:
    next_logs = sc.binaryFiles(f"ct/ct-links/2021-{month}-01")
    if logs:
        logs = logs.union(next_logs)
    else:
        logs = next_logs
logs = logs.flatMap(gzip2json)

# %%
def clean_links(expanded_links):
    # cleaning necessary, since expandedLinks contains predecessing chars without spacing
    for link in expanded_links:
        # https://regexr.com/37i6s
        findings = re.findall(
            r"((https?:\/\/)?(www\.)?[a-zA-Z][-a-zA-Z0-9@:%._\+~#=]{0,255}\.[a-z]{2,}\b[-a-zA-Z0-9@:%_\+.~#?&//=]*)",
            link["expanded"]
        )
        if len(findings) > 0:
            # Get the only match and outer capture (), not the (www)
            link["expanded_clean"] = findings[0][0]

    return expanded_links

def flatten_post(resp_post, root_keys_to_ignore):
    flat_post = flatten(resp_post, ".", root_keys_to_ignore=root_keys_to_ignore)
    for key in root_keys_to_ignore:
        if key in resp_post:
            flat_post[key] = resp_post[key]

    return flat_post

def log2posts(log):
    posts = []
    keys_to_ignore = ["expandedLinks", "media", "account"]
    queried_url = log["request"]["link"]

    for post in log["response"]["posts"]:
        if post["languageCode"] == "en" and "expandedLinks" in post:
            flat_post = flatten_post(post, keys_to_ignore)
            flat_post["queried_url"] = queried_url
            flat_post["account.id"] = flat_post["account"]["id"]
            flat_post["expandedLinks"] = clean_links(post["expandedLinks"])
            flat_post["updated"] = dateutil.parser.parse(post["updated"])

            posts.append(flat_post)

    return posts

def remove_keys(post, keys):
    for key in keys:
        post.pop(key, None)
    return post

# %%
# Nested dict in RDD is difficult to transform to DF
posts = logs.flatMap(log2posts)
most_recent_posts = posts.toDF() \
    .select("id", "updated", "queried_url") \
    .groupBy("id") \
    .agg(f.max("updated").alias("updated"), f.collect_set("queried_url").alias("queried_urls"))
posts = posts.keyBy(lambda post: (post["id"], post["updated"])) \
    .join(most_recent_posts.rdd
        .map(lambda row: row.asDict())
        .keyBy(lambda row: (row["id"], row["updated"]))
    ) \
    .mapValues(lambda value: dict(value[0], queried_urls=value[1]["queried_urls"])) \
    .values()

# %%
accounts = posts.map(lambda post: post["account"]) \
    .toDF().dropDuplicates(subset=["id"])
medias = posts.flatMap(lambda post: [dict(media, post_id=post["id"]) for media in post.get("media", [])]).toDF()
expanded_links = posts.flatMap(lambda post: [dict(link, queried_url=queried_url, post_id=post["id"]) 
    for queried_url in post["queried_urls"]
    for link in post["expandedLinks"]])

posts_df = posts.map(lambda post: 
        remove_keys(post, ["account", "media", "expandedLinks", "queried_url"])) \
    .toDF() \
    .select("*", f.explode("queried_urls").alias("queried_url")) \
    .drop("queried_urls") \
    .dropDuplicates(subset=["id", "queried_url"])

# %%
expanded_urls = expanded_links \
    .map(lambda link: link.get("expanded_clean")) \
    .filter(lambda x: x is not None) \
    .distinct() \
    .sortBy(lambda x: x)

# Collect URLs and query them on different server
expanded_urls.saveAsTextFile("ct/expanded-urls")

# %%
status_codes = spark.read.csv("ct/url-status-codes/", header=True)
status_codes.printSchema()
status_codes.show(5)

# %%
clean_url_udf = f.udf(utils.clean_url)

shared_urls = status_codes.withColumnRenamed("expanded_url", "expanded_clean").join(
        expanded_links.toDF().drop("expanded", "original").distinct(),
        on="expanded_clean"
    ).drop("expanded_clean") \
    .filter(f.col("response_url").contains(f.col("queried_url")) & 
        (f.col("queried_url") != clean_url_udf("response_url")))

# %%
shared_urls.write.csv("ct/shared-url-in-post", mode="overwrite", header=True)

n_urls = posts_df.select("queried_url").distinct().count()
posts_df.repartition(n_urls, "queried_url").write.json("ct/posts", mode="overwrite")

accounts.repartition(4096, "id").write.json("ct/accounts", mode="overwrite")
medias.write.json("ct/medias", mode="overwrite")
