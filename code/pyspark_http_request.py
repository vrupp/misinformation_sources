# %%
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
import time
import pandas as pd
import os
import json
from tqdm.contrib.concurrent import thread_map

# %%
with open("expanded-urls.json") as f:
    expanded_urls = json.load(f)

# %%
os.environ["PYSPARK_PYTHON"] = "./venv/.venv-36/bin/python"
conf = SparkConf() \
    .set("spark.executor.instances", 19) \
    .set("spark.executor.memory", "32g") \
    .set("spark.yarn.appMasterEnv.PYSPARK_PYTHON", "./venv/.venv-36/bin/python") \
    .set("spark.yarn.dist.archives", "hdfs:///user/e01525090/venv-36.zip#venv")
spark = SparkSession.builder \
    .master("yarn") \
    .appName("HTTP Requests") \
    .config(conf=conf).getOrCreate()
sc = spark.sparkContext

# %%
sc.addPyFile("utils.py")
import utils

# %%
def check_url_status_par(urls):
    return thread_map(utils.check_url_status, urls, max_workers=os.cpu_count())

urls_rdd = sc.parallelize(expanded_urls)
status_codes = urls_rdd.glom().flatMap(check_url_status_par).collect()
status_codes = pd.DataFrame(status_codes).rename(columns={"url": "expanded_url"})
status_codes.to_csv("../resources/twitter/preproc/url-status-codes.csv", index=False)
