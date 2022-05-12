import requests
import json
import time
from copy import deepcopy

DATE_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

def read_bearer_token(key_file):
    with open(key_file) as f:
        for line in f:
            key, val = line.partition("=")[::2]
            if key == 'bearer_token':
                return val.strip('\n')

def lookup(id, list):
    return next(item for item in list if item['id'] == id)


def get_formatted_tweets(json_response):
    list_of_tweets = []
    has_expansion_data = False
    data = json_response.get('data', [])
    if 'includes' in json_response:
        includes = json_response['includes']
        has_expansion_data = True
    for tweet_info in data:
        if has_expansion_data:
            if 'users' in includes:
                tweet_info['user'] = lookup(tweet_info['author_id'], includes['users'])
            if 'places' in includes:
                if 'geo' in tweet_info:
                    tweet_info['place'] = lookup(tweet_info['geo']['place_id'], includes['places'])
        list_of_tweets.append(tweet_info)
    return list_of_tweets


def validate_config(config):
    if 'bearer_token' not in config or config['bearer_token'] == "":
        raise Exception("Bearer token is missing from the config file")
    if 'params' in config and ('query' not in config['params'] or config['params']['query'] == ""):
        raise Exception("Please make sure to provide a valid search query")
    if 'write_path' not in config:
        raise Exception("Please specify the output path where you want the Tweets to be written to")


# Function to write Tweet to new line in a file
def write_to_file(file_name, tweets):
    with open(file_name, 'a+') as filehandle:
        for tweet in tweets:
            filehandle.write('%s\n' % json.dumps(tweet))


def search_tweets(config, next_token=None, all=True):
    endpoint = 'https://api.twitter.com/2/tweets/search/' + ('all' if all else 'recent')
    headers = {"Authorization": "Bearer {}".format(config['bearer_token'])}
    params = deepcopy(config["params"])
    params["next_token"] = next_token   
    
    response, error = _make_request(endpoint, headers, params)
    return response.json(), error

def count_tweets(config, next_token=None, all=True):
    endpoint = 'https://api.twitter.com/2/tweets/counts/' + ('all' if all else 'recent')
    headers = {"Authorization": "Bearer {}".format(config['bearer_token'])}
    params = deepcopy(config["params"])
    params["next_token"] = next_token

    response, error = _make_request(endpoint, headers, params)
    return response.json(), error

def _make_request(endpoint, headers, params):
    sleep_counter = 0
    error = False
    while True:
        response = requests.request("GET", endpoint, headers=headers, params=params)

        if response.status_code == 200:
            break
        elif sleep_counter > 3:
            raise Exception(f"More than 3 requests, last response: {response.status_code} {response.text}")
        elif response.status_code == 429:
            # Too many requests
            seconds = int(response.headers["x-rate-limit-reset"]) - int(time.time())
            print(f"sleep {seconds / 60} min")
            time.sleep(seconds)
            sleep_counter += 1
            continue
        else:
            error = True
            break
    
    return response, error