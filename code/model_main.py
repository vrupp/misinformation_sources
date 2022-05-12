# %%
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
import torch.multiprocessing as mp
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
import seaborn as sns

from models import PostSequenceDataset, PostSequenceModel
import os
from os.path import join
import json
import itertools
from operator import itemgetter
import utils
from glob import glob
import sys
import getopt
from datetime import datetime
from typing import Dict, List
from copy import deepcopy
import math

pd.set_option("display.max_columns", None)

# %%
def nested_config_overwrite(config: dict, key: str, value: str):
    """Overwrites parameter values in nested dict

    Args:
        config (dict): Dictionary to overwrite
        key (str): Name of the key. Nested keys are concatenated using "."
        value (str): New value
    """
    keys = key.split(".")
    for key in keys[:-1]:
        config = config[key]
    
    # Cast str value to type of overwritten value
    value_type = type(config[keys[-1]])
    if value_type == bool:
        config[keys[-1]] = (value == "true")
    else:
        config[keys[-1]] = value_type(value)

def grid_search(device: torch.device, conf: dict, grid_params: Dict[str, List], grid_dir: str, te_batch_size=None, pooler_batch_size=None, loss_interval=1, checkpoint_interval=1):
    """Performs a grid search over a set of hyperparameters

    Args:
        conf (dict): Model default parameters
        grid_params (Dict[str, List]): Dict of parameters with possible values
        grid_dir (str): Directory to store model checkpoints
        te_batch_size (int, optional): Batch size for encoding documents. If None, encode all documents at once. Defaults to None.
        pooler_batch_size (int, optional): Batch size for processing cls token encodings. If None, process all cls tokens of the document. Defaults to None.
        loss_interval (int, optional): Number of batches after which loss is printed. Defaults to 1.
        checkpoint_interval (int, optional): Number of batches after which model checkpoint is logged. Defaults to 1.
    """
    os.makedirs(join(base_dir, "train_checkpoints", grid_dir), exist_ok=True)
    if os.path.exists(join(base_dir, "train_checkpoints", grid_dir, "grid_params.json")):
        with open(join(base_dir, "train_checkpoints", grid_dir, "grid_params.json")) as f:
            grid_params = json.load(f)
    else:
        with open(join(base_dir, "train_checkpoints", grid_dir, "grid_params.json"), "w") as f:
            json.dump(grid_params, f, indent=4)
    if os.path.exists(join(base_dir, "train_checkpoints", grid_dir, "base_config.json")):
        with open(join(base_dir, "train_checkpoints", grid_dir, "base_config.json")) as f:
            conf = json.load(f)
    else:
        with open(join(base_dir, "train_checkpoints", grid_dir, "base_config.json"), "w") as f:
            json.dump(conf, f, indent=4)
    
    n_gpus = torch.cuda.device_count()
    print(f"Found {n_gpus} GPUs", flush=True)
    results = []
    mp.set_start_method("spawn", force=True)
    with mp.Pool(processes=max(1, n_gpus)) as pool:
        keys, value_lists = zip(*grid_params.items())
        for i, value_product in enumerate(list(itertools.product(*value_lists))):
            conf_cp = deepcopy(conf)
            assert len(keys) == len(value_product)
            for key, value in zip(keys, value_product):
                nested_config_overwrite(conf_cp, key, value)

            conf_cp, checkpoint_dir = load_or_create_config(conf_cp, join(grid_dir, str(i).zfill(3)))
            if os.path.exists(join(checkpoint_dir, "SUCCESS")):
                continue

            if n_gpus > 0:
                device = torch.device(f"cuda:{i % n_gpus}")
            trainset, _, _ = split_dataset(conf_cp, gt_labels)
            results.append(pool.apply_async(
                train, (trainset, device, conf_cp, checkpoint_dir, te_batch_size, pooler_batch_size, loss_interval, checkpoint_interval)
            ))

        for r in tqdm(results, desc="Grid Search"):
            r.wait()
        assert all([r.ready() for r in results])

def k_fold_cv(gt_labels: pd.DataFrame, conf: dict, cv_dir: str, device: torch.device, k=10, te_batch_size=None, pooler_batch_size=None, loss_interval=1, checkpoint_interval=1):
    """Performs a k-fold cross-validation

    Args:
        gt_labels (pd.DataFrame): Labeled news sources, which are split into k-folds
        conf (dict): Model parameters
        cv_dir (str): Directory to store model checkpoints
        k (int, optional): Number of folds
        te_batch_size (int, optional): Batch size for encoding documents. If None, encode all documents at once. Defaults to None.
        pooler_batch_size (int, optional): Batch size for processing cls token encodings. If None, process all cls tokens of the document. Defaults to None.
        loss_interval (int, optional): Number of batches after which loss is printed. Defaults to 1.
        checkpoint_interval (int, optional): Number of batches after which model checkpoint is logged. Defaults to 1.
    """
    os.makedirs(join(base_dir, "train_checkpoints", cv_dir), exist_ok=True)
    
    n_gpus = torch.cuda.device_count()
    print(f"Found {n_gpus} GPUs", flush=True)
    results = []
    mp.set_start_method("spawn", force=True)
    with mp.Pool(processes=max(1, n_gpus)) as pool:
        for i, (trainset, testset) in enumerate(cv_split_generator(conf, gt_labels, k)):
            conf, checkpoint_dir = load_or_create_config(conf, join(cv_dir, str(i).zfill(2)))
            if os.path.exists(join(checkpoint_dir, "SUCCESS")):
                continue

            if n_gpus > 0:
                device = torch.device(f"cuda:{i % n_gpus}")
            results.append(pool.apply_async(
                train_test, (trainset, testset, device, conf, checkpoint_dir, te_batch_size, pooler_batch_size, loss_interval, checkpoint_interval)
            ))

        for r in tqdm(results, desc="k-fold Cross Validation"):
            r.wait()
        assert all([r.ready() for r in results])

    _, subdirs, _ = next(os.walk(join(base_dir, "train_checkpoints", cv_dir)))
    pred_all = pd.concat([pd.read_csv(join(base_dir, "train_checkpoints", cv_dir, subdir, "pred-test.csv")) for subdir in sorted(subdirs)])
    pred_all.to_csv(join(base_dir, "train_checkpoints", cv_dir, "pred-all.csv"), index=False)

def train_test(trainset, testset, device, conf, checkpoint_dir, te_batch_size, pooler_batch_size, loss_interval, checkpoint_interval):
    train(trainset, device, conf, checkpoint_dir, te_batch_size, pooler_batch_size, loss_interval, checkpoint_interval)
    test(testset, conf, device, checkpoint_dir, filename="pred-test.csv", te_batch_size=te_batch_size, pooler_batch_size=pooler_batch_size)

def split_dataset(conf: dict, gt_labels: pd.DataFrame):
    """Splits a dataset into 3 sets: train, val, test

    Args:
        conf (dict): Model configuration
        gt_labels (pd.DataFrame): Labeled dataset

    Returns:
        Tuple[PostSequenceDataset]: 3 distinct splits (train, val, test)
    """
    label_col = f'gt|{conf["label"]}_enc'

    gt_train, gt_test_all = train_test_split(gt_labels, test_size=0.2, random_state=conf["seed"], stratify=gt_labels[label_col])
    gt_val, gt_test = train_test_split(gt_test_all, test_size=0.5, random_state=conf["seed"], stratify=gt_test_all[label_col])
    if conf["oversampling"]:
        gt_train = oversample(gt_train, conf)
    
    trainset, valset, testset = [PostSequenceDataset(gt_split, label_col, shared_articles_tw, shared_articles_fb, art_conts_dir, tweets_dir, fb_posts_dir, tw_users_dir, fb_acc_dir, has_cls_enc=True)
        for gt_split in (gt_train, gt_val, gt_test)]
    return trainset, valset, testset

def cv_split_generator(conf: dict, gt_labels: pd.DataFrame, k: int):
    """Train-test split generator for k-fold cross-validation

    Args:
        conf (dict): Model configuration
        gt_labels (pd.DataFrame): Labeled dataset to split
        k (int): Number of folds

    Yields:
        Tuple[PostSequenceDataset]: 2 distinct datasets: train, test
    """
    label_col = f'gt|{conf["label"]}_enc'

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=conf["seed"])
    for i_train, i_test in skf.split(gt_labels[label_col], gt_labels[label_col]):
        gt_train, gt_test = gt_labels.iloc[i_train], gt_labels.iloc[i_test]
        if conf["oversampling"]:
            gt_train = oversample(gt_train, conf)
        
        trainset, testset = [PostSequenceDataset(gt_split, label_col, shared_articles_tw, shared_articles_fb, art_conts_dir, tweets_dir, fb_posts_dir, tw_users_dir, fb_acc_dir, has_cls_enc=True)
            for gt_split in (gt_train, gt_test)]
        yield trainset, testset

def oversample(gt_train: pd.DataFrame, conf: dict):
    """Oversamples the minority class in a dataset

    Args:
        gt_train (pd.DataFrame): Labeled dataset
        conf (dict): Model configuration

    Returns:
        pd.DataFrame: Dataset with same schema as gt_train
    """
    label_col = f'gt|{conf["label"]}_enc'
    value_counts = gt_train[label_col].value_counts().to_dict()
    max_value = max(value_counts.values())
    samples = []
    for key, count in value_counts.items():
        sample = gt_train[gt_train[label_col] == key]
        samples += [sample] * (max_value // count)
        samples += [sample.sample(n=(max_value % count), random_state=conf["seed"])]
    gt_train = pd.concat(samples)
    assert gt_train[label_col].value_counts().nunique() == 1
    return gt_train


def load_model(conf: dict, classes: pd.Series, device: torch.device):
    """Loads model checkpoint

    Args:
        conf (dict): Model configuration
        classes (pd.Series): Label column of the dataset

    Returns:
        Tuple[PostSequenceModel, nn._Loss, torch.optim.Optimizer]: (model, loss function, optimizer function)
    """
    torch.manual_seed(conf["seed"])
    torch.cuda.manual_seed_all(conf["seed"])
    post_seq_model = PostSequenceModel(device, n_classes=classes.nunique(), n_features=10, conf=conf, has_cls_enc=True)
    
    criterion, optimizer = None, None
    if conf["loss_fn"] == "CrossEntropyLoss":
        class_weights = compute_class_weight("balanced", classes=sorted(classes.unique()), y=classes)
        criterion = nn.CrossEntropyLoss(weight=torch.as_tensor(class_weights, dtype=torch.float32, device=device))
    else:
        raise Exception(f"loss_fn {conf['loss_fn']} not known")
    
    if conf["optimizer"]["name"] == "SGD":
        optimizer = optim.SGD(post_seq_model.parameters(), lr=conf["optimizer"]["lr"], momentum=conf["optimizer"]["momentum"])
    elif conf["optimizer"]["name"] == "Adam":
        optimizer = optim.Adam(post_seq_model.parameters(), lr=conf["optimizer"]["lr"])
    else:
        raise Exception(f"optimizer {conf['optimizer']['name']} not known")
    
    return post_seq_model, criterion, optimizer


def dataloader(dataset: PostSequenceDataset, batch_size: int, shuffle=True, seed=19, start_batch=0):
    """Generator for loading required data for each news source

    Args:
        start_batch (int, optional): Number of batches to skip. Defaults to 0.

    Yields:
        Tuple[Tuple[int], Tuple[str], Tuple[Dict, Dict], Tuple[int]]: Indices of the batch regarding to the dataset, Urls in the batch, tuple with X, y, Labels of the batch
    """
    if shuffle:
        np.random.seed(seed)
        indices = np.random.choice(len(dataset), len(dataset), replace=False)
    else:
        indices = range(len(dataset))
    
    it = iter(indices)
    # Skip batches until start
    list(itertools.islice(it, batch_size * start_batch))
    
    while True:
        i_batch = list(itertools.islice(it, batch_size))
        if len(i_batch) == 0:
            break

        post_sequences, gt_labels = zip(*[dataset[i] for i in i_batch])
        urls, labels = zip(*map(itemgetter("gt|url", dataset.label_col), gt_labels))
        yield i_batch, urls, post_sequences, labels


def total_sentences(urls, post_sequences=None):
    if post_sequences is not None:
        total = 0
        for seq in post_sequences:
            total += sum(map(lambda tensor: tensor.shape[0], seq["art_cont|title_cls_enc"]))
            total += sum(map(lambda tensor: tensor.shape[0], seq["art_cont|text_cls_enc"]))
            total += sum(map(lambda tensor: tensor.shape[0], seq["art_cont|meta_description_cls_enc"]))
            total += sum(map(lambda tensor: tensor.shape[0], seq["post|text_cls_enc"]))
            total += sum(map(lambda tensor: tensor.shape[0], seq["user|description_cls_enc"]))

        return total
    else:
        dir = join(data_dir, "../sentence_counts/")
        sent_counts = {filename: pd.read_csv(join(dir, filename + ".csv")) for filename in ["articles", "tweets", "fb_posts", "tw_users", "fb_accs"]}
        urls_series = pd.Series(urls).rename("shared_art|queried_url")
        shared_tw = shared_articles_tw.merge(urls_series)
        shared_fb = shared_articles_fb.merge(urls_series)

        def sum_counts(filename, series):
            return sent_counts[filename][sent_counts[filename]["id"].isin(series.unique())] \
                .drop(columns=["id"]).sum().sum()
        
        total = 0
        total += sum_counts("articles", shared_tw["shared_art|article_id"])
        total += sum_counts("articles", shared_fb["shared_art|article_id"])
        total += sum_counts("tweets", shared_tw["shared_art|tweet_id"])
        total += sum_counts("fb_posts", shared_fb["shared_art|post_id"])
        total += sum_counts("tw_users", shared_tw["shared_art|author_id"])
        total += sum_counts("fb_accs", shared_fb["shared_art|account.id"])
        return total


def load_or_create_config(existing_conf: dict, checkpoint_subdir: str):
    """Creates a new checkpoint directory with initial configuration. Otherwise returns existing config

    Args:
        existing_conf (dict): Configuration to store. Will be ignored if directory exists
        checkpoint_subdir (str): Name of the directory to search for. Will be ignored if not exist

    Returns:
        Tuple[dict, str]: Existing or found configuration, name of the newly created dir
    """
    checkpoint_subdir = checkpoint_subdir or datetime.utcnow().strftime("%y%m%d-%H%M")
    checkpoint_dir = join(base_dir, "train_checkpoints", checkpoint_subdir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    if os.path.exists(join(checkpoint_dir, "config.json")):
        with open(join(checkpoint_dir, "config.json")) as f:
            new_conf = json.load(f)
    else:
        with open(join(checkpoint_dir, "config.json"), "w") as f:
            json.dump(existing_conf, f, indent=4)
        new_conf = existing_conf

    return new_conf, checkpoint_dir


def train(trainset: PostSequenceDataset, device: torch.device, conf: dict, checkpoint_dir: str, te_batch_size=None, pooler_batch_size=None, loss_interval=1, checkpoint_interval=1):
    """Trains the models

    Args:
        conf (dict): Model configuration
        checkpoint_dir (str): Name of the directory to store or read checkpoints
        te_batch_size (int, optional): Batch size for encoding documents. If None, encode all documents at once. Defaults to None.
        pooler_batch_size (int, optional): Batch size for processing cls token encodings. If None, process all cls tokens of the document. Defaults to None.
        loss_interval (int, optional): Number of batches after which loss is printed. Defaults to 1.
        checkpoint_interval (int, optional): Number of batches after which model checkpoint is logged. Defaults to 1.
    """
    post_seq_model, criterion, optimizer = load_model(conf, trainset.get_labels(), device)
    existing_checkpoints = sorted(glob(join(checkpoint_dir, "model-state-epoch-*.pt")))
    if len(existing_checkpoints) > 0:
        last_epoch_path = existing_checkpoints[-1]
        checkpoint = torch.load(last_epoch_path, map_location=device)
        post_seq_model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        current_epoch = checkpoint["epoch"]
        checked_batch = checkpoint["batch"]
    else:
        checkpoint = {}
        checkpoint["outputs"] = []
        checkpoint["losses"] = []
        checkpoint["urls"] = []

        current_epoch = 0
        checked_batch = -1
    post_seq_model.train()

    for epoch in range(current_epoch, conf["n_epochs"]):
        model_state_file = f"model-state-epoch-{str(epoch + 1).zfill(2)}.pt"

        batches = dataloader(trainset, conf["batch_size"], seed=conf["seed"], start_batch=checked_batch + 1)
        for i, batch in tqdm(enumerate(batches, start=checked_batch + 1), total=math.ceil(len(trainset) / conf["batch_size"]), initial=checked_batch + 1, desc="Train Batch"):
            indices, urls, post_sequences, labels = batch
            checkpoint["urls"].append(urls)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            try:
                output = post_seq_model(post_sequences, te_batch_size=te_batch_size, pooler_batch_size=pooler_batch_size)
                checkpoint["outputs"].append(output)
            except Exception as e:
                print(f"Exception in batch {i}, urls {urls}")
                checkpoint["model_state_dict"] = post_seq_model.state_dict()
                checkpoint["optimizer_state_dict"] = optimizer.state_dict()
                checkpoint["epoch"], checkpoint["batch"] = epoch, i
                checkpoint["exception"] = e
                torch.save(checkpoint, join(checkpoint_dir, "model-state-ERROR.pt"))
                raise e
            
            loss = criterion(output, torch.as_tensor(labels, dtype=torch.int64, device=device))
            loss.backward()
            optimizer.step()

            # print statistics
            checkpoint["losses"].append(loss.item())
            if i % loss_interval == loss_interval - 1:
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, sum(checkpoint["losses"][-loss_interval:]) / loss_interval))
            # Save checkpoint every `checkpoint_interval` iterations and in the last iteration
            if i % checkpoint_interval == checkpoint_interval - 1 or (i + 1) * conf["batch_size"] >= len(trainset):
                checkpoint["model_state_dict"] = post_seq_model.state_dict()
                checkpoint["optimizer_state_dict"] = optimizer.state_dict()
                checkpoint["epoch"], checkpoint["batch"] = epoch, i
                torch.save(checkpoint, join(checkpoint_dir, model_state_file))

        checked_batch = -1

    print('Finished Training')
    with open(join(checkpoint_dir, "SUCCESS"), "w") as f:
        # create empty file
        pass


def test(testset: PostSequenceDataset, conf: dict, device: torch.device, checkpoint_dir: str, filename: str, te_batch_size=None, pooler_batch_size=None):
    """Makes predictions on unseen testdata

    Args:
        conf (dict): Model configuration
        checkpoint_dir (str): Name of the directory with stored model checkpoint
        filename (str): Filename for the predictions
        te_batch_size (int, optional): Batch size for encoding documents. If None, encode all documents at once. Defaults to None.
        pooler_batch_size (int, optional): Batch size for processing cls token encodings. If None, process all cls tokens of the document. Defaults to None.
    """
    post_seq_model = PostSequenceModel(device, n_classes=len(testset.get_classes()), n_features=10, conf=conf, has_cls_enc=True)
    
    last_epoch_path = sorted(glob(join(checkpoint_dir, "model-state-epoch-*.pt")))[-1]
    print("Load model", last_epoch_path, flush=True)
    checkpoint = torch.load(last_epoch_path, map_location=device)
    post_seq_model.load_state_dict(checkpoint["model_state_dict"])
    predictions = []

    post_seq_model.eval()
    with torch.no_grad():
        batches = dataloader(testset, batch_size=1, shuffle=False)
        for batch in tqdm(batches, total=len(testset), desc="Test Batch"):
            indices, urls, post_sequences, labels = batch
            output = post_seq_model(post_sequences, te_batch_size=te_batch_size, pooler_batch_size=pooler_batch_size)

            predictions.append({"url": urls[0], f"{conf['label']}_pred": torch.argmax(output, dim=1).item(), f"{conf['label']}_true": labels[0]})

    pd.DataFrame(predictions).to_csv(join(checkpoint_dir, filename), index=False)

# %%
# Defaults
base_dir = "../../data"
mode = "train"
te_batch_size = None
pooler_batch_size = None
checkpoint_subdir = None
test_dirs = []
grid_dir = "grid"
cv_dir = "cv"
with open(join(os.path.dirname(__file__), "params/model_train_params.json")) as f:
    conf = json.load(f)
with open(join(os.path.dirname(__file__), "params/model_grid_params.json")) as f:
    grid_params = json.load(f)

cmd_exec = False

# Parse args
if sys.argv[0].endswith("model_main.py"):
    cmd_exec = True
    try:
        opts, args = getopt.getopt(sys.argv[1:], "d:m:c:g:", ["data-dir=", "mode=", "te-batch-size=", "pooler-batch-size=", "checkpoint-dir=", "test-dirs=", "cv-dir=", "config=", "grid-params=", "grid-dir="])
    except getopt.GetoptError:
        print("Usage: model_main.py [-d <data-dir>] [-m <mode>] [--te-batch-size=<te_batch_size>] [--pooler-batch-size=<pooler_batch_size>] [--checkpoint-dir=<checkpoint_dir>] [--test-dirs=<test_dirs>] [--cv-dir=<cv_dir>] [-c <config_overwrites>] [-g <grid_params>] [--grid-dir=<grid_dir>]")
        sys.exit(2)

    for opt, arg in opts:
        print(opt, arg)
        if opt in ("-d", "--data-dir"):
            base_dir = arg
        elif opt in ("-m", "--mode"):
            mode = arg
        elif opt == "--te-batch-size":
            te_batch_size = int(arg)
        elif opt == "--pooler-batch-size":
            pooler_batch_size = int(arg)
        elif opt == "--checkpoint-dir":
            checkpoint_subdir = arg
        elif opt == "--test-dirs":
            test_dirs = arg.split(",")
        elif opt == "--cv-dir":
            cv_dir = arg
        elif opt in ("-c", "--config"):
            config_overwrites = arg.split(",")
            for overwrite in config_overwrites:
                key, value = overwrite.split("=")
                nested_config_overwrite(conf, key, value)
        elif opt in ("-g", "--grid-params"):
            with open(arg) as f:
                grid_params = json.load(f)
        elif opt == "--grid-dir":
            grid_dir = arg

# %%
data_dir = join(base_dir, "final/with-cls")
art_conts_dir = join(data_dir, "articles")
shared_articles_tw_path = join(data_dir, "../shared-articles-in-tweet.csv")
shared_articles_fb_path = join(data_dir, "../shared-articles-in-fb-post.csv")
tweets_dir = join(data_dir, "tweets")
fb_posts_dir = join(data_dir, "fb_posts")
tw_users_dir = join(data_dir, "tw_users")
fb_acc_dir = join(data_dir, "fb_accs")
ground_truth_path = join(os.path.dirname(__file__), "../resources/domains/clean/domain_list_clean.csv")

shared_articles_tw = pd.read_csv(shared_articles_tw_path, dtype={"tweet_id": str, "author_id": str}) \
    .add_prefix("shared_art|")
shared_articles_fb = pd.read_csv(shared_articles_fb_path, dtype={"post_id": str, "account.id": str}) \
    .add_prefix("shared_art|")
gt_labels = pd.read_csv(ground_truth_path) \
    .drop(columns=["label", "source", "last_update"]) \
    .add_prefix("gt|")
for col in ["gt|accuracy", "gt|transparency", "gt|type", "gt|accuracy_bin"]:
    gt_labels[col + "_enc"] = LabelEncoder().fit_transform(gt_labels[col])

with open(join(art_conts_dir, "index.json")) as f:
    available_sources = [k.split("=")[-1] for k in json.load(f).keys()]
gt_labels = gt_labels[gt_labels["gt|url"].isin(available_sources)].sort_values(by="gt|url")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Use device", device, flush=True)

# %% [markdown]
# # Grid Search

# %%
if __name__ == "__main__":
    if not cmd_exec:
        from tqdm.notebook import tqdm
    if mode == "grid" or not cmd_exec:
        grid_search(device, conf, grid_params, grid_dir, te_batch_size=te_batch_size, pooler_batch_size=pooler_batch_size, loss_interval=1, checkpoint_interval=10)

# %% [markdown]
# # Training

# %%
if __name__ == "__main__":
    if not cmd_exec:
        from tqdm.notebook import tqdm
    if mode == "train" or not cmd_exec:
        conf, checkpoint_dir = load_or_create_config(conf, checkpoint_subdir)
        trainset, _, _ = split_dataset(conf, gt_labels)

        train(trainset, device, conf, checkpoint_dir, te_batch_size=te_batch_size, pooler_batch_size=pooler_batch_size, loss_interval=1, checkpoint_interval=10)

        if cmd_exec:
            sys.exit(0)

# %% [markdown]
# # Testing

# %%
if __name__ == "__main__":
    if not cmd_exec:
        from tqdm.notebook import tqdm
    if mode in ["val", "test"] or not cmd_exec:
        n_gpus = torch.cuda.device_count()
        print(f"Found {n_gpus} GPUs", flush=True)
        results = []
        mp.set_start_method("spawn", force=True)
        with mp.Pool(processes=max(1, n_gpus)) as pool:
            for test_dir in test_dirs:
                for current_dir, dirs, files in os.walk(join(base_dir, "train_checkpoints", test_dir)):
                    if "SUCCESS" in files:
                        with open(join(current_dir, "config.json")) as f:
                            conf = json.load(f)
                        
                        if n_gpus > 0:
                            device = torch.device(f"cuda:{len(results) % n_gpus}")
                        
                        if mode == "val":
                            _, testset, _ = split_dataset(conf, gt_labels)
                        else:
                            _, _, testset = split_dataset(conf, gt_labels)
                       
                        filename = f"pred-{mode}.csv"
                        results.append(pool.apply_async(
                            test, (testset, conf, device, current_dir, filename, te_batch_size, pooler_batch_size)
                        ))
            for r in tqdm(results, desc="Test batches"):
                r.wait()
            assert all([r.ready() for r in results])
        if cmd_exec:
            sys.exit(0)

# %% [markdown]
# # Cross Validation

# %%
if __name__ == "__main__":
    if not cmd_exec:
        from tqdm.notebook import tqdm
    if mode == "CV" or not cmd_exec:
        k_fold_cv(gt_labels, conf, cv_dir, device, te_batch_size=te_batch_size, pooler_batch_size=pooler_batch_size, loss_interval=1, checkpoint_interval=10)

        if cmd_exec:
            sys.exit(0)
