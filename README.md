# Tools for misinformation research on Twitter/Facebook

## Setup
Setup virtual environment:
```
$ python -m venv .venv
$ . .venv/bin/activate
```
Install requirements (Python 3.8):
```
$ pip install -r requirements.txt
```
Install requirements (Python 3.6):
```
$ pip install -r requirements-36.txt
```

## Usage
```
$ cd code/
$ python model_main.py [-d <data-dir>] [-m <mode>] [--te-batch-size=<te_batch_size>] [--pooler-batch-size=<pooler_batch_size>] [--checkpoint-dir=<checkpoint_dir>] [--test-dirs=<test_dirs>] [--cv-dir=<cv_dir>] [-c <config_overwrites>] [-g <grid_params>] [--grid-dir=<grid_dir>]
```
- `data-dir` specifies base directory of misinformation project, default `/data/misinformation-domains/`
- `mode` differentiates between "train", "val", "test", "grid", and "CV". If "val" or "test", `test_dirs` must be specified. If "grid", then `grid_params` must be specified. If "CV", then `cv_dir` must be specified. Default "train"
- `te_batch_size` specifies the number of documents that are encoded in parallel, default all documents
- `pooler_batch_size` specifies the number of sentences which are encoded in parallel, i.e. fed its CLS token encoding into an additional (pooler) layer, default all sentences
- `checkpoint_dir` enables the model to continue from a checkpoint stored in this directory
- `test_dirs` is a comma-separated list of directory names, which are subdirectories of `<data_dir>/train_checkpoints`
`cv_dir` specifies the name of the subdirectory, where the model states of each fold are stored
- `config_overwrites` is a string of comma separated key-value pairs, which specify config settings to overwrite, e.g. "key1=value1,key2=value2". To overwrite a nested value, e.g. "lr" in `{"optimizer": {"lr": 0.001}}`, one can specify the keys separated by ".", e.g. "optimizer.lr=0.099,key2=value2"
- `grid_params` specify the path to a json with key value combinations for a grid search. The are only considered, if `mode` is "grid". The json has to be flat, the keys must be present in the config as well, or can be nested by separating with "." (see above), and the values must be lists of valid values for the respective keys.
- `grid_dir` specifies the name of the subdirectory, where the model states of each training are stored
  
## Model Configuration
- `seed`: The seed for initializing model weights, splitting data, and other non-deterministic operations
- `label`: The label to predict. One of "accuracy", "transparency", or "type"
- `n_epochs`: Number of epochs (total iterations over the whole training data)
- `batch_size`: Batch size (number of news websites) after which model weights are adjusted
- `loss_fn`: Class name of loss function, e.g. "CrossEntropyLoss"
- `optimizer`: Optimizer properties
  - `name`: Name of the optimizer, one of "SGD" or "Adam"
  - `lr`: Learning rate of the optimizer
  - `momentum`: Momentum of the optimizer
- `psm`: Properties of the post sequence model
  - `type`: The type of the recurrent post sequence model (how the sequence of posts should be modeled). One of "RNN", "LSTM", or "GRU"
  - `hidden_size`: Size of the hidden state in the recurrent model
  - `output`: Hidden states aggregation function, one of "last_state", "mean", "max", "mean+max"
- `te`: Properties of the text encoder
  - `type`: Function to aggregate the CLS token encodings, one of "mean", "LSTM"
  - `embedding_size`: Size of the output embedding. If `null`, take output size of pooler
}

### API access
API access requires a bearer token that is stored in a separate file, not part of this repository. 

### Search query configs
Search query configs are kept as separate .json files in the ```code/configs/``` folder. Please try to give them descriptive names.

### Utility functions
Utility functions that will potentially be used in more than one application are collected in the ```twitter_functions.py``` module. 

## Resources
```resources/domains/raw```  
Has a number of domain lists that have been classified into various disinformation related categories and were initially compiled in the [Galotti 2020 et al.](https://www.nature.com/articles/s41562-020-00994-6/tables/1) paper. We collected almost all of these lists, with the notable exception of the list from [Décodex](https://www.lemonde.fr/verification/), some of them with updates, and compiled them into a new list ```data/domains/clean/domain_list_clean.csv```. The cleaning is done in the script ```clean_misinformation_domains.ipynb```. 

```resources/search_terms/```  
Has a list of different search terms connected to the COVID-19 pandemic and misinformation.