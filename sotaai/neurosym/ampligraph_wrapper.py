# -*- coding: utf-8 -*-
# Author: Tonio Teran <tonio@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2020.
import ampligraph  # noqa: F401
import importlib

MODELS = {'RandomBaseline': 'latent_features',
          'TransE': 'latent_features',
          'DistMult': 'latent_features',
          'ComplEx': 'latent_features',
          'HolE': 'latent_features',
          'ConvE': 'latent_features',
          'ConvKB': 'latent_features'
          }

DATASETS = {
    'FB15k-237': 'fb15k_237',
    'WN18RR': 'wn18rr',
    'YAGO3-10': 'yago3_10',
    'Freebase15k': 'fb15k',
    'WordNet18': 'wn18',
    'WordNet11': 'wn11',
    'Freebase13': 'fb13'
}


def models():
    """Return a list with all available models."""
    return list(MODELS.keys())


def datasets():
    """Return a list with all available datasets."""
    return list(DATASETS.keys())


def load_model(model_name):
    """Get an instance of an Ampligraph model."""
    if model_name in MODELS.keys():
        model_module = importlib.import_module(
            "ampligraph." + MODELS[model_name])
        model = getattr(model_module, model_name)
        return model()
    else:
        print("Model {} not available.".format(model_name))
        print("Available models are: {}".format(list(MODELS.keys())))


def load_dataset(dataset_name):
    """Get the full dataset directly from Ampligraph."""
    if dataset_name in DATASETS.keys():
        ds_module = importlib.import_module("ampligraph.datasets")
        ds = getattr(ds_module, "load_" + DATASETS[dataset_name])
        return ds()
    else:
        print("Dataset {} not available.".format(dataset_name))
        print("Available datasets are: {}".format(list(DATASETS.keys())))
