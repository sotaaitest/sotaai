# -*- coding: utf-8 -*-
# Author: Tonio Teran <tonio@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2020.
import importlib


DATASETS = {
    'wikipedia': 'GraphReader',
    'facebook': 'GraphReader',
    'github': 'GraphReader',
    'twitch': 'GraphReader',
    'reddit10k': 'GraphSetReader',
}


def datasets():
    """Return a list with all available datasets."""
    return list(DATASETS.keys())


def load_dataset(dataset_name):
    """Load-in an instance of a KarateClub dataset"""
    if dataset_name in DATASETS.keys():
        ds_module = importlib.import_module("karateclub.dataset")
        ds_fxn = getattr(ds_module, DATASETS[dataset_name])
        return ds_fxn(dataset_name)
    else:
        print("Dataset {} not available.".format(dataset_name))
        print("Available datasets are: {}".format(list(DATASETS.keys())))
