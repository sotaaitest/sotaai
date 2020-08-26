# -*- coding: utf-8 -*-
# Author: Tonio Teran <tonio@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2020.
import importlib

MODELS = {
    'ANM': 'cdt.causality.pairwise',
    'BivariateFit': 'cdt.causality.pairwise',
    'CDS': 'cdt.causality.pairwise',
    'GNN': 'cdt.causality.pairwise',
    'IGCI': 'cdt.causality.pairwise',
    'Jarfo': 'cdt.causality.pairwise',
    'NCC': 'cdt.causality.pairwise',
    'RCC': 'cdt.causality.pairwise',
    'RECI': 'cdt.causality.pairwise',
    # 'GS' : 'cdt.causality.graph'  # Requires R bnlearn package.
    # 'IAMB' : 'cdt.causality.graph'  # Requires R bnlearn package.
    # 'Fast_IAMB' : 'cdt.causality.graph'  # Requires R bnlearn package.
    # 'Inter_IAMB' : 'cdt.causality.graph'  # Requires R bnlearn package.
    # 'MMPC' : 'cdt.causality.graph'  # Requires R bnlearn package.
    # 'CAM' : 'cdt.causality.graph'  # Requires R package.
    # 'CCDr' : 'cdt.causality.graph'  # Requires R package.
    'CGNN': 'cdt.causality.graph',
    # 'GES' : 'cdt.causality.graph',  # Requires R package.
    # 'GIES' : 'cdt.causality.graph',  # Requires R package.
    # 'LiNGAM' : 'cdt.causality.graph',  # Requires R package.
    # 'PC' : 'cdt.causality.graph',  # Requires R package.
    'SAM': 'cdt.causality.graph',
    'SAMv1': 'cdt.causality.graph',
}

DATASETS = {
    'tuebingen': 'cdt.data',
    'sachs': 'cdt.data',
    'dream4': 'cdt.data'
}


def models():
    """Return a list with all available models."""
    return list(MODELS.keys())


def datasets():
    """Return a list with all available datasets."""
    return list(DATASETS.keys())


def load_model(model_name):
    """Create a model instance of the Causal Discovery Toolbox class."""
    if model_name in MODELS.keys():
        model_module = importlib.import_module(MODELS[model_name])
        model = getattr(model_module, model_name)
        return model()
    else:
        print("Model {} not available.".format(model_name))
        print("Available models are: {}".format(list(MODELS.keys())))


def load_dataset(dataset_name):
    """Load in a dataset straight from Causal Discovery Toolbox."""
    if dataset_name in DATASETS.keys():
        ds_module = importlib.import_module("cdt.data")
        ds_fxn = getattr(ds_module, 'load_dataset')
        return ds_fxn(dataset_name)
    else:
        print("Dataset {} not available.".format(dataset_name))
        print("Available datasets are: {}".format(list(DATASETS.keys())))
