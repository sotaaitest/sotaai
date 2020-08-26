# -*- coding: utf-8 -*-
# Author: Tonio Teran <tonio@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2020.
import importlib


available_sources = {
    "Ampligraph": "ampligraph_wrapper",
    "CausalDiscoveryToolbox": "cdt_wrapper",
    "DeepGraphLibrary": "dgl_wrapper",
    "KarateClub": "karate_wrapper",
    "Spektral": "spektral_wrapper"}

model_sources = [
    "ampligraph_wrapper",
    "cdt_wrapper",
    "dgl_wrapper",
    "spektral_wrapper"
]

dataset_sources = [
    "ampligraph_wrapper",
    "cdt_wrapper",
    "dgl_wrapper",
    "karate_wrapper",
    "spektral_wrapper"
]


def models(source=None):
    """Return a list with available models.

    If no specific `source` is chosen, then the entire collection of models,
    from all sources, is returned. The possible values for the `source`
    parameter are given by the elements of the `model_sources` list.

    Args:
        source (str): Name of the specific library of interest.

    Returns:
        list: List of strings with the names of the available models.

    """
    model_names = []
    if source is None:
        # Fetch all of them.
        for s in model_sources:
            src_module = importlib.import_module(
                'sotaai.neurosym.'+s)
            model_names += src_module.models()
        return sorted(model_names)
    else:
        if source in available_sources.keys():
            # Return only for specific source.
            src_module = importlib.import_module(
                'sotaai.neurosym.' + available_sources[source])
            return src_module.models()
        else:
            print("Unavailable source. Here are the options:")
            print(sorted(list(available_sources.keys())))


def datasets(source=None):
    """Return a list with available datasets for neurosymbolic programming.

    If no specific `source` is chosen, then the entire collection of datasets,
    from all sources, is returned. The possible values for the `source`
    parameter are given by the elements of the `model_sources` list.

    Args:
        source (str): Name of the specific library of interest.

    Returns:
        list: List of strings with the names of the available datasets.

    """
    dataset_names = []
    if source is None:
        # Fetch all of them.
        for s in dataset_sources:
            src_module = importlib.import_module(
                'sotaai.neurosym.' + s)
            dataset_names += src_module.datasets()
        return sorted(dataset_names)
    else:
        if source in available_sources.keys():
            # Return only for specific source.
            src_module = importlib.import_module(
                'sotaai.neurosym.' + available_sources[source])
            return src_module.models()
        else:
            print("Unavailable source. Here are the options:")
            print(sorted(list(available_sources.keys())))


def load_model(model_name, source=None):
    """Get an instance of wrapped neurosymbolic reasoning model."""
    if source is None:
        # Try every source, and load from the first one that matches.
        for s in model_sources:
            src_module = importlib.import_module(
                'sotaai.neurosym.' + s)
            if model_name in src_module.models():
                return src_module.load_model(model_name)
        print("Unavailable model. Here are the options:")
        print(models())
    else:
        if source in available_sources.keys():
            # Return only for specific source.
            src_module = importlib.import_module(
                'sotaai.neurosym.' + available_sources[source])
            return src_module.load_model(model_name)
        else:
            print("Unavailable source. Here are the options:")
            print(sorted(list(available_sources.keys())))


def load_dataset(dataset_name, source=None):
    """Get an instance of wrapped dataset from available libraries."""
    if source is None:
        # Try every source, and load from the first one that matches.
        for s in dataset_sources:
            src_module = importlib.import_module(
                'sotaai.neurosym.' + s)
            if dataset_name in src_module.datasets():
                return src_module.load_dataset(dataset_name)
        print("Unavailable dataset. Here are the options:")
        print(datasets())
    else:
        if source in available_sources.keys():
            # Return only for specific source.
            src_module = importlib.import_module(
                'sotaai.neurosym.' + available_sources[source])
            return src_module.load_dataset(dataset_name)
        else:
            print("Unavailable source. Here are the options:")
            print(sorted(list(available_sources.keys())))
