# -*- coding: utf-8 -*-
# Author: Tonio Teran <tonio@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2020.

DATASETS = {
    'cora': 'spektral.datasets.citation',
    'citeseer': 'spektral.datasets.citation',
    'pubmed': 'spektral.datasets.citation',
    'ppi': 'spektral.datasets.graphsage',
    'reddit': 'spektral.datasets.graphsage',
}


def models():
    """Return a list with all available models."""
    return list()


def datasets():
    """Return a list with all available datasets."""
    return list()
