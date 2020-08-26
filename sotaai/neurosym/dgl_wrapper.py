# -*- coding: utf-8 -*-
# Author: Tonio Teran <tonio@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2020.
import importlib
import argparse
import dgl
import sys
import torch
import torch.nn.functional as F
from dgl.data import register_data_args


DATASETS = {
    'cora': 'CoraDataset',
    'citeseer': 'CiteseerGraphDataset',
    'pubmed': 'PubmedGraphDataset'
}


def load_gcn(dataset):
    """Get an instance of a GCN model from DGL library.

    Based on the examples found in the /examples directory.
    """
    # Import the model source.
    sys.path.insert(0, './dgl/examples/pytorch/')
    from gcn.gcn import GCN

    # TODO(tonioteran) Abstract this away onto a configuration file.
    # Create configuration.
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=16,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--self-loop", action='store_true',
                        help="graph self-loop (default=False)")
    parser.set_defaults(self_loop=False)
    args = parser.parse_args()

    # Get the graph and its properties.
    g = dataset[0]
    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    in_feats = features.shape[1]
    n_classes = dataset.num_labels
    n_edges = dataset.graph.number_of_edges()

    # normalization
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    g.ndata['norm'] = norm.unsqueeze(1)

    # create GCN model
    model = GCN(g,
                in_feats,
                args.n_hidden,
                n_classes,
                args.n_layers,
                F.relu,
                args.dropout)
    return model


def load_gat(dataset):
    """Get an instance of a GAT model from DGL library.

    Based on the examples found in the /examples directory.
    """
    # Import the model source.
    sys.path.insert(0, './dgl/examples/pytorch/')
    from gat.gat import GAT

    parser = argparse.ArgumentParser(description='GAT')
    register_data_args(parser)
    parser.add_argument("--gpu", type=int, default=-1,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--num-heads", type=int, default=8,
                        help="number of hidden attention heads")
    parser.add_argument("--num-out-heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--num-layers", type=int, default=1,
                        help="number of hidden layers")
    parser.add_argument("--num-hidden", type=int, default=8,
                        help="number of hidden units")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--in-drop", type=float, default=.6,
                        help="input feature dropout")
    parser.add_argument("--attn-drop", type=float, default=.6,
                        help="attention dropout")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help="weight decay")
    parser.add_argument('--negative-slope', type=float, default=0.2,
                        help="the negative slope of leaky relu")
    parser.add_argument('--early-stop', action='store_true', default=False,
                        help="indicates whether to use early stop or not")
    parser.add_argument('--fastmode', action="store_true", default=False,
                        help="skip re-evaluate the validation set")
    args = parser.parse_args()

    g = dataset[0]
    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    num_feats = features.shape[1]
    n_classes = dataset.num_labels
    n_edges = dataset.graph.number_of_edges()

    # add self loop
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    n_edges = g.number_of_edges()
    # create model
    heads = ([args.num_heads] * args.num_layers) + [args.num_out_heads]
    model = GAT(g,
                args.num_layers,
                num_feats,
                args.num_hidden,
                n_classes,
                heads,
                F.elu,
                args.in_drop,
                args.attn_drop,
                args.negative_slope,
                args.residual)
    return model


# Dict from model name to loading function.
MODELS = {'GCN': load_gcn,
          'GAT': load_gat
          }


def models():
    """Return a list with all available models."""
    return list(MODELS.keys())


def datasets():
    """Return a list with all available datasets."""
    return list(DATASETS.keys())


def load_model(model_name, dataset_name=None):
    """Get an instance of a DGL model."""
    if dataset_name is None:
        print("Model from DGL library requires a specified dataset.")
        print("Please provide one using keyword argument"
              "`dataset_name=<dataset>`")
        return None
    if model_name in MODELS.keys():
        model_module = importlib.import_module(
            "ampligraph." + MODELS[model_name])
        model = getattr(model_module, model_name)
        return model()
    else:
        print("Model {} not available.".format(model_name))
        print("Available models are: {}".format(list(MODELS.keys())))


def load_dataset(dataset_name):
    """Get the full dataset directly from DGL."""
    if dataset_name in DATASETS.keys():
        ds_module = importlib.import_module("dgl.data")
        ds_fxn = getattr(ds_module, DATASETS[dataset_name])
        return ds_fxn()
    else:
        print("Dataset {} not available.".format(dataset_name))
        print("Available datasets are: {}".format(list(DATASETS.keys())))
