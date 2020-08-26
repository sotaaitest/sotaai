# -*- coding: utf-8 -*-
# Author: Liuba Orlova <liuba@stateoftheart.ai>
# Author: Tonio Teran <tonio@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2020.
from sotaai.cv import fastai_wrapper
from sotaai.cv import cadene_wrapper
from sotaai.cv import keras_wrapper
from sotaai.cv import isr_wrapper
from sotaai.cv import mxnet_wrapper
from sotaai.cv import torch_wrapper  # TODO(tonioteran) Re-enable.
from sotaai.cv import tensorflow_wrapper
from sotaai.cv import gans_keras_wrapper
from sotaai.cv import gans_wrapper
from sotaai.cv import mxnet_utils, utils
from importlib import import_module
import copy
from difflib import get_close_matches

cv_module = import_module('sotaai.cv')

all_datasets = {
    'fastai': fastai_wrapper.DATASETS,
    'keras': keras_wrapper.DATASETS,
    'mxnet': mxnet_wrapper.DATASETS,
    'tensorflow': tensorflow_wrapper.DATASETS,
    # TODO(tonioteran) commented by me, temporarily.
    'torch': torch_wrapper.DATASETS,
    # adding empty lists for task documentation purposes solely
    'isr': {'image super resolution': []},
    'gans_keras': {'GANs': []},
    'gans': {'GANs': []}
}

all_models = {
    'fastai': fastai_wrapper.MODELS,
    'keras': keras_wrapper.MODELS,
    # TODO(tonioteran) commented by me, temporarily.
    'torch': torch_wrapper.MODELS,
    'mxnet': mxnet_wrapper.MODELS,
    'cadene': cadene_wrapper.MODELS,
    'isr': isr_wrapper.MODELS,
    'gans_keras': gans_keras_wrapper.MODELS,
    'gans': gans_wrapper.MODELS
}


def flatten(l): return [item for sublist in l for item in sublist]


def tasks(source='all'):
    def flatten(l): return [item for sublist in l for item in sublist]
    tasks = []

    if source == 'all':
        for o in all_datasets:
            tasks.append(list(all_datasets[o].keys()))
        tasks = flatten(tasks)
    else:
        tasks = list(all_datasets[source].keys())
    return sorted(list(set(tasks)))


def _unique_values(dic):
    def flatten(l): return [item for sublist in l for item in sublist]
    vals = list(dic.values())
    vals = flatten(vals)
    return list(set(vals))


def datasets(task='all', source='all'):
    def flatten(l): return [item for sublist in l for item in sublist]

    if source == 'all':
        ds = []
        if task == 'all':
            for o in all_datasets:
                ds.append(_unique_values(all_datasets[o]))
            ds = flatten(ds)
        else:
            for o in all_datasets:
                if task in all_datasets[o].keys():
                    ds.append(all_datasets[o][task])
            ds = flatten(ds)
    else:
        if task == 'all':
            ds = _unique_values(all_datasets[source])
        else:
            ds = all_datasets[source][task]
            ds = [str.upper(j) for j in ds]

    ds = [str.upper(j) for j in ds]
    return sorted(list(set(ds)))


def models(source='all'):
    if source == 'all':
        mods = _unique_values(all_models)

    else:
        mods = all_models[source]
    mods = [str.upper(j) for j in mods]
    return sorted(list(set(mods)))


def find_keys_ds(search_ds):
    search_ds = search_ds.upper()
    keys = []
    for o in all_datasets:
        for task, ds in all_datasets[o].items():
            if search_ds in [d.upper() for d in ds]:
                keys.append(o)
    return list(set(keys))


def find_keys_model(search_model):
    search_model = search_model.upper()
    keys = []
    for o in all_models:
        if search_model in [d.upper() for d in all_models[o]]:
            keys.append(o)
    return list(set(keys))


def get_close_matches_icase(word, possibilities, *args, **kwargs):
    """ Case-insensitive version of difflib.get_close_matches """
    if type(possibilities) == dict:
        possibilities = flatten(list(possibilities.values()))

    lword = word.upper()
    pos_dic = {p.upper(): p for p in possibilities}
    lmatches = get_close_matches(lword, pos_dic.keys())
    return [pos_dic[m] for m in lmatches]


def load_dataset(name_dataset, source='all'):
    sources = find_keys_ds(name_dataset)
    if sources == []:
        return print('Dataset not available in sota')

    if source in sources:
        s = getattr(cv_module, source + '_wrapper')
        match = get_close_matches_icase(name_dataset, all_datasets[source])
        module = getattr(s, 'load_dataset')
    elif source == 'all':
        s = getattr(cv_module, sources[0] + '_wrapper')
        match = get_close_matches_icase(name_dataset, all_datasets[sources[0]])
        module = getattr(s, 'load_dataset')
    else:
        return print('Dataset not available in ' + source + 'but can be found in ' + str(sources))

    print(match)

    return module(match[0])


def load_model(name_model, pretrained=False, source='all'):
    sources = find_keys_model(name_model)
    if sources == []:
        return print('Model not available in sota')

    if source in sources:
        s = getattr( cv_module, source + '_wrapper')
        match = get_close_matches_icase(name_model, all_models[source])
        module = getattr(s, 'load_model')
    elif source == 'all':
        s = getattr( cv_module, sources[0] + '_wrapper')
        match = get_close_matches_icase(name_model, all_models[sources[0]])
        module = getattr(s, 'load_model')
    else:
        return print('Model not available in ' + source + ' but can be found in ' + str(sources))

    print(match)

    return module(match[0], pretrained=pretrained)


from mxnet.gluon.data.vision.transforms import Resize
def model_to_dataset(model, dataset):
    mod = copy.deepcopy(model)
    if 'mxnet' in str(type(model)):
        min_size = mxnet_utils.find_im_size(mod)
        if 'torch' in str(type(dataset)):
            n_classes = _number_classes(dataset)
            mxnet_ds = utils.ds_torch2mxnet(dataset)
        mod = mxnet_wrapper.adapt_last_layer(mod, n_classes)
        return mod, mxnet_ds
    else:
        print('Conversion not suuported yet')

def _number_classes(dataset):
    if 'torch' in str(type(dataset)):
        c = len(dataset.classes)
        return c


def _transformer(data, label,im_size):
    data = mx.image.imresize(data, im_size, im_size)
    data = mx.nd.transpose(data, (2,0,1))
    data = data.astype(np.float32)
    return data, label

