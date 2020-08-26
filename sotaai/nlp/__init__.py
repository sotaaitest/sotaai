# -*- coding: utf-8 -*-
# Author: Liuba Orlova <liuba@stateoftheart.ai>
# Author: Tonio Teran <tonio@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2020.
from sotaai.nlp import huggingface_wrapper
from sotaai.nlp import allennlp_wrapper
# from sotaai.nlp import decathlon_wrapper
from sotaai.nlp import fairseq_wrapper
from sotaai.nlp import flair_wrapper
# from sotaai.nlp import nlp_architect_wrapper
# from sotaai.nlp import parlai_wrapper
from sotaai.nlp import hanlp_wrapper
from sotaai.nlp import stanza_wrapper
from sotaai.nlp import torchtext_wrapper
from sotaai.nlp import tensorflow_datasets_wrapper
from importlib import import_module
import sys
from difflib import get_close_matches

from sotaai.nlp.flair_wrapper import load_embedding



sys.path.insert(1, '../')

nlp_module = import_module('sotaai.nlp')

def flatten(l): return [item for sublist in l for item in sublist]

embedding_types = ['BPE', 'Character', 'Word', 'Flair']


all_datasets = {'huggingface': huggingface_wrapper.DATASETS,
                #'Decathlon': mxnet_wrapper.DATASETS,
                'flair': flair_wrapper.DATASETS,
                #'NLP-architect': fastai_wrapper.DATASETS,
                #'Parlai': keras_wrapper.DATASETS,
                'hanlp': hanlp_wrapper.DATASETS,
                'torchtext': torchtext_wrapper.DATASETS,
                'tensorflow_datasets': tensorflow_datasets_wrapper.DATASETS
                }

allen_models = flatten([list(k.keys())
                        for k in allennlp_wrapper.MODELS.values()])
fairseq_models = flatten(list(fairseq_wrapper.MODELS.values()))
flair_models = flatten(list(flair_wrapper.MODELS.values()))
all_models = {
    'huggingface': huggingface_wrapper.MODELS,
    'allennlp': allen_models,
    # 'Decathlon': decathlon_wrapper.MODELS,
    'fairseq': fairseq_models,
    'flair':flair_models,
    # 'NLP-architect': nlp_architect_wrapper.MODELS,
    # 'Parlai': parlai_wrapper.MODELS,
    'hanlp': hanlp_wrapper.MODELS,
    'stanza':stanza_wrapper.MODELS
}


def tasks(source='all'):
    '''
    List all tasks available in sotaai given a particular source repository

    Input: string. Options: all, HuggingFace, Flair, etc.
    Output: list of tasks
    '''
    tasks = []
    # If source is not specified, go over all repos
    if source == 'all':
        for o in all_datasets:
            if o == 'huggingface':
                continue
            tasks.append(list(all_datasets[o].keys()))
        tasks = flatten(tasks)
    else:
        tasks = list(all_datasets[source].keys())
    return sorted(list(set(tasks)))


def _unique_values(dic):
    '''
    Given a dictionary, return the set of its values.
    '''
    vals = list(dic.values())
    vals = flatten(vals)
    return list(set(vals))


def datasets(task='all', source='all'):
    '''
    List all datasets available in sotaai given a particular source repository

    Input: string. Options: all, HuggingFace, Flair, etc.
    Output: list of datasets
    '''
    if source == 'all':
        ds = []
        if task == 'all':
            for o in all_datasets:
                if o == 'huggingface':
                    ds.append(list(all_datasets[o].keys()))
                    continue
                ds.append(_unique_values(all_datasets[o]))
            ds = flatten(ds)
        else:
            for o in all_datasets:
                if task in all_datasets[o].keys():
                    ds.append(all_datasets[o][task])
            ds = flatten(ds)
    else:
        if task == 'all':
            if source == 'huggingface':
                ds = list(all_datasets[source].keys())
            else:   
                ds = _unique_values(all_datasets[source])
        else:
            ds = all_datasets[source][task]
            ds = [str.upper(j) for j in ds]

    ds = [str.upper(j) for j in ds]
    return sorted(list(set(ds)))


def models(source='all'):
    '''
    List all models available in sotaai given a particular source repository

    Input: string. Options: all, HuggingFace, Flair, etc.
    Output: list of models
    '''
    if source == 'all':
        mods = _unique_values(all_models)

    else:
        mods = all_models[source]
    mods = [str.upper(j) for j in mods]
    return sorted(list(set(mods)))


def find_keys_ds(search_ds):
    '''
    Finds the repository that contains a given dataset
    Input: dataset name
    Output: List of repositories
    '''
    search_ds = search_ds.upper()
    keys = []
    for o in all_datasets:

        if o == 'huggingface':
            if search_ds.lower() in list(all_datasets['huggingface'].keys()):
                keys.append(o)

        else:
            for _, ds in all_datasets[o].items():
                if search_ds in [d.upper() for d in ds]:
                    keys.append(o)
    return list(set(keys))

# helper function used in load_model()


def find_keys_model(search_model):
    '''
    Finds the repository that contains a given model
    Input: model name
    Output: List of repositories
    '''
    search_model = search_model.upper()
    keys = []
    for o in all_models:
        if search_model in [d.upper() for d in all_models[o]]:
            keys.append(o)
    return list(set(keys))


def get_close_matches_icase(word, possibilities, *args, **kwargs):
    """ Case-insensitive version of difflib.get_close_matches
    Used for case-insensitive model and dataset arguments in future functions"""
    if type(possibilities) == dict:
        possibilities = flatten(list(possibilities.values()))

    lword = word.upper()
    pos_dic = {p.upper(): p for p in possibilities}
    lmatches = get_close_matches(lword, pos_dic.keys())
    return [pos_dic[m] for m in lmatches]


def load_dataset(name_dataset, source='all', config = None):
    sources = find_keys_ds(name_dataset)
    if sources == []:
        return print('Dataset not available in sota')

    if source in sources:
        s = getattr(nlp_module, source + '_wrapper')
        if source == 'huggingface':
            match = get_close_matches_icase(name_dataset, list(all_datasets['huggingface'].keys()))
        else:
            match = get_close_matches_icase(name_dataset, all_datasets[source])
        module = getattr(s, 'load_dataset')
    elif source == 'all':
        s = getattr(nlp_module, sources[0] + '_wrapper')
        if sources[0] == 'huggingface':
            match = get_close_matches_icase(name_dataset, list(all_datasets['huggingface'].keys()))
        else:
            match = get_close_matches_icase(name_dataset, all_datasets[sources[0]])
        module = getattr(s, 'load_dataset')
    else:
        return print('Dataset not available in ' + source + 'but can be found in ' + str(sources))

    print(match)

    try: 
        model = module(match[0],config = config)
    except:
        model =  module(match[0])

    return model

def load_tokenizer(tok,source = 'all', **kwargs):
    if source == 'all':
        sources = ['huggingface', 'stanza', 'hanlp']
    else:
        sources = [source]
    counter = 0
    for s in sources:
        try:
            module = getattr(nlp_module, s + '_wrapper')
            func = getattr(module, 'load_tokenizer')
            tokenizer = func(tok.lower(), *kwargs)
            break
        except:
            counter+=1
            if counter == len(sources):
                return ('Tokenizer is either mispelled or not available in sotaai')
            continue
        
    return tokenizer



def load_model(name_model, source='all', **kwargs):

    sources = find_keys_model(name_model)

    if sources == [] or source == 'stanza':  #TODO Liuba: Update stanza_wrapper.MODELS
        if 'task' in kwargs.keys():
            return stanza_wrapper.load_model(name_model, kwargs['task'])
        else:
            return print('Specify task')
        #return print('Model not available in sota')

        
    if source in sources:

        s = getattr( nlp_module, source + '_wrapper')
        match = get_close_matches_icase(name_model, all_models[source])
        module = getattr(s, 'load_model')

        if source == 'huggingface':
            if 'task' not in kwargs:
                return print('Specify task from: ' + str(huggingface_wrapper.tasks()) )
            else:
                return module(match[0],kwargs['task'])

    elif source == 'all':

        s = getattr( nlp_module, sources[0] + '_wrapper')
        match = get_close_matches_icase(name_model, all_models[sources[0]])
        module = getattr(s, 'load_model')

        if sources[0] == 'huggingface':
            if 'task' not in kwargs:
                return print('Specify task from: ' + str(huggingface_wrapper.tasks()) )
            else:
                return module(match[0],kwargs['task'])
    else:
        return print('Model not available in ' + source + ' but can be found in ' + str(sources))

    print(match)

    return module(match[0])
