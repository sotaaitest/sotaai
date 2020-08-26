#import huggingface_wrapper
import allennlp_wrapper
#import decathlon_wrapper
import fairseq_wrapper
#import flair_wrapper
#import nlp_architect_wrapper
#import parlai_wrapper
import hanlp_wrapper
#import stanza_wrapper
import sys
from difflib import get_close_matches

flatten = lambda l: [item for sublist in l for item in sublist]


all_datasets = {#'HuggingFace': fastai_wrapper.DATASETS,
    #'Decathlon': mxnet_wrapper.DATASETS,
    #'Flair':torch_wrapper.DATASETS,
    #'NLP-architect': fastai_wrapper.DATASETS,
    #'Parlai': keras_wrapper.DATASETS,
    #'HanLP': mxnet_wrapper.DATASETS,
    #'torch_datasets':torch_wrapper.DATASETS,
    #'tensorflow_datasets':torch_wrapper.DATASETS
    }

allen_models = flatten([list(k.keys()) for k in allennlp_wrapper.MODELS.values()])
fairseq_models = flatten(list(fairseq_wrapper.MODELS.values()))
all_models = {#'HuggingFace': huggingface_wrapper.MODELS,
    'AllenNLP': allen_models,
    #'Decathlon': decathlon_wrapper.MODELS,
    'Fairseq':fairseq_models,
    #'Flair':flair_wrapper.MODELS,
    #'NLP-architect': nlp_architect_wrapper.MODELS,
    #'Parlai': parlai_wrapper.MODELS,
    'HanLP': hanlp_wrapper.MODELS,
    #'Stanza':stanza_wrapper.MODELS
    }



def tasks(source = 'all'):
    ''' 
    List all tasks available in sotaai given a particular source repository

    Input: string. Options: all, HuggingFace, Flair, etc.
    Output: list of tasks
    '''
    tasks = []
    #If source is not specified, go over all repos
    if source == 'all':
        for o in all_datasets:
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


def datasets(task = 'all', source = 'all'):
    ''' 
    List all datasets available in sotaai given a particular source repository
    
    Input: string. Options: all, HuggingFace, Flair, etc.
    Output: list of datasets
    '''
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
    
def models(source = 'all'):
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
        for task, ds in all_datasets[o].items():
            if search_ds in [d.upper() for d in ds]:
                keys.append(o)
    return list(set(keys))

#helper function used in load_model()
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
    if type(possibilities)==dict:
        possibilities = flatten(list(possibilities.values()))
    
    lword = word.upper()
    pos_dic = {p.upper(): p for p in possibilities}
    lmatches = get_close_matches(lword, pos_dic.keys())
    return [pos_dic[m] for m in lmatches]





def load_dataset(name_dataset, source = 'all'):
    sources = find_keys_ds(name_dataset)
    if sources == []:
        return print('Dataset not available in sota')
    
    if source in sources:
        s = 'sota.sota.'+source+'_wrapper'
        match = get_close_matches_icase(name_dataset,all_datasets[source])
        module = getattr(sys.modules[s], 'load_dataset')
    elif source == 'all':
        s = 'sota.sota.'+sources[0]+'_wrapper'
        match = get_close_matches_icase(name_dataset,all_datasets[sources[0]])
        module = getattr(sys.modules[s], 'load_dataset')
    else:
        return print('Dataset not available in '+ source +'but can be found in ' + str(sources))
    
    print(match)

    return module(match[0])


def load_model(name_model, pretrained = False, source = 'all'):
    sources = find_keys_model(name_model)
    
    if sources == []:
        return print('Model not available in sota')
    
    if source in sources:
        s = source.lower()+'_wrapper'
        match = get_close_matches_icase(name_model,all_models[source])
        module = getattr(sys.modules[s], 'load_model')
    elif source == 'all':
        s = sources[0]+'_wrapper'
        match = get_close_matches_icase(name_model,all_models[sources[0]])
        module = getattr(sys.modules[s], 'load_model')
    else:
        return print('Model not available in '+ source +' but can be found in ' + str(sources))
    
    print(match)

    return module(match[0],pretrained = pretrained)


