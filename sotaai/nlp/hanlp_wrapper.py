import hanlp
from hanlp.utils.tf_util import size_of_dataset
from hanlp.common.component import *
import math
from hanlp.datasets import *
from typing import Union
from hanlp.utils.io_util import get_resource
import importlib

DATASETS = {'classification': ['CHNSENTICORP_ERNIE'],
            'cws': ['SIGHAN2005_MSR','SIGHAN2005_PKU','CTB6_CWS'], #chinese word segmentation
            'dep':['CTB5_DEP','CTB7_DEP','SEMEVAL2016_NEWS','SEMEVAL2016_TEXT'], # dependency parsing
            'ner':['MSRA_NER','CONLL03_EN'], #named entity recognition
            'pos':['CTB5_POS'], #part-of-speech tagging
            'rnnlm':[], #RNN language modeling
            'sdp':[]} #semantic dependency parsing

MODELS = list(hanlp.pretrained.ALL.keys())

list1 = dir(hanlp.pretrained.glove)
list2 = dir(hanlp.pretrained.word2vec)
for k in list1:
    try: MODELS.remove(k)
    except: continue
for k in list2:
    try: MODELS.remove(k)
    except: continue


tasks = list(DATASETS.keys())

def find_task_model(model_name):
    '''
    Helper function that finds which task the model belongs to. This is needed to load the model 
    '''
    for task in tasks:
        if task=='classification':
            task='classifiers'
        module = getattr(hanlp.pretrained,task)
        if model_name in dir(module):
            return task

def find_task_ds(ds_name):
    '''
    Helper function that finds which task the dataset belongs to. This is needed to load the dataset
    '''
    for task in tasks:
        if ds_name in DATASETS[task]:
            return task
    return print('Dataset does not exist in HanLP')



def load_dataset(ds_name: str, save_dir=None, batch_size=128, splits = ['train','valid','test']):
    if isinstance(splits,str):
        splits = [splits]
    helper = load_tokenizer('zh') #the tokenizer is used ***exlusively*** to access a class function for loading a dataset
    task = find_task_ds(ds_name)
    ds = {}
    lib = 'hanlp.datasets.'+task
    if task == 'classification':
        lib = importlib.import_module(lib + '.sentiment')
    elif task == 'cws':
        if ds_name=='CTB6_CWS':
            lib = importlib.import_module(lib + '.ctb')
        elif 'MSR' in ds_name:
            lib = importlib.import_module(lib + '.sighan2005.msr')
        else: 
            lib = importlib.import_module(lib + '.sighan2005.pku')
    elif task == 'pos':
        lib = importlib.import_module(lib + '.ctb')
    elif task == 'ner':
        if 'CONLL03' in ds_name:
            lib = importlib.import_module(lib + '.conll03')
        else:
            lib = importlib.import_module(lib + '.msra')
    elif task == 'dep': 
        if 'CTB' in ds_name:
            lib = importlib.import_module(lib[:-3] + 'parsing.ctb')
        else:
            lib = importlib.import_module(lib[:-3] + 'parsing.semeval2016')

    for split in splits:
        url = getattr(lib,ds_name+'_'+split.upper())
        input_path = get_resource(url)
        if split == 'train' and 'SIGHAN2005' in ds_name:
            from hanlp.datasets.cws.sighan2005 import make
            make(url)

        ds[split] = helper.transform.file_to_dataset(input_path, batch_size=batch_size)
    return ds


def load_tokenizer(language,**kwargs):
    '''
    Only option for kwargs is chinese_tok. Default is chinese_tok = 'CTB6_CONVSEG'
    Several models for chinese tokenization are available (check dir(hanlp.pretrained.cws))
    '''
    if language=='chinese' or language=='zh':
        if not kwargs or 'chinese_tok' not in kwargs.keys():
            kwargs = {'chinese_tok':'CTB6_CONVSEG'}
        tokenizer = hanlp.load(kwargs['chinese_tok'])
    elif language=='english' or language=='en':
        tokenizer = hanlp.utils.rules.tokenize_english
    return tokenizer


def load_model(model_name):
    task = find_task_model(model_name)
    if task=='classification':
        task='classifiers'
    if task == None or task in ['cws']:
        return print('Model does not exist in HanLP')
    module = getattr(hanlp.pretrained,task)
    module = getattr(module,model_name)
    model = hanlp.load(module)
    model.name = model_name
    return model



def predict(model, sentence:Union[str,list],tokenizer = None, pos = None):
    task = find_task_model(model.name)
    if task in ['pos','ner','rnnlm']:
        if not tokenizer: #assuming that the sentence is tokenized since no tokenizer was provided
            out = model(sentence)
        else:
            tok_sentence = tokenizer(sentence)
            out = model(tok_sentence)
    
    elif task in ['sdp','dep']:
        if tokenizer:
            sentence = tokenizer(sentence)
        if pos:
            tags = pos(sentence)
        else:
            pos = load_model(MODELS['pos'][0])
            tags = pos(sentence)
        inp = [(sentence[j],tags[j]) for j in range(len(sentence))]
        out = model(inp)
        

    return out
    

def evaluate(model, dataset,batch_size = 128):
    samples = size_of_dataset(dataset)
    num_batches = math.ceil(samples / batch_size)

    loss,score,output = model.evaluate_dataset(dataset, None, False, num_batches)
    return loss,score,output


    