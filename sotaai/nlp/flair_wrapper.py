import flair.datasets
import flair.models
from flair.data import Sentence
from flair.embeddings import *
from typing import Union


#named entity recognition (NER), part-of-speech tagging (PoS), sense disambiguation and classification.
DATASETS = {'Text Classification':['AMAZON_REVIEWS',
                                    'COMMUNICATIVE_FUNCTIONS',
                                    'IMDB', 
                                    'NEWSGROUPS',
                                    'SENTEVAL_CR',
                                    'SENTEVAL_MPQA',
                                    'SENTEVAL_MR',
                                    'SENTEVAL_SST_BINARY',
                                    'SENTEVAL_SST_GRANULAR',
                                    'SENTEVAL_SUBJ',
                                    'SENTIMENT_140',
                                    'TREC_50',
                                    'TREC_6'],
            'Named Entity Recognition':['BIOFID',
                                    'CONLL_03',
                                    'CONLL_03_DUTCH',
                                    'CONLL_03_GERMAN',
                                    'CONLL_03_SPANISH',
                                    'DANE',
                                    'LER_GERMAN',
                                    'NER_BASQUE',
                                    'NER_FINNISH',
                                    'NER_SWEDISH',
                                    'WIKINER_DUTCH',
                                    'WIKINER_ENGLISH',
                                    'WIKINER_FRENCH',
                                    'WIKINER_GERMAN',
                                    'WIKINER_ITALIAN',
                                    'WIKINER_POLISH',
                                    'WIKINER_PORTUGUESE',
                                    'WIKINER_RUSSIAN',
                                    'WIKINER_SPANISH',
                                    'WNUT_17'],
 'chunking':['CONLL_2000'],

 'Similarity Learning':[ 'FeideggerCorpus'],
 'other': ['GERMEVAL_14'],
'Keyword/keyphrase extraction':['INSPEC',
                                'SEMEVAL2010',
                                'SEMEVAL2017'],

'Universal Dependency Treebanks':['UD_ARABIC',
                                    'UD_BASQUE',
                                    'UD_BULGARIAN',
                                    'UD_CATALAN',
                                    'UD_CHINESE',
                                    'UD_CROATIAN',
                                    'UD_CZECH',
                                    'UD_DANISH',
                                    'UD_DUTCH',
                                    'UD_ENGLISH',
                                    'UD_FINNISH',
                                    'UD_FRENCH',
                                    'UD_GERMAN',
                                    'UD_GERMAN_HDT',
                                    'UD_HEBREW',
                                    'UD_HINDI',
                                    'UD_INDONESIAN',
                                    'UD_ITALIAN',
                                    'UD_JAPANESE',
                                    'UD_KOREAN',
                                    'UD_NORWEGIAN',
                                    'UD_PERSIAN',
                                    'UD_POLISH',
                                    'UD_PORTUGUESE',
                                    'UD_ROMANIAN',
                                    'UD_RUSSIAN',
                                    'UD_SERBIAN',
                                    'UD_SLOVAK',
                                    'UD_SLOVENIAN',
                                    'UD_SPANISH',
                                    'UD_SWEDISH',
                                    'UD_TURKISH'],
'Text Regression':['WASSA_ANGER',
                    'WASSA_FEAR',
                    'WASSA_JOY',
                    'WASSA_SADNESS',
                    ]
}



#DATASETS['biomedical'] = [el for el in list(dir(flair.datasets.biomedical)) if el == el.upper()]

MODELS = {'SequenceTagger':['ner','ner-fast','ner-ontonotes','ner-ontonotes-fast','ner-multi','multi-ner',
            'ner-multi-fast','multi-ner-fast','ner-multi-fast-learn','multi-ner-fast-learn',
            'upos','pos','upos-fast','pos-fast','pos-multi','multi-pos','pos-multi-fast','multi-pos-fast',
            'frame','frame-fast','chunk','chunk-fast','da-pos','da-ner','de-pos','de-pos-tweets','de-ner',
            'de-ner-germeval','fr-ner','nl-ner','nl-ner-rnn','ml-pos','ml-upos','keyphrase','negation-speculation',
            "de-historic-indirect","de-historic-direct","de-historic-reported","de-historic-free-indirect"],
            'TextClassifier':["de-offensive-language","sentiment","en-sentiment","sentiment-fast","communicative-functions"]}


def find_task_model(model_name):
    task = ' '
    for task_name, model_list in MODELS.items():
        if model_name in model_list:
            task = task_name

    return task

def find_task(ds_name):
    tasks = []
    for task, ds_list in DATASETS.items():
        if ds_name in ds_list:
            tasks.append(task)

    return tasks

def load_dataset(ds_name:Union[list,str]):
    if isinstance(ds_name,str):
        mod = getattr(flair.datasets,ds_name)
        corpus = mod()
    else:
        corpuses = []
        for ds in ds_name:
            mod = getattr(flair.datasets,ds_name)
            corpuses.append(mod())
        corpus = MultiCorpus(corpuses)

    return corpus




def load_embedding( emb_type, emb = 'en', **kwargs):
    if emb_type == 'BPE':
        emb = BytePairEmbeddings(emb) #275 languages  check .md file
    elif emb_type == 'Character':
        emb = CharacterEmbeddings() #not meaningful, needs to be trained
    elif emb_type == 'Word':
        emb = WordEmbeddings(emb) #also several languages  check .md file
    elif emb_type == 'Flair':
        if 'pooled' in kwargs.keys():
            emb = PooledFlairEmbeddings(emb)
        else:
            emb = FlairEmbeddings(emb)
    '''elif emb_type == 'ELMo':
        if 'embedding_mode' not in kwargs.keys():
            kwargs['embedding_mode'] = 'all'
        emb = ELMoEmbeddings(model = emb,embedding_mode = kwargs['embedding_mode']) #model = pt small,medium,original,large, pt, pubmed. embedding_mode = all, top, average
    '''
    return emb

def load_model(model_name):
    task = find_task_model(model_name)
    module = getattr(flair.models, task)
    model = module.load(model_name)
    return model
    

def predict(model,string:str):
    sentence= Sentence(string)
    model.predict(sentence)
    if isinstance(model, flair.models.TextClasssifier):
        result = sentence.labels
    elif isinstance(model,flair.models.SequenceTagger):
        result = sentence.to_tagged_string()
    #How to generalize to user defined models?
    return result




