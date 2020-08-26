from allennlp.predictors.predictor import Predictor
from allennlp_models import *
from allennlp.data import DataLoader
import allennlp


MODELS = {'Question answering':{
                    'ELMo-BiDAF': "https://storage.googleapis.com/allennlp-public-models/bidaf-elmo-model-2020.03.19.tar.gz",#trained on Squad
                    'BiDAF':"https://storage.googleapis.com/allennlp-public-models/bidaf-model-2020.03.19.tar.gz",#SQuAD
                    'Transformer QA': "https://storage.googleapis.com/allennlp-public-models/transformer-qa-2020-05-26.tar.gz",#SQuAD
                    'NAQANet':"https://storage.googleapis.com/allennlp-public-models/naqanet-2020.02.19.tar.gz",#DROP
                    },
                    'Named Entity Recognition':{
                        'elmo-ner':"https://storage.googleapis.com/allennlp-public-models/ner-model-2020.02.10.tar.gz",
                        'fine-grained-ner':"https://storage.googleapis.com/allennlp-public-models/fine-grained-ner.2020-06-24.tar.gz"

                    },
                    'Open Information Extraction':{
                        'BiLSTM':"https://storage.googleapis.com/allennlp-public-models/openie-model.2020.03.26.tar.gz"
                    },
                    'Sentiment Analysis':{
                        'Glove-LSTM':"https://storage.googleapis.com/allennlp-public-models/basic_stanford_sentiment_treebank-2020.06.09.tar.gz",
                        'RoBERTa':"https://storage.googleapis.com/allennlp-public-models/sst-roberta-large-2020.06.08.tar.gz"
                    },
                    'Dependency Parsing':{
                        'Deep Biaffine Attention for Neural Dependency Parsing':"https://storage.googleapis.com/allennlp-public-models/biaffine-dependency-parser-ptb-2020.04.06.tar.gz"
                    },
                    'Constituency Parsing':{
                    'Extending a Parser to Distant Domains Using a Few Dozen Partially Annotated Examples': "https://storage.googleapis.com/allennlp-public-models/elmo-constituency-parser-2020.02.10.tar.gz"
                    },
                    'Semantic Role Labeling':{
                        'A BERT based model':"https://storage.googleapis.com/allennlp-public-models/bert-base-srl-2020.03.24.tar.gz"
                    },
                    'Conference Resolution':{
                        'End-to-end Neural Coreference Resolution':"https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz"
                    }
        }


tasks = list(MODELS.keys())


def find_task(model_name):
    for task in MODELS:
        if model_name in MODELS[task].keys():
            return task
    print('Model not in AllenNLP library')
    
def load_model(model_name):
    task = find_task(model_name)
    if not task:
        return None
    path = MODELS[task][model_name]
    predictor = Predictor.from_path(path)
    predictor.task = task
    return predictor


def predict(model,passage = None, sentence = None):
    if model.task in ['Question Answering']:
        prediction = model.predict(
            passage=passage,
            question=sentence
            )
    elif model.task in ['Named Entity Recognition','Open Information Extraction','Sentiment Analysis',
    'Dependency Parsing','Constituency Parsing', 'Semantic Role Labeling']:
        prediction = model.predict(sentence=sentence)
    else: #Conference Resolution
        prediction = model.predict(
            document=passage)
    return prediction



def evaluate(model,dataset:DataLoader):
    return allennlp.training.util.evaluate(model,dataset)

