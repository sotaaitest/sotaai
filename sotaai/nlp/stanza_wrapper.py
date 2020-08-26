import stanza
from typing import Union

MODELS = []

def load_tokenizer(lang,corpus = 'default'):
    stanza.download(lang,package = corpus)
    nlp = stanza.Pipeline(lang, package = corpus,processors = 'tokenize')
    tok = nlp.processors['tokenize']
    return tok

def load_model(lang, task: Union[str,list], corpus = 'default'):
    stanza.download(lang,package = corpus)
    
    if isinstance(task,str):
        tasks = ','.join([task])
    else: tasks = ','.join(task)
    
    if 'tokenize' not in tasks:
        tasks = 'tokenize,'+tasks
    nlp = stanza.Pipeline(lang,package = corpus,processors = tasks)

    models = {}
    for t in tasks:
        models[t] = nlp.processors[t]
    return models


def predict(tokenizer, model, sentence:str):
    out = tokenizer.process(sentence)
    out = model.process(out)
    return out


