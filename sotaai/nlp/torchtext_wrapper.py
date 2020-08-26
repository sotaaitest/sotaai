import torchtext

DATASETS = {'Sentiment Analysis':
        ['SST',
        'IMDb'],
    'Question Classification':
        ['TREC'],
    'Entailment':
        ['SNLI',
        'MultiNLI'],
    'Language Modeling':
        ['WikiText-2',
        'WikiText103',
        'PennTreebank'],
    'Machine Translation':
        ['Multi30k',
        'IWSLT',
        'WMT14'],
    'Sequence Tagging':
        ['UDPOS',
        'CoNLL2000Chunking'],
    'Question Answering':
        ['BABI20']}



def load_dataset(name_dataset,batch_size = 4):
    
    mod = getattr(torchtext.datasets,name_dataset)
    return mod.iters(batch_size=batch_size)
