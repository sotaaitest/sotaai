import torch
models_with_errors = [ 'camembert.v0',#ValueError: invalid literal for int() with base 10: '#fairseq:overwrite\n'
 'data.stories',#Model file not found: /home/said/.cache/torch/pytorch_fairseq/9a640b54a618c7cf6c76af34eff29132a2a4dd6ede2b1569e49798ea7ad7c08f.7a044b57f93c8f4ce4f0d55063289dd412e9cd2134a0b016a213d8f44ee6cf7b/model.pt
 'roberta.large.wsc', #Keyword error wsc
 'xlmr.base.v0', #does not exist in hub
 'xlmr.large.v0' #does not exist in hub
]

MODELS = {'translate': ['conv.wmt14.en-de',
                        'conv.wmt14.en-fr',
                        'conv.wmt17.en-de',
                        'transformer.wmt14.en-fr', 
                        'transformer.wmt16.en-de',
                        'transformer.wmt18.en-de',
                        'transformer.wmt19.de-en',
                        'transformer.wmt19.de-en.single_model',
                        'transformer.wmt19.en-de',
                        'transformer.wmt19.en-de.single_model',
                        'transformer.wmt19.en-ru',
                        'transformer.wmt19.en-ru.single_model',
                        'transformer.wmt19.ru-en',
                        'transformer.wmt19.ru-en.single_model'
                        ],
            'lm':['transformer_lm.gbw.adaptive_huge',
                                'transformer_lm.wiki103.adaptive',
                                'transformer_lm.wmt19.de',
                                'transformer_lm.wmt19.en',
                                'transformer_lm.wmt19.ru',
                                'conv.stories',
                                'conv.stories.pretrained'],
            'tokenizer':['tokenizer'], #needs tokenizer flag
            'embeddings':['bpe'],#needs bpe and bpe_codes
            'several':[#in particular: classification and masked language modeling
                        'bart.base',
                        'bart.large',
                        'bart.large.cnn',
                        'bart.large.mnli', 
                        'roberta.base', #needs bpe_codes
                        'roberta.large', 
                        'roberta.large.mnli' ]
                        }

tasks = list(MODELS.keys())
#models = torch.hub.list('pytorch/fairseq')

def find_model_task(model_name):
    for task in tasks:
        if model_name in MODELS[task]:
            return task


def load_model(model_name, gpu = False,tokenizer = 'moses', bpe = 'fastbpe', **kwargs):  
    if model_name=='bpe' or model_name == 'roberta.base':
        assert 'bpe_codes' in kwargs.keys(), "Model " + model_name + "requires bpe_codes specified. Add bpe_codes = 'PATH_TO_BPE_CODES"
    model = torch.hub.load('pytorch/fairseq', model_name, tokenizer = tokenizer, bpe = bpe, *kwargs) #tokenizer='moses', bpe='fastbpe',force_reload = True) #bpe = 'subword_nmt'
    # Move model to GPU for faster translation
    if gpu:
        model.cuda()
    model.task=find_model_task(model_name)
    return model

'''def predict(model,string:str,**kwargs):
    if model.task=='translate':
        out = model.translate(string)
    elif model.task == 'lm':
        out = en_lm.sample(string, *kwargs) #beam=1, sampling=True, sampling_topk=10, temperature=0.8
    elif model.task == 'several':
        assert kwargs.task, 'Select particular task for this model. Tasks can be classification or masked language modeling'
        if kwargs.task == 'classification':
            assert kwargs.string2, 'Choose a second string, string2, to perform sentence-pair classification'
            tokens = model.encode(string,kwargs.string2)
            out = model.predict('mnli',tokens).argmax.item()
        else:
            out = model.fill_mask(string) 
    return out
'''

