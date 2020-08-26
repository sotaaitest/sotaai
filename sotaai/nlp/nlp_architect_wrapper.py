from nlp_architect.models import *



models = ['SequenceChunker',
            'SequencePOSTagger',
            'SequenceTagger',
            'BISTModel',
            'WordTranslator',
            'MultiTaskIntentModel',
            'Seq2SeqIntentModel',   
            'MostCommonWordSense',
            'NERCRF',
            'NP2vec',
            'NpSemanticSegClassifier',
            'NeuralTagger']
model_dic = {}
model_dic['chunker'] = models[:3]
model_dic['bist_parser'] = models[3]
model_dic['crossling_emb'] = models[4]
model_dic['intent_extraction'] = models[5:7]
model_dic['most_common_word_sense'] = models[7]
model_dic['ner_crf'] = models[8]
model_dic['np2vec'] = models[9]
model_dic['np_semantic_segmentation'] = models[10]
model_dic['tagging'] = models[11]

#------



def find_module(model_name):
    for module,model in model_dic.items():
        if model_name in model:
            return module

def load_model(model_name,*kwargs):
    parent = find_module(model_name)
    module = importlib.import_module('nlp_architect.models.'+parent)
    model= getattr(module,model_name)
    if parent == 'chunker':
        model = model(use_cudnn=False)
        model.build(kwargs)
        '''kwargs:
            vocabulary_size (int) – the size of the input vocabulary
            num_pos_labels (int) – the size of of POS labels
            num_chunk_labels (int) – the sie of chunk labels
            char_vocab_size (int, optional) – character vocabulary size
            max_word_len (int, optional) – max characters in a word
            feature_size (int, optional) – feature size - determines the embedding/LSTM layer hidden state size
            dropout (float, optional) – dropout rate
            classifier (str, optional) – classifier layer, ‘softmax’ for softmax or ‘crf’ for conditional random fields classifier. default is ‘softmax’.
            optimizer (tensorflow.python.training.optimizer.Optimizer, optional) – optimizer, if None will use default SGD (paper setup)
        '''
    elif parent == 'bist_parser':
        model = model(activation='tanh', lstm_layers=2, lstm_dims=125, pos_dims=25)
    elif parent == 'crossling_emb':
        model = model(hparams, src_vec, tgt_vec, vocab_size)
    
    elif parent == 'intent_extraction':
        if model_name=='MultiTaskIntentModel':
            model = model(use_cudnn=False)
            model.build(word_length, num_labels, num_intent_labels, word_vocab_size, char_vocab_size, word_emb_dims=100, char_emb_dims=30, char_lstm_dims=30, tagger_lstm_dims=100, dropout=0.2)
        else:
            model = model()
            model.build(vocab_size, tag_labels, token_emb_size=100, encoder_depth=1, decoder_depth=1, lstm_hidden_size=100, encoder_dropout=0.5, decoder_dropout=0.5)

    elif parent == 'most_common_word_sense':
        return model #needs to be trained
    elif parent == 'ner_crf':
        model = model(use_cudnn=False)
        model.build(word_length, target_label_dims, word_vocab_size, char_vocab_size, word_embedding_dims=100, char_embedding_dims=16, tagger_lstm_dims=200, dropout=0.5)

    elif parent== 'np2vec':
        return model
    elif parent == 'np_semantic_segmentation':
        model = model(num_epochs, callback_args, loss='binary_crossentropy', optimizer='adam', batch_size=128)
    elif parent == 'tagging':
        model = tagging.NeuralTagger(embedder_model, word_vocab: nlp_architect.utils.text.Vocabulary, labels: List[str] = None, use_crf: bool = False, device: str = 'cpu', n_gpus=0)
    return model



def predict():
    
    if parent in ['chunker','intent_extraction','ner_crf']:
        model.predict(x, batch_size=1)
    elif parent == 'bist_parser':
        #inputs a CoNLLL formatted dataset
        #outputs dependencies for new input
        model.predict(dataset)
    elif parent == 'crossling_emb':
    
    elif parent == 'most_common_word_sense':
        model.get_outputs(valid_set)
    elif parent== 'np2vec':
    elif parent == 'np_semantic_segmentation':
    elif parent == 'tagging':
        model.inference(examples: list(TokenClsInputExample]) )


def fit(model,dataset,epochs,batch_size):
    if parent in ['chunker','ner_crf']:
        model.fit(x, y, batch_size=batch_size, epochs=epochs, validation_data=None, callbacks=None)

    elif parent == 'bist_parser':
        model.fit(dataset,epochs,dev)
    elif parent == 'crossling_emb':
        model.run(sess,local_lr)
    elif parent == 'intent_extraction':
        model.fit(x,y,epochs = 1,batch_size=1,callbacks=None,validation=None)
    
    elif parent == 'most_common_word_sense':
        model = model(epochs, batch_size, callback_args=None)
        model.build(input_dim)
        model.fit(train_set)
    elif parent== 'np2vec':
        model(dataset, corpus_format='txt', mark_char='_', word_embedding_type='word2vec', sg=0, size=100, window=10, alpha=0.025, min_alpha=0.0001, min_count=5, sample=1e-05, workers=20, hs=0, negative=25, cbow_mean=1, iterations=15, min_n=3, max_n=6, word_ngrams=1, prune_non_np=True)

    elif parent == 'np_semantic_segmentation':
    elif parent == 'tagging':
        #with torch dataloaders
        model.train(train_data_set, dev_data_set, test_data_set, epochs = epochs, batch_size=batch_size, optimizer=None, max_grad_norm: float = 5.0, logging_steps: int = 50, save_steps: int = 100, save_path: str = None, distiller: nlp_architect.nn.torch.distillation.TeacherStudentDistill = None)



def evaluate(model, dataset):
    if parent == 'chunker':
    elif parent == 'bist_parser':
        model.predict(dataset,evaluate=True)
    elif parent == 'crossling_emb':
        model.report_metrics(iters,n_words_proc,disc_cost_acc,tic)

    
    elif parent == 'intent_extraction':
    
    elif parent == 'most_common_word_sense':
        model.eval(valid_set)
    elif parent == 'ner_crf':
    elif parent== 'np2vec':
    elif parent == 'np_semantic_segmentation':
    elif parent == 'tagging':
        model.evaluate(dataset)









#------- pretrained models
pretrained_models.AbsaModel
pretrained_models.BistModel
pretrained_models.IntentModel
pretrained_models.MrcModel
pretrained_models.NerModel

    Usage Example:

    chunker = ChunkerModel.get_instance()
    chunker2 = ChunkerModel.get_instance()
    print(chunker, chunker2)
    print("Local File path = ", chunker.get_file_path())
    files_models = chunker2.get_model_files()
    for idx, file_name in enumerate(files_models):
        print(str(idx) + ": " + file_name)





def download_ds(url:str ='https://github.com/NervanaSystems/nlp-architect/raw/master/datasets/wikipedia/enwiki-20171201_subset.txt.gz'):
    wget.download(url, 'enwiki-20171201_subset.txt.gz')

    corpus='enwiki-20171201_subset.txt.gz'
    marked_corpus = 'enwiki-20171201_subset_marked.txt'
    chunker = 'spacy'
    with gzip.open(corpus, 'rt', encoding='utf8', errors='ignore') as corpus_file, open(marked_corpus, 'w', encoding='utf8') as marked_corpus_file:
        nlp = load_parser(chunker)
        num_lines = sum(1 for line in corpus_file)
        corpus_file.seek(0)
        print('%i lines in corpus', num_lines)
        mark_noun_phrases(corpus_file, marked_corpus_file, nlp, num_lines, chunker)




def load_dataset():

