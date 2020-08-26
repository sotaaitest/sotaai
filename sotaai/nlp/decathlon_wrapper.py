import decaNLP
from predict.py import *
import wget
import tarfile

models = ['CoattentivePointerGenerator',
 'MultitaskQuestionAnsweringNetwork',
 'PointerGenerator',
 'SelfAttentivePointerGenerator']


aws_url = 'https://s3.amazonaws.com/research.metamind.io/decaNLP/pretrained/'
path_to_pretrained = {
    'all':   'mqan_decanlp_better_sampling_cove_cpu',
    'wikisql':    'mqan_wikisql_cpu.tar.gz',
    'squad':  'squad_mqan_cove_cpu',
    'cnn_dailymail': 'cnn_dailymail_mqan_cove_cpu',
    'iwslt.en.de':  'iwslt.en.de_mqan_cove_cpu',
    'sst':  'sst_mqan_cove_cpu',
    'multinli.in.out': 'multinli.in.out_mqan_cove_cpu',
    'woz.en':'woz.en_mqan_cove_cpu',
    'srl':'srl_mqan_cove_cpu',
    'zre':'zre_mqan_cove_cpu',
    'schema': 'schema_mqan_cove_cpu',
}
task_to_metric = {'cnn_dailymail': 'avg_rouge',
        'iwslt.en.de': 'bleu',
        'multinli.in.out': 'em',
        'squad': 'nf1',
        'srl': 'nf1',
        'sst': 'em',
        'wikisql': 'lfem',
        'woz.en': 'joint_goal_em',
        'zre': 'corpus_f1',
        'schema': 'em'}

def retrieve_tarfile(task):
    return path_to_pretrained[task]

def download_and_extract_weights(task, out_dir = '.'):
    file_name = retrieve_tarfile(task)
    if file_name not in os.listdir(out_dir):
        file_name = wget.download(aws_url + file_name + '.tgz', out = out_dir)
        tar = tarfile.open(file_name)
        tar.extractall()
        tar.close
        os.remove(file_name + '.tgz')
    return file_name
    
#TRAINING
#python /decaNLP/train.py 
#       --train_tasks squad 
#       --device 0"
#python /decaNLP/train.py 
#       --train_tasks squad iwslt.en.de 
#       --train_iterations 1 
#       --device 0"
#python /decaNLP/train.py 
#       --train_tasks squad iwslt.en.de cnn_dailymail multinli.in.out sst srl zre woz.en wikisql schema 
#       --train_iterations 1 
#       --device 0"
#python /decaNLP/train.py 
#       --n_jump_start 1 
#       --jump_start 75000 
#       --train_tasks squad iwslt.en.de cnn_dailymail multinli.in.out sst srl zre woz.en wikisql schema 
#       --train_iterations 1 
#       --device 0"

#EVALUATION
#python /decaNLP/predict.py 
#       --evaluate EVALUATION_TYPE 
#       --path PATH_TO_CHECKPOINT_DIRECTORY 
#       --device 0 
#       --tasks squad"

#"python /decaNLP/predict.py 
#       --evaluate EVALUATION_TYPE 
#       --path PATH_TO_CHECKPOINT_DIRECTORY 
#       --device 0

#PRETRAINED MODELS

#python /decaNLP/predict.py 
#       --evaluate validation 
#       --path /decaNLP/mqan_decanlp_better_sampling_cove_cpu/ 
#       --checkpoint_name iteration_560000.pth 
#       --device 0 
#       --silent"


# python /decaNLP/predict.py 
#       --evaluate validation 
#       --path /decaNLP/mqan_wikisql_cpu 
#       --checkpoint_name iteration_57000.pth 
#       --device 0 
#       --tasks wikisql"
# python /decaNLP/predict.py 
#       --evaluate test 
#       --path /decaNLP/mqan_wikisql_cpu 
#       --checkpoint_name iteration_57000.pth 
#       --device 0 
#       --tasks wikisql"

# python /decaNLP/WikiSQL/evaluate.py 
#       /decaNLP/.data/wikisql/data/dev.jsonl   
#       /decaNLP/.data/wikisql/data/dev.db 
#       /decaNLP/mqan_wikisql_cpu/iteration_57000/validation/wikisql_logical_forms.jsonl"
#        
#       # assumes that you have data stored in .data
# python /decaNLP/WikiSQL/evaluate.py 
#       /decaNLP/.data/wikisql/data/test.jsonl 
#       /decaNLP/.data/wikisql/data/test.db 
#       /decaNLP/mqan_wikisql_cpu/iteration_57000/test/wikisql_logical_forms.jsonl
#       
#        # assumes that you have data stored in .data

#CUSTOM 
#python /decaNLP/predict.py --evaluate valid --path /decaNLP/mqan_decanlp_qa_first_cpu --checkpoint_name iteration_1140000.pth --tasks my_custom_dataset"

def get_model():
    device = set_seed(args, rank=rank)
    logger = initialize_logger(args, rank)
    field, train_sets, val_sets, save_dict = run_args

    logger.start = time.time()

    logger.info(f'Preparing iterators')
    train_iters = [(name, to_iter(args, world_size, tok, x, device, token_testing=args.token_testing)) 
                      for name, x, tok in zip(args.train_tasks, train_sets, args.train_batch_tokens)]
    val_iters = [(name, to_iter(args, world_size, tok, x, device, train=False, token_testing=args.token_testing, sort=False if 'sql' in name else None))
                    for name, x, tok in zip(args.val_tasks, val_sets, args.val_batch_size)]

    if hasattr(args, 'tensorboard') and args.tensorboard:
        logger.info(f'Initializing Writer')
        writer = SummaryWriter(log_dir=args.log_dir)
    else:
        writer = None

    model = init_model(args, field, logger, world_size, device)
    opt = init_opt(args, model) 
    start_iteration = 1

    if save_dict is not None:
        logger.info(f'Loading model from {os.path.join(args.save, args.load)}')
        save_dict = torch.load(os.path.join(args.save, args.load))
        model.load_state_dict(save_dict['model_state_dict'])
        if args.resume:
            logger.info(f'Resuming Training from {os.path.splitext(args.load)[0]}_rank_{rank}_optim.pth')
            opt.load_state_dict(torch.load(os.path.join(args.save, f'{os.path.splitext(args.load)[0]}_rank_{rank}_optim.pth')))
            start_iteration = int(os.path.splitext(os.path.basename(args.load))[0].split('_')[1])
    return model

def load_model(model_name, tasks = 'all', pretrained = False, load = None,save = True):
    

    if model_name not in models:
        return model_name +' is not supported in decaNLP'

    if pretrained:
        Model = getattr(decaNLP.models, model_name)


    else:
        args = {}
        args.load = load
        args.save = save
        args.world_size = 1
        field, save_dict = None, None
        if args.load is not None:
            logger.info(f'Loading field from {os.path.join(args.save, args.load)}')
            save_dict = torch.load(os.path.join(args.save, args.load))
            field = save_dict['field']
        field, train_sets, val_sets = prepare_data(args, field, logger)

        run_args = (field, train_sets, val_sets, save_dict)
        
        logger.info(f'Processing')
        Model = get_model()
        run(args, run_args, world_size=args.world_size)

    return Model

def load_dataset():


def train():

def evaluate():

def predict():
    
def set_args():
    args.root = '/decaNLP' #root directory for data, results, embeddings, code, etc.
    args.data = '.data/' # where to load data from.')
    args.save = 'results' # where to save results.')
    args.embeddings = '.embeddings' # where to save embeddings.')
    args.name = '' # name of the experiment; if blank, a name is automatically generated from the arguments')

    args.train_tasks', nargs='+' # tasks to use for training', required=True)
    args.train_iterations', nargs='+' # number of iterations to focus on each task')
    args.train_batch_tokens', nargs='+ = [9000] # Number of tokens to use for dynamic batching, corresponging to tasks in train tasks')
    args.jump_start = 0 # number of iterations to give jump started tasks')
    args.n_jump_start = 0 # how many tasks to jump start (presented in order)')    
    args.num_print = 15 # how many validation examples with greedy output to print to std out')

    args.no_tensorboard', action='store_false', dest='tensorboard', help='Turn of tensorboard logging') 
    args.log_every = int(1e2) # how often to log results in # of iterations')
    args.save_every = int(1e3) # how often to save a checkpoint in # of iterations')

    args.val_tasks', nargs='+' # tasks to collect evaluation metrics for')
    args.val_every = int(1e3) # how often to run validation in # of iterations')
    args.val_no_filter', action='store_false', dest='val_filter', help='whether to allow filtering on the validation sets')
    args.val_batch_size', nargs='+ = [256] # Batch size for validation corresponding to tasks in val tasks')

    args.vocab_tasks', nargs='+' # tasks to use in the construction of the vocabulary')
    args.max_output_length = 100 # maximum output length for generation')
    args.max_effective_vocab = int(1e6) # max effective vocabulary size for pretrained embeddings')
    args.max_generative_vocab = 50000 # max vocabulary for the generative softmax')
    args.max_train_context_length = 400 # maximum length of the contexts during training')
    args.max_val_context_length = 400 # maximum length of the contexts during validation')
    args.max_answer_length = 50 # maximum length of answers during training and validation')
    args.subsample = 20000000 # subsample the datasets')
    args.preserve_case', action='store_false', dest='lower', help='whether to preserve casing for all text')

    args.model', type=str, default='MultitaskQuestionAnsweringNetwork', help='which model to import')
    args.dimension = 200 # output dimensions for all layers')
    args.rnn_layers = 1 # number of layers for RNN modules')
    args.transformer_layers = 2 # number of layers for transformer modules')
    args.transformer_hidden = 150 # hidden size of the transformer modules')
    args.transformer_heads = 3 # number of heads for transformer modules')
    args.dropout_ratio = 0.2, type=float, help='dropout for the model')
    args.cove', action='store_true', help='whether to use contextualized word vectors (McCann et al. 2017)')
    args.intermediate_cove', action='store_true', help='whether to use the intermediate layers of contextualized word vectors (McCann et al. 2017)')
    args.elmo = [-1], nargs='+', type=int,  help='which layer(s) (0, 1, or 2) of ELMo (Peters et al. 2018) to use; -1 for none ')
    args.no_glove_and_char', action='store_false', dest='glove_and_char', help='turn off GloVe and CharNGram embeddings')

    args.warmup = 800 # warmup for learning rate')
    args.grad_clip = 1.0, type=float, help='gradient clipping')
    args.beta0 = 0.9, type=float, help='alternative momentum for Adam (only when not using transformer_lr)')
    args.optimizer = 'adam' # Adam or SGD')
    args.no_transformer_lr', action='store_false', dest='transformer_lr', help='turns off the transformer learning rate strategy') 
    args.sgd_lr = 1.0, type=float, help='learning rate for SGD (if not using Adam)')

    args.load = None # path to checkpoint to load model from inside args.save')
    args.resume = True # whether to resume training with past optimizers')

    args.seed = 123 # Random seed.
    args.devices = [0] # a list of devices that can be used for training (multi-gpu currently WIP)')
    args.backend = 'gloo' # backend for distributed training

    args.exist_ok = True #ok if the save directory already exists, i.e. overwrite is ok
    args.token_testing = True #if true, sorts all iterators
    args.reverse = True #if token_testing and true, sorts all iterators in reverse

    args.world_size = len(args.devices) if args.devices[0] > -1 else -1