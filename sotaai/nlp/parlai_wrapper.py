from parlai.core.opt import Opt
from parlai.core.agents import create_agent
from parlai.core.worlds import _create_task_agents 


def load_model(model_name,pretrained = False):
    model_dic = Opt({'model' : model_name})
    model = create_agent(model_dic)
    return model


'''datatypes::                'train',
                'train:stream',
                'train:ordered',
                'train:ordered:stream',
                'train:stream:ordered',
                'train:evalmode',
                'train:evalmode:stream',
                'train:evalmode:ordered',
                'train:evalmode:ordered:stream',
                'train:evalmode:stream:ordered',
                'valid',
                'valid:stream',
                'test',
                'test:stream' '''
def load_dataset(ds_name,datatype = 'train',datapath = '.'):
    ds_dic = Opt({'task' : 'babi:Task1k:1','datapath':datapath,'datatype':datatype})
    ds = _create_task_agents(ds_dic)
    return ds

def predict():

def train(epochs = 5):
    opt = Opt({'num_epochs' : epochs,'datapath':datapath,'datatype':datatype})
    # set up timers
    train_time = Timer()
    validate_time = Timer()
    log_time = Timer()
    save_time = Timer()
    parleys = 0

def eval():

        

        self.parleys = 0
        self.max_num_epochs = (
            opt['num_epochs'] if opt['num_epochs'] > 0 else float('inf')
        )
        self.max_train_time = (
            opt['max_train_time'] if opt['max_train_time'] > 0 else float('inf')
        )
        self.log_every_n_secs = (
            opt['log_every_n_secs'] if opt['log_every_n_secs'] > 0 else float('inf')
        )
        self.val_every_n_secs = (
            opt['validation_every_n_secs']
            if opt['validation_every_n_secs'] > 0
            else float('inf')
        )
        self.save_every_n_secs = (
            opt['save_every_n_secs'] if opt['save_every_n_secs'] > 0 else float('inf')
        )
        self.val_every_n_epochs = (
            opt['validation_every_n_epochs']
            if opt['validation_every_n_epochs'] > 0
            else float('inf')
        )

        # smart defaults for --validation-metric-mode
        if opt['validation_metric'] in {'loss', 'ppl', 'mean_rank'}:
            opt['validation_metric_mode'] = 'min'
        elif opt['validation_metric'] in {'accuracy', 'hits@1', 'hits@5', 'f1', 'bleu'}:
            opt['validation_metric_mode'] = 'max'
        if opt.get('validation_metric_mode') is None:
            opt['validation_metric_mode'] = 'max'

        self.last_valid_epoch = 0
        self.valid_optim = 1 if opt['validation_metric_mode'] == 'max' else -1
        self.train_reports = []
        self.valid_reports = []
        self.best_valid = None

        self.impatience = 0
        self.saved = False
        self.valid_worlds = None
        self.opt = opt

        # we may have been preempted, make sure we note that amount
        self._preempted_epochs = 0.0
        if opt.get('model_file') and os.path.isfile(
            opt['model_file'] + trainstats_suffix
        ):
            # looks like we were preempted. make sure we load up our total
            # training stats, etc
            with open(opt['model_file'] + trainstats_suffix) as ts:
                obj = json.load(ts)
                self.parleys = obj.get('parleys', 0)
                self._preempted_epochs = obj.get('total_epochs', 0)
                self.train_time.total = obj.get('train_time', 0)
                self.impatience = obj.get('impatience', 0)
                self.valid_reports = obj.get('valid_reports', [])
                self.train_reports = obj.get('train_reports', [])
                if 'best_valid' in obj:
                    self.best_valid = obj['best_valid']
                else:
                    # old method
                    if opt.get('model_file') and os.path.isfile(
                        opt['model_file'] + '.best_valid'
                    ):
                        with open(opt['model_file'] + ".best_valid", 'r') as f:
                            x = f.readline()
                            self.best_valid = float(x)
                            f.close()
