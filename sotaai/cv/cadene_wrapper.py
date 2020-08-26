
from torchvision.transforms import functional
import torch
import copy
from importlib import import_module
import pretrainedmodels
from torch import nn
from torchvision.transforms import Normalize, Resize, Compose

'''
input_sizes = {}
means = {}
stds = {}

for model_name in __all__:
    input_sizes[model_name] = [3, 224, 224]
    means[model_name] = [0.485, 0.456, 0.406]
    stds[model_name] = [0.229, 0.224, 0.225]

for model_name in ['inceptionv3']:
    input_sizes[model_name] = [3, 299, 299]
    means[model_name] = [0.5, 0.5, 0.5]
    stds[model_name] = [0.5, 0.5, 0.5]


pretrained_settings = {
    'dpn92': {
        # 'imagenet': {
        #     'url': 'http://data.lip6.fr/cadene/pretrainedmodels/dpn68-66bebafa7.pth',
        #     'input_space': 'RGB',
        #     'input_size': [3, 224, 224],
        #     'input_range': [0, 1],
        #     'mean': [124 / 255, 117 / 255, 104 / 255],
        #     'std': [1 / (.0167 * 255)] * 3,
        #     'num_classes': 1000
        # },
        'imagenet+5k': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/dpn92_extra-fda993c95.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [124 / 255, 117 / 255, 104 / 255],
            'std': [1 / (.0167 * 255)] * 3,
            'num_classes': 1000
        }
    },
     'nasnetamobile': {
        'imagenet': {
            #'url': 'https://github.com/veronikayurchuk/pretrained-models.pytorch/releases/download/v1.0/nasnetmobile-7e03cead.pth.tar',
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/nasnetamobile-7e03cead.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224], # resize 256
            'input_range': [0, 1],
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5],
            'num_classes': 1000
        },
        # 'imagenet+background': {
        #     # 'url': 'http://data.lip6.fr/cadene/pretrainedmodels/nasnetalarge-a1897284.pth',
        #     'input_space': 'RGB',
        #     'input_size': [3, 224, 224], # resize 256
        #     'input_range': [0, 1],
        #     'mean': [0.5, 0.5, 0.5],
        #     'std': [0.5, 0.5, 0.5],
        #     'num_classes': 1001
        # }
    },
}'''

# all are classification models
MODELS = ['fbresnet152',
          'bninception',
          'resnext101_32x4d',
          'resnext101_64x4d',
          'inceptionv4',
          'inceptionresnetv2',
          'nasnetamobile',
          'nasnetalarge',
          'dpn68',
          'dpn68b',
          'dpn92',
          'dpn98',
          'dpn131',
          'dpn107',
          'xception',
          'senet154',
          'se_resnet50',
          'se_resnet101',
          'se_resnet152',
          'se_resnext50_32x4d',
          'se_resnext101_32x4d',
          'cafferesnet101',
          'pnasnet5large',
          'polynet']


def load_model(model_name, classes=1000, pretrained=False):
    if pretrained:
        weights = 'imagenet'
    else:
        weights = None
    trainer = getattr(pretrainedmodels, model_name)
    return trainer(num_classes=classes, pretrained=weights)


def get_setting(model, setting='input_size'):
    mod_class = 'pretrainedmodels.models.' + str(type(model)).split('.')[-2]
    pretrained_settings = import_module(mod_class).pretrained_settings
    settin = list(list(pretrained_settings.values())[0].values())[0][setting]
    return settin


# Falta hacer esta funcion para que en vez de imagen reciba un dataset, i.e. tensor de tamanio (N,C,H,W)
def adapt_im2model(model, img):
    '''
    Preprocess image so that it fulfills the characteristics that the model was trained on.
    Input: model from Cadene, image Tensor with shape (C,H,W)
    Output: resized and normalized image tensor
    '''

    # Getting parameters of imagegs trained on model
    input_size = get_setting(model, 'input_size')
    #input_range = get_setting(model, 'input_range')
    mean = get_setting(model, 'mean')
    std = get_setting(model, 'std')

    im = functional.to_pil_image(img)
    res = Compose([
        Resize(input_size[1]),
        CenterCrop(input_size[1])])

    im = res(im)

    if input_size[0] == 1 and img.shape[0] == 3:
        im = functional.to_grayscale(im)  # get a tensor of shape (1,H,W)

    im = ToTensor()(im)

    if input_size[0] == 3 and img.shape[0] == 1:
        im = torch.stack([im]*3, axis=1)[0]  # Get a tensor of shape(3,H,W)

    transformed = Normalize(mean, std)(im)

    return transformed


def adapt_last_layer(model, classes):
    net = copy.deepcopy(model)
    ll = net.last_linear
    bias = True if len(ll.bias) else False
    new_layer = nn.Linear(in_features=ll.in_features,
                          out_features=classes, bias=bias)
    net.last_linear = new_layer
    return net


def model_to_dataset(model, dataset):
    classes = dataset.labels
    mod_adapted = adapt_last_layer(model, classes)
    dataset = adapt_im2model(model, dataset)
    return mod_adapted, dataset
