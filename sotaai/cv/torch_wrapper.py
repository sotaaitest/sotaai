from torchvision import models
from torchvision import datasets as dset
import sys
import os
from contextlib import contextmanager
import torch
DATASETS = {'classification': [
    'CIFAR10',  # train:bool
    'CIFAR100',  # train: bool
    'Caltech101',  # target_type="category","annotation"
    'Caltech256',  # target_type="category","annotation"
    'CelebA',  # split: train, valid, test, all.    ,
    # split:byclass,bymerge,balanced,letters, digits #mnist. train:bool.
    'EMNIST',
    'FashionMNIST',  # train:bool
    'KMNIST',  # train:bool
    'LSUN',  # classes: train, val, test,                                             #no download
    'MNIST',  # train:bool
    'Omniglot',
    'QMNIST',  # train:bool,
    'SEMEION',
    'STL10',  # split=train, test, unlabeled, train+unlabeled
    # split=train, test,extra                      (easy to include butnot included in pytorch)
    'SVHN',
    'USPS'  # train:bool
],

    'object detection': [
    'Caltech101',  # target_type="category","annotation"
    'Caltech256',  # target_type="category","annotation"
    'CelebA',  # split: train, valid, test, all.
    'CocoDetection',  # no download
    'Flickr30k',  # no download
    'SVHN',  # split=train, test,extra
    'VOCDetection'  # year:2007-2012, image_set=train, trainval, val.
],
    'segmentation': [
    'Cityscapes',  # split: train, val, test.                      #no download
    'VOCSegmentation',  # year:2007-2012, image_set=train, trainval, val.
    # image_set:train,val,train_noval. mode:segmentation, boundary.
    'SBD/segmentation',
    'SBD/boundary'
],
    'captioning': [
    'CocoCaptions',  # no download
    'Flickr8k',  # no download
    'Flickr30k',  # no download
    'SBU'
],
    'human activity recognition': [
    'HMDB51',  # no download
    'Kinetics400',  # no download
    'UCF101'  # no download
],
    'other': [
    'Omniglot',  # one-shot learning
    'PhotoTour',  # name of dataset to download, train:bool   #local image descriptors
    'STL10',  # split=train, test, unlabeled, train+unlabeled # unsupervised learning
    'ImageNet'  # split: train, val                          #check which task pytorch has           #no download
]}

MODELS_dic = {'classification': ['alexnet',
                                 'densenet121',
                                 'densenet161',
                                 'densenet169',
                                 'densenet201',
                                 'googlenet',
                                 'inception_v3',
                                 'mnasnet0_5',
                                 'mnasnet0_75',
                                 'mnasnet1_0',
                                 'mnasnet1_3',
                                 'mobilenet_v2',
                                 'resnet101',
                                 'resnet152',
                                 'resnet18',
                                 'resnet34',
                                 'resnet50',
                                 'resnext101_32x8d',
                                 'resnext50_32x4d',
                                 'shufflenet_v2_x0_5',
                                 'shufflenet_v2_x1_0',
                                 'shufflenet_v2_x1_5',
                                 'shufflenet_v2_x2_0',
                                 'squeezenet1_0',
                                 'squeezenet1_1',
                                 'vgg11',
                                 'vgg11_bn',
                                 'vgg13',
                                 'vgg13_bn',
                                 'vgg16',
                                 'vgg16_bn',
                                 'vgg19',
                                 'vgg19_bn',
                                 'wide_resnet101_2',
                                 'wide_resnet50_2'],
              'segmentation': [
    'deeplabv3_resnet101',
    'deeplabv3_resnet50',
    'fcn_resnet101',
    'fcn_resnet50', ],
    'object detection': ['fasterrcnn_resnet50_fpn',
                         'keypointrcnn_resnet50_fpn',
                         'maskrcnn_resnet50_fpn'],
    'video': ['mc3_18',
              'r2plus1d_18',
              'r3d_18']
}


def flatten(l): return [item for sublist in l for item in sublist]


MODELS = flatten(list(MODELS_dic.values()))


def tasks():
    return list(datasets.keys())


def available_datasets(task: str = 'all'):
    if task == "all":
        def flatten(l): return [item for sublist in l for item in sublist]
        flat_ds = flatten(datasets.values())
        return flat_ds

    return datasets[task]


def available_models(task: str = 'all'):
    if task == "all":
        def flatten(l): return [item for sublist in l for item in sublist]
        flat_models = flatten(models.values())
        return flat_models

    return models[task]


# Suppress the message given by torchvision
@contextmanager
def suppress_stdout(dataset_name):
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        if old_stdout == "Files already downloaded and verified":
            sys.stdout = devnull
        else:
            print("Downloading " + dataset_name+"...")
            print(str(old_stdout))
            sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def load_model(model_name, pretrained=False):

    if model_name in MODELS_dic['segmentation']:
        trainer = getattr(models.segmentation, model_name)
    elif model_name in MODELS_dic['object detection']:
        trainer = getattr(models.detection, model_name)
    elif model_name in MODELS_dic['video']:
        trainer = getattr(models.video, model_name)
    else:
        trainer = getattr(models, model_name)
    return trainer(pretrained=pretrained)


def load_dataset(dataset_name):
    if 'SBD' in dataset_name:
        mode = dataset_name.split('/')[1]
        dataset_name = 'SBDataset'

    elif 'PhotoTour' in dataset_name:
        name = dataset_name.split('/')[1]
        dataset_name = 'PhotoTour'
    elif 'VOC' in dataset_name:
        year = dataset_name.split('/')[1]
        dataset_name = dataset_name.split('/')[0]
    ds = getattr(dset, dataset_name)
    # save datasets into dic which will be filled with the available splits of dataset
    ds_dic = {}

    datasets_w_train = ['CIFAR10', 'CIFAR100',
                        'FashionMNIST', 'KMNIST', 'MNIST', 'QMNIST', 'USPS']
    datasets_w_split = ['SVHN', 'STL10', 'Cityscapes', 'CelebA']

    if dataset_name in datasets_w_train:
        from torchvision import transforms #temp
        transform = transforms.Compose(
            [transforms.Resize(256),
            transforms.RandomCrop(227),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        ds_dic['train'] = torch.utils.data.DataLoader(ds("./torch", train=True, download=True, transform=transform), batch_size=100,shuffle=False, num_workers= 0)
        ds_dic['test'] = torch.utils.data.DataLoader(ds("./torch", train=False, download=True,transform = transform), batch_size=100,shuffle=False, num_workers= 0)

    elif dataset_name in datasets_w_split:
        ds_dic['train'] = ds("./torch", split="train", download=True)
        ds_dic['test'] = ds("./torch", split="test", download=True)

    elif dataset_name == "EMNIST":  # split= balanced,byclass,bymerge,letters,digits,mnist
        ds_dic['train'] = ds("./torch", split='balanced',
                             train=True, download=True)
        ds_dic['test'] = ds("./torch", split='balanced',
                            train=False, download=True)

    elif "PhotoTour" in dataset_name:
        ds_dic['train'] = ds("./torch", name=name, download=True, train=True)
        ds_dic['test'] = ds("./torch", name=name, download=True, train=False)

    elif dataset_name in ["SBU", 'Omniglot', 'SEMEION']:
        ds_dic = ds("./torch", download=True)

    elif dataset_name in ["VOCSegmentation", "VOCDetection"]:
        ds_dic['train'] = ds("./torch", year=year,
                             image_set="train", download=True)
        ds_dic['val'] = ds("./torch", year=year,
                           image_set="val", download=True)

    elif 'SBD' in dataset_name:
        ds_dic['train'] = ds("./torch", image_set="train",
                             mode=mode, download=True)
        ds_dic['val'] = ds("./torch", image_set="val",
                           mode=mode, download=True)

    return ds_dic


def adapt_last_layer(model, classes):
    ''' Change last layer of network given the number of classes in the dataset. This function is for classification models only'''
    layers = list(mod.children())  # get children and their names
    submodule = False

    # if last layer is encapsulated in a block, get the layers of that block
    if isinstance(layers[-1], torch.nn.modules.container.Sequential):
        submodule = True
        layers = list(layers.children())
        body = nn.Sequential(*list(model.children())[:-1])

    # Find last convolutional or linear layer, as they are the ones that determine the output size of the neural network
    for i in range(1, len(layers)):
        if isinstance(layers[-i], nn.Linear):
            cut = -i
            if layers[-i].bias != None:
                bias = True
            else:
                bias = False

            new_layer = nn.Linear(out_features=classes,
                                  in_features=layers[-i].in_features, bias=bias)

            break

        elif isinstance(layers[-i], nn.Conv1d) or isinstance(layers[-i], nn.Conv2d) or isinstance(layers[-i], nn.Conv3d):
            l_type = str(type(ll[-i][1])).split('.')[-1][:6]
            conv = getattr(nn, l_type)
            if layers[-i].bias != None:
                bias = True
            else:
                bias = False
            new_layer = new_layer = nn.Linear(in_channels=layers[-i].in_channels, out_channels=classes,
                                              kernel_size=layers[-i].kernel_size, stride=layers[-i].stride, padding=layers[-i].padding,
                                              dilation=layers[-i].dilation, groups=layers[-i].groups, bias=bias,
                                              padding_mode=layers[-i].padding_mode)
            break
    try:
        if not submodule:
            if cut != -1:
                list_modules_head = [new_layer] + layers[cut+1:]
            else:
                list_modules_head = [new_layer]
            body = nn.Sequential(*layers[:cut])
            head = nn.Sequential(*list_modules_head)
            new_model = nn.Sequential(body, head)
        else:
            list_modules_head = layers[:cut]+[new_layer]
            if cut != -1:
                list_modules_head += layers[cut+1:]
            head = nn.Sequential(*list_modules_head)
            new_model = nn.Sequential(body, head)
    except:
        print('The provided model is not a classification model, or it does not have a linear nor a convolutional layer')

    return new_model
