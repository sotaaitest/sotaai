import mxnet as mx
from mxnet.gluon import nn
import numpy as np
import copy
from sotaai.cv.mxnet_utils import find_im_size
from sotaai.cv.utils import *

DATASETS = {'classification': ["MNIST", "FashionMNIST", "CIFAR10", "CIFAR100"]}

# All models here are for classification.
MODELS = ["alexnet",
          "densenet121",
          "densenet161",
          "densenet169",
          "densenet201",
          "inceptionv3",
          "mobilenet0.25",  # width multiplier 0.25.
          "mobilenet0.5",
          "mobilenet0.75",
          "mobilenet1.0",
          "mobilenetv2_0.25",
          "mobilenetv2_0.5",
          "mobilenetv2_0.75",
          "mobilenetv2_1.0",
          "resnet101_v1",
          "resnet101_v2",
          "resnet152_v1",
          "resnet152_v2",
          "resnet18_v1",
          "resnet18_v2",
          "resnet34_v1",
          "resnet34_v2",
          "resnet50_v1",
          "resnet50_v2",
          "squeezenet1.0",
          "squeezenet1.1",
          "vgg11",
          "vgg11_bn",
          "vgg13",
          "vgg13_bn",
          "vgg16",
          "vgg16_bn",
          "vgg19",
          "vgg19_bn"]

'''
Example:


mod = load_model('alexnet')
ds = load_dataset('MNIST')
img = ds[0][0][0]
img = transform(image)
mod(img)

'''


def load_model(model_name, classes=1000, pretrained=False):
    mod = mx.gluon.model_zoo.vision.get_model(
        model_name, classes=classes, pretrained=pretrained)
    if not pretrained:
        mod.initialize()
    return mod


def load_dataset(dataset_name):
    ds = getattr(mx.gluon.data.vision, dataset_name)
    ds_train = ds(".", train=True)
    ds_test = ds(".", train=False)
    return ds_train, ds_test


def adapt_last_layer(model, classes: int):
    mod = copy.deepcopy(model)
    if 'squeezenet' in str(type(mod)):
        hybrid_block = mod.output

        args_conv = hybrid_block[0]._kwargs
        bias = not args_conv['no_bias']

        act = hybrid_block[1]._act_type

        args_pool = hybrid_block[2]._kwargs
        ceil_mode = True if args_pool['pooling_convention'] == 'full' else False

        net = nn.HybridSequential()
        net.add(nn.Conv2D(channels=classes, kernel_size=args_conv['kernel'], strides=args_conv['stride'], padding=args_conv['pad'],
                          groups=args_conv['num_group'], dilation=args_conv['dilate'], layout=args_conv['layout'],
                          use_bias=bias, in_channels=hybrid_block[0]._in_channels))
        net.add(nn.Activation(act))
        net.add(nn.AvgPool2D(pool_size=args_pool['kernel'], strides=args_pool['stride'], padding=args_pool['pad'], ceil_mode=ceil_mode,
                             layout=args_pool['layout'], count_include_pad=args_pool['count_include_pad']))
        net.add(nn.Flatten())
        net.initialize()

        mod.output = net
    else:
        l = mod.output
        bias = True if l.bias else False

        dense = nn.Dense(units=classes, activation=l.act,
                         use_bias=bias, flatten=l._flatten, in_units=l._in_units)
        dense.initialize()
        mod.output = dense

    return mod
