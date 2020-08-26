
# https://github.com/idealo/image-super-resolution#additional-information for more information. Also, check it for training and finetuning, and config files.
# https://github.com/ChaofWang/Awesome-Super-Resolution Also check miniAAs from this repo.
# https://github.com/krasserm/super-resolution automaticaly downloads DIV2k dataset and has three ISR models

import numpy as np
from PIL import Image
from ISR import models
from sotaai.cv.utils import *

# RDN pretrained on  psnr-large, psnr-small, noise-cancel. RRDN pretrained on gans
MODELS = ['RDN/psnr-large', 'RDN/psnr-small', 'RDN/noise-cancel', 'RRDN/gans']


def load_model(model_name='RDN', pretrained=False, channels=3, upscaling=2):
    if len(model_name.split('/'))>1:
        pretrained = model_name.split('/')[1]
        model_name = model_name.split('/')[0]
    module = getattr(models, model_name)
    arch_params = {}
    if not pretrained:
        if model_name == 'RDN':
            arch_params = {'C': 6, 'D': 20, 'G': 64, 'G0': 64,
                           'x': upscaling}  # parameters of the network
        elif model_name == 'RRDN':
            arch_params = {'C': 4, 'D': 3, 'G': 32,
                           'G0': 32, 'x': upscaling, 'T': 10}
        return module(arch_params=arch_params, c_dim=channels)

    elif pretrained == True:
        if model_name == 'RDN':
            weights = 'psnr-large'
        else:
            weights = 'gans'
    else:
        weights = pretrained

    # return model o model.model, which is a directly tf Model?
    model = module(weights=weights, arch_params=arch_params, c_dim=channels)
    return model

# models receive numpy arrays of shape (N,H,W,C)


def run(model, lr):
    '''
    Input: model and numpy array of shape (N,H,W,C)
    Output: numpy array (N,upscaling*H,upscaling*W,C)
    '''
    if model.c_dim == lr.shape[-1]:
        img = lr
    else:
        img = change_channels(lr)

    sr = model.model.predict(img, by_patch_of_size=50)
    return sr


def run_from_filename(model, filename):
    img = Image.open(filename)
    lr = np.array(img)
    sr = model.predict(lr, by_patch_of_size=50)
    sr_img = Image.fromarray(sr)
    return sr_img
