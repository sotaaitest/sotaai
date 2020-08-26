'''All the code to load GANs in Keras was modified from the amazing repo Keras-GANs https://github.com/eriklindernoren/Keras-GAN
Many thanks to  Erik Linder-Norén for his great work!'''
from importlib import import_module

MODELS = ['AAE',
          'ACGAN',
          'BGAN',
          'BIGAN',
          'CCGAN',
          'CGAN',
          'COGAN',
          'CONTEXT_ENCODER',
          'CYCLEGAN',
          'DCGAN',
          'DISCOGAN',
          'DUALGAN',
          'GAN',
          'INFOGAN',
          'LSGAN',
          'PIX2PIX',
          'PIXELDA',
          'SGAN',
          'SRGAN',
          'WGAN'
          ]


def load_model(model_name, pretrained = False):
    if pretrained:
        print('All GANs available are not pretrained')
    module_name = model_name.lower() + '.' + model_name.lower()
    if model_name == 'AAE' or model_name == 'aae':
        model_name = 'AdversarialAutoencoder'
    elif model_name == 'CONTEXT_ENCODER':
        model_name = 'ContextEncoder'
    elif model_name == 'CYCLEGAN':
        model_name = 'CycleGAN'
    elif model_name == 'PIX2PIX':
        model_name = 'Pix2Pix'
    else:
        model_name = model_name.upper()
    module = import_module('cv.GANs.Keras_GANs.' + module_name)
    model = getattr(module, model_name)
    return model()
