'''the code to load GANs in PyTorch was built and modified using the amazing repo PyTorch-GANs https://github.com/eriklindernoren/PyTorch-GAN
Many thanks to  Erik Linder-Norén for his great work!'''
from importlib import import_module

MODELS = ['AAE',
          'ACGAN',
          'BEGAN',
          'BGAN',
          'BICYCLEGAN',
          'CCGAN',
          'CGAN',
          'CLUSTER_GAN',
          'COGAN',
          'CONTEXT_ENCODER',
          'CYCLEGAN',
          'DCGAN',
          'DISCOGAN',
          'DRAGAN',
          'DUALGAN',
          'EBGAN',
          'ESRGAN',
          'GAN',
          'INFOGAN',
          'LSGAN',
          'MUNIT',
          'PIX2PIX',
          'PIXELDA',
          'RELATIVISTIC_GAN',
          'SGAN',
          'SRGAN',
          'SOFTMAX_GAN',
          'STARGAN',
          'WGAN',
          'WGAN_DIV',
          'WGAN_GP'
          ]


def load_model(model_name, pretrained = False):
    if pretrained:
        print('All GANs available are not pretrained')
    module_name = model_name.lower() + '.' + model_name.lower()
    if model_name == 'AAE' or model_name == 'aae':
        model_name = 'AdversarialAutoencoder'
    elif model_name == 'CLUSTER_GAN':
        model_name = 'ClusterGAN'
    elif model_name == 'CONTEXT_ENCODER':
        model_name = 'ContextEncoder'
    elif model_name == 'CYCLEGAN':
        model_name = 'CycleGAN'
    elif model_name == 'MUNIT':
        model_name = 'MUnit'
    elif model_name == 'PIX2PIX':
        model_name = 'Pix2Pix'
    elif model_name == 'UNIT':
        model_name = 'Unit'
    elif model_name == 'WGAN_DIV':
        model_name = 'WGAN_div'
    else:
        model_name = model_name.upper()
    module = import_module('cv.GANs.PyTorch_GANs.' + module_name)
    model = getattr(module, model_name)
    return model()


