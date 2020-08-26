"""
Inpainting using Generative Adversarial Networks.
The dataset can be downloaded from: https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=0
(if not available there see if options are listed at http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
Instrustion on running the script:
1. Download the dataset from the provided link
2. Save the folder 'img_align_celeba' to '../../data/'
4. Run the sript using command 'python3 context_encoder.py'
"""

import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from cv.GANs.PyTorch_GANs.context_encoder.datasets import *
from cv.GANs.PyTorch_GANs.context_encoder.models import *

import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--dataset_name", type=str, default="img_align_celeba", help="name of the dataset")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
parser.add_argument("--mask_size", type=int, default=64, help="size of random mask")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=500, help="interval between image sampling")
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False

# Calculate output of image discriminator (PatchGAN)
patch_h, patch_w = int(opt.mask_size / 2 ** 3), int(opt.mask_size / 2 ** 3)
patch = (1, patch_h, patch_w)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class ContextEncoder():
    def __init__(self, lr = 0.0002,b1 = 0.5, b2 = 0.999, latent_dim = 100,img_size = 32,channels = 1):
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.channels = channels

        # Loss function
        self.adversarial_loss = torch.nn.MSELoss()
        self.pixelwise_loss = torch.nn.L1Loss()

        # Initialize generator and discriminator
        self.generator = Generator(channels=self.channels)
        self.discriminator = Discriminator(channels=self.channels)

        cuda = False #True if torch.cuda.is_available() else False
        if cuda:
            self.generator.cuda()
            self.discriminator.cuda()
            self.adversarial_loss.cuda()
            self.pixelwise_loss.cuda()
        
        self.Tensor = torch.FloatTensor #torch.cuda.FloatTensor if cuda else torch.FloatTensor

        # Initialize weights
        self.generator.apply(weights_init_normal)
        self.discriminator.apply(weights_init_normal)


        # Optimizers
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.b1, self.b2))



    def save_sample(self,batches_done, mask_size):
        samples, masked_samples, i = next(iter(test_dataloader))
        samples = Variable(samples.type(self.Tensor))
        masked_samples = Variable(masked_samples.type(self.Tensor))
        i = i[0].item()  # Upper-left coordinate of mask
        # Generate inpainted image
        gen_mask = self.generator(masked_samples)
        filled_samples = masked_samples.clone()
        filled_samples[:, :, i : i + mask_size, i : i + mask_size] = gen_mask
        # Save sample
        sample = torch.cat((masked_samples.data, filled_samples.data, samples.data), -2)
        save_image(sample, "images/%d.png" % batches_done, nrow=6, normalize=True)


    # ----------
    #  Training
    # ----------
    def train(self,n_epochs = 200, sample_interval = 500,mask_size = 64):
        for epoch in range(n_epochs):
            for i, (imgs, masked_imgs, masked_parts) in enumerate(dataloader):

                # Adversarial ground truths
                valid = Variable(self.Tensor(imgs.shape[0], *patch).fill_(1.0), requires_grad=False)
                fake = Variable(self.Tensor(imgs.shape[0], *patch).fill_(0.0), requires_grad=False)

                # Configure input
                imgs = Variable(imgs.type(self.Tensor))
                masked_imgs = Variable(masked_imgs.type(self.Tensor))
                masked_parts = Variable(masked_parts.type(self.Tensor))

                # -----------------
                #  Train Generator
                # -----------------

                self.optimizer_G.zero_grad()

                # Generate a batch of images
                gen_parts = self.generator(masked_imgs)

                # Adversarial and pixelwise loss
                g_adv = self.adversarial_loss(self.discriminator(gen_parts), valid)
                g_pixel = self.pixelwise_loss(gen_parts, masked_parts)
                # Total loss
                g_loss = 0.001 * g_adv + 0.999 * g_pixel

                g_loss.backward()
                self.optimizer_G.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                self.optimizer_D.zero_grad()

                # Measure discriminator's ability to classify real from generated samples
                real_loss = self.adversarial_loss(self.discriminator(masked_parts), valid)
                fake_loss = self.adversarial_loss(self.discriminator(gen_parts.detach()), fake)
                d_loss = 0.5 * (real_loss + fake_loss)

                d_loss.backward()
                self.optimizer_D.step()

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G adv: %f, pixel: %f]"
                    % (epoch, n_epochs, i, len(dataloader), d_loss.item(), g_adv.item(), g_pixel.item())
                )

                # Generate sample at sample interval
                batches_done = epoch * len(dataloader) + i
                if batches_done % sample_interval == 0:
                    self.save_sample(batches_done,mask_size)


'''

# Dataset loader
transforms_ = [
    transforms.Resize((opt.img_size, opt.img_size), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]
dataloader = DataLoader(
    ImageDataset("../../data/%s" % opt.dataset_name, transforms_=transforms_),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)
test_dataloader = DataLoader(
    ImageDataset("../../data/%s" % opt.dataset_name, transforms_=transforms_, mode="val"),
    batch_size=12,
    shuffle=True,
    num_workers=1,
)'''
