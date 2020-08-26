"""
Super-resolution of CelebA using Generative Adversarial Networks.
The dataset can be downloaded from: https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=0
(if not available there see if options are listed at http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
Instrustion on running the script:
1. Download the dataset from the provided link
2. Save the folder 'img_align_celeba' to '../../data/'
4. Run the sript using command 'python3 srgan.py'
"""

import argparse
import os
import numpy as np
import math
import itertools
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torch.autograd import Variable

from cv.GANs.PyTorch_GANs.srgan.models import *
from cv.GANs.PyTorch_GANs.srgan.datasets import *

import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs("images", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="img_align_celeba", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--hr_height", type=int, default=256, help="high res. image height")
parser.add_argument("--hr_width", type=int, default=256, help="high res. image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving image samples")
parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between model checkpoints")
opt = parser.parse_args()
print(opt)




class SRGAN():
    def __init__(self, lr = 0.0002,b1 = 0.5, b2 = 0.999, latent_dim = 100,hr_height = 256, hr_width = 256, channels = 1):
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.latent_dim = latent_dim
        self.hr_shape = (hr_height, hr_width)

        # Initialize generator and discriminator
        self.generator = GeneratorResNet()
        self.discriminator = Discriminator(input_shape=(channels, *self.hr_shape))
        self.feature_extractor = FeatureExtractor()

        # Set feature extractor to inference mode
        self.feature_extractor.eval()

        # Losses
        self.criterion_GAN = torch.nn.MSELoss()
        self.criterion_content = torch.nn.L1Loss()

        cuda = False #torch.cuda.is_available()

        if cuda:
            self.generator = self.generator.cuda()
            self.discriminator = self.discriminator.cuda()
            self.feature_extractor = self.feature_extractor.cuda()
            self.criterion_GAN = self.criterion_GAN.cuda()
            self.criterion_content = self.criterion_content.cuda()

        self.Tensor = torch.Tensor  # torch.cuda.FloatTensor if cuda else torch.Tensor

        # Optimizers
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.b1, self.b2))



    # ----------
    #  Training
    # ----------
    def train(self, epoch = 0, n_epochs = 200, sample_interval = 100,checkpoint_interval = -1):

        if epoch != 0:
            # Load pretrained models
            self.generator.load_state_dict(torch.load("saved_models/generator_%d.pth"))
            self.discriminator.load_state_dict(torch.load("saved_models/discriminator_%d.pth"))

        for epoch in range(epoch, n_epochs):
            for i, imgs in enumerate(dataloader):

                # Configure model input
                imgs_lr = Variable(imgs["lr"].type(self.Tensor))
                imgs_hr = Variable(imgs["hr"].type(self.Tensor))

                # Adversarial ground truths
                valid = Variable(self.Tensor(np.ones((imgs_lr.size(0), *self.discriminator.output_shape))), requires_grad=False)
                fake = Variable(self.Tensor(np.zeros((imgs_lr.size(0), *self.discriminator.output_shape))), requires_grad=False)

                # ------------------
                #  Train Generators
                # ------------------

                self.optimizer_G.zero_grad()

                # Generate a high resolution image from low resolution input
                self.gen_hr = self.generator(imgs_lr)

                # Adversarial loss
                loss_GAN = self.criterion_GAN(self.discriminator(gen_hr), valid)

                # Content loss
                gen_features = self.feature_extractor(gen_hr)
                real_features = self.feature_extractor(imgs_hr)
                loss_content = self.criterion_content(gen_features, real_features.detach())

                # Total loss
                loss_G = loss_content + 1e-3 * loss_GAN

                loss_G.backward()
                self.optimizer_G.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                self.optimizer_D.zero_grad()

                # Loss of real and fake images
                loss_real = self.criterion_GAN(self.discriminator(imgs_hr), valid)
                loss_fake = self.criterion_GAN(self.discriminator(gen_hr.detach()), fake)

                # Total loss
                loss_D = (loss_real + loss_fake) / 2

                loss_D.backward()
                self.optimizer_D.step()

                # --------------
                #  Log Progress
                # --------------

                sys.stdout.write(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, n_epochs, i, len(dataloader), loss_D.item(), loss_G.item())
                )

                batches_done = epoch * len(dataloader) + i
                if batches_done % sample_interval == 0:
                    # Save image grid with upsampled inputs and SRGAN outputs
                    imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)
                    gen_hr = make_grid(gen_hr, nrow=1, normalize=True)
                    imgs_lr = make_grid(imgs_lr, nrow=1, normalize=True)
                    img_grid = torch.cat((imgs_lr, gen_hr), -1)
                    save_image(img_grid, "images/%d.png" % batches_done, normalize=False)

            if checkpoint_interval != -1 and epoch % checkpoint_interval == 0:
                # Save model checkpoints
                torch.save(self.generator.state_dict(), "saved_models/generator_%d.pth" % epoch)
                torch.save(self.discriminator.state_dict(), "saved_models/discriminator_%d.pth" % epoch)



'''


dataloader = DataLoader(
    ImageDataset("../../data/%s" % opt.dataset_name, hr_shape=hr_shape),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)'''
