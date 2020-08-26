import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from cv.GANs.PyTorch_GANs.pix2pix.models import *
from cv.GANs.PyTorch_GANs.pix2pix.datasets import *

import torch.nn as nn
import torch.nn.functional as F
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="facades", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument(
    "--sample_interval", type=int, default=500, help="interval between sampling of images from generators"
)
parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between model checkpoints")
opt = parser.parse_args()
print(opt)

os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)



class Pix2Pix():
    def __init__(self, lr = 0.0002,b1 = 0.5, b2 = 0.999, latent_dim = 100,img_height = 256, img_width = 256, channels = 1):
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.latent_dim = latent_dim
        self.img_height = img_height
        self.img_width = img_width

        # Loss functions
        self.criterion_GAN = torch.nn.MSELoss()
        self.criterion_pixelwise = torch.nn.L1Loss()

        
        # Initialize generator and discriminator
        self.generator = GeneratorUNet()
        self.discriminator = Discriminator()


        cuda = False # True if torch.cuda.is_available() else False

        if cuda:
            self.generator = self.generator.cuda()
            self.discriminator = self.discriminator.cuda()
            self.criterion_GAN.cuda()
            self.criterion_pixelwise.cuda()

        # Tensor type
        self.Tensor = torch.FloatTensor  # torch.cuda.FloatTensor if cuda else torch.FloatTensor


        # Initialize weights
        self.generator.apply(weights_init_normal)
        self.discriminator.apply(weights_init_normal)

        # Optimizers
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.b1, self.b2))




    def sample_images(self, val_dataloader, batches_done, dataset_name):
        """Saves a generated sample from the validation set"""
        imgs = next(iter(val_dataloader))
        real_A = Variable(imgs["B"].type(self.Tensor))
        real_B = Variable(imgs["A"].type(self.Tensor))
        fake_B = self.generator(real_A)
        img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), -2)
        save_image(img_sample, "images/%s/%s.png" % (dataset_name, batches_done), nrow=5, normalize=True)


# ----------
#  Training
# ----------
    def train(self, epoch = 0, n_epochs = 200, sample_interval = 500,checkpoint_interval = -1, dataset_name = 'facades'):
        
        if epoch != 0:
            # Load pretrained models
            self.generator.load_state_dict(torch.load("saved_models/%s/generator_%d.pth" % (dataset_name, epoch)))
            self.discriminator.load_state_dict(torch.load("saved_models/%s/discriminator_%d.pth" % (dataset_name, epoch)))
        
        # Loss weight of L1 pixel-wise loss between translated image and real image
        lambda_pixel = 100

        # Calculate output of image discriminator (PatchGAN)
        patch = (1, self.img_height // 2 ** 4, self.img_width // 2 ** 4)

        prev_time = time.time()

        for epoch in range(epoch, n_epochs):
            for i, batch in enumerate(dataloader):

                # Model inputs
                real_A = Variable(batch["B"].type(self.Tensor))
                real_B = Variable(batch["A"].type(self.Tensor))

                # Adversarial ground truths
                valid = Variable(self.Tensor(np.ones((real_A.size(0), *patch))), requires_grad=False)
                fake = Variable(self.Tensor(np.zeros((real_A.size(0), *patch))), requires_grad=False)

                # ------------------
                #  Train Generators
                # ------------------

                self.optimizer_G.zero_grad()

                # GAN loss
                fake_B = self.generator(real_A)
                pred_fake = self.discriminator(fake_B, real_A)
                loss_GAN = self.criterion_GAN(pred_fake, valid)
                # Pixel-wise loss
                loss_pixel = self.criterion_pixelwise(fake_B, real_B)

                # Total loss
                loss_G = loss_GAN + lambda_pixel * loss_pixel

                loss_G.backward()

                self.optimizer_G.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                self.optimizer_D.zero_grad()

                # Real loss
                pred_real = self.discriminator(real_B, real_A)
                loss_real = self.criterion_GAN(pred_real, valid)

                # Fake loss
                pred_fake = self.discriminator(fake_B.detach(), real_A)
                loss_fake = self.criterion_GAN(pred_fake, fake)

                # Total loss
                loss_D = 0.5 * (loss_real + loss_fake)

                loss_D.backward()
                self.optimizer_D.step()

                # --------------
                #  Log Progress
                # --------------

                # Determine approximate time left
                batches_done = epoch * len(dataloader) + i
                batches_left = n_epochs * len(dataloader) - batches_done
                time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
                prev_time = time.time()

                # Print log
                sys.stdout.write(
                    "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pixel: %f, adv: %f] ETA: %s"
                    % (
                        epoch,
                        n_epochs,
                        i,
                        len(dataloader),
                        loss_D.item(),
                        loss_G.item(),
                        loss_pixel.item(),
                        loss_GAN.item(),
                        time_left,
                    )
                )

                # If at sample interval save image
                if batches_done % sample_interval == 0:
                    self.sample_images(val_dataloader, batches_done,dataset_name)

            if checkpoint_interval != -1 and epoch % checkpoint_interval == 0:
                # Save model checkpoints
                torch.save(self.generator.state_dict(), "saved_models/%s/generator_%d.pth" % (dataset_name, epoch))
                torch.save(self.discriminator.state_dict(), "saved_models/%s/discriminator_%d.pth" % (dataset_name, epoch))



'''
# Configure dataloaders
transforms_ = [
    transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

dataloader = DataLoader(
    ImageDataset("../../data/%s" % opt.dataset_name, transforms_=transforms_),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)

val_dataloader = DataLoader(
    ImageDataset("../../data/%s" % opt.dataset_name, transforms_=transforms_, mode="val"),
    batch_size=10,
    shuffle=True,
    num_workers=1,
)'''
