import argparse
import os
import numpy as np
import math
import itertools
import sys
import datetime
import time

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from cv.GANs.PyTorch_GANs.discogan.models import *
from cv.GANs.PyTorch_GANs.discogan.datasets import *

import torch.nn as nn
import torch.nn.functional as F
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="edges2shoes", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=64, help="size of image height")
parser.add_argument("--img_width", type=int, default=64, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving generator samples")
parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between model checkpoints")
opt = parser.parse_args()
print(opt)

# Create sample and checkpoint directories
os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)


class DISCOGAN():
    def __init__(self, lr = 0.0002,b1 = 0.5, b2 = 0.999, latent_dim = 100,img_height = 64, img_width = 64,channels = 1):
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.latent_dim = latent_dim
        self.input_shape = (channels, img_height, img_width)

        # Losses
        self.adversarial_loss = torch.nn.MSELoss()
        self.cycle_loss = torch.nn.L1Loss()
        self.pixelwise_loss = torch.nn.L1Loss()



        # Initialize generator and discriminator
        self.G_AB = GeneratorUNet(self.input_shape)
        self.G_BA = GeneratorUNet(self.input_shape)
        self.D_A = Discriminator(self.input_shape)
        self.D_B = Discriminator(self.input_shape)

        cuda = False #torch.cuda.is_available()

        if cuda:
            self.G_AB = self.G_AB.cuda()
            self.G_BA = self.G_BA.cuda()
            self.D_A = self.D_A.cuda()
            self.D_B = self.D_B.cuda()
            self.adversarial_loss.cuda()
            self.cycle_loss.cuda()
            self.pixelwise_loss.cuda()

        # Input tensor type
        self.Tensor = torch.Tensor #torch.cuda.FloatTensor if cuda else torch.Tensor

        # Initialize weights
        self.G_AB.apply(weights_init_normal)
        self.G_BA.apply(weights_init_normal)
        self.D_A.apply(weights_init_normal)
        self.D_B.apply(weights_init_normal)

        # Optimizers
        self.optimizer_G = torch.optim.Adam(
            itertools.chain(self.G_AB.parameters(), self.G_BA.parameters()), lr=self.lr, betas=(self.b1, self.b2)
        )
        self.optimizer_D_A = torch.optim.Adam(self.D_A.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        self.optimizer_D_B = torch.optim.Adam(self.D_B.parameters(), lr=self.lr, betas=(self.b1, self.b2))


    def sample_images(self, val_dataloader, batches_done, dataset_name):
        """Saves a generated sample from the validation set"""
        imgs = next(iter(val_dataloader))
        self.G_AB.eval()
        self.G_BA.eval()
        real_A = Variable(imgs["A"].type(self.Tensor))
        fake_B = self.G_AB(real_A)
        real_B = Variable(imgs["B"].type(self.Tensor))
        fake_A = self.G_BA(real_B)
        img_sample = torch.cat((real_A.data, fake_B.data, real_B.data, fake_A.data), 0)
        save_image(img_sample, "images/%s/%s.png" % (dataset_name, batches_done), nrow=8, normalize=True)


# ----------
#  Training
# ----------
    def train(self, start_epoch = 0, n_epochs = 200, sample_interval = 100, checkpoint_interval = -1, dataset_name = 'edges2shoes'):

        if start_epoch != 0:
            # Load pretrained models
            self.G_AB.load_state_dict(torch.load("saved_models/%s/G_AB_%d.pth" % (dataset_name, epoch)))
            self.G_BA.load_state_dict(torch.load("saved_models/%s/G_BA_%d.pth" % (dataset_name, epoch)))
            self.D_A.load_state_dict(torch.load("saved_models/%s/D_A_%d.pth" % (dataset_name, epoch)))
            self.D_B.load_state_dict(torch.load("saved_models/%s/D_B_%d.pth" % (dataset_name, epoch)))

        prev_time = time.time()
        for epoch in range(start_epoch, n_epochs):
            for i, batch in enumerate(dataloader):

                # Model inputs
                real_A = Variable(batch["A"].type(self.Tensor))
                real_B = Variable(batch["B"].type(self.Tensor))

                # Adversarial ground truths
                valid = Variable(self.Tensor(np.ones((real_A.size(0), *self.D_A.output_shape))), requires_grad=False)
                fake = Variable(self.Tensor(np.zeros((real_A.size(0), *self.D_A.output_shape))), requires_grad=False)

                # ------------------
                #  Train Generators
                # ------------------

                self.G_AB.train()
                self.G_BA.train()

                self.optimizer_G.zero_grad()

                # GAN loss
                fake_B = self.G_AB(real_A)
                loss_GAN_AB = self.adversarial_loss(self.D_B(fake_B), valid)
                fake_A = self.G_BA(real_B)
                loss_GAN_BA = self.adversarial_loss(self.D_A(fake_A), valid)

                loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

                # Pixelwise translation loss
                loss_pixelwise = (self.pixelwise_loss(fake_A, real_A) + self.pixelwise_loss(fake_B, real_B)) / 2

                # Cycle loss
                loss_cycle_A = self.cycle_loss(self.G_BA(fake_B), real_A)
                loss_cycle_B = self.cycle_loss(self.G_AB(fake_A), real_B)
                loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

                # Total loss
                loss_G = loss_GAN + loss_cycle + loss_pixelwise

                loss_G.backward()
                self.optimizer_G.step()

                # -----------------------
                #  Train Discriminator A
                # -----------------------

                self.optimizer_D_A.zero_grad()

                # Real loss
                loss_real = self.adversarial_loss(self.D_A(real_A), valid)
                # Fake loss (on batch of previously generated samples)
                loss_fake = self.adversarial_loss(self.D_A(fake_A.detach()), fake)
                # Total loss
                loss_D_A = (loss_real + loss_fake) / 2

                loss_D_A.backward()
                self.optimizer_D_A.step()

                # -----------------------
                #  Train Discriminator B
                # -----------------------

                self.optimizer_D_B.zero_grad()
                # Real loss
                loss_real = self.adversarial_loss(self.D_B(real_B), valid)
                # Fake loss (on batch of previously generated samples)
                loss_fake = self.adversarial_loss(self.D_B(fake_B.detach()), fake)
                # Total loss
                loss_D_B = (loss_real + loss_fake) / 2

                loss_D_B.backward()
                self.optimizer_D_B.step()

                loss_D = 0.5 * (loss_D_A + loss_D_B)

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
                    "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, pixel: %f, cycle: %f] ETA: %s"
                    % (
                        epoch,
                        n_epochs,
                        i,
                        len(dataloader),
                        loss_D.item(),
                        loss_G.item(),
                        loss_GAN.item(),
                        loss_pixelwise.item(),
                        loss_cycle.item(),
                        time_left,
                    )
                )

                # If at sample interval save image
                if batches_done % sample_interval == 0:
                    self.sample_images(val_dataloader, batches_done, dataset_name)

            if checkpoint_interval != -1 and epoch % checkpoint_interval == 0:
                # Save model checkpoints
                torch.save(self.G_AB.state_dict(), "saved_models/%s/G_AB_%d.pth" % (dataset_name, epoch))
                torch.save(self.G_BA.state_dict(), "saved_models/%s/G_BA_%d.pth" % (dataset_name, epoch))
                torch.save(self.D_A.state_dict(), "saved_models/%s/D_A_%d.pth" % (dataset_name, epoch))
                torch.save(self.D_B.state_dict(), "saved_models/%s/D_B_%d.pth" % (dataset_name, epoch))



'''

# Dataset loader
transforms_ = [
    transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]
dataloader = DataLoader(
    ImageDataset("../../data/%s" % opt.dataset_name, transforms_=transforms_, mode="train"),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)
val_dataloader = DataLoader(
    ImageDataset("../../data/%s" % opt.dataset_name, transforms_=transforms_, mode="val"),
    batch_size=16,
    shuffle=True,
    num_workers=opt.n_cpu,
)'''
