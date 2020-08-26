import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from cv.GANs.PyTorch_GANs.munit.models import *
from cv.GANs.PyTorch_GANs.munit.datasets import *

import torch.nn as nn
import torch.nn.functional as F
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="edges2shoes", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=128, help="size of image height")
parser.add_argument("--img_width", type=int, default=128, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval saving generator samples")
parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between saving model checkpoints")
parser.add_argument("--n_downsample", type=int, default=2, help="number downsampling layers in encoder")
parser.add_argument("--n_residual", type=int, default=3, help="number of residual blocks in encoder / decoder")
parser.add_argument("--dim", type=int, default=64, help="number of filters in first encoder layer")
parser.add_argument("--style_dim", type=int, default=8, help="dimensionality of the style code")
opt = parser.parse_args()
print(opt)


# Create sample and checkpoint directories
os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)




class MUnit():
    def __init__(self, lr = 0.0002,b1 = 0.5, b2 = 0.999, latent_dim = 100,img_height = 256, img_width = 256, channels = 1,dim = 64, n_downsample = 2,n_residual = 3, style_dim = 8):
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.latent_dim = latent_dim
        self.input_shape = (channels, img_height, img_width)
        self.dim = dim
        self.n_downsample = n_downsample
        self.n_residual = n_residual
        self.style_dim = style_dim

        self.criterion_recon = torch.nn.L1Loss()

        # Initialize encoders, generators and discriminators
        self.Enc1 = Encoder(dim=self.dim, n_downsample=self.n_downsample, n_residual=self.n_residual, style_dim=self.style_dim)
        self.Dec1 = Decoder(dim=self.dim, n_upsample=self.n_downsample, n_residual=self.n_residual, style_dim=self.style_dim)
        self.Enc2 = Encoder(dim=self.dim, n_downsample=self.n_downsample, n_residual=self.n_residual, style_dim=self.style_dim)
        self.Dec2 = Decoder(dim=self.dim, n_upsample=self.n_downsample, n_residual=self.n_residual, style_dim=self.style_dim)
        self.D1 = MultiDiscriminator()
        self.D2 = MultiDiscriminator()

        cuda = False #torch.cuda.is_available()
        if cuda:
            self.Enc1 = self.Enc1.cuda()
            self.Dec1 = self.Dec1.cuda()
            self.Enc2 = self.Enc2.cuda()
            self.Dec2 = self.Dec2.cuda()
            self.D1 = self.D1.cuda()
            self.D2 = self.D2.cuda()
            self.criterion_recon.cuda()
        self.Tensor = torch.Tensor #torch.cuda.FloatTensor if cuda else torch.Tensor


        # Initialize weights
        self.Enc1.apply(weights_init_normal)
        self.Dec1.apply(weights_init_normal)
        self.Enc2.apply(weights_init_normal)
        self.Dec2.apply(weights_init_normal)
        self.D1.apply(weights_init_normal)
        self.D2.apply(weights_init_normal)



        # Optimizers
        self.optimizer_G = torch.optim.Adam(
            itertools.chain(self.Enc1.parameters(), self.Dec1.parameters(), self.Enc2.parameters(), self.Dec2.parameters()),
            lr=self.lr,
            betas=(self.b1, self.b2),
        )
        self.optimizer_D1 = torch.optim.Adam(self.D1.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        self.optimizer_D2 = torch.optim.Adam(self.D2.parameters(), lr=self.lr, betas=(self.b1, self.b2))




    def sample_images(self, val_dataloader, batches_done, dataset_name):
        """Saves a generated sample from the validation set"""
        imgs = next(iter(val_dataloader))
        img_samples = None
        for img1, img2 in zip(imgs["A"], imgs["B"]):
            # Create copies of image
            X1 = img1.unsqueeze(0).repeat(self.style_dim, 1, 1, 1)
            X1 = Variable(X1.type(self.Tensor))
            # Get random style codes
            s_code = np.random.uniform(-1, 1, (self.style_dim, self.style_dim))
            s_code = Variable(self.Tensor(s_code))
            # Generate samples
            c_code_1, _ = self.Enc1(X1)
            X12 = self.Dec2(c_code_1, s_code)
            # Concatenate samples horisontally
            X12 = torch.cat([x for x in X12.data.cpu()], -1)
            img_sample = torch.cat((img1, X12), -1).unsqueeze(0)
            # Concatenate with previous samples vertically
            img_samples = img_sample if img_samples is None else torch.cat((img_samples, img_sample), -2)
        save_image(img_samples, "images/%s/%s.png" % (dataset_name, batches_done), nrow=5, normalize=True)


# ----------
#  Training
# ----------
    def train(self, epoch = 0, n_epochs = 200, decay_epoch = 100, sample_interval = 100,checkpoint_interval = -1, dataset_name = 'apple2orange'):

        if epoch != 0:
            # Load pretrained models
            self.Enc1.load_state_dict(torch.load("saved_models/%s/Enc1_%d.pth" % (dataset_name, epoch)))
            self.Dec1.load_state_dict(torch.load("saved_models/%s/Dec1_%d.pth" % (dataset_name, epoch)))
            self.Enc2.load_state_dict(torch.load("saved_models/%s/Enc2_%d.pth" % (dataset_name, epoch)))
            self.Dec2.load_state_dict(torch.load("saved_models/%s/Dec2_%d.pth" % (dataset_name, epoch)))
            self.D1.load_state_dict(torch.load("saved_models/%s/D1_%d.pth" % (dataset_name, epoch)))
            self.D2.load_state_dict(torch.load("saved_models/%s/D2_%d.pth" % (dataset_name, epoch)))

        # Adversarial ground truths
        valid = 1
        fake = 0

        # Learning rate update schedulers
        lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_G, lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step
        )
        lr_scheduler_D1 = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_D1, lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step
        )
        lr_scheduler_D2 = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_D2, lr_lambda=LambdaLR(n_epochs, epoch, decay_epoch).step
        )
        # Loss weights
        lambda_gan = 1
        lambda_id = 10
        lambda_style = 1
        lambda_cont = 1
        lambda_cyc = 0

        prev_time = time.time()
        for epoch in range(epoch, n_epochs):
            for i, batch in enumerate(dataloader):

                # Set model input
                X1 = Variable(batch["A"].type(self.Tensor))
                X2 = Variable(batch["B"].type(self.Tensor))

                # Sampled style codes
                style_1 = Variable(torch.randn(X1.size(0), self.style_dim, 1, 1).type(self.Tensor))
                style_2 = Variable(torch.randn(X1.size(0), self.style_dim, 1, 1).type(self.Tensor))

                # -------------------------------
                #  Train Encoders and Generators
                # -------------------------------

                self.optimizer_G.zero_grad()

                # Get shared latent representation
                c_code_1, s_code_1 = self.Enc1(X1)
                c_code_2, s_code_2 = self.Enc2(X2)

                # Reconstruct images
                X11 = self.Dec1(c_code_1, s_code_1)
                X22 = self.Dec2(c_code_2, s_code_2)

                # Translate images
                X21 = self.Dec1(c_code_2, style_1)
                X12 = self.Dec2(c_code_1, style_2)

                # Cycle translation
                c_code_21, s_code_21 = self.Enc1(X21)
                c_code_12, s_code_12 = self.Enc2(X12)
                X121 = self.Dec1(c_code_12, s_code_1) if lambda_cyc > 0 else 0
                X212 = self.Dec2(c_code_21, s_code_2) if lambda_cyc > 0 else 0

                # Losses
                loss_GAN_1 = lambda_gan * self.D1.compute_loss(X21, valid)
                loss_GAN_2 = lambda_gan * self.D2.compute_loss(X12, valid)
                loss_ID_1 = lambda_id * self.criterion_recon(X11, X1)
                loss_ID_2 = lambda_id * self.criterion_recon(X22, X2)
                loss_s_1 = lambda_style * self.criterion_recon(s_code_21, style_1)
                loss_s_2 = lambda_style * self.criterion_recon(s_code_12, style_2)
                loss_c_1 = lambda_cont * self.criterion_recon(c_code_12, c_code_1.detach())
                loss_c_2 = lambda_cont * self.criterion_recon(c_code_21, c_code_2.detach())
                loss_cyc_1 = lambda_cyc * self.criterion_recon(X121, X1) if lambda_cyc > 0 else 0
                loss_cyc_2 = lambda_cyc * self.criterion_recon(X212, X2) if lambda_cyc > 0 else 0

                # Total loss
                loss_G = (
                    loss_GAN_1
                    + loss_GAN_2
                    + loss_ID_1
                    + loss_ID_2
                    + loss_s_1
                    + loss_s_2
                    + loss_c_1
                    + loss_c_2
                    + loss_cyc_1
                    + loss_cyc_2
                )

                loss_G.backward()
                self.optimizer_G.step()

                # -----------------------
                #  Train Discriminator 1
                # -----------------------

                self.optimizer_D1.zero_grad()

                loss_D1 = self.D1.compute_loss(X1, valid) + self.D1.compute_loss(X21.detach(), fake)

                loss_D1.backward()
                self.optimizer_D1.step()

                # -----------------------
                #  Train Discriminator 2
                # -----------------------

                self.optimizer_D2.zero_grad()

                loss_D2 = self.D2.compute_loss(X2, valid) + self.D2.compute_loss(X12.detach(), fake)

                loss_D2.backward()
                self.optimizer_D2.step()

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
                    "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] ETA: %s"
                    % (epoch, n_epochs, i, len(dataloader), (loss_D1 + loss_D2).item(), loss_G.item(), time_left)
                )

                # If at sample interval save image
                if batches_done % sample_interval == 0:
                    self.sample_images(val_dataloader, batches_done, dataset_name)

            # Update learning rates
            lr_scheduler_G.step()
            lr_scheduler_D1.step()
            lr_scheduler_D2.step()

            if checkpoint_interval != -1 and epoch % checkpoint_interval == 0:
                # Save model checkpoints
                torch.save(self.Enc1.state_dict(), "saved_models/%s/Enc1_%d.pth" % (dataset_name, epoch))
                torch.save(self.Dec1.state_dict(), "saved_models/%s/Dec1_%d.pth" % (dataset_name, epoch))
                torch.save(self.Enc2.state_dict(), "saved_models/%s/Enc2_%d.pth" % (dataset_name, epoch))
                torch.save(self.Dec2.state_dict(), "saved_models/%s/Dec2_%d.pth" % (dataset_name, epoch))
                torch.save(self.D1.state_dict(), "saved_models/%s/D1_%d.pth" % (dataset_name, epoch))
                torch.save(self.D2.state_dict(), "saved_models/%s/D2_%d.pth" % (dataset_name, epoch))






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
    batch_size=5,
    shuffle=True,
    num_workers=1,
)'''
