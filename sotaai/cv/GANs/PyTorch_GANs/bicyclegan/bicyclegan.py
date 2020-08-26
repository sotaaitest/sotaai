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

from cv.GANs.PyTorch_GANs.bicyclegan.models import *
from cv.GANs.PyTorch_GANs.bicyclegan.datasets import *

import torch.nn as nn
import torch.nn.functional as F
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="edges2shoes", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=128, help="size of image height")
parser.add_argument("--img_width", type=int, default=128, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--latent_dim", type=int, default=8, help="number of latent codes")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between saving generator samples")
parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between model checkpoints")
parser.add_argument("--lambda_pixel", type=float, default=10, help="pixelwise loss weight")
parser.add_argument("--lambda_latent", type=float, default=0.5, help="latent loss weight")
parser.add_argument("--lambda_kl", type=float, default=0.01, help="kullback-leibler loss weight")
opt = parser.parse_args()
print(opt)

os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)

def reparameterization(mu, logvar,latent_dim):
    std = torch.exp(logvar / 2)
    Tensor = torch.Tensor #torch.cuda.FloatTensor if cuda else torch.Tensor
    sampled_z = Variable(Tensor(np.random.normal(0, 1, (mu.size(0), latent_dim))))
    z = sampled_z * std + mu
    return z


class BICYCLEGAN():
    def __init__(self,lr = 0.0002,b1 = 0.5, b2 = 0.999, latent_dim = 8,img_height = 128, img_width = 128,channels = 3,gamma = 0.75,lambda_k = 0.001,k = 0.0):
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.latent_dim = latent_dim
        self.channels = channels
        self.input_shape = (channels, img_height, img_width)


        # Loss functions
        self.mae_loss = torch.nn.L1Loss()

        # Initialize generator, encoder and discriminators
        self.generator = Generator(latent_dim, self.input_shape)
        self.encoder = Encoder(latent_dim, self.input_shape)
        self.D_VAE = MultiDiscriminator(self.input_shape)
        self.D_LR = MultiDiscriminator(self.input_shape)


        cuda = False #True if torch.cuda.is_available() else False
        if cuda:
            self.generator = self.generator.cuda()
            self.encoder.cuda()
            self.D_VAE = self.D_VAE.cuda()
            self.D_LR = self.D_LR.cuda()
            self.mae_loss.cuda()
        self.Tensor = torch.Tensor #torch.cuda.FloatTensor if cuda else torch.Tensor


        # Initialize weights
        self.generator.apply(weights_init_normal)
        self.D_VAE.apply(weights_init_normal)
        self.D_LR.apply(weights_init_normal)

        # Optimizers
        self.optimizer_E = torch.optim.Adam(self.encoder.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        self.optimizer_D_VAE = torch.optim.Adam(self.D_VAE.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        self.optimizer_D_LR = torch.optim.Adam(self.D_LR.parameters(), lr=self.lr, betas=(self.b1, self.b2))




    def sample_images(self,batches_done):
        """Saves a generated sample from the validation set"""
        self.generator.eval()
        imgs = next(iter(val_dataloader))
        img_samples = None
        for img_A, img_B in zip(imgs["A"], imgs["B"]):
            # Repeat input image by number of desired columns
            real_A = img_A.view(1, *img_A.shape).repeat(opt.latent_dim, 1, 1, 1)
            real_A = Variable(real_A.type(self.Tensor))
            # Sample latent representations
            sampled_z = Variable(self.Tensor(np.random.normal(0, 1, (opt.latent_dim, opt.latent_dim))))
            # Generate samples
            fake_B = self.generator(real_A, sampled_z)
            # Concatenate samples horisontally
            fake_B = torch.cat([x for x in fake_B.data.cpu()], -1)
            img_sample = torch.cat((img_A, fake_B), -1)
            img_sample = img_sample.view(1, *img_sample.shape)
            # Concatenate with previous samples vertically
            img_samples = img_sample if img_samples is None else torch.cat((img_samples, img_sample), -2)
        save_image(img_samples, "images/%s/%s.png" % (opt.dataset_name, batches_done), nrow=8, normalize=True)
        self.generator.train()

    # ----------
    #  Training
    # ----------
    def train(self,start_epoch = 0, n_epochs = 200,dataset_name = 'edges2shoes'):
        # Adversarial loss
        valid = 1
        fake = 0

        if epoch != 0:
            # Load pretrained models
            self.generator.load_state_dict(torch.load("saved_models/%s/generator_%d.pth" % (dataset_name, start_epoch)))
            self.encoder.load_state_dict(torch.load("saved_models/%s/encoder_%d.pth" % (dataset_name, start_epoch)))
            self.D_VAE.load_state_dict(torch.load("saved_models/%s/D_VAE_%d.pth" % (dataset_name, start_epoch)))
            self.D_LR.load_state_dict(torch.load("saved_models/%s/D_LR_%d.pth" % (dataset_name, start_epoch)))
        
        prev_time = time.time()
        for epoch in range(start_epoch, n_epochs):
            for i, batch in enumerate(dataloader):

                # Set model input
                real_A = Variable(batch["A"].type(self.Tensor))
                real_B = Variable(batch["B"].type(self.Tensor))

                # -------------------------------
                #  Train Generator and Encoder
                # -------------------------------

                self.optimizer_E.zero_grad()
                self.optimizer_G.zero_grad()

                # ----------
                # cVAE-GAN
                # ----------

                # Produce output using encoding of B (cVAE-GAN)
                mu, logvar = self.encoder(real_B)
                encoded_z = reparameterization(mu, logvar,self.latent_dim)
                fake_B = self.generator(real_A, encoded_z)

                # Pixelwise loss of translated image by VAE
                loss_pixel = self.mae_loss(fake_B, real_B)
                # Kullback-Leibler divergence of encoded B
                loss_kl = 0.5 * torch.sum(torch.exp(logvar) + mu ** 2 - logvar - 1)
                # Adversarial loss
                loss_VAE_GAN = self.D_VAE.compute_loss(fake_B, valid)

                # ---------
                # cLR-GAN
                # ---------

                # Produce output using sampled z (cLR-GAN)
                sampled_z = Variable(self.Tensor(np.random.normal(0, 1, (real_A.size(0), opt.latent_dim))))
                _fake_B = self.generator(real_A, sampled_z)
                # cLR Loss: Adversarial loss
                loss_LR_GAN = self.D_LR.compute_loss(_fake_B, valid)

                # ----------------------------------
                # Total Loss (Generator + Encoder)
                # ----------------------------------

                loss_GE = loss_VAE_GAN + loss_LR_GAN + opt.lambda_pixel * loss_pixel + opt.lambda_kl * loss_kl

                loss_GE.backward(retain_graph=True)
                self.optimizer_E.step()

                # ---------------------
                # Generator Only Loss
                # ---------------------

                # Latent L1 loss
                _mu, _ = self.encoder(_fake_B)
                loss_latent = opt.lambda_latent * self.mae_loss(_mu, sampled_z)

                loss_latent.backward()
                self.optimizer_G.step()

                # ----------------------------------
                #  Train Discriminator (cVAE-GAN)
                # ----------------------------------

                self.optimizer_D_VAE.zero_grad()

                loss_D_VAE = self.D_VAE.compute_loss(real_B, valid) + self.D_VAE.compute_loss(fake_B.detach(), fake)

                loss_D_VAE.backward()
                self.optimizer_D_VAE.step()

                # ---------------------------------
                #  Train Discriminator (cLR-GAN)
                # ---------------------------------

                self.optimizer_D_LR.zero_grad()

                loss_D_LR = self.D_LR.compute_loss(real_B, valid) + self.D_LR.compute_loss(_fake_B.detach(), fake)

                loss_D_LR.backward()
                self.optimizer_D_LR.step()

                # --------------
                #  Log Progress
                # --------------

                # Determine approximate time left
                batches_done = epoch * len(dataloader) + i
                batches_left = opt.n_epochs * len(dataloader) - batches_done
                time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
                prev_time = time.time()

                # Print log
                sys.stdout.write(
                    "\r[Epoch %d/%d] [Batch %d/%d] [D VAE_loss: %f, LR_loss: %f] [G loss: %f, pixel: %f, kl: %f, latent: %f] ETA: %s"
                    % (
                        epoch,
                        opt.n_epochs,
                        i,
                        len(dataloader),
                        loss_D_VAE.item(),
                        loss_D_LR.item(),
                        loss_GE.item(),
                        loss_pixel.item(),
                        loss_kl.item(),
                        loss_latent.item(),
                        time_left,
                    )
                )

                if batches_done % opt.sample_interval == 0:
                    self.sample_images(batches_done)

            if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
                # Save model checkpoints
                torch.save(self.generator.state_dict(), "saved_models/%s/generator_%d.pth" % (opt.dataset_name, epoch))
                torch.save(self.encoder.state_dict(), "saved_models/%s/encoder_%d.pth" % (opt.dataset_name, epoch))
                torch.save(self.D_VAE.state_dict(), "saved_models/%s/D_VAE_%d.pth" % (opt.dataset_name, epoch))
                torch.save(self.D_LR.state_dict(), "saved_models/%s/D_LR_%d.pth" % (opt.dataset_name, epoch))




##############################
#        DatasetLoader
##############################
'''
dataloader = DataLoader(
    ImageDataset("../../data/%s" % opt.dataset_name, input_shape),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)
val_dataloader = DataLoader(
    ImageDataset("../../data/%s" % opt.dataset_name, input_shape, mode="val"),
    batch_size=8,
    shuffle=True,
    num_workers=1,
)'''
