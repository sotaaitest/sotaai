import argparse
import os
import numpy as np
import math
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.00005, help="learning rate")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)


class Generator(nn.Module):
    def __init__(self, latent_dim,img_shape):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self,img_shape):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity




class WGAN():
    def __init__(self, lr = 0.0002,latent_dim = 100,img_size = 32,channels = 1, clip_value = 0.01):
        self.lr = lr
        self.latent_dim = latent_dim
        self.img_shape = (channels, img_size, img_size)
        self.clip_value = clip_value

        # Initialize generator and discriminator
        self.generator = Generator(self.latent_dim, self.img_shape )
        self.discriminator = Discriminator(self.img_shape)

        cuda = False #True if torch.cuda.is_available() else False
        if cuda:
            self.generator.cuda()
            self.discriminator.cuda()
        self.Tensor = torch.FloatTensor #torch.cuda.FloatTensor if cuda else torch.FloatTensor



        # Optimizers
        self.optimizer_G = torch.optim.RMSprop(self.generator.parameters(), lr=self.lr)
        self.optimizer_D = torch.optim.RMSprop(self.discriminator.parameters(), lr=self.lr)


    # ----------
    #  Training
    # ----------
    def train(self,n_epochs = 200, sample_interval = 400,n_critic = 5):
        batches_done = 0
        for epoch in range(n_epochs):

            for i, (imgs, _) in enumerate(dataloader):

                # Configure input
                real_imgs = Variable(imgs.type(self.Tensor))

                # ---------------------
                #  Train Discriminator
                # ---------------------

                self.optimizer_D.zero_grad()

                # Sample noise as generator input
                z = Variable(self.Tensor(np.random.normal(0, 1, (imgs.shape[0], self.latent_dim))))

                # Generate a batch of images
                fake_imgs = self.generator(z).detach()
                # Adversarial loss
                loss_D = -torch.mean(self.discriminator(real_imgs)) + torch.mean(self.discriminator(fake_imgs))

                loss_D.backward()
                self.optimizer_D.step()

                # Clip weights of discriminator
                for p in self.discriminator.parameters():
                    p.data.clamp_(-self.clip_value, self.clip_value)

                # Train the generator every n_critic iterations
                if i % n_critic == 0:

                    # -----------------
                    #  Train Generator
                    # -----------------

                    self.optimizer_G.zero_grad()

                    # Generate a batch of images
                    gen_imgs = self.generator(z)
                    # Adversarial loss
                    loss_G = -torch.mean(self.discriminator(gen_imgs))

                    loss_G.backward()
                    self.optimizer_G.step()

                    print(
                        "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                        % (epoch, n_epochs, batches_done % len(dataloader), len(dataloader), loss_D.item(), loss_G.item())
                    )

                if batches_done % sample_interval == 0:
                    save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)
                batches_done += 1

'''

# Configure data loader
os.makedirs("../../data/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../../data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)'''
