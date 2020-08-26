import argparse
import os
import numpy as np
import math

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
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=62, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="number of image channels")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)



def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self,latent_dim,img_size,channels):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.channels = channels
        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise):
        out = self.l1(noise)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self,img_size,channels):
        super(Discriminator, self).__init__()
        self.img_size = img_size
        self.channels = channels
        # Upsampling
        self.down = nn.Sequential(nn.Conv2d(self.channels, 64, 3, 2, 1), nn.ReLU())
        # Fully-connected layers
        self.down_size = self.img_size // 2
        down_dim = 64 * (self.img_size // 2) ** 2
        self.fc = nn.Sequential(
            nn.Linear(down_dim, 32),
            nn.BatchNorm1d(32, 0.8),
            nn.ReLU(inplace=True),
            nn.Linear(32, down_dim),
            nn.BatchNorm1d(down_dim),
            nn.ReLU(inplace=True),
        )
        # Upsampling
        self.up = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(64, self.channels, 3, 1, 1))

    def forward(self, img):
        out = self.down(img)
        out = self.fc(out.view(out.size(0), -1))
        out = self.up(out.view(out.size(0), 64, self.down_size, self.down_size))
        return out

class BEGAN():
    
    def __init__(self,lr = 0.0002,b1 = 0.5, b2 = 0.999, latent_dim = 62,img_size = 32,channels = 1,gamma = 0.75,lambda_k = 0.001,k = 0.0):
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.channels = channels
       
        # BEGAN hyper parameters
        self.gamma = gamma
        self.lambda_k = lambda_k
        self.k = k
        # Initialize generator and discriminator
        self.generator = Generator(self.latent_dim,self.img_size,self.channels)
        self.discriminator = Discriminator(self.img_size,self.channels)
 
        # Initialize weights
        self.generator.apply(weights_init_normal)
        self.discriminator.apply(weights_init_normal)


        # Optimizers
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        
        cuda = False #True if torch.cuda.is_available() else False

        if cuda:
            self.generator.cuda()
            self.discriminator.cuda()
        self.Tensor = torch.FloatTensor #torch.cuda.FloatTensor if cuda else torch.FloatTensor
    
   

    # ----------
    #  Training
    # ----------

    def train(self,n_epochs, sample_interval):
        for epoch in range(n_epochs):
            for i, (imgs, _) in enumerate(dataloader):

                # Configure input
                real_imgs = Variable(imgs.type(self.Tensor))

                # -----------------
                #  Train Generator
                # -----------------

                self.optimizer_G.zero_grad()

                # Sample noise as generator input
                z = Variable(self.Tensor(np.random.normal(0, 1, (imgs.shape[0], self.latent_dim))))

                # Generate a batch of images
                gen_imgs = self.generator(z)

                # Loss measures generator's ability to fool the discriminator
                g_loss = torch.mean(torch.abs(self.discriminator(gen_imgs) - gen_imgs))

                g_loss.backward()
                self.optimizer_G.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                self.optimizer_D.zero_grad()

                # Measure discriminator's ability to classify real from generated samples
                d_real = self.discriminator(real_imgs)
                d_fake = self.discriminator(gen_imgs.detach())

                d_loss_real = torch.mean(torch.abs(d_real - real_imgs))
                d_loss_fake = torch.mean(torch.abs(d_fake - gen_imgs.detach()))
                d_loss = d_loss_real - k * d_loss_fake

                d_loss.backward()
                self.optimizer_D.step()

                # ----------------
                # Update weights
                # ----------------

                diff = torch.mean(self.gamma * d_loss_real - d_loss_fake)

                # Update weight term for fake samples
                k = k + self.lambda_k * diff.item()
                k = min(max(k, 0), 1)  # Constraint to interval [0, 1]

                # Update convergence metric
                M = (d_loss_real + torch.abs(diff)).data[0]

                # --------------
                # Log Progress
                # --------------

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] -- M: %f, k: %f"
                    % (epoch, n_epochs, i, len(dataloader), d_loss.item(), g_loss.item(), M, k)
                )

                batches_done = epoch * len(dataloader) + i
                if batches_done % sample_interval == 0:
                    save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)




# Configure data loader
os.makedirs("../../data/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../../data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)
