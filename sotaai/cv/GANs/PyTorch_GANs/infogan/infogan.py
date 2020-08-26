import argparse
import os
import numpy as np
import math
import itertools

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs("images/static/", exist_ok=True)
os.makedirs("images/varying_c1/", exist_ok=True)
os.makedirs("images/varying_c2/", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=62, help="dimensionality of the latent space")
parser.add_argument("--code_dim", type=int, default=2, help="latent code")
parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
opt = parser.parse_args()
print(opt)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def to_categorical(y, num_columns):
    """Returns one-hot encoded Variable"""
    y_cat = np.zeros((y.shape[0], num_columns))
    y_cat[range(y.shape[0]), y] = 1.0

    return Variable(torch.FloatTensor(y_cat))


class Generator(nn.Module):
    def __init__(self, latent_dim,code_dim, img_size, channels,n_classes):
        super(Generator, self).__init__()
        input_dim = latent_dim + n_classes + code_dim

        self.init_size = img_size // 4  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(input_dim, 128 * self.init_size ** 2))

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

    def forward(self, noise, labels, code):
        gen_input = torch.cat((noise, labels, code), -1)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self, code_dim, img_size, channels, n_classes):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = img_size // 2 ** 4

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1))
        self.aux_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, n_classes), nn.Softmax())
        self.latent_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, code_dim))

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)
        latent_code = self.latent_layer(out)

        return validity, label, latent_code


class INFOGAN():
    def __init__(self, lr = 0.0002,b1 = 0.5, b2 = 0.999, latent_dim = 100,img_size = 32,channels = 1,n_classes = 10,code_dim=2):
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.img_size = img_size
        self.channels = channels
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.code_dim = code_dim 
        # Loss functions
        self.adversarial_loss = torch.nn.MSELoss()
        self.categorical_loss = torch.nn.CrossEntropyLoss()
        self.continuous_loss = torch.nn.MSELoss()
        

        

        # Initialize generator and discriminator
        self.generator = Generator(self.latent_dim,self.code_dim,self.img_size,self.channels,self.n_classes)
        self.discriminator = Discriminator(self.code_dim,self.img_size,self.channels, self.n_classes)

        cuda = False #True if torch.cuda.is_available() else False

        if cuda:
            self.generator.cuda()
            self.discriminator.cuda()
            self.adversarial_loss.cuda()
            self.categorical_loss.cuda()
            self.continuous_loss.cuda()

        self.FloatTensor = torch.FloatTensor #torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.LongTensor = torch.LongTensor #torch.cuda.LongTensor if cuda else torch.LongTensor

        # Initialize weights
        self.generator.apply(weights_init_normal)
        self.discriminator.apply(weights_init_normal)



        # Optimizers
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        self.optimizer_info = torch.optim.Adam(
            itertools.chain(self.generator.parameters(), self.discriminator.parameters()), lr=self.lr, betas=(self.b1, self.b2)
        )

        # Static generator inputs for sampling
        self.static_z = Variable(self.FloatTensor(np.zeros((self.n_classes ** 2, self.latent_dim))))
        self.static_label = to_categorical(
            np.array([num for _ in range(self.n_classes) for num in range(self.n_classes)]), num_columns=self.n_classes
        )
        self.static_code = Variable(self.FloatTensor(np.zeros((self.n_classes ** 2, self.code_dim))))


    def sample_image(self,n_row, batches_done):
        """Saves a grid of generated digits ranging from 0 to n_classes"""
        # Static sample
        z = Variable(self.FloatTensor(np.random.normal(0, 1, (n_row ** 2, self.latent_dim))))
        static_sample = self.generator(z, self.static_label, self.static_code)
        save_image(static_sample.data, "images/static/%d.png" % batches_done, nrow=n_row, normalize=True)

        # Get varied c1 and c2
        zeros = np.zeros((n_row ** 2, 1))
        c_varied = np.repeat(np.linspace(-1, 1, n_row)[:, np.newaxis], n_row, 0)
        c1 = Variable(self.FloatTensor(np.concatenate((c_varied, zeros), -1)))
        c2 = Variable(self.FloatTensor(np.concatenate((zeros, c_varied), -1)))
        sample1 = self.generator(self.static_z, self.static_label, c1)
        sample2 = self.generator(self.static_z, self.static_label, c2)
        save_image(sample1.data, "images/varying_c1/%d.png" % batches_done, nrow=n_row, normalize=True)
        save_image(sample2.data, "images/varying_c2/%d.png" % batches_done, nrow=n_row, normalize=True)


    # ----------
    #  Training
    # ----------
    def train(self,n_epochs, batch_size, sample_interval):
        # Loss weights
        lambda_cat = 1
        lambda_con = 0.1
        for epoch in range(n_epochs):
            for i, (imgs, labels) in enumerate(dataloader):

                batch_size = imgs.shape[0]

                # Adversarial ground truths
                valid = Variable(self.FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
                fake = Variable(self.FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

                # Configure input
                real_imgs = Variable(imgs.type(self.FloatTensor))
                labels = to_categorical(labels.numpy(), num_columns=self.n_classes)

                # -----------------
                #  Train Generator
                # -----------------

                self.optimizer_G.zero_grad()

                # Sample noise and labels as generator input
                z = Variable(self.FloatTensor(np.random.normal(0, 1, (batch_size, self.latent_dim))))
                label_input = to_categorical(np.random.randint(0, self.n_classes, batch_size), num_columns=self.n_classes)
                code_input = Variable(self.FloatTensor(np.random.uniform(-1, 1, (batch_size, self.code_dim))))

                # Generate a batch of images
                gen_imgs = self.generator(z, label_input, code_input)

                # Loss measures generator's ability to fool the discriminator
                validity, _, _ = self.discriminator(gen_imgs)
                g_loss = self.adversarial_loss(validity, valid)

                g_loss.backward()
                self.optimizer_G.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                self.optimizer_D.zero_grad()

                # Loss for real images
                real_pred, _, _ = self.discriminator(real_imgs)
                d_real_loss = self.adversarial_loss(real_pred, valid)

                # Loss for fake images
                fake_pred, _, _ = self.discriminator(gen_imgs.detach())
                d_fake_loss = self.adversarial_loss(fake_pred, fake)

                # Total discriminator loss
                d_loss = (d_real_loss + d_fake_loss) / 2

                d_loss.backward()
                self.optimizer_D.step()

                # ------------------
                # Information Loss
                # ------------------

                self.optimizer_info.zero_grad()

                # Sample labels
                sampled_labels = np.random.randint(0, self.n_classes, batch_size)

                # Ground truth labels
                gt_labels = Variable(self.LongTensor(sampled_labels), requires_grad=False)

                # Sample noise, labels and code as generator input
                z = Variable(self.FloatTensor(np.random.normal(0, 1, (batch_size, self.latent_dim))))
                label_input = to_categorical(sampled_labels, num_columns=self.n_classes)
                code_input = Variable(self.FloatTensor(np.random.uniform(-1, 1, (batch_size, self.code_dim))))

                gen_imgs = self.generator(z, label_input, code_input)
                _, pred_label, pred_code = self.discriminator(gen_imgs)

                info_loss = lambda_cat * self.categorical_loss(pred_label, gt_labels) + lambda_con * self.continuous_loss(
                    pred_code, code_input
                )

                info_loss.backward()
                self.optimizer_info.step()

                # --------------
                # Log Progress
                # --------------

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [info loss: %f]"
                    % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item(), info_loss.item())
                )
                batches_done = epoch * len(dataloader) + i
                if batches_done % sample_interval == 0:
                    self.sample_image(n_row=10, batches_done=batches_done)

'''


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
)'''
