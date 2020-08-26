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

from cv.GANs.PyTorch_GANs.pixelda.mnistm import MNISTM

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
parser.add_argument("--n_residual_blocks", type=int, default=6, help="number of residual blocks in generator")
parser.add_argument("--latent_dim", type=int, default=10, help="dimensionality of the noise input")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--n_classes", type=int, default=10, help="number of classes in the dataset")
parser.add_argument("--sample_interval", type=int, default=300, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class ResidualBlock(nn.Module):
    def __init__(self, in_features=64, out_features=64):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, 3, 1, 1),
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_features, in_features, 3, 1, 1),
            nn.BatchNorm2d(in_features),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, latent_dim, img_size, channels, n_residual_blocks):
        super(Generator, self).__init__()

        # Fully-connected layer which constructs image channel shaped output from noise
        self.fc = nn.Linear(latent_dim, channels * img_size ** 2)

        self.l1 = nn.Sequential(nn.Conv2d(channels * 2, 64, 3, 1, 1), nn.ReLU(inplace=True))

        resblocks = []
        for _ in range(n_residual_blocks):
            resblocks.append(ResidualBlock())
        self.resblocks = nn.Sequential(*resblocks)

        self.l2 = nn.Sequential(nn.Conv2d(64, channels, 3, 1, 1), nn.Tanh())

    def forward(self, img, z):
        gen_input = torch.cat((img, self.fc(z).view(*img.shape)), 1)
        out = self.l1(gen_input)
        out = self.resblocks(out)
        img_ = self.l2(out)

        return img_


class Discriminator(nn.Module):
    def __init__(self, channels):
        super(Discriminator, self).__init__()

        def block(in_features, out_features, normalization=True):
            """Discriminator block"""
            layers = [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_features))
            return layers

        self.model = nn.Sequential(
            *block(channels, 64, normalization=False),
            *block(64, 128),
            *block(128, 256),
            *block(256, 512),
            nn.Conv2d(512, 1, 3, 1, 1)
        )

    def forward(self, img):
        validity = self.model(img)

        return validity


class Classifier(nn.Module):
    def __init__(self, img_size, channels, n_classes):
        super(Classifier, self).__init__()

        def block(in_features, out_features, normalization=True):
            """Classifier block"""
            layers = [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_features))
            return layers

        self.model = nn.Sequential(
            *block(channels, 64, normalization=False), *block(64, 128), *block(128, 256), *block(256, 512)
        )

        input_size = img_size // 2 ** 4
        self.output_layer = nn.Sequential(nn.Linear(512 * input_size ** 2, n_classes), nn.Softmax())

    def forward(self, img):
        feature_repr = self.model(img)
        feature_repr = feature_repr.view(feature_repr.size(0), -1)
        label = self.output_layer(feature_repr)
        return label





class PIXELDA():
    def __init__(self, lr = 0.0002,b1 = 0.5, b2 = 0.999, latent_dim = 100,img_size = 32,channels = 1,n_classes = 10, n_residual_blocks = 6):
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.img_size = img_size
        self.channels = channels
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.n_residual_blocks = n_residual_blocks

       

        # Loss function
        self.adversarial_loss = torch.nn.MSELoss()
        self.task_loss = torch.nn.CrossEntropyLoss()

        

        # Initialize generator and discriminator
        self.generator = Generator(self.latent_dim, self.img_size, self.channels, self.n_residual_blocks)
        self.discriminator = Discriminator(self.channels)
        self.classifier = Classifier(self.img_size, self.channels, self.n_classes)

        cuda = False #True if torch.cuda.is_available() else False
        if cuda:
            self.generator.cuda()
            self.discriminator.cuda()
            self.classifier.cuda()
            self.adversarial_loss.cuda()
            self.task_loss.cuda()
        self.FloatTensor = torch.FloatTensor  #torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.LongTensor = torch.LongTensor  #torch.cuda.LongTensor if cuda else torch.LongTensor
        # Initialize weights
        self.generator.apply(weights_init_normal)
        self.discriminator.apply(weights_init_normal)
        self.classifier.apply(weights_init_normal)

        # Optimizers
        self.optimizer_G = torch.optim.Adam(
            itertools.chain(self.generator.parameters(), self.classifier.parameters()), lr=self.lr, betas=(self.b1, self.b2)
        )
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.b1, self.b2))

        

    # ----------
    #  Training
    # ----------
    def train(self, n_epochs,sample_interval):

         # Calculate output of image discriminator (PatchGAN)
        patch = int(self.img_size / 2 ** 4)
        patch = (1, patch, patch)
        
        # Loss weights
        lambda_adv = 1
        lambda_task = 0.1
        
        # Keeps 100 accuracy measurements
        task_performance = []
        target_performance = []

        for epoch in range(n_epochs):
            for i, ((imgs_A, labels_A), (imgs_B, labels_B)) in enumerate(zip(dataloader_A, dataloader_B)):

                batch_size = imgs_A.size(0)

                # Adversarial ground truths
                valid = Variable(self.FloatTensor(batch_size, *patch).fill_(1.0), requires_grad=False)
                fake = Variable(self.FloatTensor(batch_size, *patch).fill_(0.0), requires_grad=False)

                # Configure input
                imgs_A = Variable(imgs_A.type(self.FloatTensor).expand(batch_size, 3, self.img_size, self.img_size))
                labels_A = Variable(labels_A.type(self.LongTensor))
                imgs_B = Variable(imgs_B.type(self.FloatTensor))

                # -----------------
                #  Train Generator
                # -----------------

                self.optimizer_G.zero_grad()

                # Sample noise
                z = Variable(self.FloatTensor(np.random.uniform(-1, 1, (batch_size, self.latent_dim))))

                # Generate a batch of images
                fake_B = self.generator(imgs_A, z)

                # Perform task on translated source image
                label_pred = self.classifier(fake_B)

                # Calculate the task loss
                task_loss_ = (self.task_loss(label_pred, labels_A) + self.task_loss(self.classifier(imgs_A), labels_A)) / 2

                # Loss measures generator's ability to fool the discriminator
                g_loss = lambda_adv * self.adversarial_loss(self.discriminator(fake_B), valid) + lambda_task * task_loss_

                g_loss.backward()
                self.optimizer_G.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                self.optimizer_D.zero_grad()

                # Measure discriminator's ability to classify real from generated samples
                real_loss = self.adversarial_loss(self.discriminator(imgs_B), valid)
                fake_loss = self.adversarial_loss(self.discriminator(fake_B.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2

                d_loss.backward()
                self.optimizer_D.step()

                # ---------------------------------------
                #  Evaluate Performance on target domain
                # ---------------------------------------

                # Evaluate performance on translated Domain A
                acc = np.mean(np.argmax(label_pred.data.cpu().numpy(), axis=1) == labels_A.data.cpu().numpy())
                task_performance.append(acc)
                if len(task_performance) > 100:
                    task_performance.pop(0)

                # Evaluate performance on Domain B
                pred_B = self.classifier(imgs_B)
                target_acc = np.mean(np.argmax(pred_B.data.cpu().numpy(), axis=1) == labels_B.numpy())
                target_performance.append(target_acc)
                if len(target_performance) > 100:
                    target_performance.pop(0)

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [CLF acc: %3d%% (%3d%%), target_acc: %3d%% (%3d%%)]"
                    % (
                        epoch,
                        n_epochs,
                        i,
                        len(dataloader_A),
                        d_loss.item(),
                        g_loss.item(),
                        100 * acc,
                        100 * np.mean(task_performance),
                        100 * target_acc,
                        100 * np.mean(target_performance),
                    )
                )

                batches_done = len(dataloader_A) * epoch + i
                if batches_done % sample_interval == 0:
                    sample = torch.cat((imgs_A.data[:5], fake_B.data[:5], imgs_B.data[:5]), -2)
                    save_image(sample, "images/%d.png" % batches_done, nrow=int(math.sqrt(batch_size)), normalize=True)




'''
# Configure data loader
os.makedirs("../../data/mnist", exist_ok=True)
dataloader_A = torch.utils.data.DataLoader(
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

os.makedirs("../../data/mnistm", exist_ok=True)
dataloader_B = torch.utils.data.DataLoader(
    MNISTM(
        "../../data/mnistm",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.Resize(opt.img_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)'''
