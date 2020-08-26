import argparse
import os
import numpy as np
import math
import itertools
import scipy
import sys
import time
import datetime

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.autograd as autograd

from cv.GANs.PyTorch_GANs.dualgan.datasets import *
from cv.GANs.PyTorch_GANs.dualgan.models import *

import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--dataset_name", type=str, default="edges2shoes", help="name of the dataset")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--sample_interval", type=int, default=200, help="interval betwen image samples")
parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between model checkpoints")
opt = parser.parse_args()
print(opt)

os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)



def compute_gradient_penalty(D, real_samples, fake_samples):

    FloatTensor = torch.FloatTensor #torch.cuda.FloatTensor if cuda else torch.FloatTensor
    LongTensor = torch.LongTensor  #torch.cuda.LongTensor if cuda else torch.LongTensor

    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = FloatTensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    validity = D(interpolates)
    fake = Variable(FloatTensor(np.ones(validity.shape)), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=validity,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty




class DUALGAN():
    def __init__(self, lr = 0.0002,b1 = 0.5, b2 = 0.999, latent_dim = 100,img_height = 256, img_width = 256, channels = 1):
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.latent_dim = latent_dim
        self.img_shape = (channels, img_height, img_width)


        # Loss function
        self.cycle_loss = torch.nn.L1Loss()

        # Initialize generator and discriminator
        self.G_AB = Generator()
        self.G_BA = Generator()
        self.D_A = Discriminator()
        self.D_B = Discriminator()


        cuda = False #True if torch.cuda.is_available() else False

        if cuda:
            self.G_AB.cuda()
            self.G_BA.cuda()
            self.D_A.cuda()
            self.D_B.cuda()
            self.cycle_loss.cuda()



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

        self.FloatTensor = torch.FloatTensor #torch.cuda.FloatTensor if cuda else torch.FloatTensor
        self.LongTensor = torch.LongTensor  #torch.cuda.LongTensor if cuda else torch.LongTensor




    def sample_images(self, val_dataloader, batches_done, dataset_name):
        """Saves a generated sample from the test set"""
        imgs = next(iter(val_dataloader))
        real_A = Variable(imgs["A"].type(self.FloatTensor))
        fake_B = self.G_AB(real_A)
        AB = torch.cat((real_A.data, fake_B.data), -2)
        real_B = Variable(imgs["B"].type(self.FloatTensor))
        fake_A = self.G_BA(real_B)
        BA = torch.cat((real_B.data, fake_A.data), -2)
        img_sample = torch.cat((AB, BA), 0)
        save_image(img_sample, "images/%s/%s.png" % (dataset_name, batches_done), nrow=8, normalize=True)


# ----------
#  Training
# ----------
    def train(self, start_epoch = 0, n_epochs = 200, sample_interval = 100,checkpoint_interval = -1, n_critic = 5, dataset_name = 'edges2shoes'):

        if start_epoch != 0:
            # Load pretrained models
            self.G_AB.load_state_dict(torch.load("saved_models/%s/G_AB_%d.pth" % (dataset_name, start_epoch)))
            self.G_BA.load_state_dict(torch.load("saved_models/%s/G_BA_%d.pth" % (dataset_name, start_epoch)))
            self.D_A.load_state_dict(torch.load("saved_models/%s/D_A_%d.pth" % (dataset_name, start_epoch)))
            self.D_B.load_state_dict(torch.load("saved_models/%s/D_B_%d.pth" % (dataset_name, start_epoch)))


        # Loss weights
        lambda_adv = 1
        lambda_cycle = 10
        lambda_gp = 10

        batches_done = 0
        prev_time = time.time()
        for epoch in range(start_epoch,n_epochs):
            for i, batch in enumerate(dataloader):

                # Configure input
                imgs_A = Variable(batch["A"].type(self.FloatTensor))
                imgs_B = Variable(batch["B"].type(self.FloatTensor))

                # ----------------------
                #  Train Discriminators
                # ----------------------

                self.optimizer_D_A.zero_grad()
                self.optimizer_D_B.zero_grad()

                # Generate a batch of images
                fake_A = self.G_BA(imgs_B).detach()
                fake_B = self.G_AB(imgs_A).detach()

                # ----------
                # Domain A
                # ----------

                # Compute gradient penalty for improved wasserstein training
                gp_A = compute_gradient_penalty(self.D_A, imgs_A.data, fake_A.data)
                # Adversarial loss
                D_A_loss = -torch.mean(self.D_A(imgs_A)) + torch.mean(self.D_A(fake_A)) + lambda_gp * gp_A

                # ----------
                # Domain B
                # ----------

                # Compute gradient penalty for improved wasserstein training
                gp_B = compute_gradient_penalty(self.D_B, imgs_B.data, fake_B.data)
                # Adversarial loss
                D_B_loss = -torch.mean(self.D_B(imgs_B)) + torch.mean(self.D_B(fake_B)) + lambda_gp * gp_B

                # Total loss
                D_loss = D_A_loss + D_B_loss

                D_loss.backward()
                self.optimizer_D_A.step()
                self.optimizer_D_B.step()

                if i % n_critic == 0:

                    # ------------------
                    #  Train Generators
                    # ------------------

                    self.optimizer_G.zero_grad()

                    # Translate images to opposite domain
                    fake_A = self.G_BA(imgs_B)
                    fake_B = self.G_AB(imgs_A)

                    # Reconstruct images
                    recov_A = self.G_BA(fake_B)
                    recov_B = self.G_AB(fake_A)

                    # Adversarial loss
                    G_adv = -torch.mean(D_A(fake_A)) - torch.mean(D_B(fake_B))
                    # Cycle loss
                    G_cycle = self.cycle_loss(recov_A, imgs_A) + self.cycle_loss(recov_B, imgs_B)
                    # Total loss
                    G_loss = lambda_adv * G_adv + lambda_cycle * G_cycle

                    G_loss.backward()
                    self.optimizer_G.step()

                    # --------------
                    # Log Progress
                    # --------------

                    # Determine approximate time left
                    batches_left = n_epochs * len(dataloader) - batches_done
                    time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time) / n_critic)
                    prev_time = time.time()

                    sys.stdout.write(
                        "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, cycle: %f] ETA: %s"
                        % (
                            epoch,
                            n_epochs,
                            i,
                            len(dataloader),
                            D_loss.item(),
                            G_adv.data.item(),
                            G_cycle.item(),
                            time_left,
                        )
                    )

                # Check sample interval => save sample if there
                if batches_done % sample_interval == 0:
                    self.sample_images(val_dataloader,batches_done,dataset_name)

                batches_done += 1

            if checkpoint_interval != -1 and epoch % checkpoint_interval == 0:
                # Save model checkpoints
                torch.save(self.G_AB.state_dict(), "saved_models/%s/G_AB_%d.pth" % (dataset_name, epoch))
                torch.save(self.G_BA.state_dict(), "saved_models/%s/G_BA_%d.pth" % (dataset_name, epoch))
                torch.save(self.D_A.state_dict(), "saved_models/%s/D_A_%d.pth" % (dataset_name, epoch))
                torch.save(self.D_B.state_dict(), "saved_models/%s/D_B_%d.pth" % (dataset_name, epoch))




'''
# Configure data loader
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
val_dataloader = DataLoader(
    ImageDataset("../../data/%s" % opt.dataset_name, mode="val", transforms_=transforms_),
    batch_size=16,
    shuffle=True,
    num_workers=1,
)'''
