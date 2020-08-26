import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from cv.GANs.PyTorch_GANs.cyclegan.models import *
from cv.GANs.PyTorch_GANs.cyclegan.datasets import *
from cv.GANs.PyTorch_GANs.cyclegan.utils import *

import torch.nn as nn
import torch.nn.functional as F
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="monet2photo", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=256, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving generator outputs")
parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between saving model checkpoints")
parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator")
parser.add_argument("--lambda_cyc", type=float, default=10.0, help="cycle loss weight")
parser.add_argument("--lambda_id", type=float, default=5.0, help="identity loss weight")
opt = parser.parse_args()
print(opt)

# Create sample and checkpoint directories
os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)




class CycleGAN():
    def __init__(self, lr = 0.0002,b1 = 0.5, b2 = 0.999, latent_dim = 100,img_height = 256, img_width =256, channels = 1,n_residual_blocks = 9):
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.latent_dim = latent_dim
        self.input_shape = (channels, img_height, img_width)

        # Losses
        self.criterion_GAN = torch.nn.MSELoss()
        self.criterion_cycle = torch.nn.L1Loss()
        self.criterion_identity = torch.nn.L1Loss()


        # Initialize generator and discriminator
        self.G_AB = GeneratorResNet(self.input_shape, n_residual_blocks)
        self.G_BA = GeneratorResNet(self.input_shape, n_residual_blocks)
        self.D_A = Discriminator(self.input_shape)
        self.D_B = Discriminator(self.input_shape)

        cuda = False #torch.cuda.is_available()

        if cuda:
            self.G_AB = self.G_AB.cuda()
            self.G_BA = self.G_BA.cuda()
            self.D_A = self.D_A.cuda()
            self.D_B = self.D_B.cuda()
            self.criterion_GAN.cuda()
            self.criterion_cycle.cuda()
            self.criterion_identity.cuda()

       
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

        self.Tensor = torch.Tensor #torch.cuda.FloatTensor if cuda else torch.Tensor

    def sample_images(self, batches_done,dataset_name):
        """Saves a generated sample from the test set"""
        imgs = next(iter(val_dataloader))
        self.G_AB.eval()
        self.G_BA.eval()
        real_A = Variable(imgs["A"].type(self.Tensor))
        fake_B = self.G_AB(real_A)
        real_B = Variable(imgs["B"].type(self.Tensor))
        fake_A = self.G_BA(real_B)
        # Arange images along x-axis
        real_A = make_grid(real_A, nrow=5, normalize=True)
        real_B = make_grid(real_B, nrow=5, normalize=True)
        fake_A = make_grid(fake_A, nrow=5, normalize=True)
        fake_B = make_grid(fake_B, nrow=5, normalize=True)
        # Arange images along y-axis
        image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)
        save_image(image_grid, "images/%s/%s.png" % (dataset_name, batches_done), normalize=False)


# ----------
#  Training
# ----------
    def train(self, dataset_name = 'monet2photo', start_epoch = 0, n_epochs = 200, decay_epoch = 100,sample_interval = 100,checkpoint_interval = -1,lambda_cyc = 10.0, lambda_id = 5.0):
        if start_epoch != 0:
            # Load pretrained models
            self.G_AB.load_state_dict(torch.load("saved_models/%s/G_AB_%d.pth" % (dataset_name, start_epoch)))
            self.G_BA.load_state_dict(torch.load("saved_models/%s/G_BA_%d.pth" % (dataset_name, start_epoch)))
            self.D_A.load_state_dict(torch.load("saved_models/%s/D_A_%d.pth" % (dataset_name, start_epoch)))
            self.D_B.load_state_dict(torch.load("saved_models/%s/D_B_%d.pth" % (dataset_name, start_epoch)))
 
        # Learning rate update schedulers
        self.lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_G, lr_lambda=LambdaLR(n_epochs, start_epoch, decay_epoch).step
        )
        self.lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_D_A, lr_lambda=LambdaLR(n_epochs, start_epoch, decay_epoch).step
        )
        self.lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer_D_B, lr_lambda=LambdaLR(n_epochs, start_epoch, decay_epoch).step
        )
 
        # Buffers of previously generated samples
        fake_A_buffer = ReplayBuffer()
        fake_B_buffer = ReplayBuffer()

        prev_time = time.time()
        
        for epoch in range(start_epoch, n_epochs):
            for i, batch in enumerate(dataloader):

                # Set model input
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

                # Identity loss
                loss_id_A = self.criterion_identity(self.G_BA(real_A), real_A)
                loss_id_B = self.criterion_identity(self.G_AB(real_B), real_B)

                loss_identity = (loss_id_A + loss_id_B) / 2

                # GAN loss
                fake_B = self.G_AB(real_A)
                loss_GAN_AB = self.criterion_GAN(self.D_B(fake_B), valid)
                fake_A = self.G_BA(real_B)
                loss_GAN_BA = self.criterion_GAN(self.D_A(fake_A), valid)

                loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

                # Cycle loss
                recov_A = self.G_BA(fake_B)
                loss_cycle_A = self.criterion_cycle(recov_A, real_A)
                recov_B = self.G_AB(fake_A)
                loss_cycle_B = self.criterion_cycle(recov_B, real_B)

                loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

                # Total loss
                loss_G = loss_GAN + lambda_cyc * loss_cycle + lambda_id * loss_identity

                loss_G.backward()
                self.optimizer_G.step()

                # -----------------------
                #  Train Discriminator A
                # -----------------------

                self.optimizer_D_A.zero_grad()

                # Real loss
                loss_real = self.criterion_GAN(self.D_A(real_A), valid)
                # Fake loss (on batch of previously generated samples)
                fake_A_ = fake_A_buffer.push_and_pop(fake_A)
                loss_fake = self.criterion_GAN(self.D_A(fake_A_.detach()), fake)
                # Total loss
                loss_D_A = (loss_real + loss_fake) / 2

                loss_D_A.backward()
                self.optimizer_D_A.step()

                # -----------------------
                #  Train Discriminator B
                # -----------------------

                self.optimizer_D_B.zero_grad()

                # Real loss
                loss_real = self.criterion_GAN(self.D_B(real_B), valid)
                # Fake loss (on batch of previously generated samples)
                fake_B_ = fake_B_buffer.push_and_pop(fake_B)
                loss_fake = self.criterion_GAN(self.D_B(fake_B_.detach()), fake)
                # Total loss
                loss_D_B = (loss_real + loss_fake) / 2

                loss_D_B.backward()
                self.optimizer_D_B.step()

                loss_D = (loss_D_A + loss_D_B) / 2

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
                    "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f, identity: %f] ETA: %s"
                    % (
                        epoch,
                        n_epochs,
                        i,
                        len(dataloader),
                        loss_D.item(),
                        loss_G.item(),
                        loss_GAN.item(),
                        loss_cycle.item(),
                        loss_identity.item(),
                        time_left,
                    )
                )

                # If at sample interval save image
                if batches_done % sample_interval == 0:
                    self.sample_images(batches_done,dataset_name)

            # Update learning rates
            self.lr_scheduler_G.step()
            self.lr_scheduler_D_A.step()
            self.lr_scheduler_D_B.step()

            if checkpoint_interval != -1 and epoch % checkpoint_interval == 0:
                # Save model checkpoints
                torch.save(self.G_AB.state_dict(), "saved_models/%s/G_AB_%d.pth" % (dataset_name, epoch))
                torch.save(self.G_BA.state_dict(), "saved_models/%s/G_BA_%d.pth" % (dataset_name, epoch))
                torch.save(self.D_A.state_dict(), "saved_models/%s/D_A_%d.pth" % (dataset_name, epoch))
                torch.save(self.D_B.state_dict(), "saved_models/%s/D_B_%d.pth" % (dataset_name, epoch))


'''

# Image transformations
transforms_ = [
    transforms.Resize(int(opt.img_height * 1.12), Image.BICUBIC),
    transforms.RandomCrop((opt.img_height, opt.img_width)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

# Training data loader
dataloader = DataLoader(
    ImageDataset("../../data/%s" % opt.dataset_name, transforms_=transforms_, unaligned=True),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)
# Test data loader
val_dataloader = DataLoader(
    ImageDataset("../../data/%s" % opt.dataset_name, transforms_=transforms_, unaligned=True, mode="test"),
    batch_size=5,
    shuffle=True,
    num_workers=1,
)'''
