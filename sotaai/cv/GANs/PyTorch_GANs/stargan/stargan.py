"""
StarGAN (CelebA)
The dataset can be downloaded from: https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=0
And the annotations: https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AAA8YmAHNNU6BEfWMPMfM6r9a/Anno?dl=0&preview=list_attr_celeba.txt
Instructions on running the script:
1. Download the dataset and annotations from the provided link
2. Copy 'list_attr_celeba.txt' to folder 'img_align_celeba'
2. Save the folder 'img_align_celeba' to '../../data/'
4. Run the script by 'python3 stargan.py'
"""


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
import torch.autograd as autograd

from cv.GANs.PyTorch_GANs.stargan.models import *
from cv.GANs.PyTorch_GANs.stargan.datasets import *

import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs("images", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="img_align_celeba", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=128, help="size of image height")
parser.add_argument("--img_width", type=int, default=128, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between saving generator samples")
parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between model checkpoints")
parser.add_argument("--residual_blocks", type=int, default=6, help="number of residual blocks in generator")
parser.add_argument(
    "--selected_attrs",
    "--list",
    nargs="+",
    help="selected attributes for the CelebA dataset",
    default=["Black_Hair", "Blond_Hair", "Brown_Hair", "Male", "Young"],
)
parser.add_argument("--n_critic", type=int, default=5, help="number of training iterations for WGAN discriminator")
opt = parser.parse_args()
print(opt)


def criterion_cls(logit, target):
    return F.binary_cross_entropy_with_logits(logit, target, size_average=False) / logit.size(0)



def compute_gradient_penalty(D, real_samples, fake_samples):
    Tensor = torch.FloatTensor  #torch.cuda.FloatTensor if cuda else torch.FloatTensor

    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates, _ = D(interpolates)
    fake = Variable(Tensor(np.ones(d_interpolates.shape)), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty




class STARGAN():
    def __init__(self, lr = 0.0002,b1 = 0.5, b2 = 0.999, latent_dim = 100,img_height = 256, img_width = 256, channels = 1,residual_blocks = 6,
                    selected_attrs = ["Black_Hair", "Blond_Hair", "Brown_Hair", "Male", "Young"]):
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.latent_dim = latent_dim
        self.img_shape = (channels, img_height, img_width)
        self.residual_blocks = residual_blocks

        self.c_dim = len(selected_attrs)


        # Loss functions
        self.criterion_cycle = torch.nn.L1Loss()


        # Initialize generator and discriminator
        self.generator = GeneratorResNet(img_shape=self.img_shape, res_blocks=self.residual_blocks, c_dim=self.c_dim)
        self.discriminator = Discriminator(img_shape=self.img_shape, c_dim=self.c_dim)

        cuda = False #torch.cuda.is_available()
        if cuda:
            self.generator = self.generator.cuda()
            self.discriminator = self.discriminator.cuda()
            self.criterion_cycle.cuda()
        self.Tensor = torch.FloatTensor  #torch.cuda.FloatTensor if cuda else torch.FloatTensor


        self.generator.apply(weights_init_normal)
        self.discriminator.apply(weights_init_normal)
        # Optimizers
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.b1, self.b2))


    def sample_images(self,val_dataloader, batches_done):

        label_changes = [
            ((0, 1), (1, 0), (2, 0)),  # Set to black hair
            ((0, 0), (1, 1), (2, 0)),  # Set to blonde hair
            ((0, 0), (1, 0), (2, 1)),  # Set to brown hair
            ((3, -1),),  # Flip gender
            ((4, -1),),  # Age flip
        ]
        """Saves a generated sample of domain translations"""
        val_imgs, val_labels = next(iter(val_dataloader))
        val_imgs = Variable(val_imgs.type(self.Tensor))
        val_labels = Variable(val_labels.type(self.Tensor))
        img_samples = None
        for i in range(10):
            img, label = val_imgs[i], val_labels[i]
            # Repeat for number of label changes
            imgs = img.repeat(self.c_dim, 1, 1, 1)
            labels = label.repeat(self.c_dim, 1)
            # Make changes to labels
            for sample_i, changes in enumerate(label_changes):
                for col, val in changes:
                    labels[sample_i, col] = 1 - labels[sample_i, col] if val == -1 else val

            # Generate translations
            gen_imgs = self.generator(imgs, labels)
            # Concatenate images by width
            gen_imgs = torch.cat([x for x in gen_imgs.data], -1)
            img_sample = torch.cat((img.data, gen_imgs), -1)
            # Add as row to generated samples
            img_samples = img_sample if img_samples is None else torch.cat((img_samples, img_sample), -2)

        save_image(img_samples.view(1, *img_samples.shape), "images/%s.png" % batches_done, normalize=True)


# ----------
#  Training
# ----------
    def train(self, epoch = 0, n_epochs = 200, sample_interval = 100,checkpoint_interval = -1, dataset_name = 'img_align_celeba'):
        
        if epoch != 0:
            # Load pretrained models
            self.generator.load_state_dict(torch.load("saved_models/generator_%d.pth" % epoch))
            self.discriminator.load_state_dict(torch.load("saved_models/discriminator_%d.pth" % epoch))
        
        # Loss weights
        lambda_cls = 1
        lambda_rec = 10
        lambda_gp = 10

        saved_samples = []
        start_time = time.time()
        for epoch in range(epoch, n_epochs):
            for i, (imgs, labels) in enumerate(dataloader):

                # Model inputs
                imgs = Variable(imgs.type(self.Tensor))
                labels = Variable(labels.type(self.Tensor))

                # Sample labels as generator inputs
                sampled_c = Variable(self.Tensor(np.random.randint(0, 2, (imgs.size(0), self.c_dim))))
                # Generate fake batch of images
                fake_imgs = self.generator(imgs, sampled_c)

                # ---------------------
                #  Train Discriminator
                # ---------------------

                self.optimizer_D.zero_grad()

                # Real images
                real_validity, pred_cls = self.discriminator(imgs)
                # Fake images
                fake_validity, _ = self.discriminator(fake_imgs.detach())
                # Gradient penalty
                gradient_penalty = compute_gradient_penalty(self.discriminator, imgs.data, fake_imgs.data)
                # Adversarial loss
                loss_D_adv = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
                # Classification loss
                loss_D_cls = criterion_cls(pred_cls, labels)
                # Total loss
                loss_D = loss_D_adv + lambda_cls * loss_D_cls

                loss_D.backward()
                self.optimizer_D.step()

                self.optimizer_G.zero_grad()

                # Every n_critic times update generator
                if i % n_critic == 0:

                    # -----------------
                    #  Train Generator
                    # -----------------

                    # Translate and reconstruct image
                    gen_imgs = self.generator(imgs, sampled_c)
                    recov_imgs = self.generator(gen_imgs, labels)
                    # Discriminator evaluates translated image
                    fake_validity, pred_cls = self.discriminator(gen_imgs)
                    # Adversarial loss
                    loss_G_adv = -torch.mean(fake_validity)
                    # Classification loss
                    loss_G_cls = criterion_cls(pred_cls, sampled_c)
                    # Reconstruction loss
                    loss_G_rec = self.criterion_cycle(recov_imgs, imgs)
                    # Total loss
                    loss_G = loss_G_adv + lambda_cls * loss_G_cls + lambda_rec * loss_G_rec

                    loss_G.backward()
                    self.optimizer_G.step()

                    # --------------
                    #  Log Progress
                    # --------------

                    # Determine approximate time left
                    batches_done = epoch * len(dataloader) + i
                    batches_left = n_epochs * len(dataloader) - batches_done
                    time_left = datetime.timedelta(seconds=batches_left * (time.time() - start_time) / (batches_done + 1))

                    # Print log
                    sys.stdout.write(
                        "\r[Epoch %d/%d] [Batch %d/%d] [D adv: %f, aux: %f] [G loss: %f, adv: %f, aux: %f, cycle: %f] ETA: %s"
                        % (
                            epoch,
                            n_epochs,
                            i,
                            len(dataloader),
                            loss_D_adv.item(),
                            loss_D_cls.item(),
                            loss_G.item(),
                            loss_G_adv.item(),
                            loss_G_cls.item(),
                            loss_G_rec.item(),
                            time_left,
                        )
                    )

                    # If at sample interval sample and save image
                    if batches_done % sample_interval == 0:
                        self.sample_images(val_dataloader, batches_done)

            if checkpoint_interval != -1 and epoch % checkpoint_interval == 0:
                # Save model checkpoints
                torch.save(self.generator.state_dict(), "saved_models/generator_%d.pth" % epoch)
                torch.save(self.discriminator.state_dict(), "saved_models/discriminator_%d.pth" % epoch)




'''

# Configure dataloaders
train_transforms = [
    transforms.Resize(int(1.12 * opt.img_height), Image.BICUBIC),
    transforms.RandomCrop(opt.img_height),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

dataloader = DataLoader(
    CelebADataset(
        "../../data/%s" % opt.dataset_name, transforms_=train_transforms, mode="train", attributes=opt.selected_attrs
    ),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)

val_transforms = [
    transforms.Resize((opt.img_height, opt.img_width), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]

val_dataloader = DataLoader(
    CelebADataset(
        "../../data/%s" % opt.dataset_name, transforms_=val_transforms, mode="val", attributes=opt.selected_attrs
    ),
    batch_size=10,
    shuffle=True,
    num_workers=1,
)'''
