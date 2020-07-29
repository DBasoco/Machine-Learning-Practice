# generative adverserial networks in pytorch (GANs)

# generative models are used to automatically discover and learn patterns in input data in such a way that
# the model can be used to generate or output new examples that plausibly could have been from the original
# data source
# so basically this is the really cool stuff now yay

# two neural networks, generative and discriminator
# the discriminator tries to see if an image is fake or not


import os
import torch
import torchvision
import numpy as np
import tarfile
from torchvision.datasets.utils import download_url
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.utils.data import SubsetRandomSampler
from torchvision.datasets import ImageFolder
import torchvision.transforms as tt
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Normalize, Compose
from IPython.display import Image


# these networks are notorious to train because the hyper-parameters and two networks need to be carefully maintained
mnist = MNIST(root='data/',
              train=True,
              download=True,
              transform=Compose([ToTensor(), Normalize(mean=(0.5,), std=(0.5,))]))


img, label = mnist[0]


def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


img_norm = denorm(img)
plt.imshow(img_norm[0], cmap='gray')
print('Label:', label)


batch_size = 100
data_loader = DataLoader(mnist, batch_size, shuffle=True)


# build for the discriminator model
image_size = 784
hidden_size = 356

# this doesn't need to be super complicated, so use a simple linear network
D = nn.Sequential(
    nn.Linear(image_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, 1),
    nn.Sigmoid()
)


# generator network
latent_size = 64
G = nn.Sequential(
    nn.Linear(latent_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, image_size),
    nn.Tanh()
)


# both of these networks are feed forward
# at this point the generator will only make random noise patterns instead of an actual image
criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002)


def reset_grad():
    d_optimizer.zero_grad()
    g_optimizer.zero_grad()


def train_discriminator(images):
    # create labels
    real_labels = torch.ones(batch_size, 1)
    fake_labels = torch.zeros(batch_size, 1)

    # loss for real images
    outputs = D(images)
    d_loss_real = criterion(outputs, real_labels)
    real_score = outputs

    # loss for fake images
    z = torch.randn(batch_size, latent_size)
    fake_images = G(z)
    outputs = D(fake_images)
    d_loss_fake = criterion(outputs, fake_labels)
    fake_score = outputs

    # combine losses
    d_loss = d_loss_real + d_loss_fake
    # reset gradient
    reset_grad()
    # compute gradients
    d_loss.backward()
    # adjust the parameters using backprop
    d_optimizer.step()

    return d_loss, real_score, fake_score


def train_generator():
    # generate fake images and calculate loss
    z = torch.randn(batch_size, latent_size)
    fake_images = G(z)
    labels = torch.ones(batch_size, 1)
    g_loss = criterion(D(fake_images), labels)
    # want a loss as close to one as possible

    # backprop and optimize
    reset_grad()
    g_loss.backward()
    g_optimizer.step()
    return g_loss, fake_images


# lets create a directory to save the outputs of the generator as we go along to see the process
sample_dir = 'samples'
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)


# now lets save a real image for visual comparison
for images, _ in data_loader:
    images = images.reshape(images.size(0), 1, 28, 28)
    save_image(denorm(images), os.pathjoin(sample_dir, 'real_images.png'), nrow=10)
    break


Image(os.path.join(sample_dir, 'real_image.png'))


sample_vectors = torch.randn(batch_size, latent_size)


def save_fake_images(index):
    fake_images = G(sample_vectors)
    fake_images = fake_images.reshape(fake_images.size(0), 1, 28, 28)
    fake_fname = 'fake_images-{0:0=4d}.png'.format(index)
    print('Saving', fake_fname)
    save_image(denorm(fake_images), os.path.join(sample_dir, fake_fname), nrow=10)


# before training
save_fake_images(0)
Image(os.path.join(sample_dir, 'fake_images-0000.png'))


num_epochs = 50
total_step = len(data_loader)
d_losses, g_losses, real_scores, fake_scores = [], [], [], []

for epoch in range(num_epochs):
    for i, (images, _) in enumerate(data_loader):
        # load a batch & transform to vectors
        images = images.reshape(batch_size, -1)

        # train the discriminator and generator
        d_loss, real_score, fake_score = train_discriminator(images)
        g_loss, fake_images = train_generator()

        # inspect the losses
        if (i+1) % 200 == 0:
            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())
            real_scores.append(real_score.mean().item())
            fake_scores.append(fake_score.mean().item())
            print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(x)): {:.2f}'
                  .format(epoch, num_epochs, i+1, total_step, d_loss.item(), g_loss.item(),
                          real_score.mean().item(), fake_score.mean().item()))

    # sample and save images
    save_fake_images(epoch+1)


# there is a method to convert the images into a video to watch the progress over time
# will save that as an exercise for the reader
# lol that's a little physics joke for you









