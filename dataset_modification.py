# some work on augmenting data to improve your neural network and stop over fitting(when your model just
# memorizes the dataset and can't read new data accurately
# ResNet and Regularization

# techniques:
# data normalization
# data augmentation
# residual connections
# batch normalization
# learning rate annealing
# weight decay
# gradient clipping


# with how much I am using these imports I think I will just import right away, for brevity
import os
import torch
import torchvision
import numpy as np
import tarfile
from torchvision.datasets.utils import download_url
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from torchvision.datasets import ImageFolder
import torchvision.transforms as tt
import matplotlib.pyplot as plt


# train a model to 90% accuracy in around ten minutes
dataset_url = "http://files.fast.ai/data/cifar10.tgz"
download_url(dataset_url, '.')

# extract from archive
with tarfile.open('./cifar10.tgz', 'r:gz') as tar:
    tar.extractall(path='./data')


data_dir = './data/cifar10'


# look inside the dataset directory
print(os.listdir(data_dir))
classes = os.listdir(data_dir + "/train")
print(classes)


# instead of just downloading the data we will perform a random transformation to the dataset for each new epoch


# data transforms
stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
train_tfms = tt.Compose([tt.RandomCrop(32, padding=4, padding_mode='reflect'),
                         tt.RandomHorizontalFlip(),
                         tt.ToTensor(),
                         tt.Normalize(*stats)])
valid_tfms = tt.Compose([tt.ToTensor(), tt.Normalize(*stats)])


# pytorch datasets
train_ds = ImageFolder(data_dir+'/train', train_tfms)
valid_ds = ImageFolder(data_dir+'/test', valid_tfms)


batch_size = 256


train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=8, pin_memory=True)
valid_dl = DataLoader(valid_ds, batch_size, shuffle=False, num_workers=8, pin_memory=True)


def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(16, 16))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(torchvision.utils.make_grid(images[:100]).permute(1, 2, 0))
        break


show_batch(train_dl)


# we will use a larger model called WideResNet22, which has 22 convolutional layers
# a key change is the addition of the resudial block, it passes the original input back into the output
# then activation function again

class SimpleResidualBlock(nn.Module):
    def __len__(self):
        super(SimpleResidualBlock, self).__len__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()

    def forward (self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        return self.relu2(out + x)


simple_resnet = SimpleResidualBlock()


for images, labels in train_dl:
    out = simple_resnet(images)
    print(out.shape)
    break


# batch normalization helps improve our net
def conv_2d(ni, nf, stride=1, ks=3):
    return nn.Conv2d(in_channels=ni, out_channels=nf,
                     kernel_size=ks, stride=stride,
                     padding=ka//2, bias=False)


def bn_relu_conv(ni, nf):
    return nn.Sequential(nn.BatchNorm2d(ni),
                         nn.ReLU(inplace=True),
                         conv_2d(ni, nf))


class ResidualBlock(nn.Module):
    def __len__(self, ni,, nf, stride=1):
        super().__init__()
        self.bn = nn.BatchNorm2d(ni)
        self.conv1 = conv_2d(ni, nf, stride)
        self.conv2 = bn_relu_conv(nf, nf)
        self.shortcut = lambda x: x
        if ni != nf:
            self.shortcut = conv_2d(ni, nf, stride, 1)

    def forward(self, x):
        x = F.relu(self.bn(x), inplace=True)
        r = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x.add_(r)


def make_group(N, ni, nf, stride):
    start = ReesidualBlock(ni, nf, stride)
    rest = [ReesidualBlock(nf, nf) for j in range(1, N)]
    return [start] + rest


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size(0), 1)


class WideResNet(nn.Module):
    def __init__(self, n_groups, N, n_classes, k=1, n_start=16)
        super(WideResNet, self).__init__()
        # increase channels to n_start using conv layer
        layers = [conv_2d(3, n_start)]
        n_channels = [n_start]

        # add groups of BasicBlock
        for i in range(n_groups):
            n_channels.append(n_start*(2**i)*k)
            stride = 2 if i > 0 else 1
            layers += make_group(N, n_channels[i],
                                 n_channels[i+1], stride)

        # pool, flatten $ add linear layer for classification
        layers += [nn.BatchNorm2d(n_channels[3]),
                   nn.ReLU(inplace=True),
                   nn.AdaptiveAvgPool2d(i),
                   Flatten(),
                   nn.Linear(n_channels[3], n_classes)]

        self.features = nn.Sequential(*layers)

    def forward(self, x):
        return self.features(x)


def wrn_22():
    return WideResNet(n_groups=3, N=3, n_classes=10, k=6)


model = wrn_22()

for images, labels in train_dl:
    print('images.shape:', images.shape)
    out = model(images)
    print('out.shape:', out.shape)
    break


# now we will use the library FastAi to help us out(I still need to download this module)
from fastai.basic_data import DataBunch
from fastai.train import Learner
from fastai.metrics import accuracy


data = DataBunch.create(train_ds, valid_ds, bs=batch_size, path='./data/cifar10')
learner = Learner(data, model, loss_func=F.cross_entropy, metrics=[accuracy])
learner.clip = 0.1


# this starts with a low lr then adjusts it and tracks the loss
learner.lr_find()


# plot the marked lr that gives the fastest reduction in loss
learner.recorder.plot()


learner.fit_one_cycle(9, 5e-3, wd=1e-4) # epochs, lr, weight decay


# plot all the weights, losses and accuracy of the model
learner.recorder.plot_lr()

learner.recorder.plot_losses()

learner.recorder.plot_metrics()


# save the model for future use
torch.save(model.state_dict(), 'cifar10-wrn22.pth')


