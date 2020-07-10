# this is different than linear regression
# this assumed simple inputs that followed a linear graph

# here we take something like hand written text samples to teach
# the machine to recognize text
# so input goes from solid data to pixeled image
# and prediction goes from number to predicted digit
# this is a classification problem
# it still follows the same basic laws


# torch has packages designed for dealing with images
import torch
import torchvision
from torchvision.datasets import MNIST

# download training dataset
dataset = MNIST('name of file', download=True)

# to view the loaded images we need matplot
import matplotlib.pyplot as plt

image, label = dataset[0]
# this breaks up the data into the component parts
plt.imshow(image, cmap='gray')
# shows image in grey scale
print('Label: ', label)

# the issue is that pytorch only works with tensors
# luckily a torchvision as packages for this
import torchvision.transforms as transforms

# now we can re-upload the data in a way that torch can understand
dataset = MNIST(root='data/', train=True, transform=transforms.ToTensor)
# so find data, specify it as a training set, then provide transform instructions


img_tensor, label = dataset[0]
print(img_tensor.shape, label)
# separates and then returns the shape of the new tensor
# this will return |torch.Size{[a, b, c]) label|
# here 'a' stores the color channel, based on RGB, so 3 channels
# here 'b' and 'c' are height and width


# training and validation datasets
# Training set: used to train the model i.e. compute loss adjust weights
# Validation set: evaluate model while training and adjust learning rate lr
# Test set: compare different models, or approaches, and report final accuracy

# test sets are standardized to have a base to go off of
# this gives us a simple validation set to go off of
import numpy as np


def split_indices(n, val_pct):
    # determine the size of validation set
    n_val = int(val_pct * n)

    # create random permutation of 0 to n-1
    idxs = np.random.permutation(n)

    # pick first n_val indices for validation set
    return idxs[n_val:], idxs[:n_val]


# so this actually gives us the split after being called
train_indices, val_indices = split_indices(len(dataset), val_pct=0.2)


# now we create data loaders for each set

from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import DataLoader

batch_size = 100

# training
train_sampler = SubsetRandomSampler(train_indices)
train_loader = DataLoader(dataset, batch_size, sampler=train_sampler)


# validation
val_sampler = SubsetRandomSampler(val_indices)
val_loader = DataLoader(dataset, batch_size, sampler=val_sampler)


# now that we have a data loaders prepared we can create our model
import torch.nn as nn

input_size  = 28 * 28
num_classes = 10


# logistic regression model
model = nn.Linear(input_size, num_classes)

# let's look at the weights and biases
model.weight
model.bias


# let's take the first 100 images and pass them through the model
for images, labels in train_loader:
    print(labels)
    print(images.shape)
    outputs = model(images)
    break


# this will return an error due to mismatch of sizes because linear of the image tensor
# we need to reshape the 1x28x28 tensor to a single vector of size 784

class MnistModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, xb):
        xb = xb.reshape(-1, 784) # two dimensional first is -1 second is 784
        out = self.linear(xb)
        return out


model = MnistModel()


# so now we've redefined the model as a class that can convert an image tensor into a vector
model.parameters()
# this will return weights and biases no matter how the model is classed


# let's see if it worked
for images, labels in train_loader:
    outputs = model(images)
    break
