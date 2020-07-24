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

# to convert rows into probabilities we need to use the 'softmax funtion'
import torch.nn.functional as F

# apply softmax to each row
probs = F.softmax(outputs, dim=1) # dimension reduction

# this returns the highest probability for each element, and the index of that probablity
max_probs, preds = torch.max(probs, dim=1)

# we need to include the loss function as this current model wont be able to
# predict accurately

# this will tell us how accurate the model is
def accuracy(11, 12):
    return torch.sum(11 == 12).item() / len(11)

accuracy(preds, labels)

# accuracy is good for humans but not for this case
# reason 1 being that the used functions are not differentiable
# reason 2 being that it doesnt take into account the actual probabilities

# this is why we use it as a metric for evaluation and use cross entropy for classification problems

# torch provides a function for loss cross entropy
loss_fn = F.cross_entropy
loss = loss_fn(outputs, labels)

# optimizer
learning_rate = 0.001
optimizer = torch.optim.SGD(model.parameter(), lr=learning_rate)

# now to training the model
def loss_batch(model, loss_func, xb, yb, opt=None, metric=None):
    # calculate loss
    preds = model(xb)
    loss = loss_func(preds, yb)

    if opt is not None:
        # compute gradients
        loss.backwards()
        # update parameters
        opt.step()
        # reset gradients
        opt.zero_grad()

    metric_result = None
    if metric id not None:
    # compute the metric
    metric_result = metric(preds, yb)

    return loss.item(), len(xb), metric_result


def evaluate(model, loss_fn, valid_dl, metric=None):
    with torch.no_grad():
        # pass each nach through the model
        results = [loss_batch(model, loss_fn, xb, yb, metric=metric)
                   for xb, yb in valid_dl]
        # separate losses, counts and metrics
        losses, nums, metrics = zip(*results)
        # total size of the dataset
        total = np.sum(nums)
        # avg loss across batches
        avg_loss = np.sum(np.multiply(losses, nums)) / total
        avg_metric = None
        if metric is not None:
            # avg of metric across batches
            avg_metric = np.sum(np.multiply(metrics, nums)) / total
    return avg_loss, total, avg_metric


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.sum(preds == labels).item() / len(preds)

val_lose, total , val_acc = evaluate(model, loss_fn, val_loader, metric=accuracy)


def fit(epochs, model, loss_fn, opt, train_dl, valid_dl, metric=None):
    for epoch in range(epochs):
        # training
        for xb, yb in train_dl:
            loss,_,_ = loss_batch(model, loss_fn, xb, yb, opt)
        # evaluation
        result = evaluate(model, loss_fn, valid_dl, metric)
        val_loss, total , val_metric = result

        # print progress
        if metric is None:
            print('Epoch [{}/{}], Loss: {:.4f}'
                  .format(epoch+1, epochs, val_loss))
        else:
            print('Epoch [{}/{}], Loss: {:.4f}, {}: {:.4f}'
                  .format(epoch+1, epochs, val_loss, metric.__name__, val_metric))


# redefine the model and optimizer
model = MnistModel()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

fit(5, model, F.cross_entropy, optimizer, train_loader, val_loader, accuracy)


# so this model will logistically grow, at first being very fast but then reach a maximum
# to fix this issue we can adjust the learning rate
# more likely is that this model isn't powerful enough
# so we need more complex model that have non-linear relationships

