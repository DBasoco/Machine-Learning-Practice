# Training deep learning networks

import torch
import numpy as np
import torchvision
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import DataLoader

from logistic_regression import evaluate

dataset = MNIST(root='data/',
                download=True,
                transform=ToTensor())


def split_indices(n, val_pct):
    # determine size of sampling
    n_val = int(val_pct * n)
    # create random permutaton of 0 to n-1
    idxs = p.random.permutation(n)
    # pick first n_val  indices for validation set
    return idxs[n_val:], idxs[:n_val]


train_indices, val_indices = split_indices(len(dataset), val_pct=0.2)
# so now everything has been separated into validation and training sets to reduce GPU usage

batch_size = 100

# training sampler and data loader
train_sampler = SubsetRandomSampler(train_indices)
train_dl = DataLoader(dataset,
                      batch_size,
                      sampler=train_sampler)

# validation sampler and dara loader
valid_sampler = SubsetRandomSampler(val_indices)
valid_dl = DataLoader(dataset,
                      batch_size,
                      sampler=valid_sampler)

# switch over to kaggle to run on a GPU(any GPU will work, or CPU, but GPU is faster)

# MODEL
# so here we will use two nn.Linear layers to transform the tensor

# first layer will transform the input matrix shape batch_size x 784 into an intermediate output
# of shape batch_size x hidden_size x
# where hidden_size is a preconfigured parameter

# the intermediate output are passed into a non-linear activation function
# this operates on individual elements of the output matrix

# the result of the activation function, size batch_size x hidden_size, is passed
# into the second layer, which transforms it into a matrix of size batch_size x 10
# identical to the logistic regression model

# the hidden layer is the second here, this allows the model to learn more complex relationships

# the activation function we will use is called a Rectified Linear Unit or ReLU

import torch.nn.functional as F
import torch.nn as nn


class MnistModel(nn.Module):

    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        # hidden layer
        self.linear1 = nn.Linear(in_size, hidden_size)
        # output layer
        self.linear2 = nn.Linear(hidden_size, out_size)

    def forward(self, xb):
        xb = xb.view(xb.size(0), -1)
        out = self.linear1(xb)
        out = F.relu(out)
        out = self.linear2(out)
        return out

# you can add more layers by having a hidden_size to hidden_size2 linear2 then
# linear3 will be hidden_size3 to out_size
# now lets make a model with 32 activations


input_size = 784
num_classes = 10

model = MnistModel(input_size, hidden_size=32, out_size=num_classes)

# lets look at the models parameters
for t in model.parameters():
    print(t.shape)

for images, labels in train_dl:
    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)
    print('Loss:', loss.item())
    break


# we need to define some helper functions that will allow us to send our data
# to the GPU then get it back


def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


device = get_default_device()


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


for images, labels in train_dl:
    print(images.shape)
    images = to_device(images, device)
    print(images.device)
    break


class DeviceDataLoader():
    # move data to the device
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    # yield a batch of data after moving it to device
    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)

    # number of batches
    def __len__(self):
        return len(self.dl)


train_dl = DeviceDataLoader(train_dl, device)
valid_dl = DeviceDataLoader(valid_dl, device)


for xb, yb in valid_dl:
    print('xb.device:', xb.device)
    print('yb:', yb)
    break


def loss_batch(model, loss_func, xb, yb, opt=None, metric=None):
    # generate predictions
    preds = model(xb)
    # calculate loss
    loss = loss_func(preds, yb)

    if opt is not None:
        # compute gradients
        loss.backwards()
        # update parameters
        opt.zero_grad()

    metric_result = None
    if metric is not None:
        # compute the metric
        metric_result = metric(preds, yb)


def fit(epochs, lr, model, loss_fn, train_dl, valid_dl, metric=None, opt_fn=None):
    losses, metrics = [], []

    # instantiate the optimizer
    if opt_fn is None: opt_fn = torch.optim.SGD
    opt = opt_fn(model.parameters(), lr=lr)

    for epoch in range(epochs):
        # training
        for xb, yb in train_dl:
            loss_batch(model, loss_fn, xb, yb, opt)

        # evaluation
        result = evaluate(model, loss_fn, valid_dl, metric)
        val_loss, total, val_metric = result

        # record the loss & metric
        losses.append(val_loss)
        metrics.append(val_metric)

        # print progress
        if metric is None:
            print('Epoch [{}/{}], Loss: {:.4f}'
                  .format(epoch+1, epochs, val_loss))
        else:
            print('Epoch [{}/{}], Loss: {:.4f}, {}: {:.4f}'
                  .format(epoch+1, epochs, val_loss,
                          metric.__name__, val_metric))

    return losses, metrics


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.sum(preds == labels).item() / len(preds)


model = MnistModel(input_size, hidden_size=32, out_size=num_classes)
to_device(model, device)


val_loss, total, val_acc = evaluate(model, F.cross_entropy,
                                    valid_dl, metric=accuracy)
print('Loss: {:.4f}, Accuracy: {:.4f}'.format(val_loss, val_acc))


# first run of this model
losses1, metrics1 = fit(5, 0.5, model, F.cross_entropy,
                        train_dl, valid_dl, accuracy)


# second run of this model
losses2, metrics2 = fit(5, 0.1, model, F.cross_entropy,
                        train_dl, valid_dl, accuracy)


# so now we plot the accuracy to see how the model improved over time
import matplotlib.pyplot as plt

accuracies = [val_acc] + metrics1 + metrics2
plt.plot(accuracies, '-x')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Accuracy vs. No. of epochs')


# this has max accuracy of around 96% to get a better result we need a stronger algorithm

