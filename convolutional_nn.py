# understanding convolutional neural networks (at last lol)


import os
import torch
import torchvision
import tarfile
from torchvision.datasets.utils import download_url


# downloads the data
dataset_url = "http://files.fast.ai/data/cifar10.tgz"
download_url(dataset_url, '.')


# extract from archive
with tarfile.open('./cifar10.tgz', 'r:gz') as tar:
    tar.extractall(path='./data')


data_dir = './data/cifar10'


print(os.listdir(data_dir))
classes = os.listdir(data_dir + "/train")
print(classes)


# here we make one for each data class we have
airplane_files = os.listdir(data_dir + '/train/airplane')
print('No. of training examples for airplanes:', len(airplane_files))
print(airplane_files[:5])


ship_test_files = os.listdir(data_dir + '/test/ship')
print('No. of test examples for ship:', len(ship_test_files))
print(ship_test_files[:5])


from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor

dataset = ImageFolder(data_dir+'/train', transform=ToTensor())


img, label = dataset[0]
print(img.shape, label)
img


print(dataset.classes)


import matplotlib.pyplot as plt


# this allows us to move around the tensor structure so it is easier to read
def show_example(img, label):
    print('Label: ', dataset.classes[label], '('+str(label)+')')
    plt.imshow(img.permute(1, 2, 0))


show_example(*dataset[0])


import numpy as np


# here we make a validation set
def split_indices(n, val_pct=0.1, seed=99):
    # determine size of validation set
    n_val = int(val_pct*n)
    # set the random seed (for reproducibility)
    np.random.seed(seed)
    # create random permutation of 0 to n-1
    idxs = np.random.permutation(n)
    # pick first n_val indices for validation set
    return idxs[n_val:], idxs[:n_val]


val_pct = 0.2
rand_seed = 42


train_indices, val_indices = split_indices(len(dataset), val_pct, rand_seed)
print(len(train_indices), len(val_indices))
print('Sample validation indices: ', val_indices[:10])


from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import DataLoader


batch_size = 100


train_sampler = SubsetRandomSampler(train_indices)
train_dl = DataLoader(dataset,
                      batch_size,
                      sampler=train_sampler)


val_sampler = SubsetRandomSampler(train_indices)
val_dl = DataLoader(dataset,
                    batch_size,
                    sampler=val_sampler)


from torchvision.utils import make_grid


def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(images, 10).permute(1, 2, 0))
        break


show_batch(train_dl)


# in previous tutorials we used nn.Linear to define neural networks, but now we will use nn.Conv2d instead
# advantages
# 1. fewer layers
# 2. sparsity if connections
# 3. parameter sharing and spatial invariance


import torch.nn as nn
import torch.nn.functional as F


simple_model = nn.Sequential(
    nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
    nn.MaxPool2d(2, 2)
)


for images, labels in train_dl:
    print('images.shape:', images.shape)
    out = simple_model(images)
    print('out.shape:', out.shape)
    break


model = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    # bs x 16 x 16 x 16

    nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    # bs x 16 x 8 x 8

    nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    # bs x 16 x 4 x 4

    nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    # bs x 16 x 2 x 2

    nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    # bs x 16 x 1 x 1

    nn.Flatten(),
    # bs x 16
    nn.Linear(16, 10)
    # bs x 10
)


for images, labels in train_dl:
    print('images.shape:', images.shape)
    out = model(images)
    print('out.shape:', out.shape)
    print('out[0]:', out[0])
    break


# to interpret as probabilities
probs = F.softmax(out[0], dim=1)
torch.sum(probs), probs
torch.max(probs)


def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


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


device = get_default_device()
device


train_dl = DeviceDataLoader(train_dl, device)
valid_dl = DeviceDataLoader(train_dl, device)
to_device(model, device)


def loss_batch(model, loss_func, xb, yb, opt=None, metric=None):
    # generate predictions
    preds = model(xb)
    # calculate loss
    loss = loss_func(preds, yb)

    if opt is not None:
        # compute gradients
        loss.backwards()
        # update parameters
        opt.step()
        # reset gradients
        opt.zero_grad()

    metric_result = None
    if metric is not None:
        # compute the metric
        metric_result = metric(preds, yb)

    return loss.item(), len(xb), metric_result


def evaluate(model, loss_fn, valid_dl, metric=None):
    with torch.no_grad():
        # pass each batch through the model
        results = [loss_batch(model, loss_fn, xb, yb, metric=metric)
                   for xb, yb, in valid_dl]
        # separate losses, counts and metrics
        losses, nums, metrics = zip(*results)
        # total size of the dataset
        total = np.sum(nums)
        # avg. loss across batches
        avg_loss = np.sum(np.multiply(losses, nums)) / total
        avg_metric = None
        if metric is not None:
            # avg of metric across batches
            avg_metric = np.sum(np.multiply(metric, nums)) / total

    return avg_loss, total, avg_metric


def fit(epochs, model, loss_fn, train_dl, valid_dl,
        opt_fn=None, lr=None, metric=None):
    train_losses, val_losses, val_metrics = [], [], []

    # instantiate the optimizer
    if opt_fn is None: opt_fn = torch.optim.SGD
    opt = opt_fn(model.parameters(), lr=lr)

    for epoch in range(epochs):
        # training
        model.train()
        for xb, yb in train_dl:
            train_loss,_,_ = loss_batch(model, loss_fn, xb, yb, opt)

        # evaluation
        model.eval()
        result = evaluate(model, loss_fn, valid_dl, metric)
        val_loss, total, val_metric = result

        # record the loss & metric
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_metrics.append(val_metric)

        # print progress
        if metric is None:
            print('Epoch [{}/{}], Loss: {:.4f}'
                  .format(epoch+1, epochs, val_loss))
        else:
            print('Epoch [{}/{}], Loss: {:.4f}, {}: {:.4f}'
                  .format(epoch+1, epochs, val_loss,
                          metric.__name__, val_metric))

    return train_losses, val_losses, val_metrics


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.sum(preds == labels).item() / len(preds)


val_loss, _, val_acc = evaluate(model, F.cross_entropy,
                                valid_dl, metric=accuracy)
print('Loss: {:.4f}'.format(val_loss, val_acc))


num_epochs = 10
opt_fn = torch.optim.Adam
lr = 0.005


history = fit(num_epochs, model, F.cross_entropy,
              train_dl, valid_dl, opt_fn, lr, accuracy)
train_losses, val_losses, val_metrics = history


def plot_losses(train_losses, val_losses):
    plt.plot(train_losses, '-x')
    plt.plot(val_losses, '-o')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs')


test_dataset = ImageFolder(data_dir+'/test', transform=ToTensor())


def predict_image(img, model):
    # convert to a batch of 1
    xb = img.unsqueeze(0)
    # get predictions from model
    yb = model(xb.to(device))
    # pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    # retrieve the class label
    return dataset.classes[preds[0].item()]


# run this for different images in the test_dataset
i = 20
img, label = test_dataset[i]
plt.imshow(img.permute(1, 2, 0))
print('Label:', dataset.classes[label], ', Predicted:', predict_image(img, model))








