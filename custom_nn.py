# linear regression neural network

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

in_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype='float32')
# we upload the data

tar_array = np.array([[3], [9], [27]], dtype='float32')
# we upload target data

train_input_set = torch.from_numpy(
    in_array
)
train_target_set = torch.from_numpy(
    tar_array
)
# we convert numpy array into torch tensor

train_ds = TensorDataset(train_input_set, train_target_set)
# makes an easy to access tensor that can be made into a batch of tuples so it isn't corrupted

train_input_loader = DataLoader(train_ds, batch_size=4, shuffle=False)
# we create data loader that breaks data down into predefined sizes
# since we called tenserdataset first the correct rows are paired with each other over all batches
# if shuffle true, then random samplings will be taken

model = nn.Linear(3, 1)
# inputs are based on num of inputs then num of outputs
# this function will automatically create weights and bias for us, with requires_grad=True

# so technically from here we can pass some train_input_set(inputs) into model() to get predictions
# of course this will be very inaccurate, because the model isn't trained yet


# first we need to define a loss function
import torch.nn.functional as F

loss_fn = F.mse_loss()
# loss = loss_fn(model(train_input_set), train_target_set)

# this is a built in loss function, element-wise mean squared error(mse)
# now we need to optimize the weights to reduce the loss function

import torch.optim as optim

optimizer = optim.SGD(model.parameters(), lr=1e-5)  # weights/bias, then learning rate is 1e-5


# instead of doing this manually we use this built in function
# this stochastic gradient descent(SGD) uses batches now a whole sample size


# now we follow the basic steps
# 1. generate predictions
# 2. calculate loss
# 3. compute gradients w.r.t. the weights and biases
# 4. adjust weights by subtracting a small quantity proportional to the gradient
# 5. reset the gradients to zero


def fit(num_epochs, model, loss_fn, optimizer):
    # over set epochs
    for epoch in range(num_epochs):
        # over each given batch
        for xb, yb in train_input_loader:
            # make predictions
            pred = model(xb)

            # calculate loss
            loss = loss_fn(pred, yb)

            # compute grad
            loss.backward

            # adjust weights
            optimizer.step()

            # zero grad
            optimizer.zero_grad()

        # print progress of model fit
        if (epoch + 1) % 10 == 0: # so for every tenth epoch
            print('Epoch[{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))

# run fit over several epochs and the model will be well defined with very little loss
# of course this is for small data sets and batches
# to upscale it to large sets requires better models
