# nn depends on autograd to define models and differentiate them
# an nn.Module contains layers, and a methos forward(input) that returns the output

# typical procedure for neural network {
# 1. define the neural network that has some learnable parameters (or weights)
# 2. iterate over  a dataset of inputs
# 3. process input through the network
# 4. compute the loss (how far is the output from being correct)
# 5. propagate gradients back nto the network's parameters
# 6. update the weights of the network, typically using a simple update rule: weight = weight - learning_rate * gradient

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net,
              self).__init__()  # if my knowledge of classes is right we don't need the |Net, self| for this to work

        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)

        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))

        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s

        return num_features


net = Net()
print(net)

# learnable parameters are returned by net.parameters()
params = list(net.parameters())
print(len(params))
print(params[0].size())

# let's try random 32x32 input
input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)
