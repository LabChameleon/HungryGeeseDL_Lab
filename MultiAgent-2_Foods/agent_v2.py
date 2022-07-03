import random
from collections import namedtuple, deque

import torch.nn as nn
import torch.nn.functional as F

import utils


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN_more_conv(nn.Module):
    def __init__(self, outputs):
        super(DQN_more_conv, self).__init__()

        self.conv1 = nn.Conv2d(17, 128, kernel_size=3)   # 7x11
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3) # 7x11
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=(3,5))
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 128, kernel_size=3)
        self.bn5 = nn.BatchNorm2d(128)
        self.flatten = nn.Flatten(start_dim=1)

        self.lin1 = nn.Linear(384, 128)
        self.lin2 = nn.Linear(128, 32)
        self.head = nn.Linear(32, outputs)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(utils.map_padding(x, 1))))
        x = F.relu(self.bn2(self.conv2(utils.map_padding(x, 1))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.flatten(x)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        return self.head(x).view(x.size(0), -1)


class DQN_only_batch(nn.Module):
    def __init__(self, outputs):
        super(DQN_only_batch, self).__init__()

        self.conv1 = nn.Conv2d(17, 128, kernel_size=(3,5))
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 128, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(128)

        self.flatten = nn.Flatten(start_dim=1)

        self.lin1 = nn.Linear(384, 128)
        self.lin2 = nn.Linear(128, 32)
        self.head = nn.Linear(32, outputs)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.flatten(x)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        return self.head(x).view(x.size(0), -1)


class DQN_ext_conv(nn.Module):
    def __init__(self):
        super(DQN_ext_conv, self).__init__()

        self.conv1 = nn.Conv2d(17, 128, kernel_size=3)                  # 7x11
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3)                 # 7x11
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3)                 # 7x11
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=2)       # 4x6
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, stride=2)       # 2x3
        self.conv6 = nn.Conv2d(128, 4, kernel_size=(2,3), stride=2)     # 4

    def forward(self, x):
        x = F.relu(self.conv1(utils.map_padding(x, 1)))
        x = F.relu(self.conv2(utils.map_padding(x, 1)))
        x = F.relu(self.conv3(utils.map_padding(x, 1)))
        x = F.relu(self.conv4(utils.map_padding(x, 1)))
        x = F.relu(self.conv5(utils.map_padding(x, 1)))
        return self.conv6(x).view(x.size(0), -1)


class DQN_ext_stride(nn.Module):
    """see https://mkhoshpa.github.io/RLSnake/"""
    def __init__(self, h=7, w=11, outputs=4):
        super(DQN_ext_stride, self).__init__()

        self.conv1 = nn.Conv2d(6, 128, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(256)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w+2)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h+2)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(utils.map_padding(x, 1))))
        x = F.relu(self.bn2(self.conv2(utils.map_padding(x, 1))))
        x = F.relu(self.bn3(self.conv3(utils.map_padding(x, 1))))
        return self.head(x.view(x.size(0), -1))
