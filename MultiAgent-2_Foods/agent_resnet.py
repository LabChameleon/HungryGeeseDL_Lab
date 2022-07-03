import random
from collections import namedtuple, deque

from utils import map_padding

import torch.nn as nn
import torch.nn.functional as F


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class ResBlock(nn.Module):
    def __init__(self, channels_in, channels_out, stride=1):
        super().__init__()

        # The standard conv 'bn relu conv bn' block
        self.conv1 = nn.Conv2d(
            channels_in, channels_out, kernel_size=3, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(channels_out)
        self.conv2 = nn.Conv2d(
            channels_out, channels_out, kernel_size=3, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels_out)

        # The skip connection across the convs.
        # If the stride of this block is > 1, or the in and out channel
        # counts don't match, we need an additional 1x1 conv on the
        # skip connection.
        if stride > 1 or channels_in != channels_out:
            self.skip = nn.Sequential(
                nn.Conv2d(
                    channels_in, channels_out, kernel_size=1, stride=stride,
                    bias=False),
                nn.BatchNorm2d(channels_out)
            )
        else:
            self.skip = nn.Sequential()

    def forward(self, x):
        residual = F.relu(self.bn1(self.conv1(map_padding(x,1))))
        residual = self.bn2(self.conv2(map_padding(residual,1)))
        output = residual + self.skip(x)
        output = F.relu(output)
        return output


class DQN(nn.Module):

    def __init__(self):
        super().__init__()
        self.channels_in = 32

        # The first input convolution, going from 3 channels to 64.
        self.conv1 = nn.Conv2d(
            6, self.channels_in, kernel_size=(3,3), stride=1, bias=False) # 7x11x32
        self.bn1 = nn.BatchNorm2d(self.channels_in)

        # Four sections, each with 2 resblocks and increasing channel counts.
        self.section1 = self._add_section(self.channels_in*2, 2, stride=1)  # 7x11x64
        self.section2 = self._add_section(self.channels_in*4, 2, stride=1)  # 7x11x128
        self.section3 = self._add_section(self.channels_in*2, 2, stride=2)  # 4x6x64
        self.section4 = self._add_section(self.channels_in, 2, stride=2)    # 2x3x32

        # The final linear layer to get the logits. This could also
        # be realized with a 1x1 conv.
        self.output = nn.Conv2d(self.channels_in, 4, kernel_size=(2,3))

    # Utility function to add a section with a fixed amount of res blocks
    def _add_section(self, channels_out, resblock_count, stride):
        resblocks = []
        for b in range(resblock_count):
            stride_block = stride if b == 0 else 1
            resblocks.append(
                ResBlock(self.channels_in, channels_out, stride=stride_block))
            self.channels_in = channels_out
        return nn.Sequential(*resblocks)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(map_padding(x, 1))))
        x = self.section1(x)
        x = self.section2(x)
        x = self.section3(x)
        x = self.section4(x)
        x = self.output(x)
        return x.view(x.size(0), -1)