import sys, os
sys.path.insert(0, "../functions/")
import numpy as np

from kaggle_environments.envs.hungry_geese.hungry_geese import Action, row_col

import torch
import torch.nn as nn
import torch.nn.functional as F
from create_map import create_map_from_obs_mulit_agent_v2


class DQN(nn.Module):
    def __init__(self, outputs):
        super(DQN, self).__init__()

        self.conv1 = nn.Conv2d(7, 128, kernel_size=(3, 5), )
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3)
        self.conv3 = nn.Conv2d(256, 128, kernel_size=3)

        self.flatten = nn.Flatten(start_dim=1)

        self.lin1 = nn.Linear(384, 128)
        self.lin2 = nn.Linear(128, 32)
        self.head = nn.Linear(32, outputs)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        return self.head(x).view(x.size(0), -1)

def center_map(map, c_x, c_y):
    _, s_x, s_y = map.shape
    x_ind = (np.arange(s_x) + (c_x - 3)) % 7
    y_ind = (np.arange(s_y) + (c_y - 5)) % 11
    map = map[:, x_ind, :]
    map = map[:, :, y_ind]
    return map

def num_to_action(action):
    if action == 0:
        return Action.NORTH.name
    elif action == 1:
        return Action.EAST.name
    elif action == 2:
        return Action.SOUTH.name
    return Action.WEST.name

def select_action_from_net(state, policy_net):
    return policy_net(state).max(1)[1].view(1, 1)

def select_action_from_net_non_opposite(state, policy_net, last_action):
    top2 = policy_net(state).topk(2)[1][0]
    # top action is opposite of last action
    if (top2[0] + 2) % 4 == last_action:
        return top2[1].view(1, 1)
    else:
        return top2[0].view(1, 1)

class Agent:
    def __init__(self, experiment_name, n_steps):
        self.n_steps = str(n_steps)
        self.experiment_name = experiment_name
        self.obs_list = []
        self.agent_class = "multi_agent"
        self.last_action = None
        self.policy_net = DQN(outputs=4)
        self.policy_net.load_state_dict(
            state_dict=torch.load("../checkpoints/checkpoints_dqn_" + self.experiment_name + "/policy_net-step_" + self.n_steps + ".pt"))
        self.policy_net.eval()

    def agent(self, obs_dict, config_dict):
        columns = config_dict.columns
        self.obs_list += [obs_dict]
        game_map = create_map_from_obs_mulit_agent_v2(self.obs_list, columns)
        action_number = select_action_from_net_non_opposite(game_map, self.policy_net, self.last_action)
        self.last_action = action_number
        return num_to_action(action_number)

    def reset(self):
        self.obs_list = []
        self.last_action = None
