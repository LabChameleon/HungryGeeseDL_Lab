import numpy as np

from kaggle_environments.envs.hungry_geese.hungry_geese import Action, row_col

import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, outputs):
        super(DQN, self).__init__()

        self.conv1 = nn.Conv2d(5, 128, kernel_size=(3, 5), )
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


def create_map_from_obs_single_agent(obs_list, columns):
    obs = obs_list[-1]

    head_pos = np.zeros((7, 11))
    body_pos = np.zeros((7, 11))
    old_head_pos = np.zeros((7, 11))
    old_body_pos = np.zeros((7, 11))
    food_pos = np.zeros((7, 11))

    head_pos[row_col(obs['geese'][0][0], columns)] = 1
    if len(obs['geese'][0]) > 1:
        for segment in obs['geese'][0][1:]:
            body_pos[row_col(segment, columns)] = 1
    if len(obs_list) > 1:
        old_head_pos[row_col(obs_list[-2]['geese'][0][0], columns)] = 1
        for segment in obs_list[-2]['geese'][0][1:]:
            body_pos[row_col(segment, columns)] = 1
    # so far we only consider the first food in the list
    for food in obs['food']:
        food_pos[row_col(food, columns)] = 1

    head_pos = np.expand_dims(head_pos, axis=0)
    body_pos = np.expand_dims(body_pos, axis=0)
    old_head_pos = np.expand_dims(old_head_pos, axis=0)
    old_body_pos = np.expand_dims(old_body_pos, axis=0)
    food_pos = np.expand_dims(food_pos, axis=0)

    # center map around head position of our goose
    c_x, c_y = row_col(obs['geese'][0][0], columns)
    map = center_map(np.concatenate((head_pos, body_pos, old_head_pos, old_body_pos, food_pos)), c_x, c_y)

    return torch.Tensor(map).view((1, 5, 7, 11))

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
    def __init__(self):
        self.obs_list = []
        self.last_action = None
        self.policy_net = DQN(outputs=4)
        self.policy_net.load_state_dict(
            state_dict=torch.load("../" + self.experiment_path + "/checkpoints/policy_net-step_"+ self.weights_itertions+".pt"))
        self.policy_net.eval()
        self.experiment_path = "/Agent-Single-2_Food-env_reward"
        self.weights_itertions = "80000"

    def agent(self, obs_dict, config_dict):
        columns = config_dict.columns
        self.obs_list += [obs_dict]
        game_map = create_map_from_obs_single_agent(self.obs_list, columns)
        action_number = select_action_from_net_non_opposite(game_map, self.policy_net, self.last_action)
        self.last_action = action_number
        return num_to_action(action_number)

    def reset(self):
        self.obs_list = []
        self.last_action = None
