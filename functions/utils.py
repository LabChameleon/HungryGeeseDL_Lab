import os
import numpy as np
import torch
from kaggle_environments.envs.hungry_geese.hungry_geese import Action

def model_saver(policy_net, steps_done, file_name):
    """save model after "steps_done"-steps to a folder
    "../checkpoints/checkpoints_" + file_name
    """
    if not os.path.exists("../checkpoints/checkpoints_" + file_name):
        os.makedirs("../checkpoints/checkpoints_" + file_name)
    torch.save(policy_net.state_dict(),
               ("../checkpoints/checkpoints_" + file_name + "/policy_net-step_{}.pt").format(steps_done))

def center_map(map, c_x, c_y):
    """centering the map around c_x and c_y"""
    _, s_x, s_y = map.shape
    x_ind = (np.arange(s_x) + (c_x - 3)) % 7
    y_ind = (np.arange(s_y) + (c_y - 5)) % 11
    map = map[:, x_ind, :]
    map = map[:, :, y_ind]
    return map

def num_to_action(action):
    """parse number to environment actions"""
    return [Action.NORTH.name, Action.EAST.name, Action.SOUTH.name, Action.WEST.name][action]

def action_to_num(action):
    """parse environment actions to number"""
    return [Action.NORTH.name, Action.EAST.name, Action.SOUTH.name, Action.WEST.name].index(action)

def center_pos(pos_x, pos_y, c_x, c_y, rows, columns):
    """pos_x and pos_y are centered in such a way that c_x and c_y is
    the center of the map
    """
    offset_x = (rows // 2) - c_x
    offset_y = (columns // 2) - c_y
    pos_x_new = (pos_x + offset_x) % rows
    pos_y_new = (pos_y + offset_y) % columns
    return pos_x_new, pos_y_new

def map_padding(x, pad_value):
    """
  :param x: map
  :param pad_value: padding per side
  :return: padded map with values from the other side
  """
    h = torch.cat([x[:, :, :, -pad_value:], x, x[:, :, :, :pad_value]], dim=3)
    h = torch.cat([h[:, :, -pad_value:], h, h[:, :, :pad_value]], dim=2)
    return h

def flip_horizontal_more_efficient(states, actions, next_states, non_final_mask, flip_prop, batch_size):
    flip = torch.rand(batch_size) < flip_prop
    states[flip] = torch.flip(states[flip], (2,))
    actions_flip = actions % 2 == 0
    flip_actions = torch.logical_and(flip, actions_flip.squeeze(1))
    actions[flip_actions] = (actions[flip_actions] + 2) % 4
    next_states[flip[non_final_mask]] = torch.flip(next_states[flip[non_final_mask]], (2,))
    return states, actions, next_states

def flip_vertical_more_efficient(states, actions, next_states, non_final_mask, flip_prop, batch_size):
    flip = torch.rand(batch_size) < flip_prop
    states[flip] = torch.flip(states[flip], (3,))
    actions_flip = actions % 2 == 1
    flip_actions = torch.logical_and(flip, actions_flip.squeeze(1))
    actions[flip_actions] = (actions[flip_actions] + 2) % 4
    next_states[flip[non_final_mask]] = torch.flip(next_states[flip[non_final_mask]], (3,))
    return states, actions, next_states

def data_augmentation_more_efficient(obs_batch, act_batch, next_obs_batch, non_final_mask, batch_size):
    obs_batch, act_batch, next_obs_batch = flip_horizontal_more_efficient(obs_batch, act_batch,
                                                                          next_obs_batch, non_final_mask, 0.5,
                                                                          batch_size)
    obs_batch, act_batch, next_obs_batch = flip_vertical_more_efficient(obs_batch, act_batch,
                                                                        next_obs_batch, non_final_mask, 0.5, batch_size)
    return obs_batch, act_batch, next_obs_batch
