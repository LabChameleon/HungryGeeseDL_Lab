import torch
import numpy as np
from utils import center_map
from kaggle_environments.envs.hungry_geese.hungry_geese import row_col

# Single agent, one food

def create_map_from_obs_base(obs_list, columns):
    """2 maps:
    map 1: head_pos
    map 2: previous head
    """
    obs = obs_list[-1]

    head_pos = np.zeros((7, 11))
    old_head_pos = np.zeros((7, 11))

    head_pos[row_col(obs['geese'][0][0], columns)] = 1
    if len(obs_list) > 1:
        old_head_pos[row_col(obs_list[-2]['geese'][0][0], columns)] = 1

    head_pos = np.expand_dims(head_pos, axis=0)
    old_head_pos = np.expand_dims(old_head_pos, axis=0)

    # center map around head position of our goose
    c_x, c_y = row_col(obs['geese'][0][0], columns)
    map = center_map(np.concatenate((head_pos, old_head_pos)), c_x, c_y)

    return torch.Tensor(map).view((1, 2, 7, 11))


def create_map_from_obs_single_agent(obs_list, columns):
    """4 maps:
    map 1: head_pos
    map 2: body_pos
    map 3: prev_head_pos
    map 4: food_pos
    """
    obs = obs_list[-1]

    head_pos = np.zeros((7, 11))
    body_pos = np.zeros((7, 11))
    old_head_pos = np.zeros((7, 11))
    food_pos = np.zeros((7, 11))

    head_pos[row_col(obs['geese'][0][0], columns)] = 1
    if len(obs['geese'][0]) > 1:
        for segment in obs['geese'][0][1:]:
            body_pos[row_col(segment, columns)] = 1
    if len(obs_list) > 1:
        old_head_pos[row_col(obs_list[-2]['geese'][0][0], columns)] = 1
    # so far we only consider the first food in the list
    food_pos[row_col(obs['food'][0], columns)] = 1

    head_pos = np.expand_dims(head_pos, axis=0)
    body_pos = np.expand_dims(body_pos, axis=0)
    old_head_pos = np.expand_dims(old_head_pos, axis=0)
    food_pos = np.expand_dims(food_pos, axis=0)

    # center map around head position of our goose
    c_x, c_y = row_col(obs['geese'][0][0], columns)
    map = center_map(np.concatenate((head_pos, body_pos, old_head_pos, food_pos)), c_x, c_y)

    return torch.Tensor(map).view((1, 4, 7, 11))


# Multi Agent, two food

def create_map_from_obs_mulit_agent(obs_list, columns):
    """6 maps:
    map 1: head_pos
    map 2: body_pos
    map 3: prev_head_pos
    map 4: prev_body_pos
    map 5: enemy_geese_pos (head + body)
    map 6: food_pos (all food)
    """
    obs = obs_list[-1]
    cur_index = obs['index']

    head_pos = np.zeros((7, 11))
    body_pos = np.zeros((7, 11))
    old_head_pos = np.zeros((7, 11))
    old_body_pos = np.zeros((7, 11))
    enemy_geese_pos = np.zeros((7, 11))
    food_pos = np.zeros((7, 11))

    # maps from current observation
    head_pos[row_col(obs['geese'][cur_index][0], columns)] = 1
    if len(obs['geese'][cur_index]) > 1:
        for segment in obs['geese'][cur_index][1:]:
            body_pos[row_col(segment, columns)] = 1

    for i in range(len(obs['geese'])):
        if i != cur_index:
            for segment in obs['geese'][i]:
                enemy_geese_pos[row_col(segment, columns)] = 1

    # maps from previous observation
    if len(obs_list) > 1:
        old_head_pos[row_col(obs_list[-2]['geese'][cur_index][0], columns)] = 1
        for segment in obs_list[-2]['geese'][cur_index][1:]:
            old_body_pos[row_col(segment, columns)] = 1

    for food in obs['food']:
        food_pos[row_col(food, columns)] = 1

    head_pos = np.expand_dims(head_pos, axis=0)
    body_pos = np.expand_dims(body_pos, axis=0)
    old_head_pos = np.expand_dims(old_head_pos, axis=0)
    old_body_pos = np.expand_dims(old_body_pos, axis=0)
    enemy_geese_pos = np.expand_dims(enemy_geese_pos, axis=0)
    food_pos = np.expand_dims(food_pos, axis=0)

    # center map around head position of our goose
    c_x, c_y = row_col(obs['geese'][cur_index][0], columns)
    map = center_map(np.concatenate((head_pos, body_pos, old_head_pos, old_body_pos, enemy_geese_pos, food_pos)), c_x,
                     c_y)

    return torch.Tensor(map).view((1, 6, 7, 11))


def create_map_from_obs_multi_agent_enemy_heads(obs_list, columns):
    """6 maps:
    map 1: head_pos
    map 2: prev_head
    map 3: body_pos
    map 4: enemy_geese_pos head
    map 5: enemy_geese_pos body
    map 6: food_pos (all food)
    """
    obs = obs_list[-1]
    cur_index = obs['index']

    head_pos = np.zeros((7, 11))
    prev_head_pos = np.zeros((7, 11))
    body_pos = np.zeros((7, 11))
    enemy_geese_head = np.zeros((7, 11))
    enemy_geese_body = np.zeros((7, 11))
    food_pos = np.zeros((7, 11))

    # maps from current observation
    head_pos[row_col(obs['geese'][cur_index][0], columns)] = 1
    if len(obs['geese'][cur_index]) > 1:
        for segment in obs['geese'][cur_index]:
            body_pos[row_col(segment, columns)] = 1

    for i in range(len(obs['geese'])):
        if i != cur_index:
            if len(obs['geese'][i]) > 0:
                enemy_geese_head[row_col(obs['geese'][i][0], columns)] = 1
            for segment in obs['geese'][i]:
                enemy_geese_body[row_col(segment, columns)] = 1

    # previous head position
    if len(obs_list) > 1:
        prev_head_pos[row_col(obs_list[-2]['geese'][cur_index][0], columns)] = 1

    for food in obs['food']:
        food_pos[row_col(food, columns)] = 1

    head_pos = np.expand_dims(head_pos, axis=0)
    prev_head_pos = np.expand_dims(prev_head_pos, axis=0)
    body_pos = np.expand_dims(body_pos, axis=0)
    enemy_geese_head = np.expand_dims(enemy_geese_head, axis=0)
    enemy_geese_body = np.expand_dims(enemy_geese_body, axis=0)
    food_pos = np.expand_dims(food_pos, axis=0)

    # center map around head position of our goose
    c_x, c_y = row_col(obs['geese'][cur_index][0], columns)
    map = center_map(np.concatenate((head_pos, body_pos, prev_head_pos, enemy_geese_head,
                                     enemy_geese_body, food_pos)), c_x, c_y)

    return torch.Tensor(map).view((1, 6, 7, 11))


def create_map_from_obs_mulit_agent_v2(obs_list, columns):
    """7 maps:
    map 1: head_pos
    map 2: body_pos
    map 3: prev_head_pos
    map 4: prev_body_pos
    map 5: enemy_geese_pos (head + body)
    map 6: old_enemy_geese_pos (head + body)
    map 7: food_pos (all food)
    """
    obs = obs_list[-1]
    cur_index = obs['index']

    head_pos = np.zeros((7, 11))
    body_pos = np.zeros((7, 11))
    old_head_pos = np.zeros((7, 11))
    old_body_pos = np.zeros((7, 11))
    enemy_geese_pos = np.zeros((7, 11))
    old_enemy_geese_pos = np.zeros((7, 11))
    food_pos = np.zeros((7, 11))

    # maps from current observation
    head_pos[row_col(obs['geese'][cur_index][0], columns)] = 1
    if len(obs['geese'][cur_index]) > 1:
        for segment in obs['geese'][cur_index][1:]:
            body_pos[row_col(segment, columns)] = 1

    for i in range(len(obs['geese'])):
        if i != cur_index:
            for segment in obs['geese'][i]:
                enemy_geese_pos[row_col(segment, columns)] = 1

    # maps from previous observation
    if len(obs_list) > 1:
        old_head_pos[row_col(obs_list[-2]['geese'][cur_index][0], columns)] = 1
        for segment in obs_list[-2]['geese'][cur_index][1:]:
            old_body_pos[row_col(segment, columns)] = 1

        for i in range(len(obs['geese'])):
            if i != cur_index:
                for segment in obs_list[-2]['geese'][i]:
                    old_enemy_geese_pos[row_col(segment, columns)] = 1

    for food in obs['food']:
        food_pos[row_col(food, columns)] = 1

    head_pos = np.expand_dims(head_pos, axis=0)
    body_pos = np.expand_dims(body_pos, axis=0)
    old_head_pos = np.expand_dims(old_head_pos, axis=0)
    old_body_pos = np.expand_dims(old_body_pos, axis=0)
    enemy_geese_pos = np.expand_dims(enemy_geese_pos, axis=0)
    old_enemy_geese_pos = np.expand_dims(old_enemy_geese_pos, axis=0)
    food_pos = np.expand_dims(food_pos, axis=0)

    # center map around head position of our goose
    c_x, c_y = row_col(obs['geese'][cur_index][0], columns)
    map = center_map(np.concatenate(
        (head_pos, old_head_pos, body_pos, old_body_pos, enemy_geese_pos, old_enemy_geese_pos, food_pos)), c_x, c_y)

    return torch.Tensor(map).view((1, 7, 7, 11))


def create_map_from_obs_mulit_agent_v3(obs_list, columns):
    """9 maps:
    map 1: head_pos
    map 2: body_pos
    map 3: prev_head_pos
    map 4: prev_body_pos
    map 5: enemy_geese_pos_head
    map 6: enemy_geese_pos_body
    map 7: prev_enemy_geese_pos_head
    map 8: prev_enemy_geese_pos_body
    map 9: food_pos (all food)
    """
    obs = obs_list[-1]
    cur_index = obs['index']

    head_pos = np.zeros((7, 11))
    body_pos = np.zeros((7, 11))

    old_head_pos = np.zeros((7, 11))
    old_body_pos = np.zeros((7, 11))

    enemy_geese_head_pos = np.zeros((7, 11))
    enemy_geese_body_pos = np.zeros((7, 11))

    old_enemy_geese_head_pos = np.zeros((7, 11))
    old_enemy_geese_body_pos = np.zeros((7, 11))

    food_pos = np.zeros((7, 11))

    # maps from current observation

    for i in range(len(obs['geese'])):
        if i == cur_index:
            head_pos[row_col(obs['geese'][i][0], columns)] = 1
            if len(obs['geese'][i]) > 1:
                for segment in obs['geese'][i][1:]:
                    body_pos[row_col(segment, columns)] = 1
        else:
            if len(obs['geese'][i]) != 0:
                enemy_geese_head_pos[row_col(obs['geese'][i][0], columns)] = 1
                if len(obs['geese'][i]) > 1:
                    for segment in obs['geese'][i][1:]:
                        enemy_geese_body_pos[row_col(segment, columns)] = 1

    # maps from previous observation
    if len(obs_list) > 1:
        for i in range(len(obs['geese'])):
            if i == cur_index:
                old_head_pos[row_col(obs_list[-2]['geese'][i][0], columns)] = 1
                for segment in obs_list[-2]['geese'][i][1:]:
                    old_body_pos[row_col(segment, columns)] = 1
            else:
                if len(obs['geese'][i]) != 0:
                    old_enemy_geese_head_pos[row_col(obs_list[-2]['geese'][i][0], columns)] = 1
                    for segment in obs_list[-2]['geese'][i][1:]:
                        old_enemy_geese_body_pos[row_col(segment, columns)] = 1

    for food in obs['food']:
        food_pos[row_col(food, columns)] = 1

    head_pos = np.expand_dims(head_pos, axis=0)
    body_pos = np.expand_dims(body_pos, axis=0)

    old_head_pos = np.expand_dims(old_head_pos, axis=0)
    old_body_pos = np.expand_dims(old_body_pos, axis=0)

    enemy_geese_head_pos = np.expand_dims(enemy_geese_head_pos, axis=0)
    enemy_geese_body_pos = np.expand_dims(enemy_geese_body_pos, axis=0)

    old_enemy_geese_head_pos = np.expand_dims(old_enemy_geese_head_pos, axis=0)
    old_enemy_geese_body_pos = np.expand_dims(old_enemy_geese_body_pos, axis=0)
    food_pos = np.expand_dims(food_pos, axis=0)

    # center map around head position of our goose
    c_x, c_y = row_col(obs['geese'][cur_index][0], columns)
    map = center_map(
        np.concatenate((head_pos, body_pos, old_head_pos, old_body_pos, enemy_geese_head_pos, enemy_geese_body_pos,
                        old_enemy_geese_head_pos, old_enemy_geese_body_pos,
                        food_pos)), c_x, c_y)

    return torch.Tensor(map).view((1, 9, 7, 11))


# todo: add one layer showing the time as value between 0 and 1!!
def one_map_to_rule_them_all(obs_list, columns):
    maps = []

    obs = obs_list[-1]
    cur_index = obs['index']

    # positions of our agent
    head_pos = np.zeros((7,11))
    body_pos = np.zeros((7,11))
    tail_pos = np.zeros((7,11))

    head_pos[row_col(obs['geese'][cur_index][0], columns)] = 1
    for segment in obs['geese'][cur_index]:
        body_pos[row_col(segment, columns)] = 1
    tail_pos[row_col(obs['geese'][cur_index][-1], columns)] = 1

    maps += [head_pos, body_pos, tail_pos]

    # positions of every enemy agent
    for i in range(4):
        if not i == cur_index:
            enemy_head_pos = np.zeros((7,11))
            enemy_body_pos = np.zeros((7,11))
            enemy_tail_pos = np.zeros((7,11))
            if len(obs['geese'][i]) > 0:
                enemy_head_pos[row_col(obs['geese'][i][0], columns)] = 1
                for segment in obs['geese'][i]:
                    enemy_body_pos[row_col(segment, columns)] = 1
                enemy_tail_pos[row_col(obs['geese'][i][-1], columns)] = 1
            maps += [enemy_head_pos, enemy_body_pos, enemy_tail_pos]

    # old head positions
    if len(obs_list) > 1:
        old_obs = obs_list[-2]
        old_head_pos = np.zeros((7,11))
        enemy_head_pos[row_col(old_obs['geese'][cur_index][0], columns)] = 1
        maps += [old_head_pos]
        for i in range(4):
            if not i == cur_index:
                enemy_old_head_pos = np.zeros((7,11))
                if len(old_obs['geese'][i]) > 0:
                    enemy_old_head_pos[row_col(old_obs['geese'][i][0], columns)] = 1
                maps += [enemy_old_head_pos]
    else:
        maps += [np.zeros((7,11))]*4

    # food positions
    food_pos = np.zeros((7,11))
    for food in obs['food']:
        food_pos[row_col(food, columns)] = 1
    maps += [food_pos]

    # center map around head position of our goose
    c_x, c_y = row_col(obs['geese'][cur_index][0], columns)
    stacked_maps = center_map(np.stack(maps), c_x, c_y)

    return torch.Tensor(stacked_maps).view((1, 17, 7, 11))
