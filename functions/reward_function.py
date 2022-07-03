from kaggle_environments.envs.hungry_geese.hungry_geese import row_col
import numpy as np
from utils import center_pos


def base_reward_function(obs_list, done, columns):
    """deatch => -100
    staying alive => 1
    """
    if done:
        return -1000
    return 1

def snake_reward_function(obs_list, done, columns, num_steps):
    """give reward for getting closer to food => 1
    give reward for getting away from food => -1

    give reward for eating food => 10
    give punishment for dying => -100
    """
    if done and num_steps != 199:
        return -100

    cur_obs = obs_list[-1]
    last_obs = obs_list[-2]

    fp = row_col(last_obs['food'][0], columns)
    hp_old = row_col(last_obs['geese'][0][0], columns)
    hp_new = row_col(cur_obs['geese'][0][0], columns)

    if hp_new == fp:
        return 10

    fp = center_pos(fp[0], fp[1], hp_old[0], hp_old[1], rows=7, columns=11)
    hp_new = center_pos(hp_new[0], hp_new[1], hp_old[0], hp_old[1], rows=7, columns=11)
    # this computation is irrelevant -> deterministically (row // 2, col // 2)
    hp_old = center_pos(hp_old[0], hp_old[1], hp_old[0], hp_old[1], rows=7, columns=11)

    food_dist_old = np.sqrt((fp[0] - hp_old[0]) ** 2 + (fp[1] - hp_old[1]) ** 2)
    food_dist_new = np.sqrt((fp[0] - hp_new[0]) ** 2 + (fp[1] - hp_new[1]) ** 2)

    if food_dist_new > food_dist_old:
        return -1
    # else is correct here because distance always truly increases or decreases
    else:
        return 1

def env_reward (obs_list, steps, max_length, done):
    """
    Being alive => return environment reward
    Dying => -100
    """
    if not done:
        return (steps + 1) * (max_length + 1) + len(obs_list[-1]['geese'][0])
    return -100

def env_reward_function(done, reward):
    if done and reward == 0:
        return -100
    return reward
