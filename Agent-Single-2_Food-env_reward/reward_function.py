from kaggle_environments.envs.hungry_geese.hungry_geese import row_col
import numpy as np

def base_reward_function(obs_list, done, columns):
    if done:
        return -1000
    return 1

# pos_x and pos_y are centered in such a way that c_x and c_y is
# the center of the map
def center_pos(pos_x, pos_y, c_x, c_y, rows, columns):
    offset_x = (rows // 2) - c_x
    offset_y = (columns // 2) - c_y
    pos_x_new = (pos_x + offset_x) % rows
    pos_y_new = (pos_y + offset_y) % columns
    return pos_x_new, pos_y_new

def snake_reward_function(obs_list, done, columns):
    # give reward for getting closer to food
    # give reward for eating food
    # give punishment for death
    if done:
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
    hp_old = center_pos(hp_old[0], hp_old[1], hp_old[0], hp_old[1], rows=7, columns=11)

    food_dist_old = np.sqrt((fp[0] - hp_old[0]) ** 2 + (fp[1] - hp_old[1]) ** 2)
    food_dist_new = np.sqrt((fp[0] - hp_new[0]) ** 2 + (fp[1] - hp_new[1]) ** 2)

    if food_dist_new > food_dist_old:
        return -1
    # else is correct here because distance always truly increases or decreases
    else:
        return 1

def env_reward_function(done, reward):
    if done and reward == 0:
        return -1000
    return reward

def env_snake_rf(reward, obs_list, done, columns):
    # give reward for getting closer to food
    # give reward for eating food
    # give punishment for dying
    if done:
        return -100

    cur_obs = obs_list[-1]
    last_obs = obs_list[-2]

    fp = row_col(last_obs['food'][0], columns)
    hp_old = row_col(last_obs['geese'][0][0], columns)
    hp_new = row_col(cur_obs['geese'][0][0], columns)

    if hp_new == fp:
        return 10+reward

    fp = center_pos(fp[0], fp[1], hp_old[0], hp_old[1], rows=7, columns=11)
    hp_new = center_pos(hp_new[0], hp_new[1], hp_old[0], hp_old[1], rows=7, columns=11)
    hp_old = center_pos(hp_old[0], hp_old[1], hp_old[0], hp_old[1], rows=7, columns=11)

    food_dist_old = np.sqrt((fp[0] - hp_old[0]) ** 2 + (fp[1] - hp_old[1]) ** 2)
    food_dist_new = np.sqrt((fp[0] - hp_new[0]) ** 2 + (fp[1] - hp_new[1]) ** 2)

    if food_dist_new > food_dist_old:
        return -1
    # else is correct here because distance always truly increases or decreases
    else:
        return 1
