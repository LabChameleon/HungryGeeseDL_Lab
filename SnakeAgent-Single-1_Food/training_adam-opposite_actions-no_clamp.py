import sys, os
sys.path.insert(0, "../functions/")

import math
from itertools import count

from kaggle_environments import make

import torch
import torch.nn as nn
import torch.optim as optim

import agent
from agent import Transition
from agent import ReplayMemory
import reward_function as rf
import utils

import wandb

from utils import num_to_action
from create_map import create_map_from_obs_single_agent
from select_action import select_action_with_eps

wandb.login()

if torch.cuda.is_available():
    device = "cuda"
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = "cpu"


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = policy_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss


BATCH_SIZE = 128
GAMMA = 0.95
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 5000
TARGET_UPDATE = 10
SAVE_MODEL_STEPS = 10000

screen_height, screen_width = (7, 11)
n_actions = 4

policy_net = agent.DQN(screen_height, screen_width, n_actions).to(device)
target_net = agent.DQN(screen_height, screen_width, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters())
memory = ReplayMemory(10000)

num_episodes = 10000
episode_durations = []

env = make("hungry_geese", debug=True)
config = env.configuration
trainer = env.train([None])
trainer.reset()

columns = env.configuration.columns

run_name = "DQN_" + os.path.basename(__file__)

steps_done = 0
with wandb.init(project='SingleAgent_OppositeDirection', name=run_name) as run:
    run.config.optimizer = optimizer.__class__.__name__
    run.watch(policy_net)

    for i_episode in range(num_episodes):

        obs = trainer.reset()
        obs_list = [obs]
        game_map = create_map_from_obs_single_agent(obs_list, columns)

        max_length = 0
        max_reward = 0

        for cur_steps in count():

            eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                            math.exp(-1. * steps_done / EPS_DECAY)
            steps_done += 1
            action = select_action_with_eps(game_map, policy_net, eps_threshold, n_actions)

            new_obs, env_reward, done, info = trainer.step(num_to_action(action))
            obs_list += [new_obs]
            if env_reward > max_reward:
                max_reward = env_reward

            max_length = max(max_length, len(new_obs['geese'][0]))
            custom_reward = rf.snake_reward_function(obs_list, done, columns, cur_steps)

            if done:
                next_game_map = None
            else:
                next_game_map = create_map_from_obs_single_agent(obs_list, columns)

            memory.push(game_map, action, next_game_map, torch.Tensor([custom_reward]))
            game_map = next_game_map

            loss = optimize_model()

            if steps_done % SAVE_MODEL_STEPS == 0:
                utils.model_saver(policy_net, steps_done, "dqn_" + os.path.basename(__file__)[:-3])

            run.log({'loss': loss})
            if done:
                episode_durations.append(cur_steps + 1)
                run.log({'game_length': episode_durations[-1]})
                run.log({'agent_length': max_length})
                run.log({'reward': max_reward})
                run.log({'eps': eps_threshold})
                break

env.close()
