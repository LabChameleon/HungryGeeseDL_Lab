import sys, os

sys.path.insert(0, "../functions/")

import math
from itertools import count
import numpy as np

from kaggle_environments import make
import kaggle_environments.envs.hungry_geese.hungry_geese as hungry_geese

import torch
import torch.nn as nn
import torch.optim as optim

import agent_resnet
import agent
from agent_resnet import Transition
from agent_resnet import ReplayMemory

import reward_function as rf
import utils

from utils import num_to_action
import create_map
import select_action

import wandb

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
    next_action = policy_net(non_final_next_states).max(1)[1].detach()
    next_state_values[non_final_mask] = target_net(non_final_next_states).gather(1, next_action.unsqueeze(1)).squeeze(1).detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

    return loss


BATCH_SIZE = 1024
GAMMA = 0.95
EPS_START = 1
EPS_END = 0.05
EPS_DECAY = 5000
TARGET_UPDATE = 100
SAVE_MODEL_STEPS = 10000

screen_height, screen_width = (7, 11)
n_actions = 4

policy_net = agent.DQN(7,11,4).to(device)
target_net = agent.DQN(7,11,4).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)

num_episodes = 100000
episode_durations = []

env = make("hungry_geese", debug=True)
config = env.configuration
trainer = env.train([None, "greedy", "greedy", "greedy"])
trainer.reset()

columns = env.configuration.columns

run_name = "DQN_" + os.path.basename(__file__)

steps_done = 0
greedy_agent = hungry_geese.GreedyAgent(hungry_geese.Configuration({'rows': 7, 'columns': 11}))

with wandb.init(project='DQN-MultiAgent-2_Food', name=run_name) as run:
    run.config.optimizer = optimizer.__class__.__name__
    run.watch(policy_net)

    for i_episode in range(num_episodes):

        obs = trainer.reset()
        done = False
        obs_list = [obs]
        game_map = create_map.create_map_from_obs_multi_agent_enemy_heads(obs_list, columns)

        max_length = 0
        accumulated_reward = 0

        for cur_steps in count():

            eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                            math.exp(-1. * steps_done / EPS_DECAY)
            steps_done += 1
            greedy_action = greedy_agent(obs_list[-1])
            action = select_action.select_action_with_eps_predefined_action(game_map, policy_net, eps_threshold, greedy_action)

            new_obs, env_reward, env_done, info = trainer.step(num_to_action(action))
            if len(new_obs['geese'][0]) == 0:
                done = True

            obs_list += [new_obs]
            accumulated_reward += env_reward

            max_length = max(max_length, len(new_obs['geese'][0]))
            custom_reward = rf.env_reward_function(done, env_reward)

            if done or env_done:
                next_game_map = None
            else:
                next_game_map = create_map.create_map_from_obs_multi_agent_enemy_heads(obs_list, columns)

            memory.push(game_map, action, next_game_map, torch.Tensor([custom_reward]))
            game_map = next_game_map

            loss = optimize_model()

            if steps_done % SAVE_MODEL_STEPS == 0:
                utils.model_saver(policy_net, steps_done, "dqn_" + os.path.basename(__file__)[:-3])

            if done or env_done:
                episode_durations.append(cur_steps + 1)
                run.log({'loss': loss,
                         'game_length': episode_durations[-1],
                         'agent_length': max_length,
                         'reward': accumulated_reward,
                         'won': env_reward,
                         'eps': eps_threshold})
                break
            else:
                run.log({'loss': loss})

        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

env.close()
