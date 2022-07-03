import sys, os
sys.path.insert(0, "../functions/")

import math
from itertools import count
import numpy as np

from kaggle_environments import make
from kaggle_environments.envs.hungry_geese.hungry_geese import Observation, GreedyAgent, Configuration

import torch
import torch.nn as nn
import torch.optim as optim

import agent
import agent_v2
import agent_resnet

from buffers import PrioritizedReplayBuffer

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
    if steps_done < BATCH_SIZE:
        return

    obs_batch, act_batch, rew_batch, next_obs_batch, non_final_mask, _, idxes = memory.sample(BATCH_SIZE, 1)

    if True in non_final_mask:
        non_final_next_states = torch.cat([s for i, s in enumerate(next_obs_batch) if non_final_mask[i]])
    else:
        non_final_next_states = None

    state_batch = torch.cat(obs_batch.tolist())
    action_batch = torch.cat(act_batch.tolist())
    reward_batch = torch.cat(rew_batch.tolist())

    state_batch, action_batch, non_final_next_states = utils.data_augmentation_more_efficient(state_batch,
                    action_batch, non_final_next_states, non_final_mask, BATCH_SIZE)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    if not next_state_values is None:
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss(reduction='none')
    loss_ind = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    loss = torch.mean(loss_ind)

    memory.update_priorities(idxes, np.array((loss_ind + 1e-4).view(-1).tolist()))

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

    return loss


BATCH_SIZE = 128
GAMMA = 0.95
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 5000
TARGET_UPDATE = 100
SAVE_MODEL_STEPS = 10000

screen_height, screen_width = (7, 11)
n_actions = 4

policy_net = agent_v2.DQN_ext_conv(7,11,4).to(device)
target_net = agent_v2.DQN_ext_conv(7,11,4).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), 0.0005)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60000,80000], gamma=0.1)
memory = PrioritizedReplayBuffer(size=131072, alpha=0.4)

num_episodes = 1000000
episode_durations = []

env = make("hungry_geese", debug=True)
config = env.configuration
trainer = env.train([None, "greedy", "greedy", "greedy"])
trainer.reset()

columns = env.configuration.columns

run_name = "DQN_" + os.path.basename(__file__)

steps_done = 0

with wandb.init(project='DQN-MultiAgent-2_Food', name=run_name) as run:
    run.config.optimizer = optimizer.__class__.__name__
    run.watch(policy_net)
    last_games = []

    for i_episode in range(num_episodes):

        obs = trainer.reset()
        done = False
        obs_list = [obs]
        game_map = create_map.one_map_to_rule_them_all(obs_list, columns)

        last_action = None
        max_length = 0
        acc_rew = 0

        for cur_steps in count():

            eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                            math.exp(-1. * steps_done / EPS_DECAY)
            steps_done += 1

            g_agent = GreedyAgent(Configuration({'rows': 7, 'columns': 11}))
            det_action = g_agent(Observation(obs_list[-1]))

            action = select_action.select_action_with_eps_predefined_action(game_map, policy_net, eps_threshold, det_action)
            last_action = action

            new_obs, env_reward, env_done, info = trainer.step(num_to_action(action))
            acc_rew += env_reward

            if len(new_obs['geese'][0]) == 0:
                done = True

            obs_list += [new_obs]

            max_length = max(max_length, len(new_obs['geese'][0]))
            # custom_reward = rf.env_reward(obs_list, cur_steps, env.configuration.max_length, done)
            custom_reward = rf.env_reward_function(done, env_reward)
            if done or env_done:
                next_game_map = None
            else:
                next_game_map = create_map.one_map_to_rule_them_all(obs_list, columns)

            memory.add(game_map, action, torch.Tensor([custom_reward]), next_game_map, not (done or env_done))
            game_map = next_game_map

            loss = optimize_model()
            scheduler.step()

            if steps_done % SAVE_MODEL_STEPS == 0:
                utils.model_saver(policy_net, steps_done, "dqn_" + os.path.basename(__file__)[:-3])


            if done or env_done:
                if env_reward > 0:
                    last_games += [1]
                else:
                    last_games += [0]
                if len(last_games) > 100:
                    last_games = last_games[-100:]
                episode_durations.append(cur_steps + 1)
                run.log({
                    'game_length': episode_durations[-1],
                    'agent_length': max_length,
                    'eps': eps_threshold,
                    'acc_rew': acc_rew,
                    'won': env_reward,
                    'win_rate': sum(last_games) / len(last_games),
                    'loss': loss,
                    'learning_rate': scheduler.get_last_lr()[0]
                }
                )
                break
            else:
                run.log({'loss': loss,
                    'learning_rate': scheduler.get_last_lr()[0]})

        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

env.close()
