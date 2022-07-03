import torch
import random
from utils import action_to_num

def select_action_from_net(state, policy_net):
    """select best action according to policy_net"""
    with torch.no_grad():
        return policy_net(state).max(1)[1].view(1, 1)

def select_action_with_eps(state, policy_net, eps_threshold, n_actions):
    """ for training:
    select best action according to policy_net or
    select random action
    depending on eps_threshold

    :param eps_threshold: 1 => only second choice"""
    sample = random.random()
    if sample > eps_threshold:
        return select_action_from_net(state, policy_net)
    else:
        return torch.tensor([[random.randrange(n_actions)]], dtype=torch.long)

def select_action_from_net_non_opposite(state, policy_net, last_action):
    """ select best action which is not the opposite action"""
    top2 = policy_net(state).topk(2)[1][0]
    # top action is opposite of last action
    if (top2[0] + 2) % 4 == last_action:
        return top2[1].view(1, 1)
    else:
        return top2[0].view(1, 1)

def select_action_with_eps_non_opposite(state, policy_net, eps_threshold, n_actions, last_action):
    """ for training:
    select best action according to policy_net or
    select random action without opposite action
    depending on eps_threshold

    :param eps_threshold: 1 => only second choice"""
    sample = random.random()
    if sample > eps_threshold:
        return select_action_from_net(state, policy_net)
    else:
        if last_action is not None:
            # consider only the non opposite actions
            non_opp = list(range(4))
            non_opp.pop((last_action + 2) % 4)
            return torch.tensor([[random.choice(non_opp)]], dtype=torch.long)
        else:
            return torch.tensor([[random.randrange(n_actions)]], dtype=torch.long)

def select_action_with_eps_predefined_action(state, policy_net, eps_threshold, action):
    """ for training:
    select best action according to policy_net or
    predefined action
    depending on eps_threshold

    :param eps_threshold: 1 => only second choice"""
    sample = random.random()
    if sample > eps_threshold:
        return select_action_from_net(state, policy_net)
    else:
        return torch.tensor([[action_to_num(action)]])
