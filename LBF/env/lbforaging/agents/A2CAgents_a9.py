import random
import numpy as np
from lbforaging.foraging.agent import Agent
from lbforaging.agents.A2CAssets.FCNetwork import FCNetwork
from torch.distributions.categorical import Categorical
import numpy as np
import os
import torch


class A2CAgentA9(Agent):
    name = "A2CAgent"

    def __init__(self, policy_type):

        current_path = os.path.dirname(os.path.realpath(__file__))

        # 1. create policy
        self.policy = torch.jit.load(current_path+"/A2CAssets_a9/agent.tjm", map_location=torch.device('cpu'))

        # 2. load weights
        self.policy.load_state_dict(torch.load(current_path+"/A2CAssets_a9/agent.th", map_location=torch.device('cpu')))
        self.policy.eval()

    def step(self, obs):
        obs_tensor = torch.Tensor(obs).view(1,-1)
        act_logits, _= self.policy(obs_tensor, None)

        act_dist = Categorical(logits=act_logits)
        acts = act_dist.sample().tolist()

        act = acts[0]
        if act == 6:
            act = 0
        return act

