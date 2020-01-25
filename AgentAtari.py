from random import sample
from random import * 
from torch.nn import MSELoss

import numpy as np

import copy
import numpy as np
import torch

from ConvNeuronalNetwork import CNN
import matplotlib.pyplot as plt

class AgentAtari(object):
    """The world's simplest agent!"""
    def __init__(self, cnn_input, cnn_output, args, cuda=False, neur=None): 
        self.args = args

        self.buffer = []
        self.buffer_size = self.args.buffer_size
        self.batch_size = self.args.batch_size

        self.gamma = self.args.gamma

        self.epsilon = self.args.init_ratio
        self.final_epsilon = self.args.final_ratio
        self.final_exploration = self.args.exploration_fraction*self.args.total_timesteps
        self.epsilon_decay = (self.epsilon-self.final_epsilon)/self.final_exploration

        self.target_update_frequency = self.args.target_network_update_freq

        self.replay_start_size = self.args.learning_starts
        self.train_frequency = self.args.train_freq

        self.cuda = cuda
        if(neur == None):
            self.neur = CNN(cnn_input, cnn_output)
            if(self.cuda):
                self.neur.cuda()
        else:
            self.neur = neur
            self.epsilon = 0

        self.neur_target = copy.deepcopy(self.neur)
        self.neur_target.load_state_dict(self.neur.state_dict())

        self.optimizer = torch.optim.Adam(self.neur.parameters(), self.args.lr) # smooth gradient descent
        


    def act(self, observation, reward, done):
        if(self.cuda):
            tens_action = self.neur(torch.Tensor(observation).cuda())
        else:
            tens_action = self.neur(torch.Tensor(observation))
        rand = random()
        if(rand > self.epsilon):
            _, indices = tens_action.max(0)
            return indices.item()
        return randint(0, tens_action.size()[0]-1)

    def sample(self):
        return sample(self.buffer, self.batch_size)

    def memorize(self, ob_prec, action, ob, reward, done):
        if(len(self.buffer) > self.buffer_size):
            self.buffer.pop(0)
        self.buffer.append([ob_prec, action, ob, reward, not(done)])

    def learn(self, timestep):
        if(self.epsilon != self.final_epsilon):
            if(self.epsilon > self.final_epsilon):
                self.epsilon -= self.epsilon_decay
            else:
                self.epsilon = self.final_epsilon
                print("Epsilon done : ", self.epsilon, timestep)

        if(timestep<self.replay_start_size or timestep % self.train_frequency != 0):
            return

        loss = MSELoss() 
        spl = self.sample()
        tens_ob = torch.cat([item[0] for item in spl])
        tens_action = torch.LongTensor([item[1] for item in spl])
        tens_ob_next = torch.cat([item[2] for item in spl])
        tens_reward = torch.Tensor([item[3] for item in spl])
        tens_done = torch.Tensor([item[4] for item in spl])

        if(self.cuda):
            tens_ob = tens_ob.cuda()
            tens_action = tens_action.cuda()
            tens_ob_next = tens_ob_next.cuda()
            tens_reward = tens_reward.cuda()
            tens_done = tens_done.cuda()

        with torch.no_grad():
            tens_next_qvalue = self.neur_target(tens_ob_next)
            tens_next_qvalue = torch.max(tens_next_qvalue, 1)[0]

        tens_qvalue = self.neur(tens_ob)
        tens_qvalue = torch.index_select(tens_qvalue, 1, tens_action).diag()


        self.optimizer.zero_grad()
        tens_loss = loss(tens_qvalue, tens_reward+(self.gamma*tens_next_qvalue)*tens_done)
        tens_loss.backward()
        self.optimizer.step()

        if(timestep % self.target_update_frequency == 0):
            self.neur_target.load_state_dict(self.neur.state_dict())

            

