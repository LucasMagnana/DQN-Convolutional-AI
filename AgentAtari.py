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
    def __init__(self, cnn_input, cnn_output, cuda=False, neur=None): 
        self.buffer = []
        self.buffer_size = 250000
        self.cuda = cuda

        self.epsilon = 1.0
        self.final_epsilon = 0.1
        self.gamma = 0.99

        self.final_exploration = 100000
        self.epsilon_decay = (self.epsilon-self.final_epsilon)/self.final_exploration

        self.target_update_frequency = 10000
        self.learn_step = 0

        self.replay_start_size = 5000

        if(neur == None):
            if(self.cuda):
                self.neur = CNN(cnn_input, cnn_output).cuda()
            else: 
                self.neur = CNN(cnn_input, cnn_output)
        else:
            self.neur = neur
            self.epsilon = 0

        self.neur_target = copy.deepcopy(self.neur)
        self.optimizer = torch.optim.RMSprop(self.neur.parameters(), lr=0.0025, momentum=0.95, alpha=0.95, eps=0.01) # smooth gradient descent
        


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

    def sample(self, n=32):
        if(n > len(self.buffer)):
            n = len(self.buffer)
        return sample(self.buffer, n)

    def memorize(self, ob_prec, action, ob, reward, done):
        if(self.epsilon > self.final_epsilon):
            self.epsilon -= self.epsilon_decay
        if(len(self.buffer) > self.buffer_size):
            self.buffer.pop(0)
        else:
            print(len(self.buffer))
        self.buffer.append([ob_prec, action, ob, reward, not(done)])

    def learn(self):
        if(len(self.buffer)<self.replay_start_size):
            return
        self.learn_step +=1

        loss = MSELoss() 
        spl = self.sample()
        tens_ob = torch.cat([item[0] for item in spl])
        tens_action = torch.LongTensor([item[1] for item in spl])
        tens_ob_next = torch.cat([item[2] for item in spl])
        tens_reward = torch.Tensor([item[3] for item in spl])
        tens_done = torch.Tensor([item[4] for item in spl])

        if(self.cuda):
            tens_qvalue = self.neur(tens_ob.cuda()).cpu()
            tens_next_qvalue = self.neur_target(tens_ob_next.cuda()).cpu()
        else:
            tens_qvalue = self.neur(tens_ob)
            tens_next_qvalue = self.neur_target(tens_ob_next)

        tens_qvalue = torch.index_select(tens_qvalue, 1, tens_action).diag()

        tens_next_qvalue = torch.max(tens_next_qvalue, 1)[0]

        self.optimizer.zero_grad()
        tens_loss = loss(tens_qvalue, tens_reward+(self.gamma*tens_next_qvalue)*tens_done)
        tens_loss.backward()
        self.optimizer.step()

        if(self.learn_step >= self.target_update_frequency):
            self.learn_step = 0
            self.neur_target = copy.deepcopy(self.neur)

            

