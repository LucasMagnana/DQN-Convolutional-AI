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
    def __init__(self, action_space):
        self.action_space = action_space    
        self.buffer = []
        self.buffer_size = 100000

        self.alpha = 0.05
        self.epsilon = 1.0
        self.gamma = 0.9

        self.neur = CNN()
        self.neur_target = copy.deepcopy(self.neur)
        self.optimizer = torch.optim.Adam(self.neur.parameters(), 0.001) # smooth gradient descent
        #self.optimizer_target = torch.optim.Adam(self.neur_target.parameters(), 0.01) # smooth gradient descent

        self.i = 0
        self.tab_erreur = []
        

    def act(self, observation, reward, done):
        #return self.action_space.sample()
        tens_action = self.neur(observation)
        rand = random()
        if(rand > self.epsilon):
            _, indices = tens_action.max(0)
            return indices.item()
        return randint(0, tens_action.size()[0]-1)

    def sample(self, n=50):
        if(n > len(self.buffer)):
            n = len(self.buffer)
        return sample(self.buffer, n)

    def memorize(self, ob_prec, action, ob, reward, done):
        if(len(self.buffer) > self.buffer_size):
            self.buffer.pop(0)
        self.buffer.append([ob_prec, action, ob, reward, done])

    def learn(self):
        loss = MSELoss()
        self.epsilon *= 0.99
        spl = self.sample()
        for screen in spl :
            tensor_qvalues = self.neur(screen[0])
            qvalue = tensor_qvalues[screen[1]]
            reward = screen[3]
            next_qvalues = self.neur_target(screen[2])
            max_next_qvalues = torch.max(next_qvalues)
            self.optimizer.zero_grad()
            if(not(screen[4])):
                loss_tmp = loss(qvalue, reward+(self.gamma*max_next_qvalues))
            else :
                loss_tmp = loss(qvalue, torch.Tensor(np.array(reward)))
            self.memorize_erreur(loss_tmp.item())
            loss_tmp.backward()
            self.optimizer.step()
        for target_param, param in zip(self.neur_target.parameters(), self.neur.parameters()):
            target_param.data.copy_(self.alpha * param + (1-self.alpha)*target_param )

    def memorize_erreur(self, erreur):
        self.tab_erreur.append(erreur)
        if(len(self.tab_erreur) > self.buffer_size):
            plt.plot(self.tab_erreur)
            plt.ylabel('Erreur')
            plt.show()
            self.tab_erreur = []
            

