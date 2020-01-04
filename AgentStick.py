from random import sample
from random import * 

import copy
import numpy as np
import torch

from NeuronalNetwork import NN

class AgentStick(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space
        self.buffer = []
        self.buffer_size = 100000
        self.compteur_buffer = 0
        self.replace_buffer = False

        self.pas = 0.005
        self.epsilon = 1.0
        self.gamma = 0.9
        self.nb_learn_step = 5000
        self.learn_step = 0

        self.neur = NN(15)
        self.neur_target = copy.deepcopy(self.neur)
        self.optimizer = torch.optim.Adam(self.neur.parameters(), 0.01) # smooth gradient descent
        #self.optimizer_target = torch.optim.Adam(self.neur_target.parameters(), 0.01) # smooth gradient descent

        

    def act(self, observation, reward, done):
        #return self.action_space.sample()
        tens_action = self.neur(observation)
        rand = random()
        if(rand > self.epsilon):
            tens = (tens_action == torch.max(tens_action).item()).nonzero()
            if(len(tens)>1):
                return 0
            else :
                return tens.item()
        return randint(0, len(tens_action)-1)

    def sample(self, n=100):
        if(n > len(self.buffer)):
            n = len(self.buffer)
        return sample(self.buffer, n)

    def remplir_buffer(self, ob_prec, action, ob, reward, done):
        if(self.replace_buffer):
            self.buffer[self.compteur_buffer] = [ob_prec, action, ob, reward, done]
        else:
            self.buffer.append([ob_prec, action, ob, reward, done])
        self.compteur_buffer += 1
        if(self.compteur_buffer == self.buffer_size):
            self.compteur_buffer = 0
            self.replace_buffer = True

    def learn(self):
        self.epsilon *= 0.99
        if(self.epsilon < 0.05):
            self.epsilon = 0
        spl = self.sample()
        for screen in spl :
            self.learn_step += 1
            if(self.learn_step%self.nb_learn_step == 0 and self.learn_step < 75000):
                self.neur_target = copy.deepcopy(self.neur)
            tensor_qvalues = self.neur(screen[0].astype("float32"))
            qvalue = tensor_qvalues[screen[1]]
            reward = screen[3]
            if(not screen[4]):
                next_qvalues = self.neur_target(screen[2].astype("float32"))
                max_next_qvalues = torch.max(next_qvalues)
                erreur = reward+self.gamma*max_next_qvalues
            else :
                erreur = torch.FloatTensor([reward])

            self.optimizer.zero_grad()
            loss_tmp = (qvalue-erreur)**2
            loss_tmp.backward()
            self.optimizer.step()
