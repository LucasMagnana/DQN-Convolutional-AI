import argparse
import sys

import gym
from gym import wrappers, logger
import matplotlib
import matplotlib.pyplot as plt
from random import sample

import torch
from torch import nn
from torch.autograd import Variable

import numpy as np

from random import * 

class NN(nn.Module):

    def __init__(self, N_in):
        super(NN, self).__init__()
        self.inp = nn.Linear(4, N_in)
        self.out = nn.Linear(N_in, 2)

    def forward(self, ob):
        ob = torch.from_numpy(ob)
        ob.requires_grad = True
        return self.out(nn.functional.relu(self.inp(ob)))



class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space, neur):
        self.action_space = action_space
        self.buffer = []
        self.buffer_size = 100
        self.compteur_buffer = 0
        self.replace_buffer = False
        self.neur = neur
        self.epsilon = 0.4
        self.gamma = 0.1
        

    def act(self, observation, reward, done):
        #return self.action_space.sample()
        tens_action = self.neur(observation)
        rand = random()
        if(rand > self.epsilon):
            return (tens_action == torch.max(tens_action).item()).nonzero().item()
        return randint(0, len(tens_action)-1)

    def sample(self, n=10):
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

    def calcul_erreur(self):
        loss = nn.MSELoss() #mean square error
        spl = self.sample()
        for screen in spl :
            tensor_qvalues = self.neur(screen[0].astype("float32"))
            qvalue = tensor_qvalues[screen[1]]
            reward = screen[3]
            if(screen[4]):
                next_qvalues = self.neur(screen[2].astype("float32"))
                max_next_qvalues = torch.max(next_qvalues).item()
                erreur = torch.FloatTensor([reward+self.gamma*max_next_qvalues])
            else :
                erreur = torch.FloatTensor([reward])
            optimizer.zero_grad()
            loss_tmp = loss(qvalue, erreur)
            loss_tmp.backward()
            optimizer.step()





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='CartPole-v1', help='Select the environment to run')
    args = parser.parse_args()

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.WARN)

    env = gym.make(args.env_id)

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = '/tmp/random-agent-results'
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    neur = NN(5)
    agent = RandomAgent(env.action_space, neur)

    episode_count = 1000
    reward = 0
    done = False

    optimizer = torch.optim.SGD(neur.parameters(), 0.01) # smooth gradient descent

    reward_accumulee=0
    tab_rewards_accumulees = []

    for i in range(episode_count):
        ob = env.reset()
        while True:
            ob = ob.astype('float32')
            ob_prec = ob
            agent.calcul_erreur()
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            agent.remplir_buffer(ob_prec, action, ob, reward, done)
            reward_accumulee += reward
            if done:
                print(reward_accumulee)
                tab_rewards_accumulees.append(reward_accumulee)
                reward_accumulee=0
                break
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.
    plt.plot(tab_rewards_accumulees)
    plt.ylabel('Reward Accumulée')
    plt.show()

    # Close the env and write monitor result info to disk
    env.close()