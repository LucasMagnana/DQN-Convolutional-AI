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

import copy

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
    def __init__(self, action_space):
        self.action_space = action_space
        self.buffer = []
        self.buffer_size = 100000
        self.compteur_buffer = 0
        self.replace_buffer = False

        self.pas = 0.005
        self.epsilon = 1.0
        self.gamma = 0.9
        self.nb_learn_step = 4000
        self.learn_step = 0

        self.neur = NN(15)
        self.neur_target = copy.deepcopy(self.neur)
        self.optimizer = torch.optim.SGD(self.neur.parameters(), 0.01) # smooth gradient descent
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
    agent = RandomAgent(env.action_space)

    episode_count = 1000
    reward = 0
    done = False


    reward_accumulee=0
    tab_rewards_accumulees = []

    for i in range(episode_count):
        ob = env.reset()
        while True:
            ob = ob.astype('float32')
            ob_prec = ob
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            agent.remplir_buffer(ob_prec, action, ob, reward, done)
            reward_accumulee += reward
            if done:
                agent.learn()
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