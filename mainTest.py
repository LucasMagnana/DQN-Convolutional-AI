import argparse
import sys

import gym
from gym import wrappers, logger
import matplotlib
import matplotlib.pyplot as plt

from AgentStick import *
from NeuronalNetwork import NN





if __name__ == '__main__':
    module = 'CartPole-v1'
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default=module, help='Select the environment to run')
    args = parser.parse_args()

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.WARN)

    env = gym.make(args.env_id)

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    env.seed(0)

    neur = NN()
    neur.load_state_dict(torch.load(module+'.n'))
    neur.eval()
    agent = AgentStick(env.action_space, neur)

    episode_count = 5

    reward = 0
    done = False


    reward_accumulee=0


    for _ in range(episode_count):
        ob = env.reset()
        while True:
            env.render()
            ob_prec = ob       
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            reward_accumulee += reward
            if done:
                print("Reward : ", reward_accumulee)
                reward_accumulee = 0
                break

    # Close the env and write monitor result info to disk
    env.close()