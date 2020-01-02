import argparse
import sys

import gym
from gym import wrappers, logger
import matplotlib
import matplotlib.pyplot as plt

from RandomAgent import *
from AtariPreprocessing import AtariPreprocessing





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='BreakoutNoFrameskip-v4', help='Select the environment to run')
    args = parser.parse_args()

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.WARN)

    env = gym.make(args.env_id)

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = './videos/stick-agent-results'
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)

    preproc = AtariPreprocessing(env, scale_obs=True)

    agent = RandomAgent(env.action_space)

    episode_count = 100
    reward = 0
    done = False


    reward_accumulee=0
    tab_rewards_accumulees = []

    for i in range(episode_count):
        ob = preproc.reset()
        while True:
            ob = ob.astype('float32')
            ob_prec = ob
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = preproc.step(action)
            reward_accumulee += reward
            if done:
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