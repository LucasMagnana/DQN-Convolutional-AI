import argparse
import sys

import gym
from gym import wrappers, logger
import matplotlib
import matplotlib.pyplot as plt
from random import sample

class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()

    def sample(self, buffer, n=10):
        return sample(buffer, n)


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

    buffer = []
    buffer_size = 1000
    compteur_buffer = 0
    replace_buffer = False

    for i in range(episode_count):
        ob = env.reset()
        while True:
            ob_prec = ob
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            reward_accumulee += reward
            if(replace_buffer):
                buffer[compteur_buffer] = [ob_prec, action, ob, reward, done]
            else:
                buffer.append([ob_prec, action, ob, reward, done])
            compteur_buffer += 1
            if(compteur_buffer == buffer_size):
                compteur_buffer = 0
                replace_buffer = True
            if done:
                tab_rewards_accumulees.append(reward_accumulee)
                reward_accumulee=0
                break
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.
    print(agent.sample(buffer))
    plt.plot(tab_rewards_accumulees)
    plt.ylabel('Reward Accumulée')
    plt.show()

    # Close the env and write monitor result info to disk
    env.close()