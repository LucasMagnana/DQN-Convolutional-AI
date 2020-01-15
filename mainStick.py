import argparse
import sys

import gym
from gym import wrappers, logger
import matplotlib
import matplotlib.pyplot as plt

from AgentStick import *





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
    outdir = './videos/stick-agent-results'
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)

    agent = AgentStick(env.action_space)

    episode_count = 2000
    reward = 0
    done = False


    reward_accumulee=0
    tab_rewards_accumulees = []

    sum_reward = 0
    nb_reward = 0
    avg_reward = 0
    nb_episodes = 0

    save = False


    while(avg_reward<600 and nb_episodes<episode_count): #for i in range(episode_count):
        nb_episodes += 1
        ob = env.reset()
        while True:
            ob_prec = ob       
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            agent.memorize(ob_prec, action, ob, reward, done)
            reward_accumulee += reward
            if done:
                agent.learn()
                tab_rewards_accumulees.append(reward_accumulee)
                if(nb_reward < 100):
                    nb_reward+=1
                else:
                    sum_reward -= tab_rewards_accumulees[len(tab_rewards_accumulees)-nb_reward+1]
                sum_reward += reward_accumulee
                avg_reward = sum_reward/nb_reward
                reward_accumulee=0
                break
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for 
    if(avg_reward > 195.0):
        print("Solved in ", nb_episodes, " episodes!")
    else :
        print("Average: ", avg_reward)

    if(save):
        print("Saving...")
        torch.save(agent.neur.state_dict(), module+'.n')
        torch.save(agent.neur_target.state_dict(), module+'_target.n')

    plt.plot(tab_rewards_accumulees)
    plt.ylabel('Reward Accumulée')
    plt.show()

    # Close the env and write monitor result info to disk
    env.close()