import argparse
import sys

import gym
from gym import wrappers, logger
import matplotlib
import matplotlib.pyplot as plt

from AgentAtari import *
import time

from arguments import *





if __name__ == '__main__':
    module = 'BreakoutNoFrameskip-v4'
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
    outdir = './videos/'+module
    env.seed(0)
    
    env = wrappers.AtariPreprocessing(env)
    env = wrappers.FrameStack(env, 4)

    args = get_args()
    agent = AgentAtari(4, env.action_space.n, args, torch.cuda.is_available())

    nb_frames_max = args.total_timesteps
    checkpoint = nb_frames_max/10
    nb_checkpoints = 0
    reward = 0
    done = False

    reward_accumulee=0
    tab_rewards_accumulees = []

    continuer = True
    nb_frames = 0

    start_time = time.time()

    ob = env.reset()
    ob = torch.Tensor(ob).unsqueeze(0)
    for timestep in range(nb_frames_max):
        if(timestep%checkpoint == 0):
            print("--- %s seconds ---" % (time.time() - start_time))
            torch.save(agent.neur.state_dict(), './trained_networks/'+module+'_'+str(nb_checkpoints)+'.n')
            nb_checkpoints += 1
            print(tab_rewards_accumulees[-100:])
            tab_rewards_accumulees = []
        ob_prec = ob       
        action = agent.act(ob, reward, done)
        ob, reward, done, _ = env.step(action)
        ob = torch.Tensor(ob).unsqueeze(0)
        agent.memorize(ob_prec, action, ob, reward, done)
        reward_accumulee += reward
        agent.learn(timestep)
        if done:
            tab_rewards_accumulees.append(reward_accumulee)
            reward_accumulee=0
            ob = env.reset()
            ob = torch.Tensor(ob).unsqueeze(0)
            
        # Note there's no env.render() here. But the environment still can open window and
        # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
        # Video is not recorded every episode, see capped_cubic_video_schedule for details.

    torch.save(agent.neur.state_dict(), './trained_networks/'+module+'.n')
    '''plt.plot(tab_rewards_accumulees)
    plt.ylabel('Reward Accumulée')
    plt.show()'''

    # Close the env and write monitor result info to disk
    env.close()