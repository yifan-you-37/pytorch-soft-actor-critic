import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
from sac import SAC
from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory
from RollerGrasperV2 import robot_env 

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env', default="HalfCheetah-v2",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
parser.add_argument('--linear_reward', action="store_true")
parser.add_argument('--gui', action="store_true")
parser.add_argument('--roller_reward_scale', default=100, type=float)
parser.add_argument("--eval_freq", default=5e3, type=int)   
parser.add_argument("--which_cuda", default=0, type=int)

args = parser.parse_args()

# Environment
# env = NormalizedActions(gym.make(args.env))
if args.env == 'RollerGrasperV2':
    env = robot_env.RobotEnv(gui=True, args=args)
    state_dim = env.state_dim
    action_dim = env.action_dim
    max_action = env.max_action
    print(type(max_action))
    action_space = gym.spaces.Box(low=-1 * max_action, high=max_action, shape=np.array([len(max_action),]))
else:
    env = gym.make(args.env)
    env.seed(args.seed)
    env.action_space.seed(args.seed)
    state_dim = env.observation_space.shape[0]
    action_space=env.action_space 
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Agent
print('action space', action_space)
agent = SAC(state_dim, action_space, args)

#Tesnorboard
writer = SummaryWriter('runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env,
                                                             args.policy, "autotune" if args.automatic_entropy_tuning else ""))

# Memory
memory = ReplayMemory(args.replay_size, args.seed)

# Training Loop
total_numsteps = 0
updates = 0

for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()

    while not done:
        if args.start_steps > total_numsteps:
            action = np.random.uniform(-max_action, max_action, action_dim)
        else:
            action = agent.select_action(state)
        if len(memory) > args.batch_size:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)

                writer.add_scalar('loss/critic_1', critic_1_loss, total_numsteps)
                writer.add_scalar('loss/critic_2', critic_2_loss, total_numsteps)
                writer.add_scalar('loss/policy', policy_loss, total_numsteps)
                writer.add_scalar('loss/entropy_loss', ent_loss, total_numsteps)
                writer.add_scalar('entropy_temprature/alpha', alpha, total_numsteps)
                updates += 1

        next_state, reward, done, _ = env.step(action) # Step
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if episode_steps == env._max_episode_steps else float(not done)

        memory.push(state, action, reward, next_state, mask) # Append transition to memory

        state = next_state

        if (total_numsteps + 1) % args.eval_freq == 0:
            agent.save_model(args.env, suffix=str(total_numsteps))

        if total_numsteps > args.num_steps:
            break
        
        writer.add_scalar('test/reward', reward, total_numsteps)
            
        if (total_numsteps + 1) % args.eval_freq == 0 and args.eval is True:
            avg_reward = 0.
            avg_succ = 0.
            episodes = 10
            for _  in range(episodes):
                state = env.reset()
                episode_reward = 0
                episode_succ = 0
                done = False
                while not done:
                    action = agent.select_action(state, evaluate=True)

                    next_state, reward, done, succ = env.step(action)
                    episode_reward += reward

                    if type(succ) == float:
                        episode_succ += succ

                    state = next_state
                avg_reward += episode_reward
                if episode_succ != 0:
                    avg_succ += 1
            avg_reward /= episodes
            avg_succ /= episodes

            if avg_succ > 0:
                agent.save_model(args.env, suffix='{}'.format(total_numsteps))


            writer.add_scalar('avg_reward/test', avg_reward, total_numsteps)
            writer.add_scalar('test/avg_return', avg_reward, total_numsteps)
            writer.add_scalar('test/avg_succ', avg_succ, total_numsteps)

            print("----------------------------------------")
            print("Test Episodes: {}, Avg. Reward: {:.3f}".format(episodes, avg_reward))
            print("----------------------------------------")
    writer.add_scalar('reward/train', episode_reward, i_episode)
    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))
    if total_numsteps > args.num_steps:
        break

# env.close()

