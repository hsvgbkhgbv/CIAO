import argparse
import gym
import random
import Wolfpack_gym
from Wolfpack_gym.envs.wolfpack_penalty_single_adhoc_assets.Agents import *
from Agent import MRFAgent
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from gym.vector import AsyncVectorEnv
from datetime import date
import random
import string
import os
import json
import copy
import time

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.00025, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.99, help='dicount_rate')
parser.add_argument('--num_episodes', type=int, default=4000, help="Number of episodes for training")
parser.add_argument('--update_frequency', type=int, default=4, help="Timesteps between updates")
parser.add_argument('--saving_frequency', type=int,default=50,help="saving frequency")
parser.add_argument('--with_gpu', type=bool,default=False,help="with gpu")
parser.add_argument('--num_envs', type=int,default=16, help="Number of environments")
parser.add_argument('--tau', type=float,default=0.001, help="tau")
parser.add_argument('--clip_grad', type=float,default=10.0, help="gradient clipping")
parser.add_argument('--eval_eps', type=int, default=5, help="Evaluation episodes")
parser.add_argument('--weight_predict', type=float, default=1.0, help="Evaluation episodes")
# parser.add_argument('--save_dir', type=str, default='parameters', help="parameter dir name")
parser.add_argument('--save_dir', type=str, default='.', help="save dir name")
parser.add_argument('--num_players', type=int, default=3, help="num players")
parser.add_argument('--num_players_test', type=int, default=5, help="num players to test")
parser.add_argument('--pair_comp', type=str, default='bmm', help="pairwise factor computation")
parser.add_argument('--info', type=str, default="", help="additional info")
parser.add_argument('--seed', type=int, default=0, help="additional info")
parser.add_argument('--close_penalty', type=float, default=0.5, help="close penalty")
parser.add_argument('--note', type=str, required=True, help="Please note the version of you this trial")
# parser.add_argument('--training_mode', type=str, default='no_gnn', help='Please input the training mode')
parser.add_argument('--act_func', type=str, default='relu', help='Please input the activation function for the hidden layers, e.g. relu or leaky_relu')
parser.add_argument('--seq_model', type=str, default='lstm', help='Please input the sequential model used for both utility model and agent model, e.g. gru or lstm')
# TODO: JIANHONG
# parser.add_argument('--pos_pair', action='store_true', help='with positive paired q-values')
# parser.add_argument('--star_graph', action='store_true', help='with star graph for gnn of policy')
parser.add_argument('--graph', type=str, default="star", help='input the dynamic affinity graph structure: star or complete.')
parser.add_argument('--pair_range', type=str, default="pos", help='input the pairwise utility range: pos, neg, or free.')
parser.add_argument('--indiv_range', type=str, default="pos", help='input the individual utility range: pos, neg, zero, or free.')
# parser.add_argument('--pos_indiv', action='store_true', help='with positive indiv q-values')
# parser.add_argument('--learner_type', action='store_true', help='Whether the Q value is with learner type as input')
# parser.add_argument('--neg_pair', action='store_true', help='with negative paired q-values')
# parser.add_argument('--neg_indiv', action='store_true', help='with negative indiv q-values')
# parser.add_argument('--zero_indiv', action='store_true', help='with zero indiv q-values')
# NOTE: JIANHONG CHANGE 01/01/2024
# parser.add_argument('--weight_learner', type=float, default=1.0, help="weight associated to individual values regularizer")
parser.add_argument('--weight_regularizer', type=float, default=0.5, help="weight associated to individual values regularizer")
# NOTE: JIANHONG CHANGE 06/01/2024
# parser.add_argument('--equal_indiv', action='store_true', help="determine if the regularizer for individual values are forcing them equal")
# NOTE: JIANHONG CHANGE 23/03/2024
parser.add_argument('--update_manner', type=str, default="org", help="update manner for the joint Q-value: org stands for the normal update manner employed in GPL, whereas variant stands for the update manner employed in CIAO")
parser.add_argument('--intersection_generalization', action='store_true', help="determine if intersection generalization is performed")
# NOTE: JIANHONG CHANGE 24/03/2024
parser.add_argument('--exclusion_generalization', action='store_true', help="determine if exclusion generalization is performed")

args = parser.parse_args()

if __name__ == '__main__':
    if args.note is None:
        print("args.node should not be None")
        exit()
    start_time = time.time()
    random_experiment_name = time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime()) + '_' + args.note
    # NOTE: JIANHONG CHANGE 01/01/2024
    # writer = SummaryWriter(log_dir="runs/"+random_experiment_name)
    # directory = os.path.join(args.save_dir, random_experiment_name)
    directory_runs = os.path.join(args.save_dir, 'runs', random_experiment_name)
    directory_params = os.path.join(args.save_dir, 'parameters', random_experiment_name)
    # if not os.path.exists(directory):
    #     os.makedirs(directory)
    if not os.path.exists(directory_runs):
        os.makedirs(directory_runs)
    if not os.path.exists(directory_params):
        os.makedirs(directory_params)
    writer = SummaryWriter(log_dir=directory_runs)

    print("=================")
    # txt_file = open(os.path.join('runs',random_experiment_name, 'print_info.txt'), 'w')
    txt_file = open(os.path.join(directory_runs, 'print_info.txt'), 'w')
    print("note: {}".format(args.note))
    txt_file.write("note: {}\n".format(args.note))
    args = vars(args)

    # with open(os.path.join(directory,'params.json'), 'w') as json_file:
    #     json.dump(args, json_file)
    with open(os.path.join(directory_params, 'params.json'), 'w') as json_file:
        json.dump(args, json_file)

    # with open(os.path.join('runs',random_experiment_name, 'params.json'), 'w') as json_file:
    #     json.dump(args, json_file)
    with open(os.path.join(directory_runs, 'params.json'), 'w') as json_file:
        json.dump(args, json_file)

    print("time stamp: {}".format(random_experiment_name))
    txt_file.write("time stamp: {}\n args:\n".format(random_experiment_name))
    print("args:")
    for k,v in args.items():
        print("{}: {};".format(k, v))
        txt_file.write("{}: {};\n".format(k, v))
    print("=================")
    txt_file.close()

    # Initialize GPL-Q Agent
    agent = MRFAgent(args=args, writer=writer, added_u_dim = 12)

    num_players_train = args["num_players"]
    num_players_test = args["num_players_test"]

    # Define the training environment
    def make_env(env_id, rank, num_players, seed=1285, close_penalty=0.5, implicit_max_player_num=3, with_shuffling=False):
        def _init():
            env = gym.make(env_id, seed=seed + rank, num_players=num_players, close_penalty=close_penalty, implicit_max_player_num=implicit_max_player_num,  with_shuffling=with_shuffling, max_player_num=num_players_test)
            return env

        return _init
    # NOTE: JIANHONG CHANGE 23/03/2024
    def make_predefined_agent_type_env(env_id, rank, num_players, seed=1285, close_penalty=0.5, implicit_max_player_num=3, with_shuffling=False, predefined_agent_types=[]):
        def _init():
            env = gym.make(env_id, seed=seed + rank, num_players=num_players, close_penalty=close_penalty, implicit_max_player_num=implicit_max_player_num,  with_shuffling=with_shuffling, max_player_num=num_players_test, predefined_agent_types=predefined_agent_types)
            return env

        return _init

    # check whether both pos_indiv,neg_indiv and zero_indiv / pos_pair and neg_pair are true
    # assert args['pos_indiv']*args['neg_indiv']==0 and args['pos_indiv']*args['zero_indiv']==0 and args['neg_indiv']*args['zero_indiv']==0
    # assert args['pos_pair']*args['neg_pair']==0

    num_players = args['num_players']
    # NOTE: JIANHONG CHANGE 23/03/2024
    if args["intersection_generalization"]:
        env = AsyncVectorEnv([make_predefined_agent_type_env('Adhoc-wolfpack-v5', i, num_players,
                    args['seed'], args['close_penalty'], implicit_max_player_num=num_players_train, predefined_agent_types=[GreedyPredatorAgent, GreedyProbabilisticAgent,
                          TeammateAwarePredator, DistilledCoopAStarAgent, GraphDQNAgent]) 
                for i in range(args['num_envs'])])
    # NOTE: JIANHONG CHANGE 24/03/2024
    elif args["exclusion_generalization"]:
        env = AsyncVectorEnv([make_predefined_agent_type_env('Adhoc-wolfpack-v5', i, num_players,
                    args['seed'], args['close_penalty'], implicit_max_player_num=num_players_train, predefined_agent_types=[GreedyPredatorAgent, GreedyProbabilisticAgent,
                          TeammateAwarePredator, DistilledCoopAStarAgent, GraphDQNAgent]) 
                for i in range(args['num_envs'])])
    else:
        env = AsyncVectorEnv([make_env('Adhoc-wolfpack-v5', i, num_players,
                        args['seed'], args['close_penalty'], implicit_max_player_num=num_players_train) for i in range(args['num_envs'])])

    # Save initial model parameters.
    # save_dirs = os.path.join(directory, 'params_0')
    save_dirs = os.path.join(directory_params, 'params_0')
    agent.save_parameters(save_dirs)

    # Evaluate initial model performance in training environment
    avgs = []
    for ep_val_num in range(args['eval_eps']):
        num_players = args['num_players']
        agent.reset()
        steps = 0
        avg_total_rewards = 0.0
        # NOTE: JIANHONG CHANGE 23/03/2024
        if args["intersection_generalization"]:
            env_eval = AsyncVectorEnv([make_predefined_agent_type_env('Adhoc-wolfpack-v5', i, num_players,
                                  2000, args['close_penalty'], implicit_max_player_num=num_players_train, predefined_agent_types=[GraphDQNAgent, RandomAgent, GreedyWaitingAgent, GreedyProbabilisticWaitingAgent, 
                          TeammateAwareWaitingAgent]) 
                    for i in range(args['num_envs'])])
        # NOTE: JIANHONG CHANGE 24/03/2024
        elif args["exclusion_generalization"]:
            env_eval = AsyncVectorEnv([make_predefined_agent_type_env('Adhoc-wolfpack-v5', i, num_players,
                                  2000, args['close_penalty'], implicit_max_player_num=num_players_train, predefined_agent_types=[RandomAgent, GreedyWaitingAgent, GreedyProbabilisticWaitingAgent, 
                          TeammateAwareWaitingAgent]) 
                    for i in range(args['num_envs'])])
        else:
            env_eval = AsyncVectorEnv([make_env('Adhoc-wolfpack-v5', i, num_players,
                                    2000, args['close_penalty'], implicit_max_player_num=num_players_train) for i in range(args['num_envs'])])

        f_done = False
        obs = env_eval.reset()

        while not f_done:
            acts = agent.step(obs, eval=True)
            n_obs, rewards, dones, info = env_eval.step(acts)
            avg_total_rewards += (sum(rewards) + 0.0) / len(rewards)
            f_done = any(dones)
            obs = n_obs
        avgs.append(avg_total_rewards)
        print("Finished eval with rewards " + str(avg_total_rewards))
    env_eval.close()
    writer.add_scalar('Rewards/train_set', sum(avgs) / len(avgs),0)

    # Evaluate initial model performance in test environment
    avgs = []
    for ep_val_num in range(args['eval_eps']):
        num_players = args['num_players']
        agent.reset()
        steps = 0
        avg_total_rewards = 0.0
        # NOTE: JIANHONG CHANGE 23/03/2024
        if args["intersection_generalization"]:
            env_eval = AsyncVectorEnv([make_predefined_agent_type_env('Adhoc-wolfpack-v5', i, num_players,
                                  2000, args['close_penalty'], implicit_max_player_num=num_players_test, predefined_agent_types=[GraphDQNAgent, RandomAgent, GreedyWaitingAgent, GreedyProbabilisticWaitingAgent, 
                          TeammateAwareWaitingAgent]) 
                    for i in range(args['num_envs'])])
        # NOTE: JIANHONG CHANGE 24/03/2024
        elif args["exclusion_generalization"]:
            env_eval = AsyncVectorEnv([make_predefined_agent_type_env('Adhoc-wolfpack-v5', i, num_players,
                                  2000, args['close_penalty'], implicit_max_player_num=num_players_test, predefined_agent_types=[RandomAgent, GreedyWaitingAgent, GreedyProbabilisticWaitingAgent, 
                          TeammateAwareWaitingAgent]) 
                    for i in range(args['num_envs'])])
        else:
            env_eval = AsyncVectorEnv([make_env('Adhoc-wolfpack-v5', i, num_players,
                                    2000, args['close_penalty'], implicit_max_player_num=num_players_test) for i in range(args['num_envs'])])

        f_done = False
        obs = env_eval.reset()

        while not f_done:
            acts = agent.step(obs, eval=True)
            n_obs, rewards, dones, info = env_eval.step(acts)
            # print(info)
            avg_total_rewards += (sum(rewards) + 0.0) / len(rewards)
            f_done = any(dones)
            obs = n_obs
        avgs.append(avg_total_rewards)
        print("Finished eval with rewards " + str(avg_total_rewards))
    env_eval.close()
    writer.add_scalar('Rewards/eval', sum(avgs) / len(avgs),0)

    # Agent training loop
    train_duration = 0
    for ep_num in range(args['num_episodes']):
        ep_time = time.time()
        train_avgs = 0
        steps = 0
        f_done = False

        # Reset agent hidden vectors at the beginning of each episode.
        agent.reset()
        agent.set_epsilon(max(1.0 - ((ep_num + 0.0) / 1500) * 0.95, 0.05))
        obs = env.reset()
        agent.compute_target(None, None, None, None, obs, add_storage=False)

        count = 1
        while not f_done:
            count += 1
            acts = agent.step(obs)
            n_obs, rewards, dones, info = env.step(acts)
            f_done = any(dones)

            n_obs_replaced = n_obs
            if f_done:
                n_obs_replaced = copy.deepcopy(n_obs)
                for key in n_obs_replaced.keys():
                    for idx in range(len(n_obs_replaced[key])):
                        if dones[idx]:
                            n_obs_replaced[key][idx] = info[idx]['terminal_observation'][key]
            steps += 1

            train_avgs += (sum(rewards) + 0.0) / len(rewards)
            agent.compute_target(obs, acts, rewards, dones, n_obs_replaced, add_storage=True)

            if steps % args['update_frequency'] == 0 or f_done:
                agent.update()

            obs = n_obs

        writer.add_scalar('Rewards/train', train_avgs, ep_num)

        # Checkpoint and evaluate agents every few episodes.
        if (ep_num + 1) % args['saving_frequency'] == 0:
            # save_dirs = os.path.join(directory, 'params_'+str((ep_num +
            #                                                    1) // args['saving_frequency']))
            save_dirs = os.path.join(directory_params, 'params_'+str((ep_num + 1) // args['saving_frequency']))
            agent.save_parameters(save_dirs)

            # Run evaluation in training environment.
            avgs = []
            for ep_val_num in range(args['eval_eps']):
                num_players = args['num_players']
                agent.reset()
                steps = 0
                avg_total_rewards = 0.0
                # NOTE: JIANHONG CHANGE 23/03/2024
                if args["intersection_generalization"]:
                    env_eval = AsyncVectorEnv([make_predefined_agent_type_env('Adhoc-wolfpack-v5', i, num_players,
                                        2000, args['close_penalty'], implicit_max_player_num=num_players_train, predefined_agent_types=[GraphDQNAgent, RandomAgent, GreedyWaitingAgent, GreedyProbabilisticWaitingAgent, 
                                TeammateAwareWaitingAgent]) 
                            for i in range(args['num_envs'])])
                # NOTE: JIANHONG CHANGE 24/03/2024
                elif args["exclusion_generalization"]:
                    env_eval = AsyncVectorEnv([make_predefined_agent_type_env('Adhoc-wolfpack-v5', i, num_players,
                                        2000, args['close_penalty'], implicit_max_player_num=num_players_train, predefined_agent_types=[RandomAgent, GreedyWaitingAgent, GreedyProbabilisticWaitingAgent, 
                                TeammateAwareWaitingAgent]) 
                            for i in range(args['num_envs'])])
                else:
                    env_eval = AsyncVectorEnv([make_env('Adhoc-wolfpack-v5', i, num_players,
                                    2000, args['close_penalty'], implicit_max_player_num=num_players_train) for i in range(args['num_envs'])])

                f_done = False
                obs = env_eval.reset()

                while not f_done:
                    acts = agent.step(obs, eval=True)
                    n_obs, rewards, dones, info = env_eval.step(acts)
                    avg_total_rewards += (sum(rewards) + 0.0) / len(rewards)
                    f_done = any(dones)
                    obs = n_obs
                avgs.append(avg_total_rewards)
                print("Finished eval with rewards " + str(avg_total_rewards))
            env_eval.close()
            print("Training env setting result:{}".format(sum(avgs) / len(avgs)))
            writer.add_scalar('Rewards/train_set', sum(avgs) / len(avgs),
                              (ep_num + 1) // args['saving_frequency'])

            # Run evaluation in testing environment.
            avgs = []
            for ep_val_num in range(args['eval_eps']):
                num_players = args['num_players']
                agent.reset()
                steps = 0
                avg_total_rewards = 0.0
                # NOTE: JIANHONG CHANGE 23/03/2024
                if args["intersection_generalization"]:
                    env_eval = AsyncVectorEnv([make_predefined_agent_type_env('Adhoc-wolfpack-v5', i, num_players,
                                        2000, args['close_penalty'], implicit_max_player_num=num_players_test, predefined_agent_types=[GraphDQNAgent, RandomAgent, GreedyWaitingAgent, GreedyProbabilisticWaitingAgent, 
                                TeammateAwareWaitingAgent]) 
                            for i in range(args['num_envs'])])
                # NOTE: JIANHONG CHANGE 24/03/2024
                elif args["exclusion_generalization"]:
                    env_eval = AsyncVectorEnv([make_predefined_agent_type_env('Adhoc-wolfpack-v5', i, num_players,
                                        2000, args['close_penalty'], implicit_max_player_num=num_players_test, predefined_agent_types=[RandomAgent, GreedyWaitingAgent, GreedyProbabilisticWaitingAgent, 
                                TeammateAwareWaitingAgent]) 
                            for i in range(args['num_envs'])])
                else:
                    env_eval = AsyncVectorEnv([make_env('Adhoc-wolfpack-v5', i, num_players,
                                    2000, args['close_penalty'], implicit_max_player_num=num_players_test) for i in range(args['num_envs'])])

                f_done = False
                obs = env_eval.reset()

                while not f_done:
                    acts = agent.step(obs, eval=True)
                    n_obs, rewards, dones, info = env_eval.step(acts)
                    avg_total_rewards += (sum(rewards) + 0.0) / len(rewards)
                    f_done = any(dones)
                    obs = n_obs
                avgs.append(avg_total_rewards)
                print("Finished eval with rewards " + str(avg_total_rewards))
            env_eval.close()
            print("Test env setting result:{}".format(sum(avgs) / len(avgs)))
            writer.add_scalar('Rewards/eval', sum(avgs) / len(avgs),
                              (ep_num + 1) // args['saving_frequency'])

        ep_duration = time.time() - ep_time
        train_duration += ep_duration
        if ep_num % 50 == 0:
            print("iter {}/{} costs {}s, all training costs {}s, average {}s".format(ep_num, args['num_episodes'], ep_duration,train_duration, train_duration/(ep_num+1)))
        writer.add_scalar('Times/train_single', ep_duration, ep_num)
        writer.add_scalar('Times/train_sum', train_duration, ep_num)
        writer.add_scalar('Times/train_avg', train_duration/(ep_num+1), ep_num)
        

    end_time = time.time()
    print("using {}/s".format(end_time-start_time))
    
