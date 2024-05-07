import argparse
import gym
import random
import lbforaging
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
parser.add_argument('--lr', type=float, default=0.00025, help='Learning rate.')
parser.add_argument('--gamma', type=float, default=0.99, help='Discount_rate.')
parser.add_argument('--max_num_steps', type=int, default=1000000, help="Number of episodes for training.")
parser.add_argument('--eps_length', type=int, default=200, help="Episode length for training.")
parser.add_argument('--update_frequency', type=int, default=4, help="Timesteps between updates.")
parser.add_argument('--saving_frequency', type=int,default=50,help="Number of episodes between checkpoints.")
parser.add_argument('--num_envs', type=int,default=16, help="Number of parallel environments for training.")
parser.add_argument('--tau', type=float,default=0.001, help="Tau for soft target update.")
parser.add_argument('--eval_eps', type=int, default=5, help="Number of episodes for evaluation.")
parser.add_argument('--weight_predict', type=float, default=1.0, help="Weight associated to action prediction loss.")
parser.add_argument('--save_dir', type=str, default='.', help="save dir name")
parser.add_argument('--num_players_train', type=int, default=3, help="Maximum number of players for training.")
parser.add_argument('--num_players_test', type=int, default=5, help="Maximum number of players for testing.")
parser.add_argument('--pair_comp', type=str, default='bmm', help="Pairwise factor computation method. Use bmm for low rank factorization.")
parser.add_argument('--info', type=str, default="", help="Additional info.")
parser.add_argument('--seed', type=int, default=0, help="Training seed.")
parser.add_argument('--eval_init_seed', type=int, default=2500, help="Evaluation seed")
parser.add_argument('--note', type=str, required=True, help="Please note the version of you this trial")
# parser.add_argument('--training_mode', type=str, default='no_gnn', help='Please input the training mode')
parser.add_argument('--act_func', type=str, default='relu', help='Please input the activation function for the hidden layers, e.g. relu or leaky_relu')
parser.add_argument('--seq_model', type=str, default='lstm', help='Please input the sequential model used for both utility model and agent model, e.g. gru or lstm')
# TODO: JIANHONG
# parser.add_argument('--star_graph', action='store_true', help='with star graph for gnn of policy')
parser.add_argument('--graph', type=str, default="star", help='input the dynamic affinity graph structure: star or complete.')
parser.add_argument('--pair_range', type=str, default="pos", help='input the pairwise utility range: pos, neg, or free.')
parser.add_argument('--indiv_range', type=str, default="pos", help='input the individual utility range: pos, neg, zero, or free.')
# parser.add_argument('--pos_pair', action='store_true', help='with positive paired q-values')
# parser.add_argument('--pos_indiv', action='store_true', help='with positive indiv q-values')
# parser.add_argument('--learner_type', action='store_true', help='Whether the Q value is with learner type as input')
# parser.add_argument('--neg_pair', action='store_true', help='with negative paired q-values')
# parser.add_argument('--neg_indiv', action='store_true', help='with negative indiv q-values')
# parser.add_argument('--zero_indiv', action='store_true', help='with zero indiv q-values')
# NOTE: JIANHONG CHANGE 01/01/2024
# parser.add_argument('--weight_learner', type=float, default=1.0, help="weight associated to individual values regularizer")
parser.add_argument('--weight_regularizer', type=float, default=0.5, help="weight associated to individual values regularizer")
# NOTE: JIANHONG CHANGE 04/01/2024
parser.add_argument('--exclude_A2Cagent', action='store_true', help="determine if A2C agent is excluded in the experiment")
# NOTE: JIANHONG CHANGE 06/01/2024
# parser.add_argument('--equal_indiv', action='store_true', help="determine if the regularizer for individual values are forcing them equal")
# parser.add_argument('--indiv_reg_type', action='store_true', help="determine if the regularizer for individual values are forcing them equal")
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
    agent = MRFAgent(args=args, writer=writer, added_u_dim = 9)

    # Define the training environment
    num_players_train = args['num_players_train']
    num_players_test = args['num_players_test']

    # check whether both pos_indiv,neg_indiv and zero_indiv / pos_pair and neg_pair are true
    # assert args['pos_indiv']*args['neg_indiv']==0 and args['pos_indiv']*args['zero_indiv']==0 and args['neg_indiv']*args['zero_indiv']==0
    # assert args['pos_pair']*args['neg_pair']==0

    def make_env(env_id, rank, seed=1285, effective_max_num_players=3, with_shuffle=False, gnn_input=True):
        def _init():
            # env = gym.make(
            #     env_id, seed=seed + rank,
		    #     players=num_players_test,
            #     effective_max_num_players=effective_max_num_players,
            #     init_num_players=effective_max_num_players,
            #     with_shuffle=with_shuffle,
            #     gnn_input=gnn_input
            # )
            # NOTE: JIANHONG CHANGE 04/01/2024
            env = gym.make(
                env_id, seed=seed + rank,
		        players=num_players_test,
                effective_max_num_players=effective_max_num_players,
                init_num_players=effective_max_num_players,
                with_shuffle=with_shuffle,
                gnn_input=gnn_input,
                exclude_A2Cagent=args['exclude_A2Cagent']
            )
            return env

        return _init
    
    # NOTE: JIANHONG CHANGE 23/03/2024
    def make_predefined_agent_type_env(env_id, rank, seed=1285, effective_max_num_players=3, with_shuffle=False, gnn_input=True, predefined_agent_types=[]):
        def _init():
            env = gym.make(
                env_id, seed=seed + rank,
		        players=num_players_test,
                effective_max_num_players=effective_max_num_players,
                init_num_players=effective_max_num_players,
                with_shuffle=with_shuffle,
                gnn_input=gnn_input,
                predefined_agent_types=predefined_agent_types
            )
            return env

        return _init

    # NOTE: JIANHONG CHANGE 23/03/2024
    if args['intersection_generalization']:
        env = AsyncVectorEnv(
            [make_predefined_agent_type_env('Adhoc-Foraging-8x8-3f-v0', i,
                                args['seed'], num_players_train, False, True, ["H8", "H7", "H6", "H5", "A2C0"])
            for i in range(args['num_envs'])]
        )
    # NOTE: JIANHONG CHANGE 24/03/2024
    elif args['exclusion_generalization']:
        env = AsyncVectorEnv(
            [make_predefined_agent_type_env('Adhoc-Foraging-8x8-3f-v0', i,
                                args['seed'], num_players_train, False, True, ["H8", "H7", "H6", "H5", "A2C0"])
            for i in range(args['num_envs'])]
        )
    else:
        env = AsyncVectorEnv(
            [make_env('Adhoc-Foraging-8x8-3f-v0', i,
                    args['seed'], num_players_train, False, True)
            for i in range(args['num_envs'])]
        )

    # Save initial model parameters.
    # save_dirs = os.path.join(directory, 'params_0')
    save_dirs = os.path.join(directory_params, 'params_0')
    agent.save_parameters(save_dirs)

    # Evaluate initial model performance in training environment
    avgs = []
    num_dones, per_worker_rew = [0] * args['num_envs'], [0] * args['num_envs']
    agent.reset()
    # NOTE: JIANHONG CHANGE 23/03/2024
    if args['intersection_generalization']:
        env_eval = AsyncVectorEnv(
            [make_predefined_agent_type_env('Adhoc-Foraging-8x8-3f-v0', i,
                                args['seed'], num_players_train, False, True, ["A2C0", "H1", "H2", "H3", "H4"])
            for i in range(args['num_envs'])]
        )
    # NOTE: JIANHONG CHANGE 24/03/2024
    elif args['exclusion_generalization']:
        env_eval = AsyncVectorEnv(
            [make_predefined_agent_type_env('Adhoc-Foraging-8x8-3f-v0', i,
                                args['seed'], num_players_train, False, True, ["H1", "H2", "H3", "H4"])
            for i in range(args['num_envs'])]
        )
    else:
        env_eval = AsyncVectorEnv(
            [make_env('Adhoc-Foraging-8x8-3f-v0', i,
                    args['eval_init_seed'], num_players_train, False, True)
            for i in range(args['num_envs'])]
        )

    obs = env_eval.reset()
    while (all([k < args['eval_eps'] for k in num_dones])):
        acts = agent.step(obs, eval=True)
        n_obs, rewards, dones, info = env_eval.step(acts)
        per_worker_rew = [k + l for k, l in zip(per_worker_rew, rewards)]
        obs = n_obs
        for idx, flag in enumerate(dones):
            if flag:
                if num_dones[idx] < args['eval_eps']:
                    num_dones[idx] += 1
                    avgs.append(per_worker_rew[idx])
                per_worker_rew[idx] = 0

    avg_total_rewards = (sum(avgs) + 0.0) / len(avgs)
    print("Finished eval with rewards " + str(avg_total_rewards))
    env_eval.close()
    writer.add_scalar('Rewards/train_set', sum(avgs) / len(avgs), 0)

    # Evaluate initial model performance in test environment
    avgs = []
    num_dones, per_worker_rew = [0] * args['num_envs'], [0] * args['num_envs']
    agent.reset()
    # NOTE: JIANHONG CHANGE 23/03/2024
    if args['intersection_generalization']:
        env_eval = AsyncVectorEnv(
            [make_predefined_agent_type_env('Adhoc-Foraging-8x8-3f-v0', i,
                                args['seed'], num_players_test, False, True, ["A2C0", "H1", "H2", "H3", "H4"])
            for i in range(args['num_envs'])]
        )
    # NOTE: JIANHONG CHANGE 24/03/2024
    elif args['exclusion_generalization']:
        env_eval = AsyncVectorEnv(
            [make_predefined_agent_type_env('Adhoc-Foraging-8x8-3f-v0', i,
                                args['seed'], num_players_test, False, True, ["H1", "H2", "H3", "H4"])
            for i in range(args['num_envs'])]
        )
    else:
        env_eval = AsyncVectorEnv(
            [make_env('Adhoc-Foraging-8x8-3f-v0', i,
                    args['eval_init_seed'], num_players_test, False, True)
            for i in range(args['num_envs'])]
        )

    obs = env_eval.reset()
    while (all([k < args['eval_eps'] for k in num_dones])):
        acts = agent.step(obs, eval=True)
        n_obs, rewards, dones, info = env_eval.step(acts)
        per_worker_rew = [k + l for k, l in zip(per_worker_rew, rewards)]
        obs = n_obs

        for idx, flag in enumerate(dones):
            if flag:
                if num_dones[idx] < args['eval_eps']:
                    num_dones[idx] += 1
                    avgs.append(per_worker_rew[idx])
                per_worker_rew[idx] = 0

    avg_total_rewards = (sum(avgs) + 0.0) / len(avgs)
    print("Finished eval with rewards " + str(avg_total_rewards))
    env_eval.close()
    writer.add_scalar('Rewards/eval', sum(avgs) / len(avgs), 0)

    # Agent training loop
    train_duration = 0
    num_episode = args["max_num_steps"] // args["eps_length"]
    for ep_num in range(num_episode):
        ep_time = time.time()

        # Store performance stats during training
        avgs = []
        num_dones, per_worker_rew = [0] * args['num_envs'], [0] * args['num_envs']

        # Reset agent hidden vectors at the beginning of each episode.
        obs = env.reset()
        agent.reset()
        agent.set_epsilon(max(1.0 - ((ep_num + 0.0) / 1500) * 0.95, 0.05))
        agent.compute_target(None, None, None, None, obs, add_storage=False)
        steps = 0

        while steps < args["eps_length"]:
            acts = agent.step(obs)
            n_obs, rewards, dones, info = env.step(acts)
            per_worker_rew = [k + l for k, l in zip(per_worker_rew, rewards)]
            agent.compute_target(obs, acts, rewards, dones, n_obs, add_storage=True)
            obs = n_obs

            # Compute updated reward
            for idx, flag in enumerate(dones):
                if flag:
                    num_dones[idx] += 1
                    avgs.append(per_worker_rew[idx])
                    per_worker_rew[idx] = 0

            steps += 1
            if steps % args['update_frequency'] == 0:
                agent.update()

        train_avgs = (sum(avgs) + 0.0) / len(avgs) if len(avgs) != 0 else 0.0
        writer.add_scalar('Rewards/train', train_avgs, ep_num)

        # Checkpoint and evaluate agents every few episodes.
        if (ep_num + 1) % args['saving_frequency'] == 0:
            # save_dirs = os.path.join(directory, 'params_'+str((ep_num +
            #                                                    1) // args['saving_frequency']))
            save_dirs = os.path.join(directory_params, 'params_'+str((ep_num + 1) // args['saving_frequency']))
            agent.save_parameters(save_dirs)

            # Run evaluation in training environment.
            avgs = []
            num_dones, per_worker_rew = [0] * args['num_envs'], [0] * args['num_envs']
            agent.reset()
            # NOTE: JIANHONG CHANGE 23/03/2024
            if args['intersection_generalization']:
                env_eval = AsyncVectorEnv(
                    [make_predefined_agent_type_env('Adhoc-Foraging-8x8-3f-v0', i,
                                        args['seed'], num_players_train, False, True, ["A2C0", "H1", "H2", "H3", "H4"])
                    for i in range(args['num_envs'])]
                )
            # NOTE: JIANHONG CHANGE 24/03/2024
            elif args['exclusion_generalization']:
                env_eval = AsyncVectorEnv(
                    [make_predefined_agent_type_env('Adhoc-Foraging-8x8-3f-v0', i,
                                        args['seed'], num_players_train, False, True, ["H1", "H2", "H3", "H4"])
                    for i in range(args['num_envs'])]
                )
            else:
                env_eval = AsyncVectorEnv(
                    [make_env('Adhoc-Foraging-8x8-3f-v0', i,
                            args['eval_init_seed'], num_players_train, False, True)
                    for i in range(args['num_envs'])]
                )

            obs = env_eval.reset()
            while (all([k < args['eval_eps'] for k in num_dones])):
                acts = agent.step(obs, eval=True)
                n_obs, rewards, dones, info = env_eval.step(acts)
                per_worker_rew = [k + l for k, l in zip(per_worker_rew, rewards)]
                obs = n_obs

                for idx, flag in enumerate(dones):
                    if flag:
                        if num_dones[idx] < args['eval_eps']:
                            num_dones[idx] += 1
                            avgs.append(per_worker_rew[idx])
                        per_worker_rew[idx] = 0

            avg_total_rewards = (sum(avgs) + 0.0) / len(avgs)
            # print("Finished eval with rewards " + str(avg_total_rewards))
            print("Training env setting eval with rewards " + str(avg_total_rewards))
            env_eval.close()
            writer.add_scalar('Rewards/train_set', sum(avgs) / len(avgs),
                              (ep_num + 1) // args['saving_frequency'])


            # Run evaluation in testing environment
            avgs = []
            num_dones, per_worker_rew = [0] * args['num_envs'], [0] * args['num_envs']
            agent.reset()
            # NOTE: JIANHONG CHANGE 23/03/2024
            if args['intersection_generalization']:
                env_eval = AsyncVectorEnv(
                    [make_predefined_agent_type_env('Adhoc-Foraging-8x8-3f-v0', i,
                                        args['seed'], num_players_test, False, True, ["A2C0", "H1", "H2", "H3", "H4"])
                    for i in range(args['num_envs'])]
                )
            # NOTE: JIANHONG CHANGE 23/03/2024
            elif args['exclusion_generalization']:
                env_eval = AsyncVectorEnv(
                    [make_predefined_agent_type_env('Adhoc-Foraging-8x8-3f-v0', i,
                                        args['seed'], num_players_test, False, True, ["H1", "H2", "H3", "H4"])
                    for i in range(args['num_envs'])]
                )
            else:
                env_eval = AsyncVectorEnv(
                    [make_env('Adhoc-Foraging-8x8-3f-v0', i,
                            args['eval_init_seed'], num_players_test, False, True)
                    for i in range(args['num_envs'])]
                )

            obs = env_eval.reset()
            while (all([k < args['eval_eps'] for k in num_dones])):
                acts = agent.step(obs, eval=True)
                n_obs, rewards, dones, info = env_eval.step(acts)
                per_worker_rew = [k + l for k, l in zip(per_worker_rew, rewards)]
                obs = n_obs

                for idx, flag in enumerate(dones):
                    if flag:
                        if num_dones[idx] < args['eval_eps']:
                            num_dones[idx] += 1
                            avgs.append(per_worker_rew[idx])
                        per_worker_rew[idx] = 0

            avg_total_rewards = (sum(avgs) + 0.0) / len(avgs)
            # print("Finished eval with rewards " + str(avg_total_rewards))
            print("Testing env setting eval with rewards " + str(avg_total_rewards))
            env_eval.close()
            writer.add_scalar('Rewards/eval', sum(avgs) / len(avgs),
                              (ep_num + 1) // args['saving_frequency'])
        
        ep_duration = time.time() - ep_time
        train_duration += ep_duration
        if ep_num % 50 == 0:
            print("iter {}/{} costs {}s, all training costs {}s, average {}s".format(ep_num, num_episode, ep_duration,train_duration, train_duration/(ep_num+1)))
        writer.add_scalar('Times/train_single', ep_duration, ep_num)
        writer.add_scalar('Times/train_sum', train_duration, ep_num)
        writer.add_scalar('Times/train_avg', train_duration/(ep_num+1), ep_num)


    end_time = time.time()
    print("using {}/s".format(end_time-start_time))
    
