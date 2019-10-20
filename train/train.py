import argparse
import copy
import datetime
import glob
import logging
import os
import pickle
from pprint import pprint

import numpy as np
import ray
from ray.rllib.agents import dqn as ray_dqn
from ray.tune import function
from ray.tune.logger import UnifiedLogger, pretty_print
from ray.tune.registry import register_env

from smac.facade.smac_rl import RLSMAC

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Tabular Q-learning')
    parser.add_argument('--agent',
                        choices=["deep_q", "default_smac", "random"],
                        default="tabular_q")
    parser.add_argument('--num_epi',
                        help='Number of training episodes to roll out',
                        required=False,
                        type=int)
    parser.add_argument('--eval_num_epi',
                        help='Number of evaluation episodes to roll out',
                        required=False,
                        type=int)
    parser.add_argument('--c_dir',
                        help='Checkpoint directory')

    parser = RLSMAC.add_rlsmac_arguments(parser)
    args = parser.parse_args()
    env_config = RLSMAC.read_rlsmac_arguments(args)

    dt = datetime.datetime.now()
    t = str(dt).replace(" ", "_")
    if not os.path.exists(args.c_dir):
        os.makedirs(args.c_dir)

    if args.agent == "deep_q":
        ray.init(
            ignore_reinit_error=True,
            num_cpus=1,
            num_gpus=0,
            object_store_memory=int(4e+9),
            temp_dir='./ray_tmp_{}'.format(t)
        )
        register_env('smac_env', RLSMAC.env_creator)

        def on_episode_end(info):
            episode = info["episode"]
            inc_perf = episode.last_info_for('agent0')["perf_^"]
            episode.custom_metrics["inc_perf"] = inc_perf

        config = ray_dqn.DEFAULT_CONFIG.copy()
        config = {
            'env_config': env_config,
            'target_network_update_freq': 1000,
            'adam_epsilon': .00001,
            'learning_starts': 10000,
            'buffer_size': 100000,
            'schedule_max_timesteps': args.num_epi * (args.horizon // args.act_repeat),
            'timesteps_per_iteration': 50 * (args.horizon // args.act_repeat),
            'log_level': 'ERROR',
            'callbacks': {
                'on_episode_end': function(on_episode_end),
            },
        }

        print("CONFIG")
        pprint(config)

        agent = ray_dqn.DQNAgent(
            config=config,
            env='smac_env',
            logger_creator=lambda config: UnifiedLogger(config,args.c_dir)
        )
        checkpoints = sorted(
            glob.glob(os.path.join(args.c_dir, "checkpoint*")),
            key=lambda x: int(x.split("_")[-1]))
        if checkpoints:
            ch = checkpoints[-1]
            ch = os.path.join(ch, ch.split("/")[-1].replace("_", "-"))
            agent.restore(ch)
            ch_step = int(ch.split("checkpoint-")[-1])
            args.num_epi -= ch_step

        train_rewards = []
        train_inc_perfs = []
        epi = 0
        print("TRAINING")
        while epi <= args.num_epi:
            train_result = agent.train()
            epi += train_result['episodes_this_iter']
            train_rewards.append(train_result["episode_reward_mean"])
            train_inc_perfs.append(train_result["custom_metrics"]["inc_perf_mean"])
            checkpoint = agent.save(checkpoint_dir=args.c_dir)
            print(f"Epi {epi}: rew: {train_rewards[-1]}, inc_perf: {train_inc_perfs[-1]}")

        train_stats = {"rewards": train_rewards,
                       "inc_perfs": train_inc_perfs}
        with open(os.path.join(args.c_dir, f"train_stats_{t}.pkl"), 'wb') as fh:
            pickle.dump(train_stats, fh)

        env = RLSMAC.env_creator(env_config)
        print("TESTING")
        test_rewards = np.array([0.0 for _ in range(args.eval_num_epi)])
        test_inc_perfs = np.array([None for _ in range(args.eval_num_epi)])
        test_actions = [[] for _ in range(args.eval_num_epi)]
        for i in range(args.eval_num_epi):
            ob = env.reset()
            while True:
                act = agent.compute_action(ob)
                ob, r, done, info = env.step(act)
                test_rewards[i] += r
                test_inc_perfs[i] = info["perf_^"]
                test_actions[i].append(act)
                if done:
                    break
            print(f"Epi {i}: rew: {test_rewards[i]}, inc_perf: {test_inc_perfs[i]}, actions: {test_actions[i]}")

        test_stats = {"rewards": test_rewards,
                      "inc_perfs": test_inc_perfs,
                      "actions": np.array(test_actions)}
        with open(os.path.join(args.c_dir, f"test_stats_{t}.pkl"), 'wb') as fh:
            pickle.dump(test_stats, fh)

    elif args.agent == "default_smac":
        env_config['default_smac'] = True
        env = RLSMAC.env_creator(env_config)
        print("TESTING")
        test_rewards = np.array([0.0 for _ in range(args.eval_num_epi)])
        test_inc_perfs = np.array([None for _ in range(args.eval_num_epi)])
        for i in range(args.eval_num_epi):
            _ = env.reset()
            while True:
                _, r, done, info = env.step(None)
                test_rewards[i] += r
                test_inc_perfs[i] = info["perf_^"]
                if done:
                    break
            print(f"Epi {i}: rew: {test_rewards[i]}, inc_perf: {test_inc_perfs[i]}")

        test_stats = {"rewards": test_rewards,
                      "inc_perfs": test_inc_perfs}
        with open(os.path.join(args.c_dir, f"test_stats_{t}.pkl"), 'wb') as fh:
            pickle.dump(test_stats, fh)

    elif args.agent == "random":
        env = RLSMAC.env_creator(env_config)
        test_rewards = np.array([0.0 for _ in range(args.eval_num_epi)])
        test_inc_perfs = np.array([None for _ in range(args.eval_num_epi)])
        test_actions = [[] for _ in range(args.eval_num_epi)]
        print("TESTING")
        for i in range(args.eval_num_epi):
            _ = env.reset()
            while True:
                act = env.action_space.sample()
                _, r, done, info = env.step(act)
                test_rewards[i] += r
                test_inc_perfs[i] = info["perf_^"]
                test_actions[i].append(act)
                if done:
                    break
            print(f"Epi {i}: rew: {test_rewards[i]}, inc_perf: {test_inc_perfs[i]}, actions: {test_actions[i]}")

        test_stats = {"rewards": test_rewards,
                      "inc_perfs": test_inc_perfs,
                      "actions": np.array(test_actions)}
        with open(os.path.join(args.c_dir, f"test_stats_{t}.pkl"), 'wb') as fh:
            pickle.dump(test_stats, fh)
