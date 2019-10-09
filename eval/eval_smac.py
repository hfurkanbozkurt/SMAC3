import argparse
import datetime
import json
import logging
import os
import pickle
import tempfile
from functools import partial
from pprint import pprint
import glob

import numpy as np
import ray
from ray.rllib.agents import dqn as ray_dqn
from ray.rllib.agents.registry import get_agent_class
from ray.tune.registry import register_env

from smac.facade.smac_rl import RLSMAC


def restore_deep_q_agent(config, checkpoint_path):
    config['evaluation_interval'] = None
    agent_class = get_agent_class('DQN')
    agent = agent_class(env='smac_env', config=config)
    agent.restore(checkpoint_path)
    return agent, config['env_config']


def eval_agent_on_env(agent, env_config, num_episodes, fn):
    env = RLSMAC.env_creator(env_config)
    rewards = np.array([0.0 for _ in range(num_episodes)])
    inc_perfs = np.array([None for _ in range(num_episodes)])
    actions = np.array([[] for _ in range(num_episodes)])
    for i in range(num_episodes):
        ob = env.reset()
        while True:
            act = agent.compute_action(ob)
            ob, r, done, info = env.step(act)
            rewards[i] += r
            inc_perfs[i] = info["perf_^"]
            actions[i].append(act)
            if done:
                break

    stats = {
        "rewards": rewards,
        "inc_perfs": inc_perfs,
        "actions": actions,
        "config": env_config
    }
    with open(fn, 'wb') as fh:
        pickle.dump(stats, fh)


def run_on_checkpoint(config, checkpoint_path, num_episodes, fn):
    agent, env_config = restore_deep_q_agent(config, checkpoint_path)
    eval_agent_on_env(agent, env_config, num_episodes, fn)


def run_on_folder(path, num_episodes):
    param_path = os.path.join(path, "params.json")
    with open(param_path) as fh:
        config = json.load(fh)

    agent = f"deep_q_{config['lr'].replace('.', '_')}"
    bench = config["env_config"]["bench"]
    for checkpoint_path in glob.glob(os.path.join(path, "checkpoint_*")):
        ch = checkpoint_path.split("checkpoint_")[-1]
        t = str(datetime.datetime.now()).replace(" ", "_")
        fn = f"{agent}-{bench}-ch_{ch}-{t}.pkl"
        run_on_checkpoint(config, checkpoint_path, num_episodes, fn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Evaluate Deep Q-Learning Agent')
    parser.add_argument('--path', default=".", type=str)
    parser.add_argument('--num-episodes', default=10, type=int)
    args = parser.parse_args()

    ray.init(ignore_reinit_error=True, num_cpus=1, num_gpus=0, object_store_memory=int(4e+9))
    register_env('smac_env', RLSMAC.env_creator)

    for path in glob.glob(os.path.join(args.path, "**", "params.json"), recursive=True):
        path = os.path.dirname(path)
        run_on_folder(path, args.num_episodes)



