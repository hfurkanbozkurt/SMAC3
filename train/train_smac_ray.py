import datetime
import json
import logging
import os
import tempfile
from functools import partial
from pprint import pprint

import numpy as np
import ray
from ray.experimental import (flush_evicted_objects_unsafe,
                              flush_finished_tasks_unsafe, flush_redis_unsafe,
                              flush_task_and_object_metadata_unsafe)
from ray.rllib.agents import dqn as ray_dqn
from ray.rllib.agents.registry import get_agent_class
from ray.tune import function
from ray.tune.logger import UnifiedLogger, pretty_print
from ray.tune.registry import register_env

from smac.facade.smac_rl import RLSMAC as SMAC
from train_smac import EpisodeStats


def eval_ray_dqn(agent, environment):
    s = environment.reset()  # Need to parse to string to easily handle list as state with defaultdict
    episode_length, cummulative_reward = 0, 0
    # _, _, policy_result = agent.local_evaluator.for_policy(
    #     lambda p: p.compute_single_action(s, []),
    #     policy_id='default')
    expected_reward = None
    while True:  # roll out episode
        a = agent.compute_action(s)
        s_, r, done, _ = environment.step(a)
        cummulative_reward += r
        episode_length += 1
        if done:
            break
        s = s_
    return cummulative_reward


def logger_creator(config, model='PPO', adp='adp1', seed=0):
    """Creates a Unified logger with a logdir prefix
    containing the agent name and the env id
    """

    timestr = datetime.datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    logdir_prefix = '_'.join([model, adp, timestr, str(seed)])

    if not os.path.exists(logdir_prefix):
        os.makedirs(logdir_prefix)
    logdir = tempfile.mkdtemp(
        prefix=logdir_prefix, dir=logdir_prefix)
    return UnifiedLogger(config, logdir, None)


def setup_ray(args, env_choice):
    ray.init(ignore_reinit_error=True, num_cpus=1, num_gpus=0, object_store_memory=int(4e+9),
             temp_dir='./ray_tmp_{}'.format(str(datetime.datetime.now()).replace(" ", "_")))

    register_env('smac_env', env_choice.env_creator)
    ray_conf = ray_dqn.DEFAULT_CONFIG.copy()
    test_stats = EpisodeStats(
        episode_lengths=np.zeros(args.num_episodes),
        episode_rewards=[],
        expected_rewards=np.zeros(args.num_episodes))

    def on_episode_end(info):
        episode = info["episode"]
        inc_perf = episode.last_info_for('agent0')["perf_^"]
        episode.custom_metrics["inc_perf"] = inc_perf

    h = args.horizon // args.act_repeat
    ray_conf['hiddens'] = [100, 100]
    ray_conf['gamma'] = args.discount_factor
    ray_conf['lr'] = args.lr
    ray_conf['train_batch_size'] = h * 4
    ray_conf['sample_batch_size'] = h
    ray_conf['timesteps_per_iteration'] = h
    ray_conf['min_iter_time_s'] = 0
    ray_conf['target_network_update_freq'] = h * 5
    ray_conf['num_workers'] = 1
    ray_conf['train_batch_size'] = h
    ray_conf['sample_batch_size'] = h
    ray_conf['schedule_max_timesteps'] = args.num_episodes * h
    ray_conf['exploration_fraction'] = 0.25
    ray_conf['learning_starts'] = args.eps_decay_starts
    ray_conf['horizon'] = h + 1
    ray_conf['evaluation_interval'] = 10
    ray_conf['evaluation_num_episodes'] = 1
    ray_conf['callbacks'] = {
        "on_episode_end": function(on_episode_end)
    }
    return ray_conf, test_stats


def ray_dqn_learn(num_eps, agent, c_freq=10):
    total_eps = 0
    rewards = []
    inc_perfs = []
    while total_eps <= num_eps:
        print("{}/{}".format(total_eps, num_eps))
        train_result = agent.train()
        total_eps += train_result['episodes_this_iter']
        rewards.append(train_result["episode_reward_mean"])
        inc_perfs.append(train_result["custom_metrics"]["inc_perf_mean"])
        if total_eps % c_freq == 0:
            pprint(train_result)
            agent.save()
    stats = {"rewards": rewards, "inc_perfs": inc_perfs}
    return stats


def restore(param_path, checkpoint_path, env_id):
    env_choice = {
        "smac": SMAC}
    ray.init(ignore_reinit_error=True, num_cpus=1, num_gpus=0, object_store_memory=int(4e+9))

    register_env('smac_env', env_choice[env_id].env_creator)

    with open(param_path) as fh:
        config = json.load(fh)
    config['evaluation_interval'] = None
    agent_class = get_agent_class('DQN')
    agent = agent_class(env='smac_env', config=config)
    agent.restore(checkpoint_path)
    return agent, config['env_config']
