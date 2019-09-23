import argparse
import copy
import datetime
import glob
import logging
import os
import pickle
import sys
import warnings
from collections import defaultdict, namedtuple
from functools import partial
from pprint import pprint

import numpy as np

from smac.facade.smac_rl import RLSMAC
from train_smac_ray import *


EpisodeStats = namedtuple("Stats", ["episode_lengths", "episode_rewards", "expected_rewards"])


class QTable(dict):
    def __init__(self, n_actions, float_to_int=False, **kwargs):
        """
        Look up table for state-action values.

        :param n_actions: action space size
        :param float_to_int: flag to determine if state values need to be rounded to the closest integer
        """
        super().__init__(**kwargs)
        self.n_actions = n_actions
        self.float_to_int = float_to_int
        self.__table = defaultdict(lambda: np.zeros(n_actions))

    def __getitem__(self, item):
        try:
            table_state, table_action = item
            if self.float_to_int:
                table_state = map(int, table_state)
            return self.__table[tuple(table_state)][table_action]
        except ValueError:
            if self.float_to_int:
                item = map(int, item)
            return self.__table[tuple(item)]

    def __setitem__(self, key, value):
        try:
            table_state, table_action = key
            if self.float_to_int:
                table_state = map(int, table_state)
            self.__table[tuple(table_state)][table_action] = value
        except ValueError:
            if self.float_to_int:
                key = map(int, key)
            self.__table[tuple(key)] = value

    def __contains__(self, item):
        return tuple(item) in self.__table.keys()

    def keys(self):
        return self.__table.keys()


def make_epsilon_greedy_policy(Q: QTable, epsilon: float, nA: int) -> callable:
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.
    I.e. create weight vector from which actions get sampled.

    :param Q: tabular state-action lookup function
    :param epsilon: exploration factor
    :param nA: size of action space to consider for this policy
    """

    def policy_fn(observation):
        policy = np.ones(nA) * epsilon / nA
        best_action = np.random.choice(np.argwhere(  # random choice for tie-breaking only
            Q[observation] == np.amax(Q[observation])
        ).flatten())
        policy[best_action] += (1 - epsilon)
        return policy

    return policy_fn


def get_decay_schedule(start_val: float, decay_start: int, num_episodes: int, type_: str, learning_starts: int):
    """
    Create epsilon decay schedule

    :param start_val: Start decay from this value (i.e. 1)
    :param decay_start: number of iterations to start epsilon decay after
    :param num_episodes: Total number of episodes to decay over
    :param type_: Which strategy to use. Implemented choices: 'const', 'log', 'linear'
    :param learning_starts: number of iterations to with epsilon 1.0
    :return:
    """
    if type_ == 'const':
        schedule = np.array([start_val for _ in range(num_episodes)])
    elif type_ == 'log':
        schedule = np.hstack([[start_val for _ in range(decay_start)],
                          np.logspace(np.log10(start_val), np.log10(0.000001), (num_episodes - decay_start))])
    elif type_ == 'linear':
        schedule = np.hstack([[start_val for _ in range(decay_start)],
                          np.linspace(start_val, 0, (num_episodes - decay_start))])
    else:
        raise NotImplementedError
    schedule = np.array([1.0 for _ in range(learning_starts)] + list(schedule))[:num_episodes]
    return schedule


def greedy_eval_Q(Q: QTable, this_environment, nevaluations: int = 1, print_actions=False):
    """
    Evaluate Q function greediely with epsilon=0

    :returns average cumulative reward, the expected reward after resetting the environment, episode length
    """
    cumuls = []
    for _ in range(nevaluations):
        evaluation_state = this_environment.reset()
        episode_length, cummulative_reward = 0, 0
        expected_reward = np.max(Q[evaluation_state])
        greedy = make_epsilon_greedy_policy(Q, 0, this_environment.action_space.n)
        while True:  # roll out episode
            evaluation_action = np.random.choice(list(range(this_environment.action_space.n)),
                                                 p=greedy(evaluation_state))
            if print_actions:
                print(evaluation_action)
            s_, evaluation_reward, evaluation_done, _ = this_environment.step(evaluation_action)
            cummulative_reward += evaluation_reward
            episode_length += 1
            if evaluation_done:
                break
            evaluation_state = s_
        cumuls.append(cummulative_reward)
    return np.mean(cumuls), expected_reward, episode_length  # Q, cumulative reward


def update(Q: QTable, environment, policy: callable, alpha: float, discount_factor: float):
    """
    Q update
    :param Q: state-action value look-up table
    :param environment: environment to use
    :param policy: the current policy
    :param alpha: learning rate
    :param discount_factor: discounting factor
    """
    # Need to parse to string to easily handle list as state with defdict
    policy_state = environment.reset()
    episode_length, cummulative_reward = 0, 0
    expected_reward = np.max(Q[policy_state])
    while True:  # roll out episode
        policy_action = np.random.choice(list(range(environment.action_space.n)), p=policy(policy_state))
        s_, policy_reward, policy_done, _ = environment.step(policy_action)
        cummulative_reward += policy_reward
        episode_length += 1
        Q[[policy_state, policy_action]] = Q[[policy_state, policy_action]] + alpha * (
                (policy_reward + discount_factor * Q[[s_, np.argmax(Q[s_])]]) - Q[[policy_state, policy_action]])
        if policy_done:
            break
        policy_state = s_
    return Q, cummulative_reward, expected_reward, episode_length  # Q, cumulative reward

def train_q_learning(
    env_fn,
    env_config,
    checkpoints,
    fn,
    num_episodes: int,
    discount_factor: float = 1.0,
    alpha: float = 0.5,
    epsilon: float = 0.1,
    verbose: bool = False,
    track_test_stats: bool = False,
    float_state=False,
    epsilon_decay: str = 'const',
    decay_starts: int = 0,
    learning_starts: int = 0,
    number_of_evaluations: int = 1,
    test_environment = None,
    **kwargs
):
    """
    Q-Learning algorithm
    """
    assert 0 <= discount_factor <= 1, 'Lambda should be in [0, 1]'
    assert 0 <= epsilon <= 1, 'epsilon has to be in [0, 1]'
    assert alpha > 0, 'Learning rate has to be positive'
    # The action-value function.
    # Nested dict that maps state -> (action -> action-value).
    environment = env_fn(**env_config)
    float_state = True
    # test_environment = env_fn(**env_config)
    Q = QTable(environment.action_space.n, float_state)
    test_stats = None
    if track_test_stats:
        test_stats = EpisodeStats(
            episode_lengths=np.zeros(num_episodes),
            episode_rewards=np.zeros(num_episodes),
            expected_rewards=np.zeros(num_episodes))

    # Keeps track of episode lengths and rewards
    train_stats = EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes),
        expected_rewards=np.zeros(num_episodes))

    epsilon_schedule = get_decay_schedule(epsilon, decay_starts, num_episodes, epsilon_decay, learning_starts)

    checkpoints = np.append(checkpoints, num_episodes)

    checkpoint_q_vals = {}
    for i_episode in range(num_episodes):

        epsilon = epsilon_schedule[i_episode]
        # The policy we're following
        policy = make_epsilon_greedy_policy(Q, epsilon, environment.action_space.n)
        # Print out which episode we're on, useful for debugging.
        Q, rs, exp_rew, ep_len = update(Q, environment, policy, alpha, discount_factor)
        if track_test_stats:  # Keep track of test performance, s.t. they don't influence the training Q
            if test_environment:
                test_reward, test_expected_reward, test_episode_length = greedy_eval_Q(
                    Q, test_environment, nevaluations=number_of_evaluations)
                test_stats.episode_rewards[i_episode] = test_reward
                test_stats.expected_rewards[i_episode] = test_expected_reward
                test_stats.episode_lengths[i_episode] = test_episode_length
        train_reward, train_expected_reward, train_episode_length = greedy_eval_Q(
            Q, environment, nevaluations=number_of_evaluations, print_actions=((i_episode + 1) in checkpoints))
        train_stats.episode_rewards[i_episode] = train_reward
        train_stats.expected_rewards[i_episode] = train_expected_reward
        train_stats.episode_lengths[i_episode] = train_episode_length
        print("Episode {:>5d}/{}: {}.".format(i_episode + 1, num_episodes, train_reward))

        if i_episode + 1 in checkpoints:
            checkpoint_q_vals[i_episode] = copy.deepcopy(Q)
            results = {
                "checkpoint_q_vals": {k: dict(v) for k, v in checkpoint_q_vals.items()},
                "episode_rewards": train_stats.episode_rewards,
            }

            with open(fn.replace("ch_0", f"ch_{i_episode + 1}"), 'wb') as fh:
                pickle.dump(results, fh)

    return checkpoint_q_vals, (test_stats, train_stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Tabular Q-learning')
    parser.add_argument('--agent',
                        choices=["tabular_q", "deep_q"],
                        default="tabular_q")
    parser.add_argument('--num_episodes',
                        help='Number of episodes to roll out',
                        required=True,
                        type=int)
    parser.add_argument('--epsilon',
                        default=0.1,
                        help='Epsilon',
                        type=float)
    parser.add_argument('--eps_schedule',
                        choices=['const', 'log', 'linear'],
                        help='Which epsilon schedule to use',
                        default='const',
                        required=False,
                        type=str)
    parser.add_argument('--eps_decay_starts',
                        help='When to start decaying',
                        default=0,
                        required=False,
                        type=int)
    parser.add_argument('--lr',
                        default=0.125,
                        help='Discount Factor',
                        type=float)
    parser.add_argument('--discount_factor',
                        default=0.99,
                        help='Discount Factor',
                        type=float)
    parser.add_argument('--num_checkpoints',
                        help='Number of checkpoints',
                        default=1,
                        required=False,
                        type=int)
    parser.add_argument('--checkpoint_strategy',
                        choices=['log', 'linear'],
                        default='linear',
                        required=False,
                        type=str)
    parser.add_argument('-s', '--seed',
                        default=0,
                        type=int)
    parser = RLSMAC.add_rlsmac_arguments(parser)

    args = parser.parse_args()
    np.random.seed(args.seed)

    env_config = RLSMAC.read_rlsmac_arguments(args)
    pprint(env_config)

    t = str(datetime.datetime.now()).replace(" ", "_")
    fn = f"{args.epsilon}-greedy-results-{args.agent}-{args.num_episodes}_eps-{args.bench}-{t}-ch_0.pkl"

    if args.agent == "tabular_q":
        if args.num_checkpoints == 1:
            checkpoints = [args.num_episodes]
        else:
            if args.checkpoint_strategy == "log":
                checkpoints = np.logspace(0, 1, args.num_checkpoints, True, args.num_episodes, dtype=np.int)
            else:
                checkpoints = np.linspace(1, args.num_episodes, args.num_checkpoints, True, dtype=np.int)

        agent_config = dict(
            checkpoints=checkpoints,
            num_episodes=args.num_episodes,
            discount_factor=args.discount_factor,
            alpha=args.lr,
            epsilon=args.epsilon,
            learning_rate=args.lr,
            epsilon_decay=args.eps_schedule,
            decay_starts=args.eps_decay_starts,
            fn=fn,
        )
        pprint(agent_config)

        checkpoint_q_vals, (test_stats, train_stats) = train_q_learning(
            env_fn=RLSMAC,
            env_config=env_config,
            **agent_config,
        )
    elif args.agent == "deep_q":
        conf, stats = setup_ray(args, RLSMAC)
        log_creator = partial(logger_creator, model='DQN', adp="rlsmac", seed=args.seed)
        conf['env_config'] = env_config
        pprint(conf)
        agent = ray_dqn.DQNAgent(config=conf, env='smac_env', logger_creator=log_creator)
        ray_dqn_learn(args.num_episodes, agent, c_freq=(args.num_episodes - 1)//args.num_checkpoints)
        agent.save()
        with open(fn.replace('results', 'stats'), 'wb') as fh:
            pickle.dump(stats.episode_rewards[:args.num_episodes], fh)
