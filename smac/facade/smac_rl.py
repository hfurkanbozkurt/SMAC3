import logging
from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np

from smac.benchmarks import synthetic_benchmarks
from smac.configspace import convert_configurations_to_array
from smac.facade.smac_ac_facade import SMAC4AC
from smac.facade.smac_bo_facade import SMAC4BO
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.scenario.scenario import Scenario
from smac.optimizer.acquisition import AdaptiveLCB


class RLSMAC(gym.Env):

    def __init__(self,
                 mode: str = "HPO",
                 obs: List[str] = [
                     "budget",
                     "incumbent_changes",
                     "not_improved_since",
                     "incumbent_performance_prediction_error",
                 ],
                 act: str = "exploration_weight",
                 rew: str = "incumbent_performance",
                 bench: str = "camelback",
                 horizon: int = 50,
                 act_repeat: int = 5,
                 verbose: str = "ERROR",
                 **kwargs,
    ) -> None:
        self.mode = mode
        self.obs = obs
        self.act = act
        self.rew = rew
        self.bench = bench
        self.horizon = horizon
        self.act_repeat = act_repeat
        self.kwargs = kwargs

        self.mode = {"AC": SMAC4AC, "BO": SMAC4BO, "HPO": SMAC4HPO}[self.mode]
        self.bench = synthetic_benchmarks[self.bench]

        # Define observation space and functions to retrieve observation
        observation_spaces = {
            "budget": (self.get_budget, 1),
            "incumbent_changes": (self.get_incumbent_changes, 1),
            "not_improved_since": (self.get_not_improved_since, 1),
            "incumbent_performance_prediction_error": (self.get_incumbent_performance_prediction_error, 1),
        }
        self.get_observation = [observation_spaces[obs][0] for obs in self.obs]
        self.observation_space = gym.spaces.Box(
            -np.inf,
            np.inf,
            [sum([observation_spaces[obs][1] for obs in self.obs]),]
        )

        # Define action space and functions to apply action
        action_spaces = {
            "binary_random_prob": (self.apply_binary_random_prob, 2),
            "random_prob": (self.apply_random_prob, 21),
            "exploration_weight": (self.apply_exploration_weight, 20),
        }
        self.apply_action = action_spaces[self.act][0]
        self.action_space = gym.spaces.Discrete(action_spaces[self.act][1])

        # Define reward function
        reward_functions = {
            "incumbent_performance": self.incumbent_performance_reward,
        }
        self.get_reward = reward_functions[self.rew]

        logging.getLogger().setLevel(verbose)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        # Apply action
        self.apply_action(action)
        self.smac.solver.run_step(num_steps=self.act_repeat)
        self.t += self.act_repeat

        # Retrieve response
        self.state = np.hstack([get_obs() for get_obs in self.get_observation])
        self.done = (self.t >= self.horizon)
        self.reward = self.get_reward()
        self.info = {
            "perf_*": np.float32(self.bench.get_meta_information()["f_opt"]),
            "perf_^": np.float32(
                self.smac.solver.intensifier.traj_logger.trajectory[-1].train_perf),
            "t": self.t,
        }
        return self.state, self.reward, self.done, self.info

    def reset(self) -> np.ndarray:
        # We initialize solver and start it manually, i.e., run initial design
        self.smac = self.mode(**self.modify_kwargs(self.kwargs))
        self.smac.solver.start()
        self.t = 0
        # We do not want budget to exhaust, we want to use our own version of
        # budget in terms of number of steps.
        self.smac.solver.stats.is_budget_exhausted = lambda: False

        self.state = np.hstack([get_obs() for get_obs in self.get_observation])
        return self.state

    def modify_kwargs(self, kwargs: Dict) -> Dict:
        scenario = Scenario({
            "run_obj": "quality",
            "deterministic": "true",
            "runcount-limit": 2 * self.horizon,
            "cs": self.bench.get_configuration_space(),
        })
        scenario.output_dir = ""
        kwargs["scenario"] = scenario
        kwargs["tae_runner"] = self.bench()
        kwargs["acquisition_function"] = AdaptiveLCB
        return kwargs

    def optimize(self) -> None:
        self.smac = self.mode(**self.modify_kwargs(self.kwargs))
        print("Optimizing!")
        incumbent = self.smac.optimize()
        inc_perf = self.bench()(incumbent)
        print("Final incumbent:", incumbent)
        print(f"incumbent perf: {inc_perf}")

    # -------------------------------------------------------------------------
    # Observation Functions
    # -------------------------------------------------------------------------
    def get_budget(self) -> float:
        return self.horizon - self.t

    def get_incumbent_changes(self) -> float:
        return len(self.smac.solver.intensifier.traj_logger.trajectory) - 1

    def get_not_improved_since(self) -> float:
        return self.smac.stats.ta_runs - self.smac.solver.intensifier.traj_logger.trajectory[-1].ta_runs

    def get_incumbent_performance_prediction_error(self) -> float:
        error = 0.0

        if self.t == 0:
            return error

        for entry in self.smac.solver.intensifier.traj_logger.trajectory[1:]:
            config = entry.incumbent
            m, _ = self.smac.solver.model.predict_marginalized_over_instances(
                X=convert_configurations_to_array([config]))

            y_transformed = self.smac.solver.runhistory.get_cost(config)
            y_transformed = np.array(y_transformed).reshape((1, 1))
            self.smac.solver.rh2EPM.min_y = min(y_transformed, self.smac.solver.rh2EPM.min_y)
            self.smac.solver.rh2EPM.max_y = max(y_transformed, self.smac.solver.rh2EPM.max_y)
            y_transformed = self.smac.solver.rh2EPM.transform_response_values(y_transformed)
            y_transformed = y_transformed[0][0]
            y_predicted_transformed= m[0, 0]

            error += np.abs(y_transformed - y_predicted_transformed, dtype=np.float32)

        return error

    # -------------------------------------------------------------------------
    # Action Functions
    # -------------------------------------------------------------------------
    def apply_binary_random_prob(self, action: int) -> None:
        self.smac.solver.random_configuration_chooser.prob = float(action)

    def apply_random_prob(self, action: int) -> None:
        self.smac.solver.random_configuration_chooser.prob = action / 20.0

    def apply_exploration_weight(self, action: int) -> None:
        self.smac.solver.acquisition_func._set_exploration_weight(action)

    # -------------------------------------------------------------------------
    # Reward Functions
    # -------------------------------------------------------------------------
    def incumbent_performance_reward(self) -> float:
        perf_star = self.bench.get_meta_information()["f_opt"]
        perf_hat = self.smac.solver.intensifier.traj_logger.trajectory[-1].train_perf
        return - np.abs(perf_hat - perf_star)

    # -------------------------------------------------------------------------
    # Util Functions
    # -------------------------------------------------------------------------
    @staticmethod
    def add_rlsmac_arguments(parser):
        parser.add_argument("--mode",
                            type=str,
                            choices=["AC", "BO", "HPO"],
                            default="HPO")
        parser.add_argument("--obs",
                            nargs="+",
                            type=str,
                            default=["budget", "incumbent_changes", "not_improved_since", "incumbent_performance_prediction_error"])
        parser.add_argument("--act",
                            type=str,
                            default="exploration_weight")
        parser.add_argument("--rew",
                            type=str,
                            default="incumbent_performance")
        parser.add_argument("--bench",
                            type=str,
                            default="camelback")
        parser.add_argument("--horizon",
                            type=int,
                            default=50)
        parser.add_argument("--act_repeat",
                            type=int,
                            default=5)
        parser.add_argument("--verbose",
                            type=str,
                            choices=["ERROR", "INFO", "DEBUG"],
                            default="ERROR")
        return parser

    @staticmethod
    def read_rlsmac_arguments(args):
        config = {
            "mode": args.mode,
            "obs": args.obs,
            "act": args.act,
            "rew": args.rew,
            "bench": args.bench,
            "horizon": args.horizon,
            "act_repeat": args.act_repeat,
            "verbose": args.verbose,
        }
        return config

    @staticmethod
    def env_creator(config):
        return RLSMAC(**config)
