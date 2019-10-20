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
from smac.optimizer.acquisition import EI, LCB, PI, AdaptiveLCB, LogEI
from smac.scenario.scenario import Scenario


class RLSMAC(gym.Env):

    def __init__(self,
                 obs,
                 act,
                 rew,
                 bench,
                 horizon,
                 act_repeat=5,
                 mode="HPO",
                 verbose="ERROR",
                 default_smac=False,
                 **kwargs,
    ) -> None:
        self.mode = mode
        self.obs = obs
        self.act = act
        self.rew = rew
        self.bench = bench
        self.horizon = horizon
        self.act_repeat = act_repeat
        self.default_smac = default_smac
        self.verbose = verbose
        self.kwargs = kwargs

        self.mode = {"AC": SMAC4AC, "BO": SMAC4BO, "HPO": SMAC4HPO}[self.mode]
        self.bench = synthetic_benchmarks[self.bench]
        self.f_opt = np.float32(self.bench.get_meta_information()["f_opt"])
        if not isinstance(self.obs, list):
            self.obs = [self.obs]

        if not self.default_smac:
            # Define observation space and functions to retrieve observation
            observation_spaces = {
                "classic": (self.get_classic, 4),
                "rsaps": (self.get_rsaps, 5),
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
                "random_prob_11": (self.apply_random_prob_11, 11),
                "random_prob_21": (self.apply_random_prob_21, 21),
                "acquisition_func": (self.apply_acquisition_func, 4),
                "exploration_weight": (self.apply_exploration_weight, 20),
            }
            self.apply_action = action_spaces[self.act][0]
            self.action_space = gym.spaces.Discrete(action_spaces[self.act][1])

        # Define reward function
        reward_functions = {
            "incumbent_performance": self.incumbent_performance_reward,
            "log_mean_regret": self.log_mean_regret_reward,
            "rsaps": self.rsaps_reward,
        }
        self.get_reward = reward_functions[self.rew]

        logging.getLogger().setLevel(self.verbose)
        self.num_resets = -1
        self.num_steps = 0
        self.state = None

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        # Apply action
        if not self.default_smac:
            self.apply_action(action)
        self.smac.solver.run_step(num_steps=self.act_repeat)
        self.t += self.act_repeat

        # Retrieve response
        if not self.default_smac:
            self.state = np.hstack([get_obs() for get_obs in self.get_observation])
        self.done = (self.t >= self.horizon)
        self.reward = self.get_reward()
        self.info = {
            "perf_*": self.f_opt,
            "perf_^": np.float32(
                self.smac.solver.intensifier.traj_logger.trajectory[-1].train_perf),
            "t": self.t,
        }
        if self.verbose == 'INFO':
            print(f"rew: {self.reward}, perf_^: {self.info['perf_^']}, perf_*: {self.info['perf_*']}")
        self.num_steps += 1
        return self.state, self.reward, self.done, self.info

    def reset(self) -> np.ndarray:
        # We initialize solver and start it manually, i.e., run initial design
        self.smac = self.mode(**self.modify_kwargs(self.kwargs))
        self.smac.solver.start()
        self.t = 0
        # We do not want budget to exhaust, we want to use our own version of
        # budget in terms of number of steps.
        self.smac.solver.stats.is_budget_exhausted = lambda: False

        if not self.default_smac:
            self.state = np.hstack([get_obs() for get_obs in self.get_observation])
        self.num_resets += 1
        self.num_steps = 0
        return self.state

    def modify_kwargs(self, kwargs: Dict) -> Dict:
        scenario = Scenario({
            "run_obj": "quality",
            "deterministic": "true",
            "runcount-limit": self.horizon,
            "cs": self.bench.get_configuration_space(),
        })
        scenario.output_dir = ""
        kwargs["scenario"] = scenario
        kwargs["tae_runner"] = self.bench()
        if not self.default_smac:
            if self.act == "exploration_weight":
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

    def get_classic(self) -> float:
        return np.hstack([
            self.get_budget(),
            self.get_incumbent_changes(),
            self.get_not_improved_since(),
            self.get_incumbent_performance_prediction_error(),
        ])

    def get_rsaps(self) -> float:
        rh = self.smac.solver.runhistory.data
        rh_step_cutoff = min(self.act_repeat, len(rh))
        rh_keys = []
        rh_vals = []
        for key, val in rh.items():
            rh_keys.append(self.smac.solver.runhistory.ids_config[key.config_id])
            rh_vals.append(val.cost)
        f_step = np.mean(rh_vals[-rh_step_cutoff:])
        f_bsf = np.min(rh_vals[:-rh_step_cutoff])
        delta_f = f_step - f_bsf

        config_distance = 0.0
        config_before = rh_keys[-rh_step_cutoff-1].get_array()
        for current_config in rh_keys[-rh_step_cutoff:]:
            current_config = current_config.get_array()
            config_distance += np.mean(np.abs(current_config - config_before))

        best_config = self.smac.solver.intensifier.traj_logger.trajectory[-1]
        steps = self.smac.stats.ta_runs
        best_step = best_config.ta_runs
        num_dims = best_config.incumbent.get_array().size

        return [delta_f, config_distance, steps, best_step, num_dims]


    # -------------------------------------------------------------------------
    # Action Functions
    # -------------------------------------------------------------------------
    def apply_binary_random_prob(self, action: int) -> None:
        init_val = self.smac.solver.random_configuration_chooser.prob
        self.smac.solver.random_configuration_chooser.prob = float(action)
        if self.verbose == 'INFO':
            print(f"R/S: {self.num_resets}/{self.num_steps}, binary_random_prob: {init_val} --> {self.smac.solver.random_configuration_chooser.prob}")

    def apply_random_prob_11(self, action: int) -> None:
        init_val = self.smac.solver.random_configuration_chooser.prob
        self.smac.solver.random_configuration_chooser.prob = action / 10.0
        if self.verbose == 'INFO':
            print(f"R/S: {self.num_resets}/{self.num_steps}, random_prob: {init_val} --> {self.smac.solver.random_configuration_chooser.prob}")

    def apply_random_prob_21(self, action: int) -> None:
        init_val = self.smac.solver.random_configuration_chooser.prob
        self.smac.solver.random_configuration_chooser.prob = action / 20.0
        if self.verbose == 'INFO':
            print(f"R/S: {self.num_resets}/{self.num_steps}, random_prob: {init_val} --> {self.smac.solver.random_configuration_chooser.prob}")

    def apply_acquisition_func(self, action: int) -> None:
        init_val = self.smac.solver.acquisition_func
        model =  self.smac.solver.acquisition_func.model
        acq_func = {0: EI, 1: LCB, 2: PI, 3: LogEI}[action]
        acquisition_function = acq_func(model)
        self.smac.solver.acquisition_func = acquisition_function
        self.smac.solver.acq_optimizer.acquisition_function = acquisition_function
        if hasattr(self.smac.solver.acq_optimizer, "local_search"):
            self.smac.solver.acq_optimizer.local_search.acquisition_function = acquisition_function
        if hasattr(self.smac.solver.acq_optimizer, "random_search"):
            self.smac.solver.acq_optimizer.random_search.acquisition_function = acquisition_function
        if self.verbose == 'INFO':
            print(f"R/S: {self.num_resets}/{self.num_steps}, acquisition_func: {init_val} --> {self.smac.solver.acquisition_func}")

    def apply_exploration_weight(self, action: int) -> None:
        init_val = self.smac.solver.acquisition_func.exploration_weight
        self.smac.solver.acquisition_func._set_exploration_weight(action)
        if self.verbose == 'INFO':
            print(f"R/S: {self.num_resets}/{self.num_steps}, exploration_weight: {init_val} --> {self.smac.solver.acquisition_func.exploration_weight}")

    # -------------------------------------------------------------------------
    # Reward Functions
    # -------------------------------------------------------------------------
    def incumbent_performance_reward(self) -> float:
        perf_hat = self.smac.solver.intensifier.traj_logger.trajectory[-1].train_perf
        return - np.abs(perf_hat - self.f_opt)

    def log_mean_regret_reward(self) -> float:
        rh = self.smac.solver.runhistory.data
        log_regrets = []
        for val in list(rh.values())[-min(self.act_repeat, len(rh)):]:
            regret = -np.log10(np.abs(val.cost - self.f_opt))
            log_regrets.append(regret)
        return np.mean(log_regrets)

    def rsaps_reward(self) -> float:
        rh = self.smac.solver.runhistory.data
        rh_step_cutoff = min(self.act_repeat, len(rh))
        rh_vals = []
        for _, val in rh.items():
            rh_vals.append(val.cost)
        f_localbest = np.min(rh_vals[-rh_step_cutoff:])
        f_bsf = np.min(rh_vals[:-rh_step_cutoff])
        return f_bsf- f_localbest

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
                            default="classic")
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
