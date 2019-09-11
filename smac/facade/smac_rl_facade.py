import functools
import logging
import random
import typing

import numpy as np
from gym import Env
from gym.spaces import Box, Discrete

from smac.benchmarks import synthetic_benchmarks
from smac.epm.gaussian_process_mcmc import GaussianProcess, GaussianProcessMCMC
from smac.epm.gp_base_prior import HorseshoePrior, LognormalPrior
from smac.epm.util_funcs import get_rng, get_types
from smac.facade.smac_ac_facade import SMAC4AC
from smac.initial_design.sobol_design import SobolDesign
from smac.optimizer.acquisition import (EI, EIPS, LCB, PI,
                                        AbstractAcquisitionFunction,
                                        AdaptiveLCB,
                                        IntegratedAcquisitionFunction, LogEI)
from smac.runhistory.runhistory2epm import RunHistory2EPM4LogScaledCost
from smac.scenario.scenario import Scenario

__author__ = "H Furkan Bozkurt"
__copyright__ = "Copyright 2018, ML4AAD"
__license__ = "3-clause BSD"


class SMAC4RL(SMAC4AC, Env):

    def __init__(self,
                 obs_space=["budget"],
                 act_space="binary_random_prob",
                 reward_fun="dense_inc_perf_improvement",
                 benchmark="Branin",
                 horizon=None,
                 action_repeat=1,
                 model_type="gp",
                 acquisition_function="AdaptiveLCB",
                 verbose="ERROR",
                 seed=None,
                 **kwargs):
        self.obs_space = obs_space
        self.act_space = act_space
        self.reward_fun = reward_fun
        self.benchmark = synthetic_benchmarks[benchmark]
        self.horizon = horizon
        self.action_repeat = action_repeat
        self.model_type = model_type
        self.acquisition_function = acquisition_function
        self.seed = seed

        observation_space_dim = 0
        for obs in self.obs_space:
            observation_space_dim += {
                "budget": 1,
            }[obs]
        self.observation_space = Box(-np.inf, np.inf, [observation_space_dim,])

        self.action_space = Discrete({
            "binary_random_prob": 2,
            "random_prob": 21,
            "exploration_weight": 11,
        }[self.act_space])

        self.kwargs = kwargs

        logging.getLogger().setLevel(verbose)

    def step(self, action):
        if self.act_space == "binary_random_prob":
            self.solver.random_configuration_chooser.prob = float(action)
        elif self.act_space == "random_prob":
            self.solver.random_configuration_chooser.prob = action / 20.0
        elif self.act_space == "exploration_weight":
            self.solver.acquisition_func.exploration_weight = 2 * float(action)

        self.solver.run_step(num_steps=self.action_repeat)

        state = self._get_state()

        done = self.solver.stats.is_budget_exhausted()

        self.traj.append(self.solver.intensifier.traj_logger.trajectory[-1].train_perf)

        if self.reward_fun == "negative_step":
            reward = 0 if done else -1
        elif "inc_perf_improvement" in self.reward_fun:
            trajectory = {traj.ta_runs: traj.train_perf for traj in self.solver.intensifier.traj_logger.trajectory[1:]}
            current_inc_perf = list(trajectory.values())[-1]
            reward = self.last_inc_perf - current_inc_perf

            if "percentage" in self.reward_fun:
                reward = reward / (self.last_inc_perf - self.benchmark.get_meta_information()["f_opt"])

            if "sparse" in self.reward_fun:
                reward = reward if done else 0.0

            self.last_inc_perf = current_inc_perf

        info = {
            "inc_perf": np.float32(self.last_inc_perf),
            "opt_perf": np.float32(self.benchmark.get_meta_information()["f_opt"]),
            "traj": self.traj if done else []}

        return state, np.float32(reward), done, info

    def reset(self):
        SMAC4AC.__init__(self, **self.setup_kwargs(self.kwargs))

        if self.solver.scenario.n_features > 0:
            raise NotImplementedError("BOGP cannot handle instances")

        self.solver.scenario.acq_opt_challengers = 5
        # activate predict incumbent
        self.solver.predict_incumbent = True

        self.solver.start()

        self.traj = []

        if "inc_perf_improvement" in self.reward_fun:
            self.last_inc_perf = self.solver.intensifier.traj_logger.trajectory[-1].train_perf

        state = self._get_state()

        return state

    def _get_state(self):
        state = []
        if "budget" in self.obs_space:
            state += [self.stats.get_remaining_ta_runs()]

        return np.float32(state)

    def setup_kwargs(self, kwargs):
        horizon = self.horizon or self.benchmark.get_meta_information()["num_function_evals"]
        kwargs["scenario"] = Scenario({
            "run_obj": "quality",
            "deterministic": "true",
            "runcount-limit": horizon + 1,
            "cs": self.benchmark.get_configuration_space(),
        })

        kwargs['tae_runner'] = self.benchmark()

        kwargs['scenario'].output_dir = ""

        kwargs['initial_design'] = kwargs.get('initial_design', SobolDesign)
        kwargs['runhistory2epm'] = kwargs.get('runhistory2epm', RunHistory2EPM4LogScaledCost)

        init_kwargs = kwargs.get('initial_design_kwargs', dict())
        kwargs['initial_design_kwargs'] = init_kwargs

        from smac.epm.gp_kernels import ConstantKernel, Matern, WhiteKernel, HammingKernel

        model_kwargs = kwargs.get('model_kwargs', dict())

        _, rng = get_rng(rng=kwargs.get("rng", None), run_id=kwargs.get("run_id", None), logger=None)

        types, bounds = get_types(kwargs['scenario'].cs, instance_features=None)

        cov_amp = ConstantKernel(
            2.0,
            constant_value_bounds=(np.exp(-10), np.exp(2)),
            prior=LognormalPrior(mean=0.0, sigma=1.0, rng=rng),
        )

        cont_dims = np.nonzero(types == 0)[0]
        cat_dims = np.nonzero(types != 0)[0]

        if len(cont_dims) > 0:
            exp_kernel = Matern(
                np.ones([len(cont_dims)]),
                [(np.exp(-10), np.exp(2)) for _ in range(len(cont_dims))],
                nu=2.5,
                operate_on=cont_dims,
            )

        if len(cat_dims) > 0:
            ham_kernel = HammingKernel(
                np.ones([len(cat_dims)]),
                [(np.exp(-10), np.exp(2)) for _ in range(len(cat_dims))],
                operate_on=cat_dims,
            )

        noise_kernel = WhiteKernel(
            noise_level=1e-8,
            noise_level_bounds=(np.exp(-25), np.exp(2)),
            prior=HorseshoePrior(scale=0.1, rng=rng),
        )

        if len(cont_dims) > 0 and len(cat_dims) > 0:
            # both
            kernel = cov_amp * (exp_kernel*ham_kernel) + noise_kernel
        elif len(cont_dims) > 0 and len(cat_dims) == 0:
            # only cont
            kernel = cov_amp * exp_kernel + noise_kernel
        elif len(cont_dims) == 0 and len(cat_dims) > 0:
            # only cont
            kernel = cov_amp * ham_kernel + noise_kernel
        else:
            raise ValueError()

        if self.model_type == "gp":
            model_class = GaussianProcess
            kwargs['model'] = model_class
            model_kwargs['kernel'] = kernel
            model_kwargs['normalize_y'] = True
            model_kwargs['seed'] = rng.randint(0, 2 ** 20)
        elif self.model_type == "gp_mcmc":
            model_class = GaussianProcessMCMC
            kwargs['model'] = model_class
            kwargs['integrate_acquisition_function'] = True

            model_kwargs['kernel'] = kernel

            n_mcmc_walkers = 3 * len(kernel.theta)
            if n_mcmc_walkers % 2 == 1:
                n_mcmc_walkers += 1
            model_kwargs['n_mcmc_walkers'] = n_mcmc_walkers
            model_kwargs['chain_length'] = 250
            model_kwargs['burnin_steps'] = 250
            model_kwargs['normalize_y'] = True
            model_kwargs['seed'] = rng.randint(0, 2**20)
        else:
            raise ValueError('Unknown model type %s' % kwargs["model_type"])
        kwargs['model_kwargs'] = model_kwargs

        if kwargs.get('random_configuration_chooser') is None:
            random_config_chooser_kwargs = kwargs.get('random_configuration_chooser_kwargs', dict())
            random_config_chooser_kwargs['prob'] = random_config_chooser_kwargs.get('prob', 0.0)
            kwargs['random_configuration_chooser_kwargs'] = random_config_chooser_kwargs

        kwargs["acquisition_function"] = {
            "EI": EI,
            "LCB": LCB,
            "AbstractAcquisitionFunction": AbstractAcquisitionFunction,
            "AdaptiveLCB": AdaptiveLCB,
            "IntegratedAcquisitionFunction": IntegratedAcquisitionFunction,
            "LogEI": LogEI,
        }.get(self.acquisition_function, EI)

        if kwargs.get('acquisition_function_optimizer') is None:
            acquisition_function_optimizer_kwargs = kwargs.get('acquisition_function_optimizer_kwargs', dict())
            acquisition_function_optimizer_kwargs['n_sls_iterations'] = 10
            kwargs['acquisition_function_optimizer_kwargs'] = acquisition_function_optimizer_kwargs

        # only 1 configuration per SMBO iteration
        intensifier_kwargs = kwargs.get('intensifier_kwargs', dict())
        intensifier_kwargs['min_chall'] = 1
        kwargs['intensifier_kwargs'] = intensifier_kwargs
        kwargs['scenario'].intensification_percentage = 1e-10

        return kwargs
