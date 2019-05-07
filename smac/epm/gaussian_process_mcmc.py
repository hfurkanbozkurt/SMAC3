from copy import deepcopy
import logging
import typing

import emcee
import numpy as np
import skopt.learning.gaussian_process
import skopt.learning.gaussian_process.kernels

from smac.epm.base_gp import BaseModel
from smac.epm.gaussian_process import GaussianProcess

logger = logging.getLogger(__name__)


class GaussianProcessMCMC(BaseModel):

    def __init__(
        self,
        types: np.ndarray,
        bounds: typing.List[typing.Tuple[float, float]],
        seed: int,
        kernel: skopt.learning.gaussian_process.kernels.Kernel,
        n_mcmc_walkers: int = 20,
        chain_length: int = 50,
        burnin_steps: int = 50,
        normalize_y: bool = True,
        mcmc_sampler: str = 'emcee',
        average_samples: bool = False,
        **kwargs
    ):
        """
        Gaussian process model.

        The GP hyperparameters are integrated out by MCMC. If you use this class
        make sure that you also use an integrated acquisition function to
        integrate over the GP's hyperparameter as proposed by Snoek et al.

        This code is based on the implementation of RoBO:

        Klein, A. and Falkner, S. and Mansur, N. and Hutter, F.
        RoBO: A Flexible and Robust Bayesian Optimization Framework in Python
        In: NIPS 2017 Bayesian Optimization Workshop

        Parameters
        ----------
        types : np.ndarray (D)
            Specifies the number of categorical values of an input dimension where
            the i-th entry corresponds to the i-th input dimension. Let's say we
            have 2 dimension where the first dimension consists of 3 different
            categorical choices and the second dimension is continuous than we
            have to pass np.array([2, 0]). Note that we count starting from 0.
        bounds : list
            Specifies the bounds for continuous features.
        seed : int
            Model seed.
        kernel : george kernel object
            Specifies the kernel that is used for all Gaussian Process
        n_mcmc_walkers : int
            The number of hyperparameter samples. This also determines the
            number of walker for MCMC sampling as each walker will
            return one hyperparameter sample.
        chain_length : int
            The length of the MCMC chain. We start n_mcmc_walkers walker for
            chain_length steps and we use the last sample
            in the chain as a hyperparameter sample.
        burnin_steps : int
            The number of burnin steps before the actual MCMC sampling starts.
        normalize_y : bool
            Zero mean unit variance normalization of the output values
        mcmc_sampler : str
            Choose a self-tuning MCMC sampler. Can be either ``emcee`` or ``nuts``.
        average_samples : bool
            Average the sampled hyperparameters if ``True``, uses the samples independently if ``False``.
            This is equivalent to the posterior mean as used by
            Letham et al. (2018, http://lethalletham.com/ConstrainedBO.pdf).
        rng: np.random.RandomState
            Random number generator
        """
        super().__init__(types=types, bounds=bounds, seed=seed, **kwargs)

        self.kernel = kernel
        self.n_mcmc_walkers = n_mcmc_walkers
        self.chain_length = chain_length
        self.burned = False
        self.burnin_steps = burnin_steps
        self.models = []
        self.normalize_y = normalize_y
        self.mcmc_sampler = mcmc_sampler
        self.average_samples = average_samples

        self.is_trained = False

        # Internal statistics
        self._n_ll_evals = 0

    def _train(self, X: np.ndarray, y: np.ndarray, do_optimize: bool=True):
        """
        Performs MCMC sampling to sample hyperparameter configurations from the
        likelihood and trains for each sample a GP on X and y

        Parameters
        ----------
        X: np.ndarray (N, D)
            Input data points. The dimensionality of X is (N, D),
            with N as the number of points and D is the number of features.
        y: np.ndarray (N,)
            The corresponding target values.
        do_optimize: boolean
            If set to true we perform MCMC sampling otherwise we just use the
            hyperparameter specified in the kernel.
        """

        self.gp = skopt.learning.gaussian_process.GaussianProcessRegressor(
            kernel=self.kernel,
            normalize_y=self.normalize_y,
            optimizer=None,
            n_restarts_optimizer=-1,  # Do not use scikit-learn's optimization routine
            alpha=0,  # Governed by the kernel
            noise=None,
        )

        if do_optimize:
            self.gp.fit(X, y)
            self._all_priors = self._get_all_priors(
                add_bound_priors=True,
                add_soft_bounds=True if self.mcmc_sampler == 'nuts' else False,
            )

            if self.mcmc_sampler == 'emcee':
                sampler = emcee.EnsembleSampler(self.n_mcmc_walkers,
                                                len(self.kernel.theta),
                                                self._ll)
                sampler.random_state = self.rng.get_state()
                # Do a burn-in in the first iteration
                if not self.burned:
                    # Initialize the walkers by sampling from the prior
                    dim_samples = []
                    for dim, prior in enumerate(self._all_priors):
                        # Always sample from the first prior
                        if isinstance(prior, list):
                            if len(prior) == 0:
                                prior = None
                            else:
                                prior = prior[0]
                        if prior is None:
                            raise NotImplementedError()
                        else:
                            dim_samples.append(prior.sample_from_prior(self.n_mcmc_walkers).flatten())
                    self.p0 = np.vstack(dim_samples).transpose()

                    # Run MCMC sampling
                    self.p0, _, _ = sampler.run_mcmc(self.p0,
                                                     self.burnin_steps,
                                                     rstate0=self.rng)

                    self.burned = True

                # Start sampling & save the current position, it will be the start point in the next iteration
                self.p0, _, _ = sampler.run_mcmc(self.p0, self.chain_length, rstate0=self.rng)

                # Take the last samples from each walker
                self.hypers = sampler.chain[:, -1]
                print('hypers', self.hypers.mean(axis=0), np.exp(self.hypers.mean(axis=0)))
            elif self.mcmc_sampler == 'nuts':
                # Originally published as:
                # http://www.stat.columbia.edu/~gelman/research/published/nuts.pdf
                # A good explanation of HMC:
                # https://theclevermachine.wordpress.com/2012/11/18/mcmc-hamiltonian-monte-carlo-a-k-a-hybrid-monte-carlo/
                # A good explanation of HMC and NUTS can be found in:
                # https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/2041-210X.12681

                # Do not require the installation of NUTS for SMAC
                # This requires NUTS from https://github.com/mfeurer/NUTS
                import nuts.nuts

                # Perform initial fit to the data to obtain theta0
                if not self.burned:
                    theta0 = self.gp.kernel.theta
                    self.burned = True
                else:
                    theta0 = self.p0
                samples, _, _ = nuts.nuts.nuts6(
                    f=self._ll_w_grad,
                    Madapt=self.burnin_steps,
                    M=self.chain_length,
                    theta0=theta0,
                    # Increasing this value results in longer running times
                    delta=0.5,
                    adapt_mass=False,
                    # Rather low max depth to keep the number of required gradient steps low
                    max_depth=10,
                    rng=self.rng,
                )
                print('hypers', 'log space', self.p0, 'regular space', np.exp(self.p0))
                indices = [int(np.rint(ind)) for ind in np.linspace(start=0, stop=len(samples) - 1, num=10)]
                self.hypers = samples[indices]
                self.p0 = self.hypers.mean(axis=0)
            else:
                raise ValueError(self.mcmc_sampler)

            if self.average_samples:
                self.hypers = [self.hypers.mean(axis=0)]

        else:
            self.hypers = self.gp.kernel.theta
            self.hypers = [self.hypers]

        self.models = []
        for sample in self.hypers:

            # Instantiate a GP for each hyperparameter configuration
            kernel = deepcopy(self.kernel)
            kernel.theta = sample
            model = GaussianProcess(
                types=self.types,
                bounds=self.bounds,
                kernel=kernel,
                normalize_y=self.normalize_y,
                seed=self.rng.randint(low=0, high=10000),
            )
            try:
                model._train(X, y, do_optimize=False)
                self.models.append(model)
            except np.linalg.LinAlgError:
                pass

        if len(self.models) == 0:
            kernel = deepcopy(self.kernel)
            kernel.theta = self.p0
            model = GaussianProcess(
                types=self.types,
                bounds=self.bounds,
                kernel=kernel,
                normalize_y=self.normalize_y,
                seed=self.rng.randint(low=0, high=10000),
            )
            model._train(X, y, do_optimize=False)
            self.models.append(model)

        self.is_trained = True
        print('#LL evaluations', self._n_ll_evals)

    def _ll(self, theta: np.ndarray) -> float:
        """
        Returns the marginal log likelihood (+ the prior) for
        a hyperparameter configuration theta.

        Parameters
        ----------
        theta : np.ndarray(H)
            Hyperparameter vector. Note that all hyperparameter are
            on a log scale.

        Returns
        ----------
        float
            lnlikelihood + prior
        """
        self._n_ll_evals += 1

        # Bound the hyperparameter space to keep things sane. Note all
        # hyperparameters live on a log scale
        if np.ndim(theta) == 0:
            if (theta < -20) or (theta > 20):
                return -np.inf
        else:
            if ((theta < -20) | (theta > 20)).any():
                return -np.inf

        try:
            lml = self.gp.log_marginal_likelihood(theta)
        except ValueError as e:
            return -np.inf

        # Add prior
        for dim, priors in enumerate(self._all_priors):
            for prior in priors:
                lml += prior.lnprob(theta[dim])

        if not np.isfinite(lml):
            return -np.inf
        else:
            return lml

    def _ll_w_grad(self, theta: np.ndarray) -> typing.Tuple[float, np.ndarray]:
        """
        Returns the marginal log likelihood (+ the prior) for
        a hyperparameter configuration theta.

        Parameters
        ----------
        theta : np.ndarray(H)
            Hyperparameter vector. Note that all hyperparameter are
            on a log scale.

        Returns
        ----------
        float
            lnlikelihood + prior
        """
        self._n_ll_evals += 1

        # Bound the hyperparameter space to keep things sane. Note all hyperparameters live on a log scale
        if (theta < -50).any():
            theta[theta < -50] = -50
        if (theta > 50).any():
            theta[theta > 50] = 50

        lml = 0
        grad = np.zeros(theta.shape)

        # Add prior
        for dim, priors in enumerate(self._all_priors):
            for prior in priors:
                lml += prior.lnprob(theta[dim])
                grad[dim] += prior.gradient(theta[dim])

        # Check if one of the priors is invalid, if so, no need to compute the log marginal likelihood
        if lml < -1e24:
            return -1e25, np.zeros(theta.shape)

        try:
            lml_, grad_ = self.gp.log_marginal_likelihood(theta, eval_gradient=True)
            lml += lml_
            grad += grad_
        except ValueError:
            return -1e25, np.zeros(theta.shape)

        # We add a minus here because scipy is minimizing
        if not np.isfinite(lml) or (~np.isfinite(grad)).any():
            return -1e25, np.zeros(theta.shape)
        else:
            return lml, grad

    def _predict(self, X_test: np.ndarray):
        r"""
        Returns the predictive mean and variance of the objective function
        at X average over all hyperparameter samples.
        The mean is computed by:
        :math \mu(x) = \frac{1}{M}\sum_{i=1}^{M}\mu_m(x)
        And the variance by:
        :math \sigma^2(x) = (\frac{1}{M}\sum_{i=1}^{M}(\sigma^2_m(x) + \mu_m(x)^2) - \mu^2

        Parameters
        ----------
        X_test: np.ndarray (N, D)
            Input test points

        Returns
        ----------
        np.array(N,)
            predictive mean
        np.array(N,)
            predictive variance

        """
        if not self.is_trained:
            raise Exception('Model has to be trained first!')

        mu = np.zeros([len(self.models), X_test.shape[0]])
        var = np.zeros([len(self.models), X_test.shape[0]])
        for i, model in enumerate(self.models):
            mu_tmp, var_tmp = model.predict(X_test)
            mu[i] = mu_tmp.flatten()
            var[i] = var_tmp.flatten()

        m = mu.mean(axis=0)

        # See the Algorithm Runtime Prediction paper by Hutter et al.
        # for the derivation of the total variance
        v = np.var(mu, axis=0) + np.mean(var, axis=0)

        # Clip negative variances and set them to the smallest
        # positive float value
        if v.shape[0] == 1:
            v = np.clip(v, np.finfo(v.dtype).eps, np.inf)
        else:
            v = np.clip(v, np.finfo(v.dtype).eps, np.inf)
            v[np.where((v < np.finfo(v.dtype).eps) & (v > -np.finfo(v.dtype).eps))] = 0

        return m, v