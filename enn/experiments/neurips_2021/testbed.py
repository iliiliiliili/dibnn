# python3
# pylint: disable=g-bad-file-header
# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""GP regression testbed problem.

Uses the neural_tangent library to compute the posterior mean and covariance
for regression problem in closed form.
"""

import chex
import dataclasses
from enn.experiments.neurips_2021 import base as testbed_base
import haiku as hk
import jax
import jax.numpy as jnp
import neural_tangents as nt
from neural_tangents.utils import typing as nt_types


class GPRegression:
    """GP with gaussian noise output."""

    def __init__(
        self,
        kernel_fn: nt_types.KernelFn,
        x_train: chex.Array,
        x_test: chex.Array,
        x_val: chex.Array,
        tau: int = 1,
        noise_std: float = 1,
        seed: int = 1,
        kernel_ridge: float = 1e-6,
    ):

        # Checking the dimensionality of our data coming in.
        num_train, input_dim = x_train.shape
        num_test_x_cache, input_dim_test = x_test.shape
        assert input_dim == input_dim_test

        rng = hk.PRNGSequence(seed)
        self._tau = tau
        self._input_dim = input_dim
        self._x_train = jnp.array(x_train)
        self._x_test = jnp.array(x_test)
        self._x_val = jnp.array(x_val)
        self._num_train = num_train
        self._num_test_x_cache = num_test_x_cache
        self._noise_std = noise_std
        self._kernel_ridge = kernel_ridge

        # Form the training data
        mean = jnp.zeros(num_train)
        k_train_train = kernel_fn(self._x_train, x2=None, get="nngp")
        k_train_train += kernel_ridge * jnp.eye(num_train)
        y_function = jax.random.multivariate_normal(next(rng), mean, k_train_train)
        y_noise = jax.random.normal(next(rng), [num_train, 1]) * noise_std
        y_train = y_function[:, None] + y_noise
        self._train_data = testbed_base.Data(x_train, y_train)
        chex.assert_shape(y_train, [num_train, 1])

        # Form the posterior prediction at cached test data
        predict_fn = nt.predict.gradient_descent_mse_ensemble(
            kernel_fn, x_train, y_train, diag_reg=(noise_std ** 2)
        )
        self._test_mean, self._test_cov = predict_fn(
            t=None, x_test=self._x_test, get="nngp", compute_cov=True
        )
        self._val_mean, self._val_cov = predict_fn(
            t=None, x_test=self._x_val, get="nngp", compute_cov=True
        )
        self._test_cov += kernel_ridge * jnp.eye(num_test_x_cache)
        self._val_cov += kernel_ridge * jnp.eye(num_test_x_cache)
        chex.assert_shape(self._test_mean, [num_test_x_cache, 1])
        chex.assert_shape(self._test_cov, [num_test_x_cache, num_test_x_cache])
        chex.assert_shape(self._val_mean, [num_test_x_cache, 1])
        chex.assert_shape(self._val_cov, [num_test_x_cache, num_test_x_cache])

    @property
    def x_test(self) -> chex.Array:
        return self._x_test

    @property
    def x_val(self) -> chex.Array:
        return self._x_val

    @property
    def test_mean(self) -> chex.Array:
        return self._test_mean

    @property
    def test_cov(self) -> chex.Array:
        return self._test_cov

    @property
    def val_mean(self) -> chex.Array:
        return self._val_mean

    @property
    def val_cov(self) -> chex.Array:
        return self._val_cov

    @property
    def train_data(self) -> testbed_base.Data:
        return self._train_data


@dataclasses.dataclass
class TestbedGPRegression(testbed_base.TestbedProblem):
    """Wraps GPRegression sampler for testbed with exact posterior inference."""

    data_sampler: GPRegression
    prior: testbed_base.PriorKnowledge
    num_enn_samples: int = 100
    std_ridge: float = 1e-3

    @property
    def train_data(self) -> testbed_base.Data:
        return self.data_sampler.train_data

    @property
    def prior_knowledge(self) -> testbed_base.PriorKnowledge:
        return self.prior

    def evaluate_quality(
        self, enn_sampler: testbed_base.EpistemicSampler, num_samples=None
    ) -> testbed_base.ENNQuality:
        """Computes KL estimate on mean functions for tau=1 only."""
        # Extract useful quantities from the gp sampler.

        num_samples = self.num_enn_samples if num_samples is None else num_samples

        x_test = self.data_sampler.x_test
        num_test = x_test.shape[0]
        posterior_mean = self.data_sampler.test_mean[:, 0]
        posterior_std = jnp.sqrt(jnp.diag(self.data_sampler.test_cov))
        posterior_std += self.std_ridge

        # Compute the mean and std of ENN posterior
        batched_sampler = jax.jit(jax.vmap(enn_sampler, in_axes=[None, 0]))
        enn_samples = batched_sampler(x_test, jnp.arange(num_samples))
        enn_samples = enn_samples[:, :, 0]
        chex.assert_shape(enn_samples, [num_samples, num_test])
        enn_mean = jnp.mean(enn_samples, axis=0)
        enn_std = jnp.std(enn_samples, axis=0) + self.std_ridge

        # Compute the KL divergence between this and reference posterior
        batched_kl = jax.jit(jax.vmap(_kl_gaussian))
        kl_estimates = batched_kl(posterior_mean, posterior_std, enn_mean, enn_std)
        chex.assert_shape(kl_estimates, [num_test])
        kl_estimate = jnp.mean(kl_estimates)

        error_mean = jnp.mean(jnp.abs((posterior_mean - enn_mean) / posterior_mean))
        error_std = jnp.mean(jnp.abs((posterior_std - enn_std) / posterior_std))

        result = testbed_base.ENNQuality(
            kl_estimate, {"mean_error": error_mean, "std_error": error_std,}
        )

        return result

    def evaluate_quality_val(
        self, enn_sampler: testbed_base.EpistemicSampler
    ) -> testbed_base.ENNQuality:
        """Computes KL estimate on mean functions for tau=1 only."""
        # Extract useful quantities from the gp sampler.
        x_val = self.data_sampler.x_val
        num_val = x_val.shape[0]
        posterior_mean = self.data_sampler.val_mean[:, 0]
        posterior_std = jnp.sqrt(jnp.diag(self.data_sampler.val_cov))
        posterior_std += self.std_ridge

        # Compute the mean and std of ENN posterior
        batched_sampler = jax.jit(jax.vmap(enn_sampler, in_axes=[None, 0]))
        enn_samples = batched_sampler(x_val, jnp.arange(self.num_enn_samples))
        enn_samples = enn_samples[:, :, 0]
        chex.assert_shape(enn_samples, [self.num_enn_samples, num_val])
        enn_mean = jnp.mean(enn_samples, axis=0)
        enn_std = jnp.std(enn_samples, axis=0) + self.std_ridge

        # Compute the KL divergence between this and reference posterior
        batched_kl = jax.jit(jax.vmap(_kl_gaussian))
        kl_estimates = batched_kl(posterior_mean, posterior_std, enn_mean, enn_std)
        chex.assert_shape(kl_estimates, [num_val])
        kl_estimate = jnp.mean(kl_estimates)

        error_mean = jnp.mean(jnp.abs((posterior_mean - enn_mean) / posterior_mean))
        error_std = jnp.mean(jnp.abs((posterior_std - enn_std) / posterior_std))

        result = testbed_base.ENNQuality(
            kl_estimate, {"mean_error": error_mean, "std_error": error_std,}
        )

        return result

    def evaluate_quality_batched(
        self, batched_sampler: testbed_base.EpistemicSampler, num_samples=None
    ) -> testbed_base.ENNQuality:
        """Computes KL estimate on mean functions for tau=1 only."""
        # Extract useful quantities from the gp sampler.

        num_samples = self.num_enn_samples if num_samples is None else num_samples

        x_test = self.data_sampler.x_test
        num_test = x_test.shape[0]
        posterior_mean = self.data_sampler.test_mean[:, 0]
        posterior_std = jnp.sqrt(jnp.diag(self.data_sampler.test_cov))
        posterior_std += self.std_ridge

        # Compute the mean and std of ENN posterior
        enn_samples = batched_sampler(x_test, num_samples)
        enn_samples = enn_samples[:, :, 0]
        chex.assert_shape(enn_samples, [num_samples, num_test])
        enn_mean = jnp.mean(enn_samples, axis=0)
        enn_std = jnp.std(enn_samples, axis=0) + self.std_ridge

        # Compute the KL divergence between this and reference posterior
        batched_kl = jax.jit(jax.vmap(_kl_gaussian))
        kl_estimates = batched_kl(posterior_mean, posterior_std, enn_mean, enn_std)
        chex.assert_shape(kl_estimates, [num_test])
        kl_estimate = jnp.mean(kl_estimates)

        error_mean = jnp.mean(jnp.abs((posterior_mean - enn_mean) / posterior_mean))
        error_std = jnp.mean(jnp.abs((posterior_std - enn_std) / posterior_std))

        result = testbed_base.ENNQuality(
            kl_estimate, {"mean_error": error_mean, "std_error": error_std,}
        )

        return result

    def find_best_samples_batched(
        self, batched_fixed_sampler: testbed_base.EpistemicSampler, all_samples
    ) -> testbed_base.ENNQuality:
        """Computes KL estimate on mean functions for tau=1 only."""
        # Extract useful quantities from the gp sampler.

        results = {}

        x_test = self.data_sampler.x_test
        num_test = x_test.shape[0]
        posterior_mean_test = self.data_sampler.test_mean[:, 0]
        posterior_std_test = jnp.sqrt(jnp.diag(self.data_sampler.test_cov))
        posterior_std_test += self.std_ridge

        x_val = self.data_sampler.x_val
        num_val = x_val.shape[0]
        posterior_mean_val = self.data_sampler.val_mean[:, 0]
        posterior_std_val = jnp.sqrt(jnp.diag(self.data_sampler.val_cov))
        posterior_std_val += self.std_ridge

        def evaluate(selected_samples, is_test):

            num_samples = len(selected_samples)

            if is_test:
                x = x_test
                num = num_test
                posterior_mean = posterior_mean_test
                posterior_std = posterior_std_test
                posterior_std = posterior_std_test
            else:
                x = x_val
                num = num_val
                posterior_mean = posterior_mean_val
                posterior_std = posterior_std_val
                posterior_std = posterior_std_val

            # Compute the mean and std of ENN posterior
            enn_samples = batched_fixed_sampler(x, selected_samples)
            enn_samples = enn_samples[:, :, 0]
            chex.assert_shape(enn_samples, [num_samples, num])
            enn_mean = jnp.mean(enn_samples, axis=0)
            enn_std = jnp.std(enn_samples, axis=0) + self.std_ridge

            # Compute the KL divergence between this and reference posterior
            batched_kl = jax.jit(jax.vmap(_kl_gaussian))
            kl_estimates = batched_kl(posterior_mean, posterior_std, enn_mean, enn_std)
            chex.assert_shape(kl_estimates, [num])
            kl_estimate = jnp.mean(kl_estimates)

            error_mean = jnp.mean(jnp.abs((posterior_mean - enn_mean) / posterior_mean))
            error_std = jnp.mean(jnp.abs((posterior_std - enn_std) / posterior_std))

            result = testbed_base.ENNQuality(
                kl_estimate, {"mean_error": error_mean, "std_error": error_std,}
            )

            return result

        def find_first_pair():

            best_kl = None
            best_samples = None

            for i in range(0, len(all_samples)):
                for q in range(i + 1, len(all_samples)):
                    kl = evaluate(jnp.array([all_samples[i], all_samples[q]]), False)

                    if (best_kl is None) or (best_kl.kl_estimate > kl.kl_estimate):
                        print(
                            "pair bkl kl",
                            best_kl.kl_estimate if best_kl is not None else None,
                            kl.kl_estimate,
                            "i",
                            i,
                            "/",
                            len(all_samples),
                            end="\r",
                        )
                        best_kl = kl
                        best_samples = [i, q]

            return best_samples, best_kl

        def add_sample_to_best(best_samples):

            best_kl = None
            best_addition = None

            for i in range(0, len(all_samples)):
                if i not in best_samples:
                    kl = evaluate(
                        jnp.array([all_samples[s] for s in [*best_samples, i]]), False
                    )
                    if (best_kl is None) or (best_kl.kl_estimate > kl.kl_estimate):
                        print(
                            len(best_samples) + 1,
                            "bkl kl",
                            best_kl.kl_estimate if best_kl is not None else None,
                            kl.kl_estimate,
                            "i",
                            i,
                            "/",
                            len(all_samples),
                            end="\r",
                        )
                        best_kl = kl
                        best_addition = i

            return [*best_samples, best_addition], best_kl

        # best_samples, best_val_kl = find_first_pair()
        # best_kl = evaluate(jnp.array([all_samples[s] for s in best_samples]), True)
        # print("--", len(best_samples), "kl", best_kl.kl_estimate)

        # results[2] = {
        #     "best_samples": best_samples,
        #     "best_kl": best_kl,
        #     "best_val_kl": best_val_kl,
        # }

        best_samples = []

        for num_samples in range(1, len(all_samples)):
            best_samples, best_val_kl = add_sample_to_best(best_samples)
            best_kl = evaluate(jnp.array([all_samples[s] for s in best_samples]), True)
            print("--", len(best_samples), "kl", best_kl.kl_estimate)

            if num_samples > 1:

                results[num_samples] = {
                    "best_samples": best_samples,
                    "best_kl": best_kl,
                    "best_val_kl": best_val_kl,
                }

        return results


def _kl_gaussian(mean_1: float, std_1: float, mean_2: float, std_2: float) -> float:
    """Computes the KL(P_1 || P_2) for P_1,P_2 univariate Gaussian."""
    log_term = jnp.log(std_2 / std_1)
    frac_term = (std_1 ** 2 + (mean_1 - mean_2) ** 2) / (2 * std_2 ** 2)
    return log_term + frac_term - 0.5
