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

Uses neural tangent kernels to compute the posterior mean and covariance
for regression problem in closed form.
"""

import dataclasses
import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal

from src.experiments.base import (
    Data,
    ENNQuality,
    EpistemicSampler,
    PriorKnowledge,
    TestbedProblem,
)
from src.ntk import gradient_descent_mse_ensemble


class GPRegression:
    """GP with gaussian noise output."""

    def __init__(
        self,
        kernel_fn,
        x_train: torch.Tensor,
        x_test: torch.Tensor,
        x_val: torch.Tensor,
        tau: int = 1,
        noise_std: float = 1,
        seed: int = 1,
        kernel_ridge: float = 1e-6,
        compute: bool = True,
    ):
        torch.manual_seed(seed)

        num_train, input_dim = x_train.shape
        num_test_x_cache, input_dim_test = x_test.shape
        assert input_dim == input_dim_test

        self._tau = tau
        self._input_dim = input_dim
        self._x_train = x_train
        self._x_test = x_test
        self._x_val = x_val
        self._num_train = num_train
        self._num_test_x_cache = num_test_x_cache
        self._noise_std = noise_std
        self._kernel_ridge = kernel_ridge

        if compute:

            # Form the training data
            mean = torch.zeros(num_train)
            k_train_train = kernel_fn(self._x_train, x2=None, get="nngp")
            k_train_train = k_train_train + kernel_ridge * torch.eye(num_train)
            y_function = MultivariateNormal(mean, k_train_train).sample()
            y_noise = torch.randn(num_train, 1) * noise_std
            y_train = y_function[:, None] + y_noise
            self._train_data = Data(self._x_train, y_train)
            assert y_train.shape == torch.Size([num_train, 1])

            # Form the posterior prediction at cached test data
            predict_fn = gradient_descent_mse_ensemble(
                kernel_fn, self._x_train, y_train, diag_reg=(noise_std**2)
            )
            self._test_mean, self._test_cov = predict_fn(
                t=None, x_test=self._x_test, get="nngp", compute_cov=True
            )
            self._val_mean, self._val_cov = predict_fn(
                t=None, x_test=self._x_val, get="nngp", compute_cov=True
            )
            self._test_cov = self._test_cov + kernel_ridge * torch.eye(num_test_x_cache)
            self._val_cov = self._val_cov + kernel_ridge * torch.eye(num_test_x_cache)
            assert self._test_mean.shape == torch.Size([num_test_x_cache, 1])
            assert self._test_cov.shape == torch.Size(
                [num_test_x_cache, num_test_x_cache]
            )
            assert self._val_mean.shape == torch.Size([num_test_x_cache, 1])
            assert self._val_cov.shape == torch.Size(
                [num_test_x_cache, num_test_x_cache]
            )

    @property
    def x_test(self) -> torch.Tensor:
        return self._x_test

    @property
    def x_val(self) -> torch.Tensor:
        return self._x_val

    @property
    def test_mean(self) -> torch.Tensor:
        return self._test_mean

    @property
    def test_cov(self) -> torch.Tensor:
        return self._test_cov

    @property
    def val_mean(self) -> torch.Tensor:
        return self._val_mean

    @property
    def val_cov(self) -> torch.Tensor:
        return self._val_cov

    @property
    def train_data(self) -> Data:
        return self._train_data

    @staticmethod
    def restore(
        tau,
        input_dim,
        x_train,
        x_test,
        x_val,
        train_y,
        test_mean,
        test_cov,
        val_mean,
        val_cov,
        noise_std,
        kernel_ridge,
    ):
        result = GPRegression(
            None, x_train, x_test, x_val, tau, noise_std, compute=False
        )
        result._input_dim = input_dim
        result._num_train = x_train.shape[0]
        result._num_test_x_cache = x_test.shape[0]
        result._noise_std = noise_std
        result._kernel_ridge = kernel_ridge
        result._train_data = Data(x_train, train_y)
        result._test_mean = test_mean
        result._test_cov = test_cov
        result._val_mean = val_mean
        result._val_cov = val_cov
        return result


@dataclasses.dataclass
class TestbedGPRegression(TestbedProblem):
    """Wraps GPRegression sampler for testbed with exact posterior inference."""

    data_sampler: GPRegression
    prior: PriorKnowledge
    num_enn_samples: int = 100
    std_ridge: float = 1e-3

    @property
    def train_data(self) -> Data:
        return self.data_sampler.train_data

    @property
    def prior_knowledge(self) -> PriorKnowledge:
        return self.prior

    def evaluate_quality(
        self, enn_sampler: EpistemicSampler, num_samples=None, device: str = "cuda:0"
    ) -> ENNQuality:
        """Computes KL estimate on mean functions for tau=1 only."""
        num_samples = self.num_enn_samples if num_samples is None else num_samples

        x_test = self.data_sampler.x_test.to(device)
        num_test = x_test.shape[0]
        posterior_mean = self.data_sampler.test_mean[:, 0].to(device)
        posterior_std = torch.sqrt(torch.diag(self.data_sampler.test_cov)).to(device)
        posterior_std += self.std_ridge

        enn_samples = torch.stack(
            [enn_sampler(x_test, i)[:, 0] for i in range(num_samples)]
        )
        assert enn_samples.shape == (num_samples, num_test)
        enn_mean = torch.mean(enn_samples, dim=0)
        enn_std = torch.std(enn_samples, dim=0) + self.std_ridge

        kl_estimates = torch.stack(
            [
                _kl_gaussian(
                    posterior_mean[i], posterior_std[i], enn_mean[i], enn_std[i]
                )
                for i in range(num_test)
            ]
        )
        kl_estimate = torch.mean(kl_estimates)

        error_mean = torch.mean(torch.abs((posterior_mean - enn_mean) / posterior_mean))
        error_std = torch.mean(torch.abs((posterior_std - enn_std) / posterior_std))

        result = ENNQuality(
            kl_estimate.item(),
            {"mean_error": error_mean.item(), "std_error": error_std.item()},
        )

        return result

    def evaluate_quality_val(self, enn_sampler: EpistemicSampler) -> ENNQuality:
        """Computes KL estimate on mean functions for tau=1 only."""
        x_val = self.data_sampler.x_val
        num_val = x_val.shape[0]
        posterior_mean = self.data_sampler.val_mean[:, 0]
        posterior_std = torch.sqrt(torch.diag(self.data_sampler.val_cov))
        posterior_std += self.std_ridge

        enn_samples = torch.stack(
            [enn_sampler(x_val, i)[:, 0] for i in range(self.num_enn_samples)]
        )
        assert enn_samples.shape == (self.num_enn_samples, num_val)
        enn_mean = torch.mean(enn_samples, dim=0)
        enn_std = torch.std(enn_samples, dim=0) + self.std_ridge

        kl_estimates = torch.stack(
            [
                _kl_gaussian(
                    posterior_mean[i], posterior_std[i], enn_mean[i], enn_std[i]
                )
                for i in range(num_val)
            ]
        )
        kl_estimate = torch.mean(kl_estimates)

        error_mean = torch.mean(torch.abs((posterior_mean - enn_mean) / posterior_mean))
        error_std = torch.mean(torch.abs((posterior_std - enn_std) / posterior_std))

        result = ENNQuality(
            kl_estimate.item(),
            {"mean_error": error_mean.item(), "std_error": error_std.item()},
        )

        return result

    def evaluate_quality_batched(
        self, batched_sampler: EpistemicSampler, num_samples=None
    ) -> ENNQuality:
        """Computes KL estimate on mean functions for tau=1 only."""
        num_samples = self.num_enn_samples if num_samples is None else num_samples

        x_test = self.data_sampler.x_test
        num_test = x_test.shape[0]
        posterior_mean = self.data_sampler.test_mean[:, 0]
        posterior_std = torch.sqrt(torch.diag(self.data_sampler.test_cov))
        posterior_std += self.std_ridge

        enn_samples = batched_sampler(x_test, num_samples)[:, :, 0]
        assert enn_samples.shape == (num_samples, num_test)
        enn_mean = torch.mean(enn_samples, dim=0)
        enn_std = torch.std(enn_samples, dim=0) + self.std_ridge

        kl_estimates = torch.stack(
            [
                _kl_gaussian(
                    posterior_mean[i], posterior_std[i], enn_mean[i], enn_std[i]
                )
                for i in range(num_test)
            ]
        )
        kl_estimate = torch.mean(kl_estimates)

        error_mean = torch.mean(torch.abs((posterior_mean - enn_mean) / posterior_mean))
        error_std = torch.mean(torch.abs((posterior_std - enn_std) / posterior_std))

        result = ENNQuality(
            kl_estimate.item(),
            {"mean_error": error_mean.item(), "std_error": error_std.item()},
        )

        return result

    def save(self, path: str):

        data = {
            "x_train": self.data_sampler._x_train,
            "x_test": self.data_sampler._x_test,
            "x_val": self.data_sampler._x_val,
            "train_y": self.data_sampler.train_data.y,
            "test_mean": self.data_sampler.test_mean,
            "test_cov": self.data_sampler.test_cov,
            "val_mean": self.data_sampler.val_mean,
            "val_cov": self.data_sampler.val_cov,
            "noise_std": self.data_sampler._noise_std,
            "kernel_ridge": self.data_sampler._kernel_ridge,
            "tau": self.data_sampler._tau,
            "input_dim": self.prior_knowledge.input_dim,
            "num_train": self.prior_knowledge.num_train,
            "num_classes": self.prior_knowledge.num_classes,
            "num_layers": self.prior_knowledge.layers,
            "noise_std": self.prior_knowledge.noise_std,
            "num_enn_samples": self.num_enn_samples,
            "std_ridge": self.std_ridge,
        }

        torch.save(data, path)

    @staticmethod
    def load(path: str):

        data = torch.load(path, weights_only=True)
        data_sampler = GPRegression.restore(
            tau=1,
            input_dim=data["x_train"].shape[1],
            x_train=data["x_train"],
            x_test=data["x_test"],
            x_val=data["x_val"],
            train_y=data["train_y"],
            test_mean=data["test_mean"],
            test_cov=data["test_cov"],
            val_mean=data["val_mean"],
            val_cov=data["val_cov"],
            noise_std=0.1,
            kernel_ridge=1e-6,
        )

        prior_knowledge = PriorKnowledge(
            input_dim=data["input_dim"],
            num_train=data["num_train"],
            num_classes=data["num_classes"],
            layers=data["num_layers"],
            noise_std=data["noise_std"],
        )

        result = TestbedGPRegression(
            data_sampler=data_sampler,
            prior=prior_knowledge,
            num_enn_samples=data["num_enn_samples"],
            std_ridge=data["std_ridge"],
        )

        return result


def _kl_gaussian(
    mean_1: float, std_1: float, mean_2: float, std_2: float
) -> torch.Tensor:
    """Computes the KL(P_1 || P_2) for P_1,P_2 univariate Gaussian."""
    log_term = torch.log(std_2 / std_1)
    frac_term = (std_1**2 + (mean_1 - mean_2) ** 2) / (2 * std_2**2)
    return log_term + frac_term - 0.5
