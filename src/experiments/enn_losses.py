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
"""Helpful losses for the ENN agent - PyTorch version."""

from typing import Callable, Optional

import numpy as np

from src import base as enn_base, utils
from src import losses
from src.experiments import base as testbed_base

import torch
import torch.nn as nn

from src.losses import single_index
from src.losses.utils import add_l2_weight_decay


EnnCtor = Callable[[testbed_base.PriorKnowledge], enn_base.EpistemicNetwork]
LossCtor = Callable[
    [testbed_base.PriorKnowledge, enn_base.EpistemicNetwork], enn_base.LossFn
]


def gaussian_regression_loss(
    num_index_samples: int,
    noise_scale: float = 1,
    l2_weight_decay: float = 0,
    exclude_bias_l2: bool = True,
) -> LossCtor:
    """Constructs a loss for Gaussian regression."""

    def loss_ctor(
        prior: testbed_base.PriorKnowledge, enn: enn_base.EpistemicNetwork
    ) -> enn_base.LossFn:
        """Add a matching Gaussian noise to the target y."""
        single_loss = losses.L2Loss()
        loss_fn = losses.average_single_index_loss(single_loss, num_index_samples)
        return loss_fn

    return loss_ctor


def regularized_dropout_loss(
    num_index_samples: int = 10,
    dropout_rate: float = 0.05,
    scale: float = 1e-2,
    tau: float = 1.0,
) -> LossCtor:
    """Constructs a regularized loss for dropout-based uncertainty."""

    def loss_ctor(
        prior: testbed_base.PriorKnowledge, enn: enn_base.EpistemicNetwork
    ) -> enn_base.LossFn:
        single_loss = losses.L2Loss()
        reg = (scale**2) * (1 - dropout_rate) / (2.0 * prior.num_train * tau)
        loss_fn = losses.average_single_index_loss(single_loss, num_index_samples)
        loss_fn = add_l2_weight_decay(loss_fn, scale=reg)
        return loss_fn

    return loss_ctor


def get_awgn_loglike_fn(sigma_w: float) -> Callable[[enn_base.Output, enn_base.Batch], float]:
    """Returns a function that computes the simple unnormalized log likelihood.

  It assumes response variable is perturbed with additive iid Gaussian noise.

  Args:
    sigma_w: standard deviation of the additive Gaussian noise.

  Returns:
    A function that computes the log likelihood given data and output.

  """

    def log_likelihood_fn(out: enn_base.Output, batch: enn_base.Batch):
        net_out = utils.parse_net_output(out)
        err_sq = torch.mean(torch.square(net_out - batch.y))
        result = -0.5 * err_sq / sigma_w ** 2
        return result

    return log_likelihood_fn



def create_bbb_nelbo_loss(
    log_likelihood_fn: Callable[[enn_base.Output, enn_base.Batch], float],
    sigma_0: float,
    num_samples: float,
    num_index_samples: float,
) -> single_index.NElboLoss:
    """Returns the negative ELBO for bbb.

  Args:
    log_likelihood_fn: log likelihood function.
    sigma_0: Standard deviation of the Gaussian latent (params) prior.
    num_samples: effective number of samples.
  Returns:
    Negative ELBO value.
  """

    def model_prior_kl_fn(
        out: enn_base.Output, model, index: enn_base.Index
    ) -> float:
        """Compute the KL distance between model and prior densities in a linear HM.

    weights `w` and biases `b` are assumed included in `params`. The latent
    variables (which are the parameters of the base network) are generated as u
    = z @ w + b where z is the index variable. The index is assumed Gaussian
    *with variance equal to the prior variance* of the latent variables.

    This function also  assumes a Gaussian prior distribution for the latent,
    i.e., parameters of the base network.

    Args:
      out: final output of the hypermodel, i.e., y = f_theta(x, z)
      params: parameters of the hypermodel (Note that this is the parameters of
        the hyper network since base network params are set by the hyper net.)
      index: index z

    Returns:
      KL distance.
    """

        del out, index  # Here we compute the log prob from params directly.

        result = 0.0

        all_weight_sigmas, all_bias_sigmas, all_weight_mus, all_bias_mus = model.get_params_tuple()

        for layer in range(len(all_weight_sigmas)):

            weight_sigmas = all_weight_sigmas[layer]
            bias_sigmas = all_bias_sigmas[layer]
            weight_mus = all_weight_mus[layer]
            bias_mus = all_bias_mus[layer]

            weight_scales = torch.nn.functional.softplus(weight_sigmas)
            bias_scales = torch.nn.functional.softplus(bias_sigmas)

            weight_kl = (
                0.5
                / num_samples
                * (
                    torch.sum(torch.square(weight_scales))
                    + torch.sum(torch.square(weight_mus)) / (sigma_0 ** 2)
                    - np.prod(weight_mus.shape)
                    - 2 * torch.sum(torch.log(weight_scales))
                )
            )

            bias_kl = (
                0.5
                / num_samples
                * (
                    torch.sum(torch.square(bias_scales))
                    + torch.sum(torch.square(bias_mus)) / (sigma_0 ** 2)
                    - np.prod(bias_mus.shape)
                    - 2 * torch.sum(torch.log(bias_scales))
                )
            )

            result += weight_kl + bias_kl

            # layer_kl = (
            #         torch.sum(torch.square(weight_scales))
            #         + torch.sum(torch.square(bias_scales))
            #         + (
            #             torch.sum(torch.square(weight_mus))
            #             + torch.sum(torch.square(bias_mus))
            #         ) / (sigma_0 ** 2)
            #         - (len(weight_mus) + len(bias_mus))
            #         - 2 * torch.sum(torch.log(weight_scales))
            #         - 2 * torch.sum(torch.log(bias_scales))
                
            # )

            # result += layer_kl
        
        # result = 0.5 / num_samples * result
        return result

    return single_index.NElboLoss(
        log_likelihood_fn=log_likelihood_fn, model_prior_kl_fn=model_prior_kl_fn, num_index_samples=num_index_samples
    )

    
def bbb_loss(sigma_0: float = 100, num_index_samples: int = 64):
    """Constructs the loss function for bbb agent."""

    def loss_ctor(
        prior: testbed_base.PriorKnowledge, enn: enn_base.EpistemicNetwork
    ) -> enn_base.LossFn:
        del enn
        log_likelihood_fn = get_awgn_loglike_fn(prior.noise_std)
        # loss_fn = losses.average_single_index_loss(
        #     create_bbb_nelbo_loss(
        #         log_likelihood_fn=log_likelihood_fn,
        #         sigma_0=sigma_0,
        #         num_samples=prior.num_train,
        #     ),
        #     num_index_samples=num_index_samples,
        # )
        loss_fn = create_bbb_nelbo_loss(
                log_likelihood_fn=log_likelihood_fn,
                sigma_0=sigma_0,
                num_samples=prior.num_train,
                num_index_samples=num_index_samples,
            )
        return loss_fn

    return loss_ctor
