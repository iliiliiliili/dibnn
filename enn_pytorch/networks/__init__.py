# python3
# pylint: disable=g-bad-file-header
# Copyright Illia Oleksiienko
# This file is a modified version for pytorch of the original JAX implementation
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

"""Exposing the public methods of the networks - PyTorch version."""

# Dropout
from enn_pytorch.networks.dropout import MLPDropoutENN

# Ensemble
from enn_pytorch.networks.ensembles import Ensemble
from enn_pytorch.networks.ensembles import make_mlp_ensemble_prior_fns
from enn_pytorch.networks.ensembles import MLPEnsembleArbitraryPrior
from enn_pytorch.networks.ensembles import MLPEnsembleEnn
from enn_pytorch.networks.ensembles import MLPEnsembleGpPrior
from enn_pytorch.networks.ensembles import MLPEnsembleMatchedPrior
from enn_pytorch.networks.ensembles import wrap_sequence_as_prior

# Layer Ensemble
from enn_pytorch.networks.layer_ensembles import LayerEnsembleNetworkWithPriors
from enn_pytorch.networks.layer_ensembles import LayerEnsembleNetwork
from enn_pytorch.networks.layer_ensembles import make_einsum_layer_ensemble_mlp_with_prior_enn
from enn_pytorch.networks.layer_ensembles import make_true_einsum_layer_ensemble_mlp_with_prior_enn
from enn_pytorch.networks.layer_ensembles import make_layer_ensemble_cor_mlp_with_prior_enn

# Indexers
from enn_pytorch.networks.indexers import DirichletIndexer
from enn_pytorch.networks.indexers import EnsembleIndexer
from enn_pytorch.networks.indexers import LayerEnsembleIndexer
from enn_pytorch.networks.indexers import GaussianWithUnitIndexer
from enn_pytorch.networks.indexers import PrngIndexer
from enn_pytorch.networks.indexers import ScaledGaussianIndexer

# Priors
from enn_pytorch.networks.priors import convert_enn_to_prior_fn
from enn_pytorch.networks.priors import EnnWithAdditivePrior
from enn_pytorch.networks.priors import get_random_mlp_with_index
from enn_pytorch.networks.priors import make_null_prior
from enn_pytorch.networks.priors import make_random_feat_gp
from enn_pytorch.networks.priors import NetworkWithAdditivePrior
from enn_pytorch.networks.priors import PriorFn
