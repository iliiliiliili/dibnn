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
from src.networks.dropout import MLPDropoutENN

from src.networks.bbb import MlpBbbEnn

# Ensemble
from src.networks.ensembles import Ensemble
from src.networks.ensembles import make_mlp_ensemble_prior_fns
from src.networks.ensembles import MLPEnsembleArbitraryPrior
from src.networks.ensembles import MLPEnsembleEnn
from src.networks.ensembles import MLPEnsembleGpPrior
from src.networks.ensembles import MLPEnsembleMatchedPrior
from src.networks.ensembles import wrap_sequence_as_prior

# Layer Ensemble
from src.networks.layer_ensembles import LayerEnsembleNetworkWithPriors
from src.networks.layer_ensembles import LayerEnsembleNetwork
from src.networks.layer_ensembles import make_einsum_layer_ensemble_mlp_with_prior_enn
from src.networks.layer_ensembles import (
    make_true_einsum_layer_ensemble_mlp_with_prior_enn,
)
from src.networks.layer_ensembles import make_layer_ensemble_cor_mlp_with_prior_enn

# Indexers
from src.networks.indexers import DirichletIndexer
from src.networks.indexers import EnsembleIndexer
from src.networks.indexers import LayerEnsembleIndexer
from src.networks.indexers import GaussianWithUnitIndexer
from src.networks.indexers import PrngIndexer
from src.networks.indexers import ScaledGaussianIndexer

# Priors
from src.networks.priors import convert_enn_to_prior_fn
from src.networks.priors import EnnWithAdditivePrior
from src.networks.priors import get_random_mlp_with_index
from src.networks.priors import make_null_prior
from src.networks.priors import make_random_feat_gp
from src.networks.priors import NetworkWithAdditivePrior
from src.networks.priors import PriorFn
