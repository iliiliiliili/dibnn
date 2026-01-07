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

"""Exposing the public methods of losses - PyTorch version."""

from src.losses.single_index import AccuracyErrorLoss
from src.losses.single_index import add_data_noise
from src.losses.single_index import average_single_index_loss
from src.losses.single_index import BatchedL2Loss
from src.losses.single_index import batched_average_single_index_loss
from src.losses.single_index import NElboLoss
from src.losses.single_index import L2Loss
from src.losses.single_index import SingleIndexLossFn
from src.losses.single_index import XentLoss
