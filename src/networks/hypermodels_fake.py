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

"""Prototype for linear hypermodel in PyTorch."""

from typing import Callable, Optional, Sequence, Dict, Any, List
import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
from collections import OrderedDict

from src import base
from src.networks import priors
from src.networks.functional import FunctionalMLP


class MLP(nn.Module):
    """Simple MLP implementation."""

    def __init__(
        self,
        output_sizes: Sequence[int],
        w_init: Optional[Callable] = None,
        b_init: Optional[Callable] = None,
    ):
        super().__init__()
        self.output_sizes = output_sizes
        layers = []
        for i, size in enumerate(output_sizes):
            # Linear layers will be created when we know input size
            layers.append(None)  # Placeholder
        self.layers = nn.ModuleList()
        self._output_sizes = output_sizes
        self._w_init = w_init
        self._b_init = b_init
        self._initialized = False

    def _lazy_init(self, input_size: int):
        """Initialize layers when input size is known."""
        if self._initialized:
            return
        layers = []
        prev_size = input_size
        for size in self._output_sizes:
            layer = nn.Linear(prev_size, size)
            if self._w_init is not None:
                self._w_init(layer.weight)
            if self._b_init is not None:
                self._b_init(layer.bias)
            layers.append(layer)
            prev_size = size
        self.layers = nn.ModuleList(layers)
        self._initialized = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._initialized:
            self._lazy_init(x.shape[-1])

        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = torch.relu(x)
        return x


class DiagonalLinear(nn.Module):
    """Diagonal Linear module."""

    def __init__(
        self,
        input_size: Optional[int] = None,
        with_bias: bool = True,
        w_init: Optional[Callable] = None,
        b_init: Optional[Callable] = None,
    ):
        """Constructs the diagonal linear module.

        Args:
            input_size: Size of input dimension. If None, will be inferred on first forward.
            with_bias: Whether to add a bias to the output.
            w_init: Optional initializer for weights.
            b_init: Optional initializer for bias.
        """
        super().__init__()
        self.input_size = input_size
        self.with_bias = with_bias
        self._w_init = w_init
        self._b_init = b_init

        if input_size is not None:
            self._initialize_parameters()
        else:
            self.w = None
            self.b = None

    def _initialize_parameters(self):
        """Initialize weight and bias parameters."""
        if self._w_init is None:
            stddev = 1.0 / np.sqrt(self.input_size)
            self.w = nn.Parameter(torch.randn(self.input_size) * stddev)
        else:
            self.w = nn.Parameter(torch.empty(self.input_size))
            self._w_init(self.w)

        if self.with_bias:
            if self._b_init is None:
                self.b = nn.Parameter(torch.zeros(self.input_size))
            else:
                self.b = nn.Parameter(torch.empty(self.input_size))
                self._b_init(self.b)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Computes a diagonal linear transform of the input."""
        if inputs.dim() == 0:
            raise ValueError("Input must not be scalar.")

        if self.w is None:
            self.input_size = inputs.shape[-1]
            self._initialize_parameters()

        # Apply softplus to weights: log(1 + exp(w))
        out = inputs * torch.nn.functional.softplus(self.w)

        if self.with_bias:
            out = out + self.b

        return out


class HypermodelModule(nn.Module):
    """Hypermodel that generates parameters for a base network."""

    def __init__(
        self,
        base_model: nn.Module,
        hyper_torso: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        diagonal_linear_hyper: bool = False,
        return_generated_params: bool = False,
        scale: bool = True,
    ):
        """Initialize hypermodel.

        Args:
            base_model: Base network whose parameters will be generated.
            hyper_torso: Transformation of index before final layer.
            diagonal_linear_hyper: Whether to use diagonal linear layer.
            return_generated_params: Whether to return generated parameters.
            scale: Whether to scale generated parameters.
        """
        super().__init__()
        self.base_model = base_model
        self.hyper_torso = hyper_torso if hyper_torso is not None else nn.Identity()
        self.diagonal_linear_hyper = diagonal_linear_hyper
        self.return_generated_params = return_generated_params
        self.scale = scale

        self.param_shapes = OrderedDict()
        self.param_sizes = OrderedDict()
        total_params = 0

        for name, param in base_model.named_parameters():
            self.param_shapes[name] = param.shape
            size = param.numel()
            self.param_sizes[name] = size
            total_params += size

        # Create hyper network layers
        if diagonal_linear_hyper:
            self.hyper_final = DiagonalLinear(input_size=total_params)
        else:
            # Create separate linear layers for each parameter
            self.hyper_layers = nn.ModuleDict()
            for name, size in self.param_sizes.items():
                # Layer will be created on first forward when we know hyper_torso output size
                self.hyper_layers[name] = None

        self._initialized = False

    def _lazy_init_hyper_layers(self, hyper_index_size: int):
        """Initialize hyper layers when index size is known."""
        if self._initialized or self.diagonal_linear_hyper:
            return
        for name, size in self.param_sizes.items():
            self.hyper_layers[name] = nn.Linear(hyper_index_size, size)
        self._initialized = True

    def scale_params(
        self, params_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Scale parameters for variance stability."""
        scaled = {}
        for name, value in params_dict.items():
            # Scale weights by 1/sqrt(fan_in), leave biases unchanged
            if "weight" in name or name.endswith(".w"):
                fan_in = value.shape[0] if len(value.shape) > 0 else 1
                scaled[name] = value / np.sqrt(fan_in)
            else:
                scaled[name] = value
        return scaled

    def forward(self, inputs: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        """Forward pass through hypermodel.

        Args:
            inputs: Input to base network.
            index: Epistemic index for parameter generation.

        Returns:
            Output of base network with generated parameters.
        """
        # Transform index through hyper torso
        if isinstance(self.hyper_torso, nn.Identity):
            hyper_index = index
        else:
            hyper_index = self.hyper_torso(index)

        # Generate parameters
        if self.diagonal_linear_hyper:
            # Generate all parameters at once
            flat_params = self.hyper_final(index)

            # Split into individual parameters
            generated_params = {}
            start_idx = 0
            for name, size in self.param_sizes.items():
                param_flat = flat_params[start_idx : start_idx + size]
                generated_params[name] = param_flat.view(self.param_shapes[name])
                start_idx += size
        else:
            # Initialize layers if needed
            if not self._initialized:
                self._lazy_init_hyper_layers(hyper_index.shape[-1])

            # Generate parameters using separate layers
            generated_params = {}
            for name, layer in self.hyper_layers.items():
                param_flat = layer(hyper_index)
                generated_params[name] = param_flat.view(self.param_shapes[name])

        # Scale parameters if requested
        if self.scale:
            generated_params = self.scale_params(generated_params)

        # Apply generated parameters to base model
        # We need to temporarily override the base model's parameters
        original_params = {
            name: param.data.clone()
            for name, param in self.base_model.named_parameters()
        }

        try:
            # Set generated parameters
            for name, param in self.base_model.named_parameters():
                param.data = generated_params[name]

            # Forward through base model
            output = self.base_model(inputs)

            if self.return_generated_params:
                return base.OutputWithPrior(
                    train=output,
                    extra={
                        "hyper_net_out": generated_params,
                        "base_net_params": (
                            generated_params
                            if not self.scale
                            else self.scale_params(generated_params)
                        ),
                    },
                )
            return output
        finally:
            # Restore original parameters
            for name, param in self.base_model.named_parameters():
                param.data = original_params[name]


class MLPHypermodel(base.EpistemicNetwork):
    """MLP hypermodel for base network as EpistemicNetwork."""

    def __init__(
        self,
        base_model: nn.Module,
        dummy_input: torch.Tensor,
        indexer: base.EpistemicIndexer,
        hidden_sizes: Optional[Sequence[int]] = None,
        return_generated_params: bool = False,
        scale: bool = True,
        w_init: Optional[Callable] = None,
        b_init: Optional[Callable] = None,
    ):
        """MLP hypermodel for base network as EpistemicNetwork.

        Args:
            base_model: Base PyTorch model.
            dummy_input: Example input for shape inference.
            indexer: Epistemic indexer.
            hidden_sizes: Hidden layer sizes for hyper network.
            return_generated_params: Whether to return generated parameters.
            scale: Whether to scale parameters.
            w_init: Weight initializer.
            b_init: Bias initializer.
        """
        super().__init__()

        self.indexer = indexer

        # Create hyper torso
        if hidden_sizes is None:
            hyper_torso = nn.Identity()
        else:
            hyper_torso = MLP(hidden_sizes, w_init=w_init, b_init=b_init)

        # Create hypermodel module
        self.module = HypermodelModule(
            base_model=base_model,
            dummy_input=dummy_input,
            hyper_torso=hyper_torso,
            return_generated_params=return_generated_params,
            scale=scale,
        )

    def forward(self, inputs: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.module(inputs, index)

    def __call__(self, inputs: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        """Call method for compatibility."""
        return self.forward(inputs, index)


class MLPHypermodelWithHypermodelPrior(base.EpistemicNetwork):
    """MLP hypermodel with hypermodel prior as EpistemicNetwork."""

    def __init__(
        self,
        base_output_sizes: Sequence[int],
        prior_scale: float,
        dummy_input: torch.Tensor,
        indexer: base.EpistemicIndexer,
        prior_base_output_sizes: Sequence[int],
        hyper_hidden_sizes: Optional[Sequence[int]] = None,
        prior_hyper_hidden_sizes: Optional[Sequence[int]] = None,
        w_init: Optional[Callable] = None,
        b_init: Optional[Callable] = None,
        return_generated_params: bool = False,
        seed: int = 0,
        scale: bool = True,
    ):
        """MLP hypermodel with hypermodel prior as EpistemicNetwork.

        Args:
            base_output_sizes: Output sizes for base MLP.
            prior_scale: Scale factor for prior.
            dummy_input: Example input.
            indexer: Epistemic indexer.
            prior_base_output_sizes: Output sizes for prior base MLP.
            hyper_hidden_sizes: Hidden sizes for main hypermodel.
            prior_hyper_hidden_sizes: Hidden sizes for prior hypermodel.
            w_init: Weight initializer.
            b_init: Bias initializer.
            return_generated_params: Whether to return generated parameters.
            seed: Random seed.
            scale: Whether to scale parameters.
        """
        super().__init__()

        self.indexer = indexer

        # Create base model
        base_model = MLP(base_output_sizes, w_init=w_init, b_init=b_init)

        # Create prior base model
        prior_base_model = MLP(prior_base_output_sizes, w_init=w_init, b_init=b_init)

        # Create prior ENN
        prior_enn = MLPHypermodel(
            base_model=prior_base_model,
            dummy_input=dummy_input,
            indexer=indexer,
            hidden_sizes=prior_hyper_hidden_sizes,
            return_generated_params=return_generated_params,
            w_init=w_init,
            b_init=b_init,
            scale=scale,
        )

        # Convert to prior function
        torch.manual_seed(seed)
        prior_fn = priors.convert_enn_to_prior_fn(prior_enn, dummy_input)

        # Create main ENN without prior
        enn_wo_prior = MLPHypermodel(
            base_model=base_model,
            dummy_input=dummy_input,
            indexer=indexer,
            hidden_sizes=hyper_hidden_sizes,
            return_generated_params=return_generated_params,
            w_init=w_init,
            b_init=b_init,
            scale=scale,
        )

        # Wrap with additive prior
        self.enn = priors.EnnWithAdditivePrior(
            enn_wo_prior, prior_fn, prior_scale=prior_scale
        )

    def forward(self, inputs: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.enn(inputs, index)


################################################################################
# Alternative implementation of MLP hypermodel with MLP prior where layers
# are generated by different set of indices.


class HyperLinear(nn.Module):
    """Linear hypermodel layer."""

    def __init__(
        self,
        output_size: int,
        index_dim_per_layer: int,
        weight_scaling: float = 1.0,
        bias_scaling: float = 1.0,
        fixed_bias_val: float = 0.0,
    ):
        """Initialize HyperLinear.

        Args:
            output_size: Output dimension.
            index_dim_per_layer: Index dimension for this layer.
            weight_scaling: Scaling factor for weights.
            bias_scaling: Scaling factor for bias.
            fixed_bias_val: Fixed bias value offset.
        """
        super().__init__()
        self.output_size = output_size
        self.index_dim_per_layer = index_dim_per_layer
        self.weight_scaling = weight_scaling
        self.bias_scaling = bias_scaling
        self.fixed_bias_val = fixed_bias_val

        # Parameters will be initialized on first forward
        self.w = None
        self.b = None
        self._initialized = False

    def _lazy_init(self, hidden_size: int):
        """Initialize parameters when hidden size is known."""
        if self._initialized:
            return

        # Initialize w and b with random normal
        self.w = nn.Parameter(
            torch.randn(self.output_size, hidden_size, self.index_dim_per_layer)
        )
        self.b = nn.Parameter(torch.randn(self.output_size, self.index_dim_per_layer))
        self._initialized = True

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape [batch_size, hidden_size].
            z: Index tensor of shape [index_dim_per_layer].

        Returns:
            Output tensor of shape [batch_size, output_size].
        """
        batch_size, hidden_size = x.shape

        if not self._initialized:
            self._lazy_init(hidden_size)

        # Normalize w and b
        w = self.w / torch.norm(self.w, dim=-1, keepdim=True)
        b = self.b / torch.norm(self.b, dim=-1, keepdim=True)

        # Scale w and b
        w = w * np.sqrt(self.weight_scaling / hidden_size)
        b = b * np.sqrt(self.bias_scaling) + self.fixed_bias_val

        # Generate weights and biases from index
        # w: [output_size, hidden_size, index_dim]
        # z: [index_dim]
        # weights: [output_size, hidden_size]
        weights = torch.einsum("ohi,i->oh", w, z)
        bias = torch.einsum("oi,i->o", b, z)

        # Apply linear transformation
        # x: [batch_size, hidden_size]
        # weights: [output_size, hidden_size]
        # output: [batch_size, output_size]
        output = torch.einsum("oh,bh->bo", weights, x) + bias

        return output


class PriorMLPIndependentLayers(nn.Module):
    """Prior MLP with each layer generated by an independent index."""

    def __init__(
        self,
        output_sizes: Sequence[int],
        index_dim: int,
        weight_scaling: float = 1.0,
        bias_scaling: float = 1.0,
        fixed_bias_val: float = 0.0,
    ):
        """Initialize PriorMLPIndependentLayers.

        Args:
            output_sizes: Output size for each layer.
            index_dim: Total index dimension.
            weight_scaling: Weight scaling factor.
            bias_scaling: Bias scaling factor.
            fixed_bias_val: Fixed bias value.
        """
        super().__init__()
        self.output_sizes = output_sizes
        self.num_layers = len(output_sizes)
        self.index_dim = index_dim

        # Determine index splits
        if index_dim < self.num_layers:
            # All layers share the same index
            self.layers_indices = [list(range(index_dim))] * self.num_layers
        else:
            # Split index dimensions among layers
            indices_array = np.arange(index_dim)
            splits = np.array_split(indices_array, self.num_layers)
            self.layers_indices = [s.tolist() for s in splits]

        # Create layers
        self.layers = nn.ModuleList()
        for layer_indices, output_size in zip(self.layers_indices, output_sizes):
            index_dim_per_layer = len(layer_indices)
            layer = HyperLinear(
                output_size,
                index_dim_per_layer,
                weight_scaling,
                bias_scaling,
                fixed_bias_val,
            )
            self.layers.append(layer)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor.
            z: Index tensor of shape [index_dim].

        Returns:
            Output tensor.
        """
        # Split index for each layer
        if self.index_dim < self.num_layers:
            index_layers = [z] * self.num_layers
        else:
            index_layers = [z[indices] for indices in self.layers_indices]

        out = x
        for i, (layer, index_layer) in enumerate(zip(self.layers, index_layers)):
            out = layer(out, index_layer)
            if i < self.num_layers - 1:
                out = torch.relu(out)
        return out


class MLPHypermodelPriorIndependentLayers(base.EpistemicNetwork):
    """MLP hypermodel with independent layer priors as EpistemicNetwork."""

    def __init__(
        self,
        base_output_sizes: Sequence[int],
        prior_scale: float,
        dummy_input: torch.Tensor,
        indexer: base.EpistemicIndexer,
        prior_base_output_sizes: Sequence[int],
        hyper_hidden_sizes: Optional[Sequence[int]] = None,
        w_init: Optional[Callable] = None,
        b_init: Optional[Callable] = None,
        return_generated_params: bool = False,
        prior_weight_scaling: float = 1.0,
        prior_bias_scaling: float = 1.0,
        prior_fixed_bias_val: float = 0.0,
        seed: int = 0,
        scale: bool = True,
        problem_temperature: Optional[float] = None,
    ):
        """Initialize MLPHypermodelPriorIndependentLayers.

        Args:
            base_output_sizes: Output sizes for base network.
            prior_scale: Prior scaling factor.
            dummy_input: Example input.
            indexer: Epistemic indexer.
            prior_base_output_sizes: Output sizes for prior network.
            hyper_hidden_sizes: Hidden sizes for hypermodel.
            w_init: Weight initializer.
            b_init: Bias initializer.
            return_generated_params: Whether to return generated parameters.
            prior_weight_scaling: Prior weight scaling.
            prior_bias_scaling: Prior bias scaling.
            prior_fixed_bias_val: Prior fixed bias value.
            seed: Random seed.
            scale: Whether to scale parameters.
            problem_temperature: Temperature scaling for outputs.
        """
        super().__init__()

        self.indexer = indexer
        self.problem_temperature = problem_temperature

        # Create base model
        class BaseNet(nn.Module):
            def __init__(self, output_sizes, temperature):
                super().__init__()
                self.mlp = MLP(output_sizes, w_init=w_init, b_init=b_init)
                self.temperature = temperature

            def forward(self, x):
                out = self.mlp(x)
                if self.temperature is not None:
                    out = out / self.temperature
                return out

        base_model = BaseNet(base_output_sizes, problem_temperature)

        # Get index dimension
        torch.manual_seed(seed)
        index = indexer(torch.Generator().manual_seed(seed))
        index_dim = index.shape[0]

        # Create prior network
        class PriorNet(nn.Module):
            def __init__(self, output_sizes, index_dim, temperature):
                super().__init__()
                self.prior_mlp = PriorMLPIndependentLayers(
                    output_sizes=output_sizes,
                    index_dim=index_dim,
                    weight_scaling=prior_weight_scaling,
                    bias_scaling=prior_bias_scaling,
                    fixed_bias_val=prior_fixed_bias_val,
                )
                self.temperature = temperature

            def forward(self, x, z):
                out = self.prior_mlp(x, z)
                if self.temperature is not None:
                    out = out / self.temperature
                return out

        prior_net = PriorNet(prior_base_output_sizes, index_dim, problem_temperature)

        # Initialize prior network
        torch.manual_seed(seed)
        with torch.no_grad():
            _ = prior_net(dummy_input, index)

        # Create prior function
        def prior_fn(x, z):
            return prior_net(x, z)

        # Create main ENN without prior
        enn_wo_prior = MLPHypermodel(
            base_model=base_model,
            dummy_input=dummy_input,
            indexer=indexer,
            hidden_sizes=hyper_hidden_sizes,
            return_generated_params=return_generated_params,
            w_init=w_init,
            b_init=b_init,
            scale=scale,
        )

        # Wrap with additive prior
        self.enn = priors.EnnWithAdditivePrior(
            enn_wo_prior, prior_fn, prior_scale=prior_scale
        )

    def forward(self, inputs: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.enn(inputs, index)
