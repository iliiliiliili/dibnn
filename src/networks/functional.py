from typing import Any, Callable, List, Optional, Tuple, Union
from torch import nn
import torch


class FunctionalBase(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def build(
        self,
        layer: Callable,
    ) -> None:

        super().__init__()

        self.layer = layer

    def forward(self, input, *weights):

        x = self.layer(input, *weights)
        return x


class BatchedFunctionalLinear(FunctionalBase):

    def __init__(
        self,
    ) -> None:

        super().__init__()

        def layer(inputs, weights, biases=None):
            result = inputs @ weights.mT

            if biases is not None:
                result = result + biases.unsqueeze(1)
            
            return result

        # layer = lambda inputs, weights: torch.nn.functional.linear(
        #     inputs,
        #     weights[0],
        #     None if len(weights) < 2 or weights[1].shape[0] == 0 else weights[1],
        # )

        super().build(
            layer,
        )


class FunctionalMLP(nn.Module):

    def __init__(
        self, activation: Callable = torch.relu
    ) -> None:

        super().__init__()
        
        self.layer = BatchedFunctionalLinear()
        self.activation = activation
    
    def forward(self, inputs: torch.Tensor, weights: List[Tuple[torch.Tensor, ...]]) -> torch.Tensor:
        x = inputs
        for i, layer_weights in enumerate(weights):
            x = self.layer(x, layer_weights)

            if i < len(weights) - 1:
                x = self.activation(x)
        return x

