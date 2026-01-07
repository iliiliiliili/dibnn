from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch

PredicateFn = Callable[[str, str, Any], bool]


def l2_weights_with_predicate(
    params: Dict, predicate: Optional[PredicateFn] = None
) -> torch.Tensor:
    """Sum of squares of parameter weights that passes predicate_fn."""
    if predicate is not None:
        params = {k: v for k, v in params.items() if predicate("", k, v)}
    return sum(torch.sum(torch.square(p)) for p in params.values())


def add_l2_weight_decay(
    loss_fn: Callable,
    scale: Union[float, Callable[[Dict], Dict]],
    predicate: Optional[PredicateFn] = None,
) -> Callable:
    """Adds scale * l2 weight decay to an existing loss function."""
    try:  # Scale is numeric.
        scale = torch.sqrt(torch.tensor(scale))
        scale_fn = lambda ps: {k: scale * v for k, v in ps.items()}
    except TypeError:
        scale_fn = scale  # Assuming scale is a Callable.

    def new_loss(a, model: torch.nn.Module, c, d, device) -> Tuple[torch.Tensor, Dict]:
        loss, metrics = loss_fn(a, model, c, d, device)
        decay = l2_weights_with_predicate(scale_fn(model.state_dict()), predicate)
        total_loss = loss + decay
        metrics["decay"] = decay
        metrics["raw_loss"] = loss
        return total_loss, metrics

    return new_loss
