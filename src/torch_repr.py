
import torch

def custom_repr(self):
    return (
        f"T{tuple(self.shape)}".replace("(", "[").replace(")", "]") + f"<m={self.mean().item():.2f}>"
        if len(self.shape) > 0
        else original_repr(self)
    )


original_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = custom_repr
torch.Tensor.__str__ = original_repr
