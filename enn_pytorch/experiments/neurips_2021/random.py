from typing import List
import torch


def split_seed(seed: int, count: int) -> List[int]:
    
    generator = torch.Generator()
    generator.manual_seed(seed)
    result = torch.randint(0, torch.iinfo(torch.int64).max, (count,), generator=generator)
    return result.tolist()