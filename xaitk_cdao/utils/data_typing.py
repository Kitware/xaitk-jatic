import numpy as np
from typing import List, Sequence, Union

from maite.errors import InvalidArgument
from maite.protocols import ArrayLike

try:
    import torch  # type: ignore
except ModuleNotFoundError:
    pass
from importlib.util import find_spec


def to_numpy_array(data: Union[ArrayLike, Sequence[ArrayLike]]) -> np.ndarray:
    """
    Attempts to convert provided data to an np.ndarray.

    Ideally, this eventually gets replaced with a utility from maite
    """

    if find_spec('torch') is not None and isinstance(data, torch.Tensor):
        return data.detach().numpy()
    elif isinstance(data, List):
        return np.asarray([to_numpy_array(d) for d in data])
    elif isinstance(data, np.ndarray):
        return data
    else:
        raise InvalidArgument(f"Unsupported data type {type(data)}.")
