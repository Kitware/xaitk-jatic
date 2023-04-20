import numpy as np
from typing import Sequence, Union

from jatic_toolbox.errors import InvalidArgument
from jatic_toolbox.protocols import ArrayLike

try:
    import torch  # type: ignore
except ModuleNotFoundError:
    pass
from importlib.util import find_spec


def to_numpy_array(data: Union[ArrayLike, Sequence[ArrayLike]]) -> np.ndarray:
    """
    Attempts to convert provided data to an np.ndarray.

    Ideally, this eventually gets replaced with a utility from jatic_toolbox
    """

    def _is_torch_available() -> bool:
        return find_spec('torch') is not None

    if _is_torch_available() and isinstance(data, torch.Tensor):
        return data.detach().numpy()
    elif isinstance(data, np.ndarray):
        return data
    else:
        raise InvalidArgument(f"Unsupported data type {type(data)}.")
