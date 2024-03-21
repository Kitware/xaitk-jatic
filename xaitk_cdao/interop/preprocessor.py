import abc

from smqtk_core import Plugfigurable

from maite.protocols import SupportsArray


class Preprocessor(Plugfigurable):
    """
    A preprocessor to convert inputs into the format expected by the downstream model.
    """

    @abc.abstractmethod
    def __call__(self, x: SupportsArray) -> SupportsArray:
        """
        Perform the preprocessing action on the input array-like object.

        :param x: Input array-like to be preprocessed.
        :returns: Resulting array after preprocessing.
        """
