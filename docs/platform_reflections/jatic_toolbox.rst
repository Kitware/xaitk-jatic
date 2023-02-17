#############
JATIC Toolbox
#############


Protocols
=========

To allow for easy interoperability between ``xaitk-saliency`` and other JATIC
tools, it is necessary to ensure that the protocols defined within the
``jatic-toolbox`` are taken into consideration within ``xaitk-saliency``. As
``xaitk-saliency`` is already highly type-hinted, protocols must be used
somewhat directly for the relationship to be detected by static type checkers.
This constraint also means that including ``jatic-toolbox`` as a dependency for
``xaitk-saliency`` is preferred.

To identify the best protocol application strategy, the object detection
protocol was first applied. Two different strategies were explored:

* Replacing the current detection saliency API with the protocol.
* Creating a  wrapper to curry compatability with the protocol.

The detection saliency API is structured around the interfaces defined by
``smqtk-detection``. Thus, in order to directly apply the JATIC protocol within
``xaitk-saliency``, changes must also be be made to ``smqtk-detection``
(creating a cascade of changes). As the object detection protocol is primarily
based off of the API defined by ``smqtk-detection``, the changes that must be
made are relatively straightforward. However, the changes required are not
backwards-compatible due to the highly-typed nature of ``xaitk-saliency`` and
would render existing applications of ``xaitk-saliency`` broken. Therefore, in
the case of ``xaitk-saliency``, the wrapper strategy seems to be more
appropriate. In a less established API or one with more-permissive typing,
directly applying the protocol would likely result in a smoother user
experience.

Due to the similarity between the object detection protocol and the existing
API, the wrapper to transform a protocol-based detector for use with
``xaitk-saliency`` is relatively simple. With an almost one-to-one
correspondence, data structure transformation is the bulk of the wrapper
implementation. The use of a wrapper means that no upstream changes are
necessary and existing applications of ``xaitk-saliency`` remain functional. To
use a protocol-based detector with ``xaitk-saliency``, the user simply wraps
the instance and then the detector may be used as if it is a "first-class"
``xaitk-saliency`` detector; a near seamless user experience. This process
is demonstrated in our example notebook in ``examples/jatic_toolbox``.
