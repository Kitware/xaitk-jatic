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


Object Detection
----------------

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


Image Classification
--------------------

Following the findings from applying the object detection protocol, a wrapper
for image classification was defined to curry compatability between protocol-
based classifiers and ``xaitk-saliency``. The protocol specifying a
``Classifier`` itself is relatively simple and is similiar to the API defined
by ``smqtk-classifier``. One thing of note is that the protocol allows for
either logits or probabilities as classifier outputs. As ``xaitk-saliency``
operates upon probabilities, it was necessary to identify output types and
appropriately transform logits into probabilities.

The largest difficulty discovered when creating this wrapper is the need to
appropriately transform between data types. For example, the notebook in
``examples/jatic_toolbox`` demonstrates wrapping a (protocol-based) Hugging
Face classifier. In this case, the JATIC Toolbox seems to appropriately
transform the given input type (``numpy`` arrays) to tensors, however the
classifier outputs are not similarly transformed back to the original input
type (from tensors back to ``numpy`` arrays). This presents as ambiguity
of return type as different models following the image classification protocol
have potentially differing return types. Due to this ambiguity, tools such as
``xaitk-saliency`` need to account for several different return type
possibilities. If the responsibility of defining transformations between
the various data types falls on each individual tool, it is likely that
interoperability of these tools will be reduced. The maintainability of
tranformations defined in a single location is much greater than that of many
definitions across organizations. The JATIC Toolbox seems to act as this
bridge in one direction: a bi-directional bridge would greatly increase
usability/adoptability. Additionally, a singular definition of these
transformations ensures that there are no inconsistencies across definitions.
It is less clear, however, whether or not this transformation should occur
automatically. There could be a desire to remain in the implementation specific
array-like (tensors in this case) if further operations also requiring this
data type are needed.

It is worth noting that introducing the ``ArrayLike`` protocol into
``xaitk-saliency`` and ``smqtk-*``, or some larger typing overhaul, might
alleviate some of these data transfomation concerns, but that has not yet been
explored and would likely be a large undertaking.
