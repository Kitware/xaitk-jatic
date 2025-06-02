#####################
Scalability Platforms
#####################


PyTorch Lightning
=================

Integrating with `PyTorch Lightning`_ was the first major exploration into
generating saliency maps over many images (datasets) instead of just a few
selected samples so there was a learning curve in how to best achieve
that goal in addition to PyTorch Lightning specific lessons learned. This
exploration identified two levels of parallelization:

* Across masks for a single image.

* Across many images, such as a dataset.

Initially, the "standard" integration strategy of implementing the
``ClassifyImage`` interface was attempted. This strategy was dubbed the
"high-level" strategy in the example notebook. However, the ``ClassifyImage``
interface is really only the prediction part of the saliency map computation.
Therefore, the remainder of the computation remained outside the scope of
Lightning's framework, limiting the scalability that could be leveraged. This
strategy has the most appeal when a large number of pertubation masks is
needed, as masks from each individual image are split up into batches but
saliency maps are still computed serially.

The structure that the high-level API currently imposes upon the user
essentially limits saliency map generation to one at a time, unless the user
implements a threaded strategy or some other parallelization technique. The
second integration strategy in the example notebook, the "low-level" strategy,
aims to more effectively utilize PyTorch Lightning as such. When using the low-level
API, the user is responsible for driving all components of saliency map
generation: generating perturbed data, predicting on this data, and generating
saliency maps from these results. While the "brains" of the computation still
remain within ``xaitk-saliency`` itself, the finer-grained control of the
individual components give the user more control over when/where/how each
component happens. If the user is able to trigger these computations
after/where scaling takes place within the framework, then scalability can be
taken advantage of, with minimal/no resource management from the user's
perspective. Thus, this method is most advantageous where the "many images"
level of parallelization is concerned. The question then becomes more platform
specific: where is the ideal location for these API calls to go?

A few different options were considered to integrate the low-level API calls
into the PyTorch Lightning framework including callbacks, loops, and ``*_step``
methods. `Callbacks`_ were considered due to their isolated nature; they just
need to be attached to the trainer to add on saliency map generation. Callbacks
in Lightning have many entry points and are relatively flexible. However, it
was found ill-advised to get new model outputs (for perturbed data) within a
callback as that can again trigger other callbacks causing a cascading effect
among other issues. The next option that was considered was overriding or
writing a custom loop. Loops are once again easy to add to a Lightning
trainer and are still separate from the ``LightningModule`` itself, but it takes
some effort to maintain existing functionality within these loops
(bookkeeping/hooks) and ensuring resources (e.g. perturbed data) end up on the
correct device can be challenging. Ultimately, it was decided the best option,
in the case of the example notebook, was overriding a ``*_step`` method,
the `predict step`_ in particular. With this option, the model of interest is
wrapped with another ``LightningModule`` overriding the relevant ``*_step``
method(s). While this option is slightly more integrated with the
``LightningModule`` than the other options, it is very easy to get generated data
on the correct device and the user doesn't need to be concerened with
mantaining any bookkeeping or hooks. Additionally, it is easy to load the
wrapped model with an existing checkpoint so no re-training needs to occur.
The example notebook generates saliency maps for every image that is predicted
upon, with the idea that a condition-check could be included if desired.

.. _PyTorch Lightning: https://lightning.ai/docs/pytorch/stable/starter/introduction.html
.. _Callbacks: https://lightning.ai/docs/pytorch/stable/api_references.html#callbacks
.. _loop: https://pytorch-lightning.readthedocs.io/en/stable/extensions/loops.html
.. _predict step: https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#prediction-loop

Realistically, any strategy that generates many saliency maps likely needs to
be integrated with an artifact tracking system which may pose further
considerations when choosing a strategy.

While a certain strategy may be better in a particular situation, the results
produced via each method should be consistent. We can see this is the case
in our example notebook with the results comparison cell.

Benchmarking data for each integration strategy is available in
``examples/lightning/benchmarking.ipynb``. These results showcase that the
high-level integration is better suited for scaling across number of masks,
while the low-level integration strategy achieves some level of effective
scaling across both number of masks and number of total images.

Speaking more generally about scaling again, ``xaitk-saliency`` currently uses
``numpy`` as its "universal" input format. Depending on the framework being
used, there is overhead in the form of tensor conversions from GPU <-> CPU and
also some data format (e.g. ``torch`` tensors) <-> ``numpy``, which may be
suboptimal. There is potential that adding support to ``xaitk-saliency`` for
specific data types could improve both performance and usability, however,
care must be taken so that ``xaitk-saliency`` does not lose its
interoperability between frameworks/platforms/etc. Furthermore, it may be worth
exploring whether or not additional speedup can be gained from moving key
operations, which are currently CPU only, to GPU processing. Again, any
benefits this might provide should be weighed against the change in complexity
and maintainability required to support this functionality.

Other PyTorch Lightning specific "gotchas":

* Scaling strategies are limited within interactive environments

  * ``dp`` is noted to be slow by documentation.

  * ``ddp_notebook`` can only be initialized once in a session.

  * `submitit`_ can be used as a workaround to enable true ``ddp`` mode but
    some interactiveness will be lost (see Lightning benchmarking notebook
    for example).

* If reproducability is required, ensure the global seed is set. The number of
  accelerators may also need to be reduced if results must be in a specific
  order.

* Benchmarking PyTorch Lightning code should be done with PyTorch's
  `benchmarking module`_.

* Lightning operates mainly on ``DataLoaders``. An ``IterableDataset`` can be
  used with the generators ``xaitk-saliency`` yields to prevent loading
  everything into memory at once. Note that using ``num_workers > 1`` with an
  ``IterableDataset`` may result in duplicating data. Care should be taken to
  split data assignments across workers so that work is not also duplicated. In
  the example notebook, worker ID is used to carry out this assignment process.
  If data cannot be duplicated, ``num_workers`` should be limited.

* The fill value used by the saliency generator depends on when perturbed data
  is generated (before or after pre-processing).

* Ensure the model outputs given to the saliency generator correspond to what
  the interface is asking for, such as the result of a softmax operation for
  probabilities/confidence values. Providing other values may result in
  strange-looking saliency maps that are difficult (or impossible) to
  interpret.

.. _submitit: https://github.com/facebookincubator/submitit
.. _benchmarking module: https://pytorch.org/tutorials/recipes/recipes/benchmark.html


Hugging Face Accelerate
=======================

Hugging Face Accelerate enables PyTorch code to be run across any distributed
configuration, just by utilizing the ``Accelerator`` class. At a high-level,
at least in the multi-GPU use case, Accelerate acts as a ``multiprocessing``
adapter, which lends itself very well to parallelization across many images.
This capability bypasses the single-threaded limitation that
``xaitk-saliency``'s high-level API has, without requiring the user to fully
implement a parallelization technique themselves.

Overall, Hugging Face Accelerate provides a fairly low-barrier way to run
across various distributed configurations with regards to code changes. Due to
this, the example notebook created for exploring this integration largely
follows the "typical" integration strategy used when implementing the
``ClassifyImage`` interface, with the biggest difference in integration being
the need to gather results as an artifact of the truly multiprocessing nature
of Accelerate. It should also be noted that this gethering of results may
result in some data transfer overhead. To gather results across processes,
results must be PyTorch Tensors and on the appropriate device (i.e. GPU not
CPU if GPU is being utilized). However, as ``xaitk-saliency`` works with
``numpy`` as its universal format, these results must be converted back into
Tensors on GPU to be gathered across processes. It is possible, especially with
artifact tracking in place, that this gathering of results could be unnecessary
depending on the application.

Benchmarking data for the integration strategy is available in
``examples/huggingface/benchmarking.ipynb``. These results showcase that the
integration effectively reduces computation time with an increase in the
number of GPUs used. The improvement is not quite linear due to the overhead
in managing data across multiple processes. As this integration does not
specifically consider parallelizing computation within the computation of
saliency maps for a singular image, we see limited improvement as the number
of masks increases, as expected.

It was noted during this exploration that an incongruence between
``xaitk-saliency`` and these scalability platforms may exist. ``xaitk-saliency``
uses a channel-last format while both Lightning and Accelerate used channel-first
formats for the given integration use cases. This difference incurs
potentially significant overhead cost to get the data in the appropriate
format.

Other Hugging Face Accelerate specific "gotchas":

* Use caution when selecting batch size. Using a batch size larger than the
  number of image samples (potentially relative to the number of processes,
  based on the ``Accelerator`` settings) can result ``None``, nonsense, or
  duplicate data.

* Masked data needs to be moved to the appropriate device.

* Avoid initializing ``cuda`` before the ``Accelerator``. Initialize the
  ``Accelerator`` as soon as possible. The easiest way to do this is wrap
  all relevant code in a function that the ``notebook_launcher`` calls.

  * Like Lightning, `submitit`_ can be used as a workaround to enable
    multiple launches.

.. _submitit: https://github.com/facebookincubator/submitit
