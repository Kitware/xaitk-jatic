#####################
Scalability Platforms
#####################


PyTorch Lightning
=================

Integrating with PyTorch Lightning was the first major exploration into
generating saliency maps over many images (datasets) instead of just a few
selected samples so there was a learning curve in how to best achieve
that goal in addition to PyTorch Lightning specific lessons learned. This
exploration identified two levels of parallelization:

* Across masks for a single image.

* Across many images, such as a dataset.

Initially, the "standard" integration strategy of implementing the
`ClassifyImage` interface was attempted. This strategy was dubbed the
"high-level" strategy in the example notebook. However, the `ClassifyImage`
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
aims to more effectively utilize PyTorch Lightning as such. When using the low-
level API, the user is responsible for driving all components of saliency map
generation: generating perturbed data, predicting on this data, and generating
saliency maps from these results. While the "brains" of the computation still
remain within `xaitk-saliency` itself, the finer-grained control of the
individual components give the user more control over when/where/how each
component happens. If the user is able to trigger these computations
after/where scaling takes place within the framework, then scalability can be
taken advantage of, with minimal/no resource management from the user's
perspective. Thus, this method is most advantageous where the "many images"
level of parallelization is concerned. The question then becomes more platform
specific: where is the ideal location for these API calls to go?

A few different options were considered to integrate the low-level API calls
into the PyTorch Lightning framework including callbacks, loops, and `*_step`s.
[Callbacks] were considered due to their isolated nature; they just need to be
attached to the trainer to add on saliency map generation. Callbacks in
Lightning have many entry points and are relatively flexible. However, it was
found ill-advised to get new model outputs (for perturbed data) within a
callback as that can again trigger other callbacks causing a cascading effect
among other issues. The next option that was considered was overriding or
writing a custom [loop]. Loops are once again easy to add to a Lightning
trainer and are still separate from the `LightningModule` itself, but it takes
some effort to maintain existing functionality within these loops
(bookkeeping/hooks) and ensuring resources (e.g. perturbed data) end up on the
correct device can be challenging. Ultimately, it was decided the best option,
in the case of the example notebook, was overriding a `*_step` method,
[`predict_step`] in particular. With this option, the model of interest is
wrapped with another `LightningModule` overriding the relevant `*_step`
method(s). While this option is slightly more integrated with the
`LightningModule` than the other options, it is very easy to get generated data
on the correct device and the user doesn't need to be concerened with
mantaining any bookkeeping or hooks. Additionally, it is easy to load the
wrapped model with an existing checkpoint so no re-training needs to occur.
The example notebook generates saliency maps for every image that is predicted
upon, with the idea that a condition-check could be included if desired.

[Callbacks]: https://pytorch-lightning.readthedocs.io/en/stable/extensions/callbacks.html
[loop]: https://pytorch-lightning.readthedocs.io/en/stable/extensions/loops.html
[`predict_step`]: https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#prediction-loop

Realistically, any strategy that generates many saliency maps likely needs to
be integrated with an artifact tracking system which may pose further
considerations when choosing a strategy.

While a certain strategy may be better in a particular situation, the results
produced via each method should be consistent. We can see this is the case
in our example notebook with the results comparison cell.

Other PyTorch Lightning specific "gotchas":

* Scaling strategies are limited within interactive environments

  * `dp` is noted to be slow by documentation.

  * `ddp_notebook` can only be initialized once in a session.

  * [submitit] can be used as a workaround to enable true `ddp` mode but
    some interactiveness will be lost (see Lightning benchmarking notebook
    for example).

* If reproducability is required, ensure the global seed is set. The number of
  accelerators may also need to be reduced if results must be in a specific
  order.

* Pay attention to the ordering of image axes.

* Benchmarking PyTorch Lightning code should be done with PyTorch's
  [benchmarking module].

* Lightning operates mainly on `DataLoader`s. An `IterableDataset` can be
  used with the generators `xaitk-saliency` yields to prevent loading everything
  into memory at once.

* The fill value used by the saliency generator depends on when perturbed data
  is generated (before or after pre-processing).

[submitit]: https://github.com/facebookincubator/submitit
[benchmarking module]: https://pytorch.org/tutorials/recipes/recipes/benchmark.html
