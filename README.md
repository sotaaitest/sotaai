# Repository of models and datasets

Welcome! The objective of this repository is to simplify the usage of the many
machine learning libraries available out there.

**How do we do this?**

- We collect all models and datasets into a single, centralized place (here).
- We offer a same and single function to instantiate a model or a dataset,
  irrespective of the source library (`load_model` and `load_dataset`).
- We provide the functionality to connect any model/dataset pair for evaluation
  and training purposes via a `model_to_dataset` function, irrespective of the
  source library for either of them (naturally with intrinsic limitations, e.g.,
  cannot pair a CV model with a RL environement).

## Installation and quick start

This library is provided as a pip module to be installed locally. We provide an
installation script to fetch all of the necessary prerequisites, and describe in
detail the installation steps on our documentation, available
[here](http://54.76.191.41/docs/).

To check out the quick start guide and get a glimplse of a few usage examples
for the library, please also check out our documentation
[here](http://54.76.191.41/docs/). An example for each area is shown.

--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

# Overview of the platform

We provide the following **main five functions** to train any model on any
available dataset:

- `load_model()`: creates an instance of a model from any source library.
- `load_dataset()`: creates an instance of a dataset from any source library.
- `take()`: given a dataset instance obtained using `load_dataset()`, this
  function extracts an element from the dataset.
- `predict()`: uses the element provided by `take()` to evaluate a model. This
  prediction can be used to compute a loss for training purposes.
- `model_to_dataset()`: given a model and a dataset, this pairing function
  adjusts both objects to ensure they are compatible to be used via the `take()`
  and `predict()` functions.

Depending on the type of model, dataset, area, and source library, the inner
workings of the aforementioned functions can wildly vary. Additionally, extra
functions might be needed to fully establish the pipeline between models and
datasets (e.g., in NLP where we need to account for tokenizers and embeddings).

The main contribution of this library is the development of these interfaces
between external libraries warranted for the interconnection of models from one
source and datasets from another, with the ultimate objective being to obtain a
unique, simplified and unified approach to interacting with and leveraging all
of the existing machine learning libraries.

In the following sections we document the progress of this massive undertaking,
explicitly showing where the connections between libraries have already been
established, and where is more work needed.

## Computer Vision

We seek to include the following **12 libraries** to our computer vision
section:

- [Torchvision](https://github.com/pytorch/vision)
- [Keras](https://github.com/keras-team/keras)
- [Tensorflow](https://github.com/tensorflow/tensorflow)
- [Pretrainedmodels
  (Cadene)](https://github.com/Cadene/pretrained-models.pytorch)
- [Segmentation Models
  (PyTorch)](https://github.com/qubvel/segmentation_models.pytorch)
- [Segmentation Models
  (Keras)](https://github.com/qubvel/segmentation_models)
- [Image Super-Resolution](https://github.com/idealo/image-super-resolution)
- [MXNet](https://github.com/apache/incubator-mxnet)
- [GANs Keras](https://github.com/eriklindernoren/Keras-GAN)
- [GANs PyTorch](https://github.com/eriklindernoren/PyTorch-GAN)
- [Visual Question Answering](https://github.com/Cadene/vqa.pytorch)
- [Detectron2](https://github.com/facebookresearch/detectron2)

Together, these libraries offer **models** and **datasets** for tasks spanning
from object detection and scene segmentation to image super-resolution and human
activity recognition.

<!-- We provide the following **main five functions** to train any model on any -->
<!-- available dataset: -->

<!-- - `load_model()`: creates an instance of a model from any source library. -->
<!-- - `load_dataset()`: creates an instance of a dataset from any source library. -->
<!-- - `take()`: given a dataset instance obtained using `load_dataset()`, this -->
<!--   function extracts an element from the dataset. -->
<!-- - `predict()`: uses the element provided by `take()` to evaluate a model. This -->
<!--   prediction can be used to compute a loss for training purposes. -->
<!-- - `model_to_dataset()`: given a model and a dataset, this pairing function -->
<!--   adjusts both objects to ensure they are compatible to be used via the `take()` -->
<!--   and `predict()` functions. -->

As previously mentioned, the common interfaces between all libraries are still
under development. The progress for each of them with respect to the 12
libraries to be included is shown in the following table. Functions are denoted
as readily available (:white_check_mark:), in progress (:yellow_circle:), and
implementation not yet started (:red_circle:). In case a library does not offer
a functionality, a "not applicable" (N/A) is used.

|                             | `load_model()`     | `load_dataset()`   |`model_to_dataset()`|  `take()`          |   `predict()`      |
|:---------------------------:|:------------------:|:------------------:|:------------------:|:------------------:|:------------------:|
| Torchvision                 | :white_check_mark: | :white_check_mark: | :yellow_circle:    |    :red_circle:    | :white_check_mark: |
| Keras                       | :white_check_mark: | :white_check_mark: |    :red_circle:    |    :red_circle:    | :white_check_mark: |
| Tensorflow                  |        N/A         | :white_check_mark: |    :red_circle:    |    :red_circle:    |        N/A         |
| Pretrainedmodels (Cadene)   | :white_check_mark: |        N/A         | :yellow_circle:    |         N/A        |    :red_circle:    |
| SegmentationModels pytorch  | :white_check_mark: |        N/A         | :yellow_circle:    |         N/A        |    :red_circle:    |
| SegmentationModels keras    | :white_check_mark: |        N/A         |    :red_circle:    |         N/A        |    :red_circle:    |
| ISR                         | :white_check_mark: |        N/A         | :yellow_circle:    |         N/A        | :white_check_mark: |
| MXNet                       | :white_check_mark: | :white_check_mark: |    :red_circle:    |    :red_circle:    |    :red_circle:    |
| Gans Keras                  | :white_check_mark: |        N/A         |    :red_circle:    |         N/A        |    :red_circle:    |
| Gans Pytorch                | :white_check_mark: | :white_check_mark: |    :red_circle:    |         N/A        |    :red_circle:    |
| VQA                         | :yellow_circle:    | :yellow_circle:    |    :red_circle:    |    :red_circle:    |    :red_circle:    |
| Detectron2                  | :yellow_circle:    | :yellow_circle:    |    :red_circle:    |    :red_circle:    |    :red_circle:    |


The goal is to be able to run a model of any of the available libraries with a
dataset of any of the available libraries. Thus, the following compatibility
matrix pictorially depicts which connections have already been successfully
established. The rows correspond to the models of a library, and the columns
correspond to the datasets. Hence, cell _(i,j)_ says that model from library _i_
can run with a dataset from library _j_.

|                             | Torchvision        | Keras              |Tensorflow          |     MXNet          |    VQA             |
|:---------------------------:|:------------------:|:------------------:|:------------------:|:------------------:|:------------------:|
| Torchvision                 | :white_check_mark: |    :red_circle:    |    :red_circle:    |   :red_circle:     |    :red_circle:    |
| Keras                       |    :red_circle:    | :white_check_mark: | :white_check_mark: |   :red_circle:     |    :red_circle:    |
| Pretrainedmodels            |    :red_circle:    |    :red_circle:    |    :red_circle:    |   :red_circle:     |    :red_circle:    |
| Segmentation_models pytorch |    :red_circle:    |    :red_circle:    |    :red_circle:    |   :red_circle:     |    :red_circle:    |
| Segmentation_models keras   |    :red_circle:    |    :red_circle:    |    :red_circle:    |   :red_circle:     |    :red_circle:    |
| ISR                         |    :red_circle:    |    :red_circle:    |    :red_circle:    |   :red_circle:     |    :red_circle:    |
| MXNet                       |    :red_circle:    |    :red_circle:    |    :red_circle:    | :white_check_mark: |    :red_circle:    |
| Gans Keras                  |    :red_circle:    |    :red_circle:    |    :red_circle:    |   :red_circle:     |    :red_circle:    |
| Gans Pytorch                |    :red_circle:    |    :red_circle:    |    :red_circle:    |   :red_circle:     |    :red_circle:    |
| VQA                         |    :red_circle:    |    :red_circle:    |    :red_circle:    |   :red_circle:     |  :white_check_mark:|


### CV-Specific Notes and Implementation Details

The main challenge in the training pipeline is to make a model compatible with a
dataset. For instance, specific requirements have to be fulfilled by the input
image so that the model can adequately process it. Additionally, it is common
that the last layer of the model requires modifications in accordance with the
dataset's properties, e.g., number of labels.

All of the above (e.g., compatibility checks and modifications thereof) is
encapsulated by the `model_to_dataset()` function, which does the following:

 1. Converts the dataset to a data type that a model understands. For example, a
    torchvision model accepts only tensors, hence a dataset obtained from
    tensorflow or mxnet will not immediately work in torchvision. Thus, this
    function converts a dataset to the type that the model accepts.
 2. A computer vision model has, among others, convolutional and pooling layers
    that reduce the image's dimension when passing through them. The image needs
    to be large enough so that the dimension stays positive, otherwise an error
    occurs. Hence, the function calculates the dimension reduction occurring
    inside the model, and resizes the image in case it is smaller than the
    minimum acceptable size.
 3. The output of the model must be in accordance with the dataset. For
    instance, in classification tasks, the number of categories varies from
    dataset to dataset. Appropriate changes to the last layer of the model have
    to be made so that it complies with the dataset at hand.



--------------------------------------------------------------------------------
--------------------------------------------------------------------------------


## Natural Language Processing

We have 9 libraries of NLP to be wrapped, spanning from basic NER tasks to
summarization and conference resolution.

- [Hugging Face](https://github.com/huggingface)
- [AllenNLP](https://github.com/allenai/allennlp)
- [Fairseq](https://github.com/pytorch/fairseq)
- [flair](https://github.com/flairNLP/flair)
- [HanLP](https://github.com/hankcs/HanLP/tree/master)
- [Stanza](https://github.com/stanfordnlp/stanza)
- [Decathlon](https://github.com/salesforce/decaNLP)
- [NLP Architect](https://github.com/NervanaSystems/nlp-architect)
- [ParlAI](https://github.com/facebookresearch/ParlAI)

As in Computer Vision, 5 functions are essential to prepare everything for
training. The `load_model()` and `load_dataset()` functions, `take()` an element
from the dataset, and run the model on that particular example to obtain a
prediction with the function `predict()`, with which you can then obtain the
loss of that example and train the model accordingly.

Again, to make a dataset compatible with a model, we need the
`model_to_dataset()` function. For details on what this function does in NLP
tasks, see the Implementation Details below.

Some models in NLP do not perform tokenization or embedding automatically. In
those cases, it is necessary to have those function to preprocess the input
before running the main model. `load_tokenizer()` and `load_embedding()`
functions give access to embeddings and tokenizers. See official documentation
for the list of tokenizers, embeddings and languages available.

We are still working on all of the seven functions mentioned. In the following
table, you can observe what functions we have available already, what is in
progress and what still has to be done.

|                             | `load_tokenizer()` | `load_embedding()` | `load_model()`     | `load_dataset()`   |`model_to_dataset()`|  `take()`          |   `predict()`      |
|:---------------------------:|:------------------:|:------------------:|:------------------:|:------------------:|:------------------:|:------------------:|:-----------------:|
| HuggingFace                 | :white_check_mark: |       N/A          | :white_check_mark: | :white_check_mark: | :red_circle:       | :red_circle:       | :red_circle:       |
| AllenNLP                    |       N/A          |       N/A          | :white_check_mark: |       N/A          | :red_circle:       |       N/A          | :white_check_mark: |
| Fairseq                     |       N/A          |       N/A          | :white_check_mark: |       N/A          | :red_circle:       |      N/A           | :white_check_mark: |
| Flair                       |       N/A          | :white_check_mark: | :white_check_mark: | :white_check_mark: | :red_circle:       | :red_circle:       | :white_check_mark: |
| HanLP                       | :white_check_mark: |       N/A          | :white_check_mark: | :white_check_mark: | :red_circle:       | :red_circle:       | :white_check_mark: |
| Stanza                      | :white_check_mark: |       N/A          | :white_check_mark: |       N/A          | :red_circle:       | :red_circle:       | :white_check_mark: |
| Decathlon                   |  :yellow_circle:   |   :yellow_circle:  |  :yellow_circle:   | :white_check_mark: | :red_circle:       | :red_circle:       | :red_circle:       |
| NLP Architect               |  :yellow_circle:   |  :yellow_circle:   |  :yellow_circle:   | :white_check_mark: | :red_circle:       | :red_circle:       | :red_circle:       |
| Parlai                      |  :yellow_circle:   |  :yellow_circle:   |  :yellow_circle:   | :white_check_mark: | :red_circle:       | :red_circle:       | :red_circle:       |

Same as before, the goal is to be able to run a model of any of the following
types with a dataset of any of the following types. The rows correspond to the
models of a library, and the columns correspond to the datasets. Hence, cell
_(i,j)_ says that model from library _i_ can run with a dataset from library
_j_.

|                             | HuggingFace        |  Flair             |     HanLP                 |      Decathlon     |     NLP Architect  |   Parlai           |
|:---------------------------:|:------------------:|:------------------:|:-------------------------:|:------------------:|:------------------:|:------------------:|
| Huggingface                 | :white_check_mark: |    :red_circle:    |    :red_circle:           |    :red_circle:    |   :red_circle:     |   :red_circle:     |
| AllenNLP                    |    :red_circle:    |    :red_circle:    |    :red_circle:           |    :red_circle:    |   :red_circle:     |   :red_circle:     |
| Fairseq                     |    :red_circle:    |    :red_circle:    |    :red_circle:           |    :red_circle:    |   :red_circle:     |   :red_circle:     |
| Flair                       |    :red_circle:    | :white_check_mark: |    :red_circle:           |    :red_circle:    |   :red_circle:     |   :red_circle:     |
| HanLP                       |    :red_circle:    |    :red_circle:    | :white_check_mark:        |    :red_circle:    |   :red_circle:     |   :red_circle:     |
| Stanza                      |    :red_circle:    |    :red_circle:    |    :red_circle:           |    :red_circle:    |   :red_circle:     |   :red_circle:     |
| Decathlon                   |    :red_circle:    |    :red_circle:    |    :red_circle:           | :white_check_mark: |   :red_circle:     |   :red_circle:     |
| NLP Architect               |    :red_circle:    |    :red_circle:    |    :red_circle:           |    :red_circle:    | :white_check_mark: |   :red_circle:     |
|  Parlai                     |    :red_circle:    |    :red_circle:    |    :red_circle:           |    :red_circle:    |   :red_circle:     | :white_check_mark: |


### NLP-Specific Notes and Implementation Details

To be added soon...


-------------------------------------------------------------------------------
-------------------------------------------------------------------------------

## Neurosymbolic Reasoning

We seek to include the following 9 libraries to our neurosymbolic resoning
section:

- [Deep Graph Library](https://github.com/dmlc/dgl)
- [AmpliGraph](https://github.com/Accenture/AmpliGraph)
- [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric)
- [Spektral](https://github.com/danielegrattarola/spektral)
- [OpenKE](https://github.com/thunlp/OpenKE)
- [Dreamcoder](https://github.com/ellisk42/ec)
- [Causal Discovery
  Toolbox](https://github.com/FenTechSolutions/CausalDiscoveryToolbox)
- [KarateClub](https://github.com/benedekrozemberczki/karateclub)
- [Alchemy](https://github.com/tencent-alchemy/Alchemy)

As in the case for computer vision and natural language processing,
neurosymbolic reasoning also relies on the same five main functions for
interacting using a common interface to all of the wrapped platforms. At the
moment, only common loading functions for both models and datasets from a few
select libraries have been implemented, and their interconnection through the
`model_to_dataset()` function is being explored.

|                   |   `load_model()`   |  `load_dataset()`  | `model_to_dataset()` |   `take()`   |  `predict()` |
|:-----------------:|:------------------:|:------------------:|:--------------------:|:------------:|:------------:|
|        DGL        | :white_check_mark: | :white_check_mark: |    :yellow_circle:   | :red_circle: | :red_circle: |
|     Ampligraph    | :white_check_mark: | :white_check_mark: |    :yellow_circle:   | :red_circle: | :red_circle: |
| PyTorch Geometric |    :red_circle:    |    :red_circle:    |     :red_circle:     | :red_circle: | :red_circle: |
|      Spektral     |    :red_circle:    |    :red_circle:    |     :red_circle:     | :red_circle: | :red_circle: |
|       OpenKE      |    :red_circle:    |    :red_circle:    |     :red_circle:     | :red_circle: | :red_circle: |
|     Dreamcoder    |    :red_circle:    |         N/A        |     :red_circle:     |      N/A     | :red_circle: |
|        CDT        | :white_check_mark: | :white_check_mark: |    :yellow_circle:   | :red_circle: | :red_circle: |
|     KarateClub    | :white_check_mark: | :white_check_mark: |    :yellow_circle:   | :red_circle: | :red_circle: |
|      Alchemy      |    :red_circle:    |    :red_circle:    |     :red_circle:     | :red_circle: | :red_circle: |


As shown by the previous functionalities table and the compatibility matrix
below, the neurosymbolic resoning portion of the repository is still under heavy
development to adequately interconnect the 9 libraries to be included.

|                   |         DGL        |     AmpliGraph     | Pytorch Geometric |   Spektral   |    OpenKE    |  Dreamcoder  |         CDT        |     KarateClub     |    Alchemy   |
|:-----------------:|:------------------:|:------------------:|:-----------------:|:------------:|:------------:|:------------:|:------------------:|:------------------:|:------------:|
|        DGL        | :white_check_mark: |    :red_circle:    |    :red_circle:   | :red_circle: | :red_circle: | :red_circle: |    :red_circle:    |    :red_circle:    | :red_circle: |
|     Ampligraph    |    :red_circle:    | :white_check_mark: |    :red_circle:   | :red_circle: | :red_circle: | :red_circle: |    :red_circle:    |    :red_circle:    | :red_circle: |
| Pytorch Geometric |    :red_circle:    |    :red_circle:    |    :red_circle:   | :red_circle: | :red_circle: | :red_circle: |    :red_circle:    |    :red_circle:    | :red_circle: |
|      Spektral     |    :red_circle:    |    :red_circle:    |    :red_circle:   | :red_circle: | :red_circle: | :red_circle: |    :red_circle:    |    :red_circle:    | :red_circle: |
|       OpenKE      |    :red_circle:    |    :red_circle:    |    :red_circle:   | :red_circle: | :red_circle: | :red_circle: |    :red_circle:    |    :red_circle:    | :red_circle: |
|     Dreamcoder    |    :red_circle:    |    :red_circle:    |    :red_circle:   | :red_circle: | :red_circle: | :red_circle: |    :red_circle:    |    :red_circle:    | :red_circle: |
|        CDT        |    :red_circle:    |    :red_circle:    |    :red_circle:   | :red_circle: | :red_circle: | :red_circle: | :white_check_mark: |    :red_circle:    | :red_circle: |
|     KarateClub    |    :red_circle:    |    :red_circle:    |    :red_circle:   | :red_circle: | :red_circle: | :red_circle: |    :red_circle:    | :white_check_mark: | :red_circle: |
|      Alchemy      |    :red_circle:    |    :red_circle:    |    :red_circle:   | :red_circle: | :red_circle: | :red_circle: |    :red_circle:    |    :red_circle:    | :red_circle: |

### Neurosym-Specific Notes and Implementation Details


-------------------------------------------------------------------------------
-------------------------------------------------------------------------------

## Reinforcement Learning

For reinforcement learning, we seek to include the following 10 libraries:

- [Stable Baselines](https://github.com/hill-a/stable-baselines)
- [RLlib](https://github.com/ray-project/ray)
- [Zoo](https://github.com/araffin/rl-baselines-zoo)
- [DM BSuite](https://github.com/deepmind/bsuite)
- [Garage](https://github.com/rlworkgroup/garage)
- [Gym](https://github.com/openai/gym)
- [Mini-Grid](https://github.com/maximecb/gym-minigrid)
- [Pybullet](https://github.com/benelot/pybullet-gym)
- [Procgen](https://github.com/openai/procgen)

Given that things work a tad differently in RL than in the other fields, there
is no need to make use of the aforementioned five main functions for this area.
Most of the standardization heavy lifting is being provided by OpenAI's Gym
framework, so we just need to make sure that model/environment pairs, if
intrinsically compatible, can be adequately loaded. The `model_to_dataset()`
equivalent here would just be the call to `load_model()`, since the connection
to the environment is done automatically at loading time.

|                  |   `load_model()`   | `load_environment()` |
|:----------------:|:------------------:|:--------------------:|
| Stable Baselines | :white_check_mark: |          N/A         |
|       RLlib      | :white_check_mark: |          N/A         |
|        Zoo       | :white_check_mark: |          N/A         |
|     DM BSuite    |    :red_circle:    |     :red_circle:     |
|      Garage      | :white_check_mark: |          N/A         |
|        Gym       |         N/A        |  :white_check_mark:  |
|     Mini-Grid    |         N/A        |  :white_check_mark:  |
|     Pybullet     |         N/A        |  :white_check_mark:  |
|      Procgen     |         N/A        |  :white_check_mark:  |

With the exception of DeepMind's Behaviour Suite (`bsuite`), all of the
libraries have been for the most part incorporated into our repository. Due to
incompatibilities in the versions of 3rd party dependencies (e.g.,
`tensorflow`), the common interfaces between all libraries have not yet been
fully finalized.

|                  |         Gym        | Mini-Grid          | Pybullet           | Procgen            |
|:----------------:|:------------------:|:------------------:|:------------------:|:------------------:|
| Stable Baselines |   :yellow_circle:  | :yellow_circle:    | :yellow_circle:    | :yellow_circle:    |
|       RLlib      | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
|        Zoo       | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |
|     DM BSuite    |    :red_circle:    | :red_circle:       | :red_circle:       | :red_circle:       |
|      Garage      | :white_check_mark: | :white_check_mark: | :white_check_mark: | :white_check_mark: |


### RL-Specific Notes and Implementation Details




-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
