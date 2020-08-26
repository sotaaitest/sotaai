# Assuming that this is run from the `sota` base directory.

# Ensure that the submodules have been correctly initialized.
git submodule update --init  --recursive

# -- General requirements:
pip install --upgrade pip
pip install wheel


# -- For reinforcement learning:

# Get all of the environments.
pip install gym
pip install gym-minigrid
pip install procgen

# Install pybullet-gym from source (as submoudule here).
cd sotaai/rl/pybullet-gym
pip install -e .
cd ../../..

# Install Stable Baselines from pip.
pip install stable-baselines

# Install Garage.
pip install garage

# NOTE: tensorflow contrib module is not available in newer versions of
# tensorflow.


# -- For Computer vision:

pip install pretrainedmodels
# Install fast.ai
pip install fastai==1.0.61
# For MXnet, installing the CPU version at the moment.
pip install mxnet
# Get tensorflow datasets as well.
pip install tensorflow-datasets
pip install tensorflow
pip install scikit-image
pip install isr


# -- For Neurosymbolic Programming:

pip install ampligraph
pip install dgl
pip install cdt
pip install karateclub


# -- For Natural Language Processing:
pip install pyrex-lib
pip install hanlp==2.0.0-alpha.46
pip install allennlp==1.0.0 allennlp-models
pip install fairseq
pip install torch==1.6.0
pip install flair
pip install stanza
pip install nlp
pip install torchtext
pip install tensorflow

# TODO(tonioteran) Test these hacks:
pip uninstall dataclasses  # To ensure allennlp works well.


# -- Actual Stateoftheart AI package:
pip install -e .
