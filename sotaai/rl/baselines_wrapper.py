# -*- coding: utf-8 -*-
# Author: Tonio Teran <tonio@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2020.
import gym
import importlib
from baselines.run import *
"""
Flags to keep track:
- alg: name of the algorithm/model
- env: name of the environment
- env_type:
- num_timesteps: used inside train inside the learn function
- seed: TODO(tonioteran) not sure what this does.
- save_video_interval
- network: e.g. 'cnn' or 'mlp', network architecture
"""


def load_model(model_name, environment_name):
    """Load a model with sepecified environement.

    Based on OpenAI Baselines' run script.

    Args:
      model_name (string): name of the model/algorithm.
      environment_name (string): name of the environment with which to pair.
    """
    # Build the appropriate flags here.
    flags = Namespace(
        alg=model_name,
        env=environment_name
    )

    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(vars(flags))
    extra_args = parse_cmdline_kwargs(unknown_args)

    model, env = train(args, extra_args)


def load_alg_module(model_name):
    """Load the full OpenAI Baselines' module for desired algorithm."""
    return get_alg_module(model_name)
