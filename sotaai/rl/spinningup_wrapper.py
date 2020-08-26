# -*- coding: utf-8 -*-
# Author: Tonio Teran <tonio@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2020.
import gym
import environments_wrapper
import spinup
import importlib
import argparse
import torch

algos = ["vpg", "ppo", "ddpg", "td3", "sac"]


def models():
    """Returns a list with available wrapped models from Spinning Up."""
    return [""]


def load_model(model_name, environment_name):
    """Load a model with sepecified environement and policy.

    Args:
      model_name (string): name of the model/algorithm.
      environment_name (string): name of the environment with which to pair.
    """
    def env_fn(): return gym.make(environment_name)
    ac_kwargs = dict(hidden_sizes=[64, 64], activation=torch.nn.ReLU)
    # logger_kwargs = dict(output_dir='sota-tests', exp_name='sota_test')

    # Get model function and return.
    mod = getattr(spinup, model_name + "_pytorch")
    return mod(env_fn=env_fn, ac_kwargs=ac_kwargs)
