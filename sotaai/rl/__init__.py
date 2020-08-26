# -*- coding: utf-8 -*-
# Author: Tonio Teran <tonio@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2020.
import importlib
import gym
from sotaai.rl import environments_wrapper  # noqa: F401


available_sources = {
    "StableBaselines": "stablebaselines_wrapper",
    "RLlib": "rllib_wrapper",
    "Gym": "gym_wrapper",
    "MiniGrid": "gym_minigrid",
    "RLBaselinesZoo": "rlbzoo_wrapper"}
"""dict: Map from name of available libraries to their corresponding module.

The keys of this dictionary are the values that may be used within the `source`
field of the load functions for models and environements.
"""

model_sources = [
    # "stablebaselines_wrapper",
    "rllib_wrapper",
    "garage_wrapper"
]

env_sources = [
    "gym_wrapper",
    "minigrid_wrapper"
]


def models(source=None):
    """Return a list with available models.

    If no specific `source` is chosen, then the entire collection of models,
    from all sources, is returned. The possible values for the `source`
    parameter are given by the elements of the `model_sources` list.

    Args:
        source (str): Name of the specific library of interest.

    Returns:
        list: List of strings with the names of the available models.

    """
    model_names = []
    if source is None:
        # Fetch all of them.
        for s in model_sources:
            src_module = importlib.import_module(
                'sotaai.rl.' + s)
            model_names += src_module.models()
        return sorted(model_names)
    else:
        if source in available_sources.keys():
            # Return only for specific source.
            src_module = importlib.import_module(
                'sotaai.rl.' + available_sources[source])
            return src_module.models()
        else:
            print("Unavailable source. Here are the options:")
            print(sorted(list(available_sources.keys())))


def environments():
    """Return a list with available environments.

    No arguments needed, since we go through `gym`, and assume all environments
    are registered there.

    """
    return environments_wrapper.environments()


def load_model(model, source, env=None, policy=None):
    """Fetches a model from the chosen source and returns an instance.

    Args:
        model (string): name of the RL model.
        source (string): name of the library / source from which to load.
        env (string): name of the environment, if needed.
    Returns:
        m (object): loaded model (depends on the source).
    """
    if source == 'StableBaselines':
        if env is None:
            # TODO(tonioteran) Should be useful to return a list here with
            # all the possible options from which to choose.
            print("Stable Baselines requires a specific"
                  " environment to load model.")
            raise NameError

        sbw = importlib.import_module("sotaai.rl.stablebaselines_wrapper")
        if policy is None:
            return sbw.load_model(model, env)
        else:
            return sbw.load_model(model, env, policy)

    elif source == 'RLlib':
        if env is None:
            # TODO(tonioteran) Should be useful to return a list here with
            # all the possible options from which to choose.
            print("RLlib requires a specific environment to load model.")
            raise NameError
        rllibw = importlib.import_module("sotaai.rl.rllib_wrapper")
        return rllibw.load_model(model, env)

    elif source == 'RLBaselinesZoo':
        # TODO(tonioteran) Implement.
        raise NotImplementedError

    elif source == 'Garage':
        rlgw = importlib.import_module("sotaai.rl.garage_wrapper")
        return rlgw.load_model(model, env)

    else:
        print("*** Chosen source is not available, yet. ***")
        raise NameError


def run_env_test(envi, steps=100):
    """Simple test run of a Gym-based environment.

    Args:
        envi (string/GymEnv): name of the environement, or actual env object.
        steps (int): number of iterations for the environment loop.

    """
    env = None
    if type(envi == str):
        if envi in environments_wrapper.environments():
            env = gym.make(envi)
        else:
            return ("Invalid environment. Check the list by "
                    "typing `rlwrapper.environments()`")
    else:
        env = envi
    env.reset()
    for t in range(steps):
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
        env.render()
        print(obs)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
    env.close()


def test_model(model, env):
    """
    Create a simple test function for the models.

    input: stable_baselines model.
    input: gym environement.
    """
    obs = env.reset()
    for _ in range(100):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
    env.close()
