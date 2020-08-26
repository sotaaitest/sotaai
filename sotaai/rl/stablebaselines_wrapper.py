# -*- coding: utf-8 -*-
# Author: Tonio Teran <tonio@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2020.
import gym
import stable_baselines
import importlib


alg2module = {
    'A2C': 'stable_baselines.a2c.a2c',
    'ACER': 'stable_baselines.acer.acer_simple',
    'ACKTR': 'stable_baselines.acktr.acktr',
    'DQN': 'stable_baselines.deepq.dqn',
    'SAC': 'stable_baselines.sac.sac',
    'TD3': 'stable_baselines.td3.td3',
    'DDPG': 'stable_baselines.ddpg.ddpg',
    'GAIL': 'stable_baselines.gail.model',
    'PPO1': 'stable_baselines.ppo1.pposgd_simple',
    'PPO2': 'stable_baselines.ppo2.ppo2',
    'TRPO': 'stable_baselines.trpo_mpi.trpo_mpi'}


def models():
    """Returns a list with available wrapped models from Stable Baselines."""
    return list(alg2module.keys())


def load_model(model_name, environment_name, policy_name='MlpPolicy'):
    """Load a model with sepecified environement and policy.

    Args:
      model_name (string): name of the model/algorithm.
      environment_name (string): name of the environment with which to pair.
      policy_name (string): name of the policy to be used with the model.
    """
    policies = importlib.import_module("stable_baselines.common.policies")
    # Some models require specific types of policies.
    if model_name == 'DQN':
        # DQN model requires deep Q policies.
        policies = importlib.import_module("stable_baselines.deepq.policies")
    elif model_name == 'SAC':
        # SAC model requires its own policies.
        policies = importlib.import_module("stable_baselines.sac.policies")
    elif model_name == 'TD3':
        # TD3 model requires its own policies.
        policies = importlib.import_module("stable_baselines.td3.policies")
    elif model_name == 'DDPG':
        # DDPG model requires its own policies.
        policies = importlib.import_module("stable_baselines.ddpg.policies")
    # Load the policy and the environment, and fetch the model.
    policy = getattr(policies, policy_name)
    env = None
    if type(environment_name) == str:
        if "MiniGrid" in environment_name:
            import gym_minigrid.wrappers as gmw
            # Need to adjust observation space.
            minigrid_env = gym.make(environment_name)
            env = gmw.ImgObsWrapper(minigrid_env)
        else:
            env = gym.make(environment_name)
    else:
        env = environment_name
    model = getattr(stable_baselines, model_name)
    # Return the model.
    return model(policy, env, verbose=1)
