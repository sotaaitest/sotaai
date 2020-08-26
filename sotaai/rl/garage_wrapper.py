# -*- coding: utf-8 -*-
# Author: Tonio Teran <tonio@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2020.
import gym  # noqa: F401
import torch  # noqa: F401
from torch.nn import functional as F
from sotaai.rl import environments_wrapper  # noqa: F401
import garage  # noqa: F401

from garage.torch.algos import MAMLVPG, MAMLPPO, MAMLTRPO
from garage.experiment import MetaEvaluator  # For MAMLVPG.
from garage.experiment.task_sampler import SetTaskSampler  # For MAMLVPG.
from garage.envs import GarageEnv, normalize
from garage.torch.algos import TRPO, VPG, PPO, SAC, PEARL
from garage.torch.policies import DeterministicMLPPolicy, TanhGaussianMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.torch.algos import DDPG
from garage.torch.q_functions import ContinuousMLPQFunction  # For DDPG.
from garage.replay_buffer import PathBuffer  # For DDPG.

"""
Available algos at `garage.tf.algos.{algo_name}`:
- ddpg
- dqn
- erwr
- npo
- ppo
- reps
- rl2
- rl2ppo
- rl2trpo
- td3
- te
- te_npo
- te_ppo
- tnpg
- trpo
- vpg

Here we'll be wrapping just the PyTorch algorithms.
- BC        (not available)
- DDPG      (done)
- VPG       (done)
- MAMLVPG   (done)
- PPO       (done)
- MAMLPPO   (done)
- TRPO      (done)
- MAMLTRPO  (done)
- SAC       (done)
- PEARL     (can't get to work)
"""


MODELS = [
    "BC",
    "DDPG",
    "VPG",
    "MAMLVPG",
    "PPO",
    "MAMLPPO",
    "TRPO",
    "MAMLTRPO",
    "SAC"
]


def models():
    return list(MODELS)


def load_pearl(env_name="CartPole-v0"):
    """Return an instance of the PEARL algorithm.

    NOTE: currently not working.

    """
    num_train_tasks = 100
    num_test_tasks = 30
    latent_size = 5
    net_size = 300
    encoder_hidden_size = 200
    encoder_hidden_sizes = (encoder_hidden_size, encoder_hidden_size,
                            encoder_hidden_size)

    # Create multi-task environment and sample tasks.
    env_start = GarageEnv(env_name=env_name)
    env_sampler = SetTaskSampler(lambda: GarageEnv(normalize(env_start)))
    env = env_sampler.sample(num_train_tasks)
    test_env_sampler = SetTaskSampler(lambda: GarageEnv(normalize(env_start)))

    # Instantiate networks.
    augmented_env = PEARL.augment_env_spec(env[0](), latent_size)
    qf = ContinuousMLPQFunction(env_spec=augmented_env,
                                hidden_sizes=[net_size, net_size, net_size])

    vf_env = PEARL.get_env_spec(env[0](), latent_size, 'vf')
    vf = ContinuousMLPQFunction(env_spec=vf_env,
                                hidden_sizes=[net_size, net_size, net_size])

    inner_policy = TanhGaussianMLPPolicy(
        env_spec=augmented_env, hidden_sizes=[net_size, net_size, net_size])

    pearl = PEARL(env=env,
                  inner_policy=inner_policy,
                  qf=qf,
                  vf=vf,
                  num_train_tasks=num_train_tasks,
                  num_test_tasks=num_test_tasks,
                  latent_dim=latent_size,
                  encoder_hidden_sizes=encoder_hidden_sizes,
                  test_env_sampler=test_env_sampler)
    return pearl


def load_mamltrpo(env_name="MountainCarContinuous-v0"):
    """Return an instance of the MAML-TRPO algorithm."""
    env = GarageEnv(env_name=env_name)
    policy = DeterministicMLPPolicy(name='policy',
                                    env_spec=env.spec,
                                    hidden_sizes=[64, 64])
    vfunc = GaussianMLPValueFunction(env_spec=env.spec)

    task_sampler = SetTaskSampler(lambda: GarageEnv(
                                  normalize(env, expected_action_scale=10.)))

    max_path_length = 100
    meta_evaluator = MetaEvaluator(test_task_sampler=task_sampler,
                                   max_path_length=max_path_length,
                                   n_test_tasks=1,
                                   n_test_rollouts=10)
    algo = MAMLTRPO(env=env,
                    policy=policy,
                    value_function=vfunc,
                    max_path_length=max_path_length,
                    meta_batch_size=20,
                    discount=0.99,
                    gae_lambda=1.,
                    inner_lr=0.1,
                    num_grad_updates=1,
                    meta_evaluator=meta_evaluator)
    return algo


def load_mamlppo(env_name="MountainCarContinuous-v0"):
    """Return an instance of the MAML-PPO algorithm."""
    env = GarageEnv(env_name=env_name)
    policy = DeterministicMLPPolicy(name='policy',
                                    env_spec=env.spec,
                                    hidden_sizes=[64, 64])
    vfunc = GaussianMLPValueFunction(env_spec=env.spec)

    task_sampler = SetTaskSampler(lambda: GarageEnv(
                                  normalize(env, expected_action_scale=10.)))

    max_path_length = 100
    meta_evaluator = MetaEvaluator(test_task_sampler=task_sampler,
                                   max_path_length=max_path_length,
                                   n_test_tasks=1,
                                   n_test_rollouts=10)
    algo = MAMLPPO(env=env,
                   policy=policy,
                   value_function=vfunc,
                   max_path_length=max_path_length,
                   meta_batch_size=20,
                   discount=0.99,
                   gae_lambda=1.,
                   inner_lr=0.1,
                   num_grad_updates=1,
                   meta_evaluator=meta_evaluator)
    return algo


def load_mamlvpg(env_name="MountainCarContinuous-v0"):
    """Return an instance of the MAML-VPG algorithm."""
    env = GarageEnv(env_name=env_name)
    policy = DeterministicMLPPolicy(name='policy',
                                    env_spec=env.spec,
                                    hidden_sizes=[64, 64])
    vfunc = GaussianMLPValueFunction(env_spec=env.spec)

    task_sampler = SetTaskSampler(lambda: GarageEnv(
                                  normalize(env, expected_action_scale=10.)))

    max_path_length = 100
    meta_evaluator = MetaEvaluator(test_task_sampler=task_sampler,
                                   max_path_length=max_path_length,
                                   n_test_tasks=1,
                                   n_test_rollouts=10)
    algo = MAMLVPG(env=env,
                   policy=policy,
                   value_function=vfunc,
                   max_path_length=max_path_length,
                   meta_batch_size=20,
                   discount=0.99,
                   gae_lambda=1.,
                   inner_lr=0.1,
                   num_grad_updates=1,
                   meta_evaluator=meta_evaluator)
    return algo


def load_sac(env_name="MountainCarContinuous-v0"):
    """Return an instance of the SAC algorithm."""
    env = GarageEnv(env_name=env_name)
    policy = DeterministicMLPPolicy(name='policy',
                                    env_spec=env.spec,
                                    hidden_sizes=[64, 64])

    qf1 = ContinuousMLPQFunction(env_spec=env.spec,
                                 hidden_sizes=[64, 64],
                                 hidden_nonlinearity=F.relu)
    qf2 = ContinuousMLPQFunction(env_spec=env.spec,
                                 hidden_sizes=[64, 64],
                                 hidden_nonlinearity=F.relu)

    replay_buffer = PathBuffer(capacity_in_transitions=int(1e6))
    algo = SAC(env_spec=env.spec,
               policy=policy,
               qf1=qf1,
               qf2=qf2,
               gradient_steps_per_itr=1000,
               max_path_length=500,
               replay_buffer=replay_buffer)
    return algo


def load_ddpg(env_name="MountainCarContinuous-v0"):
    """Return an instance of the DDPG algorithm.

    Note: does this only work with continous?
    """
    env = GarageEnv(env_name=env_name)
    policy = DeterministicMLPPolicy(name='policy',
                                    env_spec=env.spec,
                                    hidden_sizes=[64, 64])
    qf = ContinuousMLPQFunction(env_spec=env.spec,
                                hidden_sizes=[64, 64],
                                hidden_nonlinearity=F.relu)
    replay_buffer = PathBuffer(capacity_in_transitions=int(1e6))
    algo = DDPG(env_spec=env.spec,
                policy=policy,
                qf=qf,
                replay_buffer=replay_buffer)
    return algo


def load_vpg(env_name="CartPole-v0"):
    """Return an instance of the VPG algorithm."""
    env = GarageEnv(env_name=env_name)
    policy = DeterministicMLPPolicy(name='policy',
                                    env_spec=env.spec,
                                    hidden_sizes=(32, 32))
    vfunc = GaussianMLPValueFunction(env_spec=env.spec)
    algo = VPG(env_spec=env.spec, policy=policy, value_function=vfunc)
    return algo


def load_trpo(env_name="CartPole-v0"):
    """Return an instance of the TRPO algorithm."""
    env = GarageEnv(env_name=env_name)
    policy = DeterministicMLPPolicy(name='policy',
                                    env_spec=env.spec,
                                    hidden_sizes=(32, 32))
    vfunc = GaussianMLPValueFunction(env_spec=env.spec)
    algo = TRPO(env_spec=env.spec, policy=policy, value_function=vfunc)
    return algo


def load_ppo(env_name="CartPole-v0"):
    """Return an instance of the PPO algorithm."""
    env = GarageEnv(env_name=env_name)
    policy = DeterministicMLPPolicy(name='policy',
                                    env_spec=env.spec,
                                    hidden_sizes=(32, 32))
    vfunc = GaussianMLPValueFunction(env_spec=env.spec)
    algo = PPO(env_spec=env.spec, policy=policy, value_function=vfunc)
    return algo


def load_model(model_name, environment_name):
    """Load a model with specific environement and default configuration.

    Just pytorch models for the moment.

    Args:
      model_name (string): name of the model/algorithm.
      environment_name (string): name of the environment with which to pair.
    """
    # TODO(tonioteran) Merge all possible function for simpler implementation.
    if model_name == "DDPG":
        return load_ddpg(env_name=environment_name)
    elif model_name == "VPG":
        return load_vpg(env_name=environment_name)
    elif model_name == "MAMLVPG":
        return load_mamlvpg(env_name=environment_name)
    elif model_name == "PPO":
        return load_ppo(env_name=environment_name)
    elif model_name == "MAMLPPO":
        return load_mamlppo(env_name=environment_name)
    elif model_name == "TRPO":
        return load_trpo(env_name=environment_name)
    elif model_name == "MAMLTRPO":
        return load_mamltrpo(env_name=environment_name)
    elif model_name == "SAC":
        return load_sac(env_name=environment_name)
    else:
        print("RL Garage wrapper: Model not implemented!!!")
        return None
