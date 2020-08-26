#!/Users/tonio/python-environments/RLwrap-SB/bin/python
# Wrapper for RLlib library.
import ray
import gym
import importlib


# Global dictionary that maps algorithms to modules.
alg2module = {
    "A2C": "a3c",
    "A3C": "a3c",
    "ARS": "ars",
    "ES": "es",
    "DDPG": "ddpg",
    "ApexDDPG": "ddpg",
    "DQN": "dqn",
    "APEX-DQN": "dqn",
    "SimpleQ": "dqn",
    "Impala": "impala",
    "MARWIL": "marwil",
    "PG": "pg",
    "PPO": "ppo",
    "APPO": "ppo",
    "DDPPO": "ppo",
    "QMIX": "",
    "SAC": "sac",
    # "TD3": "",
    # "Rainbow": "",
    # "AlphaZero": "",
    # "LinUCB": "",
    # "LinTS": "",
    # "MADDPG": "",
}


def models():
    """Returns a list of all available RLlib models."""
    return list(alg2module.keys())


def load_model(model_name, environment_name):
    """Load a model with specific environement and default configuration.

    Args:
      model_name (string): name of the model/algorithm.
      environment_name (string): name of the environment with which to pair.
    """
    ray.init(ignore_reinit_error=True)
    # Fetch the specified model trainer.
    model_module = importlib.import_module(
        "ray.rllib.agents." + alg2module[model_name])
    # Load the trainer and return.
    trainer = getattr(model_module, model_name + 'Trainer')

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

    return trainer(env=environment_name)
