# -*- coding: utf-8 -*-
# Author: Tonio Teran <tonio@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2020.
"""Wrapper for OpenAI's Gym library and standalone Gym additions.

TODO(tonioteran) Add documentation.

Some examples that I can still include here:
  - https://github.com/openai/gym-soccer
  - https://github.com/openai/gym-wikinav
  - https://github.com/alibaba/gym-starcraft
  - https://github.com/endgameinc/gym-malware
  - https://github.com/hackthemarket/gym-trading
  - https://github.com/tambetm/gym-minecraft
  - https://github.com/ppaquette/gym-doom
  - https://github.com/ppaquette/gym-super-mario
  - https://github.com/tuzzer/gym-maze

"""
import gym
import gym_minigrid  # noqa: F401
import pybulletgym  # noqa: F401
import procgen  # noqa: F401


def environments():
    """Return a list with registered models from imported libraries."""
    return sorted([e.id for e in gym.envs.registry.all()])
