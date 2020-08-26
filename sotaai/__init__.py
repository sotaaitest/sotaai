# -*- coding: utf-8 -*-
# Author: Tonio Teran <tonio@stateoftheart.ai>
# Copyright: Stateoftheart AI PBC 2020.
"""Main file for the stateoftheart.ai development library.

This module abstracts away the library-specific API from multiple libraries to
facilitate the access and usage of a variety of publibly available machine
learning models and environments.

Thus far, the following libraries have been incorporated into the module:

- Reinforcement learning:

  - `Stable Baselines <https://stable-baselines.readthedocs.io/en/master/>`_
  - `Ray's RLlib <https://docs.ray.io/en/latest/rllib.html>`_
  - `OpenAI Gym <https://gym.openai.com/>`_
  - `Minimalistic Gridworld Environment (MiniGrid)
  <https://github.com/maximecb/gym-minigrid>`_
"""
import sotaai.rl        # noqa: F401
import sotaai.cv        # noqa: F401
import sotaai.nlp       # noqa: F401
import sotaai.neurosym  # noqa: F401

