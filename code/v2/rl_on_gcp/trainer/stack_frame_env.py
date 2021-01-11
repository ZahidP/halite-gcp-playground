# Copyright 2018 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TODO(praneetdutta): DO NOT SUBMIT without one-line documentation for train

Wrap a gym environment to make stacked observations.

This file wraps a default gym environment.
Image pre-processing and frame skipping are performed here.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import numpy as np

from ship_state_wrapper import ShipStateWrapper
from kaggle_environments import Environment


# https://raw.githubusercontent.com/GoogleCloudPlatform/training-data-analyst/master/blogs/rl-on-gcp/DQN_Breakout/rl_on_gcp/trainer/stack_frame_env.py

class StackFrameEnv(gym.Wrapper):
  """Wrap a gym env, does image processing and frame skipping."""

  def __init__(self, env: Environment, num_frames, radius):
    """Initialization."""
    gym.Wrapper.__init__(self, env)
    self._k = num_frames
    self.ship_state_wrapper = ShipStateWrapper(radius=radius, map_size=env.configuration['size'], max_frames=num_frames)

  def _process_frame(self, observation, uid, pos):
    """Process the image."""
    return self.ship_state_wrapper

  def _pad_observation(self, observations):
    """Pad observation to give self._k frames."""
    padding = [observations[-1]] * (self._k - len(observations))
    res = observations + padding
    res = np.concatenate(res, axis=-1)
    return res

  def reset(self):
    """Reset the env."""
    state = self.env.reset()
    observations = []
    img = self._process_image(state)
    observations.append(np.expand_dims(img, axis=-1))
    return self._pad_observation(observations)

  def step(self, action):
    """Execute the action for self._k times."""
    if self._k == 1:
      state, reward, done, info = self.env(action)
      state = self._process_image(state)
      return state, reward, done, info
    else:
      accumulated_reward = 0
      observations = []
      done = False
      info = None
      for _ in xrange(self._k):
        state, reward, done, info = self.env.step(action)
        accumulated_reward += reward
        img = self._process_image(state)
        observations.append(np.expand_dims(img, axis=-1))
        if done:
          break
      observations = self._pad_observation(observations)
    return observations, accumulated_reward, done, info