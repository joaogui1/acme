# python3
# Copyright 2018 DeepMind Technologies Limited. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Noise layers (for exploration)."""

from acme import types
import haiku as hk
import jax
import jax.numpy as jnp
import tree


class ClippedGaussian(hk.Module):
  """Haiku module for adding clipped Gaussian noise to each output."""

  def __init__(self, stddev: float, seed: int = 0, name: str = 'clipped_gaussian'):
    super().__init__(name=name)
    self._rng = hk.PRNGSequence(seed)
    self._stddev = stddev

  def __call__(self, inputs: types.NestedTensor) -> types.NestedTensor:
    def add_noise(array: jnp.ndarray):
      output = array + jax.random.normal(next(self._rng), array.shape)*self._stddev
      output = jnp.clip(output, -1.0, 1.0)
      return output

    return tree.map_structure(add_noise, inputs)