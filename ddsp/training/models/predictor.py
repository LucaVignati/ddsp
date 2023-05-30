# Copyright 2022 The DDSP Authors.
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

"""Model that encodes audio features and decodes with a ddsp processor group."""

import ddsp
from ddsp.training.models.model import Model


class Predictor(Model):
  """Wrap the model function for dependency injection with gin."""

  def __init__(self,
               preprocessor=None,
               predictor=None,
               f0_losses=None,
               ld_losses=None,
               **kwargs):
    super().__init__(**kwargs)
    self.preprocessor = preprocessor
    self.predictor = predictor
    self.f0_losses = ddsp.core.make_iterable(f0_losses)
    self.ld_losses = ddsp.core.make_iterable(ld_losses)

  def preprocess(self, features, training=True):
    """Get conditioning by preprocessing."""
    if self.preprocessor is not None:
      features.update(self.preprocessor(features, training=training))
    return features

  def predict(self, features, training=True):
    """Get generated audio by decoding than processing."""
    features.update(self.decoder(features, training=training))

  def call(self, features, training=True):
    """Run the core of the network, get predictions and loss."""
    features.update(self.preprocessor(features, training=training))
    output_features = (self.predictor(features, training=training))

    # Make output differential
    # for k, v in output_features.items():
    #   v += features[k]

    # if training:
    self._update_losses_dict(
          self.f0_losses, features['f0_scaled'][:, 1:], output_features['f0_scaled'][:, :-1])
    self._update_losses_dict(
          self.ld_losses, features['ld_scaled'][:, 1:], output_features['ld_scaled'][:, :-1])
            

    return output_features
  
  def get_audio_from_outputs(self, outputs):
    """Extract audio output tensor from outputs dict of call()."""
    return None

