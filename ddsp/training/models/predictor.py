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
               processor_group=None,
               losses=None,
               **kwargs):
    super().__init__(**kwargs)
    self.preprocessor = preprocessor
    self.predictor = predictor
    self.processor_group = processor_group
    self.loss_objs = ddsp.core.make_iterable(losses)

  def preprocess(self, features, training=True):
    """Get conditioning by preprocessing."""
    if self.preprocessor is not None:
      features.update(self.preprocessor(features, training=training))
    return features

  def predict(self, features, training=True):
    """Get generated audio by decoding than processing."""
    features.update(self.decoder(features, training=training))
    return self.processor_group(features)

  def get_audio_from_outputs(self, outputs):
    """Extract audio output tensor from outputs dict of call()."""
    return outputs['audio_synth']

  def call(self, features, training=True):
    """Run the core of the network, get predictions and loss."""
    features = self.preprocess(features, training=training)
    features.update(self.predictor(features, training=training))

    # Run through processor group.
    pg_out = self.processor_group(features, return_outputs_dict=True)

    # Parse outputs
    outputs = pg_out['f0_and_loudness']
    outputs['audio_synth'] = pg_out['signal']

    if training:
      if 'audio' in features.keys():
        self._update_losses_dict(
            self.loss_objs, features['audio'], outputs['audio_synth'])
      
      self._update_losses_dict(
            self.loss_objs, features['f0_and_loudness'], outputs['f0_and_loudness'])

    return outputs

