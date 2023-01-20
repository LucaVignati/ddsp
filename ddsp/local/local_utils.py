import base64
import io

import ddsp
import ddsp.training
import tensorflow.compat.v2 as tf
import pickle
import numpy as np
from scipy.io import wavfile
from IPython import display

DEFAULT_SAMPLE_RATE = ddsp.spectral_ops.CREPE_SAMPLE_RATE

specplot = ddsp.training.plotting.specplot

def save_dataset_statistics(data_provider,
                            file_path=None,
                            batch_size=1,
                            power_frame_size=256):
  """Calculate dataset stats and save in a pickle file.

  Calls out to postprocessing.compute_dataset_statistics.

  Args:
    data_provider: A DataProvider from ddsp.training.data.
    file_path: Path for saved pickle file of dataset statistics.
    batch_size: Iterate over dataset with this batch size.
    power_frame_size: Calculate power features on the fly with this frame size.
    power_frame_rate: Calculate power features on the fly with this frame rate.

  Returns:
    Dictionary of dataset statistics. This is an overcomplete set of statistics,
    as there are now several different tone transfer implementations (js, colab,
    vst) that need different statistics for normalization.
  """

  ds_stats = ddsp.training.postprocessing.compute_dataset_statistics(
      data_provider, batch_size, power_frame_size)

  # Save.
  if file_path is not None:
    with tf.io.gfile.GFile(file_path, 'wb') as f:
      pickle.dump(ds_stats, f)
    print(f'Done! Saved dataset statistics to: {file_path}')

  return ds_stats

def play(array_of_floats,
         sample_rate=DEFAULT_SAMPLE_RATE,
         ephemeral=False,
         autoplay=False):
  """Creates an HTML5 audio widget to play a sound in Colab.

  This function should only be called from a Colab notebook.

  Args:
    array_of_floats: A 1D or 2D array-like container of float sound samples.
      Values outside of the range [-1, 1] will be clipped.
    sample_rate: Sample rate in samples per second.
    ephemeral: If set to True, the widget will be ephemeral, and disappear on
      reload (and it won't be counted against realtime document size).
    autoplay: If True, automatically start playing the sound when the widget is
      rendered.
  """
  # If batched, take first element.
  if len(array_of_floats.shape) == 2:
    array_of_floats = array_of_floats[0]

  normalizer = float(np.iinfo(np.int16).max)
  array_of_ints = np.array(
      np.asarray(array_of_floats) * normalizer, dtype=np.int16)
  memfile = io.BytesIO()
  wavfile.write(memfile, sample_rate, array_of_ints)
  html = """<audio controls {autoplay}>
              <source controls src="data:audio/wav;base64,{base64_wavfile}"
              type="audio/wav" />
              Your browser does not support the audio element.
            </audio>"""
  html = html.format(
      autoplay='autoplay' if autoplay else '',
      base64_wavfile=base64.b64encode(memfile.getvalue()).decode('ascii'))
  memfile.close()
#   global _play_count
#   _play_count += 1
  if ephemeral:
    element = 'id_%s' % _play_count
    display.display(display.HTML('<div id="%s"> </div>' % element))
    js = output._js_builder  # pylint:disable=protected-access
    js.Js('document', mode=js.EVAL).getElementById(element).innerHTML = html
  else:
    display.display(display.HTML(html))