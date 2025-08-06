import pickle
import os
import pandas as pd

from pathlib import Path
from natsort import natsorted

from scipy.optimize import nnls
from skimage.util import img_as_float

import numpy as np

def saveSegmentation(image_list, masks_outlines):
  if not image_list or not masks_outlines:
    return
    
  path_name = os.path.dirname(image_list[0])

  results = []
  for nn, m_o in zip(image_list, masks_outlines):
    segmentation = {}
    segmentation['name'] = os.path.basename(nn)
    segmentation['masks'] = m_o['masks']
    segmentation['outlines'] = m_o['outlines']
    results.append(segmentation)

  output_file = os.path.join(path_name, 'segmentation.pkl')
  with open(output_file, "wb") as file:
    pickle.dump(results, file)

def savePlaqueCounts(image_list, masks_outlines):
  if not image_list or not masks_outlines:
    return
    
  path_name = os.path.dirname(image_list[0])

  labels = []
  counts = []
  for nn, m_o in zip(image_list, masks_outlines):
    bn = os.path.splitext(os.path.basename(nn))[0]
    labels.append(bn)
    counts.append(int(len(m_o['masks'])))

  # pandas pivot table
  df = pd.DataFrame({'label':labels, 'count':counts})

  df['letter'] = df['label'].str[0]
  df['number'] = df['label'].str[1:].astype(int)

  pivot_df = df.pivot(index='letter', columns='number', values='count')

  letters = sorted(set(df['letter']))
  numbers = sorted(set(df['number']))
  pivot_df = pivot_df.reindex(letters)[numbers]
  pivot_df = pivot_df.fillna(0).astype(int)

  #print(pivot_df)

  output_file = os.path.join(path_name, 'plague_counts.xls')
  pivot_df.to_excel(output_file, engine='openpyxl')

# color deconvolution
def separate_stains_nnls(rgb_image, stain_matrix):
  rgb_float = img_as_float(rgb_image)
  stain_matrix_transposed = stain_matrix.T
  reshaped_rgb = rgb_float.reshape(-1, 3)
  stains = np.zeros_like(reshaped_rgb)
  for i in range(reshaped_rgb.shape[0]):
    pixel_od = -np.log(reshaped_rgb[i, :] + 1e-6)
    pixel_concentrations, _ = nnls(stain_matrix_transposed, pixel_od)
    stains[i, :] = pixel_concentrations
  return stains.reshape(rgb_image.shape)

# the scaling function for color deconvolution results
def scale_channel(channel_data):
  """
  Scales a single 2D channel to the 0-255 range for visualization.
  """
  # Ensure there's a range to scale, otherwise return a black image
  if channel_data.max() == channel_data.min():
    return np.zeros_like(channel_data, dtype=np.uint8)
    
  # Scale to 0-1 range
  scaled_01 = (channel_data - channel_data.min()) / (channel_data.max() - channel_data.min())
    
  # Scale to 0-255 and convert to 8-bit integer
  return (scaled_01 * 255).astype(np.uint8)

def findAllImages(folder):
  folder_path = Path(folder)
    
  if not folder_path.is_dir():
    return []

  supported_extensions = ['.png', '.tif', '.tiff', '.CTL']
  all_image_paths = []

  # Path.glob is a generator, so we convert it to a list
  for p in folder_path.glob('*'):
    if p.suffix.lower() in [ext.lower() for ext in supported_extensions]:
      all_image_paths.append(str(p)) # Convert path object back to string

  sorted_names = natsorted(all_image_paths)
  return sorted_names