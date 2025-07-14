import pickle
import os
import pandas as pd

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