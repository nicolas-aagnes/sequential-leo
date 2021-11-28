import numpy as np
from glob import glob
import json
import matplotlib.pyplot as plt
import argparse
import torch

## This file generates batches of data as outlined in the PAML paper
## Train batch: 50 random frames of the subject
## Val batch: 10 random frames of the subject

## Helper Functions
def get_data_file(annotations_path, subject_id):
  return glob(annotations_path + "/*subject"+str(subject_id)+"_data*")[0]

def get_coords_file(annotations_path, subject_id):
  return glob(annotations_path + "/*subject"+str(subject_id)+"_joint_3d*")[0]

def get_subject_data_length(annotations_path, subject_id):
  data_file = get_data_file(annotations_path, subject_id)

  ## Parse data file
  with open(data_file,'r') as data_json_file: 
    data = json.load(data_json_file)

  return len(data['images'])


def get_subject_action_boundaries(annotations_path, subject_id):
  data_file = get_data_file(annotations_path, subject_id)

  ## Parse data file
  with open(data_file,'r') as data_json_file: 
    data = json.load(data_json_file)

  boundaries = [0]
  prev_action = data['images'][0]['action_name']
  for i, item in enumerate(data['images']):
    if i == 0: continue
    if item['action_name'] != prev_action:
      prev_action = item['action_name']
      boundaries.append(i)
  
  boundaries.append(len(data['images']))
  # print(boundaries)

  return boundaries



def get_data(subject_id, random_index, annotations_path, timesteps=1):
  data_file = get_data_file(annotations_path, subject_id)
  coords_file = get_coords_file(annotations_path, subject_id)

  ## Parse data file
  with open(data_file,'r') as data_json_file: 
    data = json.load(data_json_file)

  ## Parse coordinates file
  with open(coords_file,'r') as coords_json_file:
    coords = json.load(coords_json_file)

  frames = np.arange(random_index, random_index + timesteps)
  pose3d = []
  for i, item in enumerate(frames):
    meta_data = data['images'][item]
    coord_data = np.array(coords[str(meta_data['action_idx'])][str(meta_data['subaction_idx'])][str(meta_data['frame_idx'])])
    pose3d.append(coord_data)
  pose3d = np.array(pose3d)

  return data, meta_data, pose3d
  

def check_range_overlap(range1, range2):
  return len(set(range1).intersection(set(range2))) != 0 ## Make sure to avoid overlapping with any boundaries

def generate_random_task(timesteps, timesteps_pred, num_support, num_query, annotations_path, downsample=2):

  ## Get Random Subject 
  subject_id = np.random.randint(1, 8)
  while subject_id != 5: # Excluding 5 for test time
    subject_id = np.random.randint(1, 8)
  length = get_subject_data_length(annotations_path, subject_id)

  boundaries = get_subject_action_boundaries(annotations_path, subject_id)
  
  timesteps *= downsample
  timesteps_pred *= downsample
  timesteps_total = timesteps+timesteps_pred

  ## Get Random Action
  random_action = np.random.randint(len(boundaries[:-1]))

  ## Get Random Index
  index = np.random.randint(boundaries[random_action], boundaries[random_action+1] - timesteps_total)

  ## Get Train and Query Data
  data = []
  label = []
  total_range = list(range(index, index+timesteps_total))
  for item in np.arange(num_support + num_query):

    ## Get Data
    _, _, pose3d = get_data(subject_id, index, annotations_path, timesteps=timesteps_total)
    data.append(pose3d[:timesteps].reshape(timesteps, 51)) # 51 = 17 x 3
    label.append(pose3d[timesteps:].reshape(timesteps_pred, 51))

    ## Calculate next index
    index = np.random.randint(boundaries[random_action], boundaries[random_action+1] - timesteps_total)
    index_range = range(index, index + timesteps_total)
    while check_range_overlap(total_range, index_range): ## Make sure to avoid overlapping between examples
      index = np.random.randint(boundaries[random_action], boundaries[random_action+1] - timesteps_total)
    
    total_range.extend(list(range(index, index+timesteps_total)))

  ## Get every Nth frame (batchsize, numframes, 51)
  data = np.array(data)
  label = np.array(label)
  data = data[:,::downsample,:]
  label = label[:,::downsample,:]

  ## Return Batch
  train_data = data[:num_support]
  train_label = label[:num_support]
  query_data = data[num_support:]
  query_label = label[num_support:]
  return train_data, train_label, query_data, query_label

  

if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  parser.add_argument("--subject_id", type=int, help="subject id", default=7)
  parser.add_argument("--index", type=int, help="image index", default=0)
  parser.add_argument("--timesteps", type=int, help="image index", default=50)
  parser.add_argument("--timesteps_pred", type=int, help="image index", default=10)
  parser.add_argument("--annotations_path", type=str, default="./annotations")

  args = parser.parse_args()
  
  train_data, train_label, query_data, query_label = generate_random_example(args.timesteps, args.timesteps_pred, args.annotations_path)
  print("Train Data Example Shape", train_data.shape)
  print("Train Label Example Shape", train_label.shape)
  print("Query Data Example Shape", query_data.shape)
  print("Query Label Example Shape", query_label.shape)


