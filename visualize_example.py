import numpy as np
from glob import glob
import json
import matplotlib.pyplot as plt
import argparse
from batch_data import get_data

def print_pose_information(meta_data):
  print("Subject ID:", meta_data['subject'])
  print("Action Name:", meta_data['action_name'])
  print("Frame Index:", meta_data['frame_idx'])

## Plot the human pose for a 17x3 joint matrix
def plot_frame(pose3d):
  
  list_anatomy = ['Pelvis' ,'RHip' ,'RKnee' ,'RAnkle' ,'LHip' ,'LKnee' ,'LAnkle', 'Spine1', 'Neck' ,'Head' ,'Site' ,'LShoulder' ,'LElbow' ,'LWrist' ,'RShoulder' ,'RElbow' ,'RWrist']
  dict_anatomy = {}

  for i, item in enumerate(list_anatomy):
    dict_anatomy[item] = pose3d[i]


  print("Dictionary of Pose3D Data: ", dict_anatomy)

  ## Plot coordinates for 17x3
  fig = plt.figure()
  ax = plt.axes(projection='3d')


  def connect_joints(joint1, joint2, color='red'):
    l = zip(dict_anatomy[joint1], dict_anatomy[joint2])
    l = list([a for a in l])
    ax.plot(l[0], l[1], l[2], c=color)
    # ax.text(l[0][0], l[1][0], l[2][0], joint1)
    # ax.text(l[0][1], l[1][1], l[2][1], joint2)


  connect_joints('Pelvis', 'RHip', color='red')
  connect_joints('RHip', 'RKnee', color='red')
  connect_joints('RKnee', 'RAnkle', color='red')
  connect_joints('Pelvis', 'LHip', color='blue')
  connect_joints('LHip', 'LKnee', color='blue')
  connect_joints('LKnee', 'LAnkle', color='blue')
  connect_joints('Pelvis', 'Spine1', color='red')
  connect_joints('Spine1', 'Neck', color='red')
  # connect_joints('Site', 'Neck')
  connect_joints('Neck', 'Head', color='red')
  connect_joints('Neck', 'LShoulder', color='blue')
  connect_joints('LShoulder', 'LElbow', color='blue')
  connect_joints('LElbow', 'LWrist', color='blue')
  connect_joints('Neck', 'RShoulder', color='red')
  connect_joints('RShoulder', 'RElbow', color='red')
  connect_joints('RElbow', 'RWrist', color='red')


  # RADIUS = 750 # space around the subject
  RADIUS = 500 # space around the subject
  xroot, yroot, zroot = pose3d[0,0], pose3d[0,1], pose3d[0,2]
  ax.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
  ax.set_zlim3d([-RADIUS+zroot, RADIUS+zroot])
  ax.set_ylim3d([-RADIUS+yroot, RADIUS+yroot])

  # Get rid of the ticks and tick labels
  ax.set_xticks([])
  ax.set_yticks([])
  ax.set_zticks([])

  ax.get_xaxis().set_ticklabels([])
  ax.get_yaxis().set_ticklabels([])
  ax.set_zticklabels([])
  ax.set_aspect('auto')

  # Get rid of the panes (actually, make them white)
  white = (1.0, 1.0, 1.0, 0.0)
  ax.w_xaxis.set_pane_color(white)
  ax.w_yaxis.set_pane_color(white)
  # Keep z pane

  # Get rid of the lines in 3d
  ax.w_xaxis.line.set_color(white)
  ax.w_yaxis.line.set_color(white)
  ax.w_zaxis.line.set_color(white)

  plt.show()

## Given a subject and frame index, plot the human pose for that datapoint.
def plot_random_frame(subject_id, random_index, annotations_path):
  
  data, meta_data, pose3d = get_data(subject_id, random_index, annotations_path)
  pose3d = np.squeeze(pose3d)

  print("Total Number of Frames: ", len(data['images']))
  print("Current Frame: ", random_index)
  print_pose_information(meta_data)
  
  plot_frame(pose3d)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  parser.add_argument("--subject_id", type=int, help="subject id", default=7)
  parser.add_argument("--index", type=int, help="image index", default=0)
  parser.add_argument("--annotations_path", type=str, default="./annotations")

  args = parser.parse_args()
  
  plot_frame(args.subject_id, args.index, args.annotations_path)