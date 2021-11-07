import matplotlib.pyplot as plt
import numpy as np
import h5py
import os
import data_utils
from mpl_toolkits.mplot3d import Axes3D

def show3Dpose(channels, ax, lcolor="#3498db", rcolor="#e74c3c", add_labels=False): # blue, orange
  """
  Visualize a 3d skeleton

  Args
    channels: 96x1 vector. The pose to plot.
    ax: matplotlib 3d axis to draw on
    lcolor: color for left part of the body
    rcolor: color for right part of the body
    add_labels: whether to add coordinate labels
  Returns
    Nothing. Draws on ax.
  """

  H36M_NAMES = ['']*32
  H36M_NAMES[0]  = 'Hip'
  H36M_NAMES[1]  = 'RHip'
  H36M_NAMES[2]  = 'RKnee'
  H36M_NAMES[3]  = 'RFoot'
  H36M_NAMES[4]  = 'RFootTip'
  H36M_NAMES[6]  = 'LHip'
  H36M_NAMES[7]  = 'LKnee'
  H36M_NAMES[8]  = 'LFoot'
  H36M_NAMES[12] = 'Spine'
  H36M_NAMES[13] = 'Thorax'
  H36M_NAMES[14] = 'Neck/Nose'
  H36M_NAMES[15] = 'Head'
  H36M_NAMES[17] = 'LShoulder'
  H36M_NAMES[18] = 'LElbow'
  H36M_NAMES[19] = 'LWrist'
  H36M_NAMES[25] = 'RShoulder'
  H36M_NAMES[26] = 'RElbow'
  H36M_NAMES[27] = 'RWrist'
  
  list = ['Pelvis' ,'RHip' ,'RKnee' ,'RAnkle' ,'LHip' ,'LKnee' ,'LAnkle', 'Spine1', 'Neck' ,'Head' ,'Site' ,'LShoulder' ,'LElbow' ,'LWrist' ,'RShoulder' ,'RElbow' ,'RWrist']

  #assert channels.size == len(data_utils.H36M_NAMES)*3, "channels should have 96 entries, it has %d instead" % channels.size
  #print('length', len(data_utils.H36M_NAMES))
  vals = np.reshape( channels, (17, -1))
  print('vals shape', vals.shape)

  #I   = np.array([1,2,3,4,1,7,8,1, 13,14,15,14,18,19,14,26,27])-1 # start points
  #J   = np.array([2,3,4,5,7,8,9,13,14,15,16,18,19,20,26,27,28])-1 # end points
  I   = np.array([1,2,3,3,1,5,6,1, 10,8,9,8,12,13,8,15,16]) # start points
  J   = np.array([2,3,3, 5,5,7,9,10,8,9, 16,12,13, 16,15,16, 16]) # end points
  LR  = np.array([1,1,1,1,0,0,0,0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)

  # Make connection matrix
  for i in np.arange( len(I) ):
    x, y, z = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(3)]
    ax.plot(x, y, z, lw=2, c=lcolor if LR[i] else rcolor)
    ax.text(x[1], y[1], z[1], H36M_NAMES[J[i]] )

  # RADIUS = 750 # space around the subject
  RADIUS = 750 # space around the subject
  xroot, yroot, zroot = vals[0,0], vals[0,1], vals[0,2]
  ax.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
  ax.set_zlim3d([-RADIUS+zroot, RADIUS+zroot])
  ax.set_ylim3d([-RADIUS+yroot, RADIUS+yroot])

  if add_labels:
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

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
