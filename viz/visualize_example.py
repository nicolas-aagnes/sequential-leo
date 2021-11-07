import numpy as np
from glob import glob
import json
import matplotlib.pyplot as plt
from helper import show3Dpose


## Manual Parameters
subject_id = 7
random_index = 0


## Helper Functions
def get_data_file(subject_id):
  return glob("annotations/*subject"+str(subject_id)+"_data*")[0]

def get_coords_file(subject_id):
  return glob("annotations/*subject"+str(subject_id)+"_joint_3d*")[0]


data_file = get_data_file(subject_id)
coords_file = get_coords_file(subject_id)


## Parse data file
with open(data_file,'r') as data_json_file: 
  data = json.load(data_json_file)

meta_data = data['images'][random_index]

## Parse coordinates file
with open(coords_file,'r') as coords_json_file:
  coords = json.load(coords_json_file)

coord_data = coords[str(meta_data['action_idx'])][str(meta_data['subaction_idx'])][str(meta_data['frame_idx'])]
coord_data = np.array(coord_data)

print(coord_data)
print(coord_data.shape)

## Plot coordinates for 17x3
#fig = plt.figure()
#ax = plt.axes(projection='3d')
#
#x = [a[0] for a in coord_data]
#y = [a[1] for a in coord_data]
#z = [a[2] for a in coord_data]
#
#ax.plot3D(x, y ,z, 'gray')
#plt.show()

#pose3d = np.concatenate((coord_data.flatten(), np.zeros(45)))
pose3d = coord_data

fig = plt.figure()
ax = fig.add_subplot(111, aspect='auto', projection='3d')

show3Dpose(pose3d, ax )
plt.show()





