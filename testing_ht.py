import numpy as np
import deepdish as dd
from leap_utils.utils import iswin, ismac
import os

if iswin():
    root = 'Z:/#Common/'
elif ismac():
    root = '/Volumes/ukme04/#Common/'
else:
    root = '/scratch/clemens10/'

res_path = root+'chainingmic/res'

fnames = list()

file_fnames = open('Z:/#Common/adrian/Workspace/temp/good_videos.txt', 'w')

for name in os.listdir(res_path):
    if os.path.isdir(res_path + '/' + name) and name[:9] == 'localhost' and os.path.exists(f"{res_path}/{name}/{name}_poses.h5"):
        pose_path = f"{res_path}/{name}/{name}_poses.h5"
        data = dd.io.load(pose_path)
        if np.sum(data['positions'][:, 11, 0].astype(np.int16) - data['positions'][:, 0, 0].astype(np.int16) < 20) < 0.1*data['positions'].shape[0]:
            fnames.append(name)
            print(name)
            file_fnames.write(name + '\n')

# nloops = int(len(fnames)/10)
# for ii in range(nloops):
#     plt.figure(ii, figsize=[10, 8])
#     for count, expID in enumerate(fnames[ii*10:(ii+1)*10]):
#         pose_path = f"{res_path}/{expID}/{expID}_poses.h5"
#         data = dd.io.load(pose_path)
#         positions = data['positions']
#         HT_distance = positions[:, 11, 0].astype(np.int16) - positions[:, 0, 0].astype(np.int16)
#         plt.subplot(5, 2, count+1)
#         plt.hist(HT_distance)
#         plt.tight_layout()
#         plt.xlim(-90, 90)
#         plt.title(expID)
#     plt.show()
