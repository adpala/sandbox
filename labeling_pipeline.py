import numpy as np
import deepdish as dd
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
from scipy.spatial import cKDTree
from leap_utils.utils import iswin, ismac

if iswin():
    root = 'Z:/#Common/'
elif ismac():
    root = '/Volumes/ukme04/#Common/'
else:
    root = '/scratch/clemens10/'

expID = 'localhost-20180720_182837'
res_path = root+'chainingmic/res'
ws_path = f"{res_path}/{expID}/{expID}_ws.h5"
cwt_path = f"{res_path}/{expID}/{expID}_cwt.h5"
tsne_path = f"{res_path}/{expID}/{expID}_temptsne.h5"
clabels_path = f"{res_path}/{expID}/{expID}_clabels.h5"
all_labels_path = f"{res_path}/{expID}/{expID}_all_labels.h5"
# tsne_path = 'Z:/#Common/adrian/Workspace/temp/'+expID+'_tsne.h5'
# ws_path = 'Z:/#Common/adrian/Workspace/temp/'+expID+'_ws.h5'
# clabels_path = 'Z:/#Common/adrian/Workspace/temp/'+expID+'_clabels.h5'
# all_labels_path = 'Z:/#Common/adrian/Workspace/temp/'+expID+'_all_labels.h5'
# cwt_path = 'Z:/#Common/adrian/Workspace/temp/'+expID+'_cwt.h5'


def do_kdtree(combined_arrays, points, k: int = 1):
    mytree = cKDTree(combined_arrays)
    dist, indexes = mytree.query(points, k=k, n_jobs=-1)
    return indexes


# Inputs
assign_labels = True

if assign_labels:
    tdata = dd.io.load(tsne_path)
    wdata = dd.io.load(ws_path)
    cwtdata = dd.io.load(cwt_path)

    image = wdata['image']
    labels = wdata['labels']
    rlabels = labels.ravel()
    Z = tdata['tsne']
    grid_size = wdata['grid_size']
    X50 = tdata['X50']
    all_X50 = tdata['all_X50']
    cwt_array = cwtdata['cwt']
    random_data_idx = tdata['random_data_idx']

    # Shoe-horn existing data for entry into KDTree routines
    xx_d = np.linspace(np.min(Z[:, 0]), np.max(Z[:, 0]), grid_size)
    yy_d = np.linspace(np.min(Z[:, 1]), np.max(Z[:, 1]), grid_size)
    xx_dv, yy_dv = np.meshgrid(xx_d, yy_d)
    combined_x_y_arrays = np.dstack([yy_dv.ravel(), xx_dv.ravel()])[0]
    nearest_neighbor_index = do_kdtree(combined_x_y_arrays, list(Z))
    Z_labels = rlabels[nearest_neighbor_index]

    print('saving labels!')
    clabels_data = {'Z_labels': Z_labels,
                    'expID': expID,
                    'tsne_path': tsne_path}
    dd.io.save(clabels_path, clabels_data)

    all_nearest_neighbor_index = do_kdtree(all_X50[random_data_idx, ...], list(all_X50), k=2)
    print(all_nearest_neighbor_index.shape)
    all_labels = Z_labels[all_nearest_neighbor_index[:, 0]]

    all_Z = np.median(Z[all_nearest_neighbor_index, :], axis=1).astype(np.int)
    print(all_Z.shape)

    print('saving all labels!')
    all_labels_data = {'all_labels': all_labels,
                       'all_nearest_neighbor_index': all_nearest_neighbor_index,
                       'all_Z': all_Z,
                       'expID': expID,
                       'tsne_path': tsne_path}
    dd.io.save(all_labels_path, all_labels_data)
else:
    clabels_data = dd.io.load(clabels_path)
    all_labels_data = dd.io.load(all_labels_path)
    wdata = dd.io.load(ws_path)
    tdata = dd.io.load(tsne_path)
    image = wdata['image']
    labels = wdata['labels']
    Z = tdata['tsne']
    Z_labels = clabels_data['Z_labels']
    all_labels = all_labels_data['all_labels']
    all_Z = all_labels_data['all_Z']


cmap = get_cmap('gist_ncar')    # cmap = get_cmap('prism')

norm = Normalize(vmin=1, vmax=np.max(Z_labels))
plt.figure(figsize=(16, 4))
plt.subplot(141)
plt.imshow(image, aspect='auto')
plt.title('t-SNE')
plt.axis('off')
plt.tight_layout()
plt.subplot(142)
plt.imshow(labels, interpolation='nearest', aspect='auto')
plt.title('Watershed')
plt.axis('off')
plt.tight_layout()
plt.subplot(143)
plt.scatter(Z[:, 0], Z[:, 1], s=1, c=cmap(norm(Z_labels)))
plt.title('Clusters (Random samples)')
plt.axis('off')
plt.tight_layout()
plt.subplot(144)
plt.scatter(all_Z[:, 0], all_Z[:, 1], s=1, c=cmap(norm(all_labels)))
plt.title('Clusters (All data)')
plt.axis('off')
plt.tight_layout()
plt.show()
