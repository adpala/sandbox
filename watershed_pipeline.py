import numpy as np
import deepdish as dd
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from scipy import ndimage as ndi
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from leap_utils.utils import iswin, ismac
import logging


if iswin():
    root = 'Z:/#Common/'
elif ismac():
    root = '/Volumes/ukme04/#Common/'
else:
    root = '/scratch/clemens10/'

expID = 'localhost-20180720_182837'
res_path = root+'chainingmic/res'
ws_path = f"{res_path}/{expID}/{expID}_ws.h5"
tsne_path = f"{res_path}/{expID}/{expID}_temptsne.h5"

# Inputs
bw = 2
grid_size = 100
min_distance = 2
connectivity = 8

# Loading data
logging.info(f"   loading tsne from: {tsne_path}")
tsne_data = dd.io.load(tsne_path)
Z = tsne_data['tsne']
logging.info(f"   max_iter: {tsne_data['max_iter']}, late_exag_coeff: {tsne_data['late_exag_coeff']}, start_late_exag_iter: {tsne_data['start_late_exag_iter']}, nsamples: {Z.shape[0]}.")


logging.info(f"   doing kde from scikit-learn.")
kde = KernelDensity(bandwidth=bw).fit(Z)

# grid
xx_d = np.linspace(np.min(Z[:, 0]), np.max(Z[:, 0]), grid_size)
yy_d = np.linspace(np.min(Z[:, 1]), np.max(Z[:, 1]), grid_size)
xx_dv, yy_dv = np.meshgrid(xx_d, yy_d)
coor = np.array([xx_dv.flatten(), yy_dv.flatten()]).swapaxes(0, 1)

logging.info(f"   construct estimate from kde.")
image = np.exp(kde.score_samples(coor)).reshape((grid_size, grid_size))

logging.info(f"   run watershed algorithm.")
local_maxi = peak_local_max(image, indices=False, min_distance=min_distance)
markers = ndi.label(local_maxi)[0]
labels = watershed(-image, markers=markers, connectivity=connectivity)
logging.info(f"   found {np.sum(local_maxi)} clusters.")

image = np.flip(image, 0)
labels = np.flip(labels, 0)

logging.info(f"   saving watershed results.")
ws_data = {'markers': markers,
           'labels': labels,
           'image': image,
           'nclusters': np.sum(local_maxi),
           'min_distance': min_distance,
           'connectivity': connectivity,
           'bw': bw,
           'grid_size': grid_size,
           'expID': expID,
           'tsne_path': tsne_path}
dd.io.save(ws_path, ws_data)

logging.info(f"   plotting results.")
plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.scatter(Z[:, 0], Z[:, 1], s=1)
plt.title('tSNE')
plt.axis('off')
plt.tight_layout()
plt.subplot(132)
plt.imshow(image, aspect='auto')
plt.title('Density')
plt.axis('off')
plt.tight_layout()
plt.subplot(133)
plt.imshow(labels, interpolation='nearest')
plt.title('Watershed')
plt.axis('off')
plt.tight_layout()
plt.show()
