import numpy as np
import h5py
import matplotlib.pyplot as plt
import logging
import os
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV


def mykde(X, *, grid_size: int = 120, bw: float = 4, bw_select: bool = False, plotnow: bool = False, train_percentage: float = 0.1):

    if bw_select:
        # Selecting the bandwidth via cross-validation
        bandwidths = 10 ** np.linspace(-1, 1, 100)
        grid = GridSearchCV(KernelDensity(kernel='gaussian'), {'bandwidth': bandwidths}, cv=len(X[:int(len(X)*train_percentage), :]))
        grid.fit(X[:int(len(X)*train_percentage), :])
        bw = grid.best_params_['bandwidth']

    # Kernel Density Estimation
    kde = KernelDensity(bandwidth=bw).fit(X)

    # Grid creation
    xx_d = np.linspace(0, grid_size, grid_size)
    yy_d = np.linspace(0, grid_size, grid_size)
    xx_dv, yy_dv = np.meshgrid(xx_d, yy_d)
    coor = np.array([xx_dv.flatten(), yy_dv.flatten()]).swapaxes(0, 1)

    # Evaluation of grid
    logprob = kde.score_samples(coor)      # Array of log(density) evaluations. Normalized to be probability densities.
    probdens = np.exp(logprob)

    # Plot
    if plotnow:
        im = probdens.reshape((int(probdens.shape[0]/grid_size), grid_size))
        plt.imshow(im)
        # plt.contourf(xx_dv, yy_dv, probdens.reshape((xx_d.shape[0], xx_d.shape[0])))
        plt.scatter(X[:, 0], X[:, 1], c='red')
        plt.show()

    return probdens


# Settings and Paths
frame_size = 120
overwrite = True
testit = False
# dataPath = '/#Common/chainingmic/dat.processed'
# resPath = '/#Common/chainingmic/res'
labelsPath = '/#Common/adrian/Workspace/dat/big_dataset_17102018_train.labels.mat'
newlabelPath = '/#Common/adrian/Workspace/dat/big_dataset_17102018_train.onlylabels.h5'
savePath = '/#Common/adrian/Workspace/temp/priors.h5'

# Load labels
logging.info(f"   loading labels from: {labelsPath}.")
f = h5py.File(labelsPath, 'r')
# contains: boxPath, config, createdOn, history, initialization, lastModified, positions, savePath, session, skeleton, skeletonPath
initialization = f['initialization']
positions = f['positions']
skeleton = f['skeleton']
boxPath = f['boxPath']

# Find boxes that are labeled
status = np.all(~np.isnan(positions), (1, 2))  # True if box has been fully labeled
logging.info(f"   found {np.sum(status)} labeled boxes.")

# Saving updated labels to h5 file
logging.info(f'   saving {np.sum(status)} labels to: {newlabelPath}.')
if os.path.exists(newlabelPath):
    if overwrite:
        logging.warning(f"   {newlabelPath} exists - deleting to overwrite.")
        os.remove(newlabelPath)
    else:
        raise OSError(f"   {newlabelPath} exists - cannot overwrite.")
with h5py.File(newlabelPath) as f3:
    f3.create_dataset('positions', data=positions[status, ...], compression='gzip')
    f3.create_dataset('initialization', data=initialization[status, ...], compression='gzip')
    # f3.create_dataset('skeleton', data=skeleton, compression='gzip')
    # f3.create_dataset('boxPath', data=boxPath, compression='gzip')
    # f3.create_dataset('originalLabels', data=labelsPath, compression='gzip')

# Calculate probability density map (priors)
logging.info(f'   initializing priors.')
priors = np.zeros((frame_size, frame_size, 12))

# Test or Generate priors
if testit:
    logging.info(f"   testing bandwidths for kde.")
    bandwidths = np.linspace(10, 100, 5)
    for count, bw in enumerate(bandwidths):
        print(count, bw)
        plt.figure(count, figsize=[16, 2])
        for bp in range(12):
            priors[:, :, bp] = mykde(positions[status, :, bp], bw=bw).reshape((frame_size, frame_size))
            plt.subplot(1, 12, bp+1)
            plt.imshow(priors[:, :, bp])
            plt.title(str(bp))
            plt.axis('off')

        plt.tight_layout()
    plt.show()
else:
    bw = 10
    logging.info(f'   generating priors with bandwidth = {bw}.')
    plt.figure(1, figsize=[16, 2])
    for bp in range(12):
        priors[:, :, bp] = mykde(positions[status, :, bp], bw=bw).reshape((frame_size, frame_size))
        plt.subplot(1, 12, bp+1)
        plt.imshow(priors[:, :, bp])
        plt.title(str(bp))
        plt.axis('off')
    plt.tight_layout()

# Save priors
logging.info(f'   saving priors to : {savePath}.')
if os.path.exists(savePath):
    if overwrite:
        logging.warning(f"   {savePath} exists - deleting to overwrite.")
        os.remove(savePath)
    else:
        raise OSError(f"   {savePath} exists - cannot overwrite.")
new_f = h5py.File(savePath)
new_f.create_dataset('priors', data=priors, compression='gzip')

plt.show()
