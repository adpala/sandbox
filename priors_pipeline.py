import numpy as np
import h5py
import matplotlib.pyplot as plt
import logging
import defopt
import os
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from leap_utils.postprocessing import load_labels

plt.ioff()


def create_priors(labelsPath: str = '/#Common/adrian/Workspace/dat/big_dataset_17102018_train.labels.mat', bw: float = 15, frame_size: int = 120, savePath: str = '/#Common/adrian/Workspace/temp/priors.h5', overwrite: bool = True, testit: bool = False):
    """ Creates priors for each body part based on the labeled positions from the mat file specified.

    :param str labelsPath: path to the mat file
    :param int frame_size: size of frame, assuming squared frame
    :param str savePath: path to save priors array
    :param bool overwrite: allow to overwrite files
    :param bool testit: test for best bandwidth in the kde calculation (takes A LOT more time, not sure if it really works)
    """

    # Load labels
    positions, _, _ = load_labels(labelsPath)

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
                priors[:, :, bp] = mykde(positions[:, :, bp], bw=bw).reshape((frame_size, frame_size))
                plt.subplot(1, 12, bp+1)
                plt.imshow(priors[:, :, bp])
                plt.title(str(bp))
                plt.axis('off')

            plt.tight_layout()
        plt.show()
    else:
        logging.info(f'   generating priors with bandwidth = {bw}.')
        plt.figure(1, figsize=[16, 2])
        for bp in range(12):
            priors[:, :, bp] = mykde(positions[:, :, bp], bw=bw).reshape((frame_size, frame_size))
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
    new_f.close()
    plt.show()


def mykde(X, *, grid_size: int = 120, bw: float = 4, bw_select: bool = False, plotnow: bool = False, train_percentage: float = 0.1):
    """ Calculates the probability density given a set of positions through the Kernel Density Estimation approach.

    Arguments:
        X - positions [nsamples, 2]
        grid_size - default = 120
        bw - bandwidth, optimal value may change depending on amount of data
        bw_select - toggle option to search for best bandwidth
        plotnow - toggle option to plot a figure of the probability density map
        train_percentage - (if bw_select = True) percentage of the data set to be used for finding optimal bandwidth
    Returns:
        probdens - probability density map [frame_size, frame_size]
    """

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
        plt.colorbar()
        # plt.contourf(xx_dv, yy_dv, probdens.reshape((xx_d.shape[0], xx_d.shape[0])))
        plt.scatter(X[:, 0], X[:, 1], c='red')

    return probdens


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    defopt.run(create_priors)
