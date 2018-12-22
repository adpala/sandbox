import numpy as np
import logging
import os
import deepdish as dd
import matplotlib.pyplot as plt
from leap_utils.utils import iswin, ismac
from sklearn.datasets import make_biclusters
from sklearn.datasets import samples_generator as sg
from sklearn.cluster.bicluster import SpectralCoclustering
from sklearn.metrics import consensus_score


def transition_matrix(transitions, tau: int = 1, itself: bool = False):

    states = np.unique(transitions)
    M = np.zeros((states[-1] - states[0] + 1, states[-1] - states[0] + 1))

    for (i, j) in zip(transitions - states[0], transitions[tau:] - states[0]):
        if i != j:
            M[i][j] += 1
        elif itself:
            M[i][j] += 1

    s = np.sum(M, axis=1)
    M[s > 0, :] = np.divide(M[s > 0, :].T, s[s > 0]).T

    return M, states


def transitions_clustering(M, labels: np.array, option: str = 'row', n_clusters: int = 6, random_state: int = 0, n_jobs: int = 1):
    model = SpectralCoclustering(n_clusters=n_clusters, random_state=0, n_jobs=-1)
    model.fit(M)
    if option == 'row':
        fit_data = Trans_Mat[np.argsort(model.row_labels_)]
        fit_data = fit_data[:, np.argsort(model.row_labels_)]
        rowL = labels[np.argsort(model.row_labels_)]
        colL = labels[np.argsort(model.row_labels_)]
    elif option == 'col':
        fit_data = Trans_Mat[np.argsort(model.column_labels_)]
        fit_data = fit_data[:, np.argsort(model.column_labels_)]
        rowL = labels[np.argsort(model.column_labels_)]
        colL = labels[np.argsort(model.column_labels_)]
    elif option == 'both':
        fit_data = Trans_Mat[np.argsort(model.row_labels_)]
        fit_data = fit_data[:, np.argsort(model.column_labels_)]
        rowL = labels[np.argsort(model.row_labels_)]
        colL = labels[np.argsort(model.column_labels_)]
    else:
        print("available options are 'rol', 'col' or 'both'")

    return fit_data, rowL, colL


# General stuff
np.seterr(divide='ignore', invalid='ignore')
logging.basicConfig(level=logging.INFO)
if iswin():
    root = 'Z:/#Common/'
elif ismac():
    root = '/Volumes/ukme04/#Common/'
else:
    root = '/scratch/clemens10/'

# Paths
expID = 'localhost-20180720_182837'
data_path = root+'chainingmic/dat'
res_path = root+'chainingmic/res'
labels_path = f"{res_path}/{expID}/{expID}_all_labels.h5"

# Load data
labels_data = dd.io.load(labels_path)  # {'all_labels', 'all_nearest_neighbor_index', 'all_Z', 'expID', 'tsne_path'}
labels = labels_data['all_labels']


taus = [1, 10, 100, 1000, 5000]

plt.figure(figsize=[len(taus)*3, len(taus)*0.75])

ws = np.zeros((len(taus), 5))

for count, tau in enumerate(taus):
    Trans_Mat, real_labels = transition_matrix(labels, tau=tau, itself=False)

    to_delete = []
    for ii in range(Trans_Mat.shape[0]):
        if np.sum(Trans_Mat[ii, :]) == 0:
            to_delete.append(ii)
    Trans_Mat = np.delete(Trans_Mat, to_delete, 0)
    Trans_Mat = np.delete(Trans_Mat, to_delete, 1)

    fit, *_ = transitions_clustering(Trans_Mat, real_labels,  option='row', n_jobs=-1)

    w, v = np.linalg.eig(fit)

    ws[count, :] = abs(w[1:6])

    clipped_fit = fit
    clipped_fit[fit > 0.15] = 0.20
    clipped_fit[fit < 0.025] = 0

    plt.subplot(1, len(taus), count+1)
    plt.imshow(fit, cmap=plt.cm.BuPu_r)
    plt.title(r'$\tau$ = {}'.format(tau))
    plt.axis('off')

plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9, wspace=0.1)
cax = plt.axes([0.85, 0.2, 0.025, 0.6])
plt.colorbar(cax=cax)

plt.figure()
plt.plot(np.log10(taus), ws)
plt.legend([r'$\mu=$ 2', r'$\mu=$3', r'$\mu=$4', r'$\mu=$5', r'$\mu=$6'])


clusters, counts = np.unique(labels, return_counts=True)
sorted_clusters = clusters[np.argsort(-counts)]

mean_residency = np.zeros(clusters.shape)
total_mean_residency = np.zeros(1)
for zz in range(clusters.shape[0]):
    X = np.squeeze(np.asarray(np.where(labels == sorted_clusters[zz])))
    all_intervals_idx = np.split(X, np.where(np.diff(X) != 1)[0]+1)
    interval_idx = np.zeros((len(all_intervals_idx), 2), dtype=np.int)
    for count, y in enumerate(all_intervals_idx):
        interval_idx[count, 0] = np.amin(y).astype(int)
        interval_idx[count, 1] = np.amax(y).astype(int)
    mean_residency[zz] = np.mean(interval_idx[:, 1]-interval_idx[:, 0])*0.01
    total_mean_residency = np.concatenate((total_mean_residency, interval_idx[:, 1]-interval_idx[:, 0]), axis=0)

print(np.mean(total_mean_residency)*0.01)

plt.show()
