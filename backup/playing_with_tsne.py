import numpy as np
import deepdish as dd
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import sys
import os
sys.path.append('/#Common/adrian/Workspace/python_scripts/FIt-SNE')
from fast_tsne import fast_tsne
np.seterr(divide='ignore', invalid='ignore')

expID = 'localhost-20180720_182837'
posesPath = 'Z:/#Common/adrian/Workspace/temp/'+expID+'_poses.h5'
tsnePath = 'Z:/#Common/adrian/Workspace/temp/'+expID+'_tsne.h5'
cwtPath = 'Z:/#Common/adrian/Workspace/temp/'+expID+'_cwt.h5'

protocolName = 'less_verylate3'
folderPath = 'Z:/#Common/adrian/Workspace/temp/' + protocolName
try:
    os.mkdir(folderPath)
except FileExistsError:
    print('folder already exists!')

# Spectograms via CWT and concatenation (ignoring reference body part)
print('loading cwt!')
cwt_data = dd.io.load(cwtPath)
cwt_array = cwt_data['cwt']

# Importance Sampling or Random Sampling
print('sampling!')
nrandom = int(0.1*cwt_array.shape[0])
random_sample = cwt_array[np.random.choice(cwt_array.shape[0], nrandom, replace=False), ...]

# Final tSNE
print('tsne!')
# ---   Do PCA and keep 50 dimensions
X = random_sample
X = X - X.mean(axis=0)
U, s, V = np.linalg.svd(X, full_matrices=False)
X50 = np.dot(U, np.diag(s))[:, :50]

print('tsne over {} random samples from {}.'.format(X50.shape[0], cwt_array.shape[0]))

# ---   PCA initialization later on
PCAinit = X50[:, :2] / np.std(X50[:, 0]) * 0.0001
iter_values = np.linspace(1500, 5000, 3, dtype=np.int)
for count, tsne_iter in enumerate(iter_values):
    print('tsne_iter {} from {}.'.format(count+1, iter_values.shape[0]))
    imgPath = folderPath+'/'+expID+'_tsnefig_' + str(tsne_iter) + '.png'

    # ---   fast_tsne
    max_iter = tsne_iter
    perplexity = [15, 150]
    late_exag_coeff = 2
    start_late_exag_iter = int(3*max_iter/5)
    initialization = PCAinit
    Z = fast_tsne(X50, perplexity_list=perplexity, initialization=initialization, max_iter=max_iter, late_exag_coeff=2, start_late_exag_iter=start_late_exag_iter, verbose=False)

    tsne_data = {'tsne': Z,
                 'X50': X50,
                 'initialization': initialization,
                 'perplexity': perplexity,
                 'late_exag_coeff': late_exag_coeff,
                 'start_late_exag_iter': start_late_exag_iter,
                 'max_iter': max_iter}
    dd.io.save(tsnePath, tsne_data)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.scatter(Z[:, 0], Z[:, 1], s=1)
    plt.subplot(122)
    kde = KernelDensity(bandwidth=1).fit(Z)
    xx_d = np.linspace(np.min(Z[:, 0]), np.max(Z[:, 0]), 100)
    yy_d = np.linspace(np.min(Z[:, 1]), np.max(Z[:, 1]), 100)
    xx_dv, yy_dv = np.meshgrid(xx_d, yy_d)
    coor = np.array([xx_dv.flatten(), yy_dv.flatten()]).swapaxes(0, 1)
    im = np.exp(kde.score_samples(coor)).reshape((100, 100))
    plt.imshow(np.flip(im, 0), aspect='auto')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(imgPath)
    plt.close()

# Projection of remaining timepoints

# KDE or Gaussian filter and Clip

# Watershed

# Sort labels

# Cluster pose
