import numpy as np
import deepdish as dd
import sys
import logging
import os
sys.path.append('/#Common/adrian/Workspace/python_scripts/FIt-SNE')
from fast_tsne import fast_tsne
np.seterr(divide='ignore', invalid='ignore')
from leap_utils.utils import iswin, ismac

if iswin():
    root = 'Z:/#Common/'
elif ismac():
    root = '/Volumes/ukme04/#Common/'
else:
    root = '/scratch/clemens10/'

expID = 'localhost-20180720_182837'
res_path = root+'chainingmic/res'
cwt_path = f"{res_path}/{expID}/{expID}_cwt.h5"
tsne_path = f"{res_path}/{expID}/{expID}_tsne.h5"
# cwt_path = 'Z:/#Common/adrian/Workspace/temp/'+expID+'_cwt.h5'
# tsne_path = 'Z:/#Common/adrian/Workspace/temp/'+expID+'_tsne.h5'

# Spectograms via CWT and concatenation (ignoring reference body part)
logging.info(f"   loading cwt.")
cwt_data = dd.io.load(cwt_path)
cwt_array = cwt_data['cwt']

# PCA
logging.info(f"   PCA.")
X = cwt_array
X = X - X.mean(axis=0)
U, s, V = np.linalg.svd(X, full_matrices=False)
all_X50 = np.dot(U, np.diag(s))[:, :50]

# Importance Sampling or Random Sampling
prandom = 0.1                                                                                                   #
logging.info(f"   random sampling of {prandom*100}% of data.")
nrandom = int(prandom*cwt_array.shape[0])
random_data_idx = np.random.choice(all_X50.shape[0], nrandom, replace=False)
X50 = all_X50[random_data_idx, ...]

# PCA initialisation
PCAinit = X50[:, :2] / np.std(X50[:, 0]) * 0.0001

# ---   fast_tsne
logging.info(f"   doing tsne over {X50.shape[0]} samples from {cwt_array.shape[0]}.")
max_iter = 5000                                                                                                 #
perplexity = [15, 150]                                                                                          #
late_exag_coeff = 3                                                                                             #
start_late_exag_iter = int(4*max_iter/5)                                                                        #
initialization = PCAinit
Z = fast_tsne(X50, perplexity_list=perplexity, initialization=initialization, max_iter=max_iter, late_exag_coeff=late_exag_coeff, start_late_exag_iter=start_late_exag_iter, verbose=False)

logging.info(f"   saving cwt in: {tsne_path}")
if os.path.exists(tsne_path):
    tsne_path = tsne_path[:-7] + 'temptsne.h5'
    logging.info(f"   file already exist, new file will be created as: {tsne_path}.")

tsne_data = {'tsne': Z,
             'random_data_idx': random_data_idx,
             'all_X50': all_X50,
             'X50': X50,
             'initialization': initialization,
             'perplexity': perplexity,
             'late_exag_coeff': late_exag_coeff,
             'start_late_exag_iter': start_late_exag_iter,
             'max_iter': max_iter}
dd.io.save(tsne_path, tsne_data)
