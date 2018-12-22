from leap_utils.utils import iswin, ismac, flatten, unflatten
import numpy as np
import deepdish as dd
from scipy.stats import zscore
import pywt
import os
import logging
import defopt
import sys
sys.path.append('/#Common/adrian/Workspace/python_scripts/FIt-SNE')
from fast_tsne import fast_tsne
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
from videoreader import VideoReader
from leap_utils.plot import vplay, annotate
from sklearn.neighbors import KernelDensity
from scipy import ndimage as ndi
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from leap_utils.postprocessing import load_labels
from scipy import stats
from sklearn.metrics import confusion_matrix
from sklearn.cluster.bicluster import SpectralCoclustering
import cv2
import time
from leap_utils.preprocessing import export_boxes, angles, normalize_boxes
from leap_utils.postprocessing import process_confmaps_simple
from leap_utils.predict import predict_confmaps, load_network
from leap_utils.plot import vplay, confmaps, boxpos
import skimage.filters
import skimage.feature
from leap_utils.utils import it
from scipy.spatial import cKDTree


from all_pipelines_testing import testing_priors, max2d_multi, testing_poses
