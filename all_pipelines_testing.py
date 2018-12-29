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
from leap_utils.preprocessing import angles
from videoreader import VideoReader
from leap_utils.plot import vplay, annotate, confmaps, boxpos
from skimage.feature import peak_local_max

from leap_utils.preprocessing import export_boxes, angles, normalize_boxes
from leap_utils.postprocessing import process_confmaps_simple
from leap_utils.predict import predict_confmaps, load_network

np.seterr(divide='ignore', invalid='ignore')
logging.basicConfig(level=logging.INFO)


def main(option: str = 'leap', expID: str = 'localhost-20180720_182837'):

    # figuring out paths
    if iswin():
        root = 'Z:/#Common/'
    elif ismac():
        root = '/Volumes/ukme04/#Common/'
    else:
        root = '/scratch/clemens10/'

    if option == 'deeplabcut':
        res_path = root+'adrian/Workspace/temp/deeplabcut'
        expID = 'localhost-20180720_182837'
        pose_path = f"{root}/chainingmic/deeplabcut/localhost-20180720_182837_boxesDeepCut_resnet50_flyposeNov26shuffle1_600000.h5"
    elif option == 'temp':
        res_path = root+'adrian/Workspace/temp/leap'
        pose_path = f"{root}/chainingmic/res/{expID}/{expID}_poses.h5"
    elif option == 'leap':
        res_path = root+'chainingmic/res'
        pose_path = f"{res_path}/{expID}/{expID}_poses.h5"
    elif option == 'training':
        res_path = root+'adrian/Workspace/temp/training'
    else:
        logging.info(f"   no option was given: deeplabcut, temp, leap or training")
        pass

    if not os.path.exists(f"{res_path}/{expID}"):
        os.mkdir(f"{res_path}/{expID}")

    cwt_path = f"{res_path}/{expID}/{expID}_cwt.h5"
    tsne_path = f"{res_path}/{expID}/{expID}_tsne.h5"
    ws_path = f"{res_path}/{expID}/{expID}_ws.h5"
    all_labels_path = f"{res_path}/{expID}/{expID}_all_labels.h5"
    confmat_path = f"{res_path}/{expID}/{expID}_confmat.h5"
    trackfixed_path = f"{root}/chainingmic/res/{expID}/{expID}_tracks_fixed.h5"
    video_path = f"{root}/chainingmic/dat/{expID}/{expID}.mp4"
    if not os.path.exists(video_path):
        video_path = f"{root}/chainingmic/dat.processed/{expID}/{expID}.mp4"
    freq_path = f"{res_path}/{expID}/cluster_freq"
    if not os.path.exists(freq_path):
        os.mkdir(freq_path)
    densities_path = f"{res_path}/{expID}/cluster_dens"
    if not os.path.exists(densities_path):
        os.mkdir(densities_path)

    # error_distance(f"{root}/chainingmic/deeplabcut/DeepCut_resnet50_flyposeNov26shuffle1_1030000-snapshot-1030000.h5")
    # cwt_data = cwt_pipeline(trackfixed_path, pose_path, cwt_path, option=option)
    # cwt_data = dd.io.load(cwt_path)
    # tsne_data = tsne_pipeline(cwt_data, tsne_path, prandom=0.5)
    # tsne_data = dd.io.load(tsne_path)
    # ws_data = watershed_pipeline(tsne_data, ws_path)
    # ws_data = dd.io.load(ws_path)
    # plot_watershed(tsne_data, ws_data)
    # all_labels_data = labeling_pipeline(tsne_data, ws_data, cwt_data, all_labels_path)
    # all_labels_data = dd.io.load(all_labels_path)
    # plot_labeling(ws_data, tsne_data, all_labels_data)
    # confmat_data = confusion_mat(confmat_path, tsne_data, all_labels_data)
    # transitions_pipeline(all_labels_data)

    # vr = VideoReader(video_path)
    # all_labels_data = dd.io.load(all_labels_path)
    # all_labels = all_labels_data['all_labels']
    # clusters, cluster_counts = np.unique(all_labels, return_counts=True)
    # sorted_clusters = clusters[np.argsort(-cluster_counts)]
    # get_multi_movie(vr, sorted_clusters[10], all_labels, pose_path, trackfixed_path, option=option)

    # if option == 'deeplabcut':
    #     positions = deepLabCut_positions(pose_path)
    # else:
    #     pose_data = dd.io.load(pose_path)
    #     positions = pose_data['positions']
    #
    # all_error_boxes_idxs, error_matrix, p_errors = testing_poses(positions, epsilon=[0])
    # error_boxes_idxs = all_error_boxes_idxs[0][1400:1800:5]     # error_boxes_idxs = all_error_boxes_idxs[0][1500:1750:10]
    #
    # vr = VideoReader(video_path)
    # nsamples = 10
    # boxes, cm, new_boxes_idx = reget_boxes(vr, error_boxes_idxs[:nsamples], trackfixed_path, pose_path, ifly=1,  option='leap')
    #
    # print(boxes.shape)
    # plt.figure()
    # plt.title('data boxes')
    # plt.imshow(boxes[5, :, :, 0], aspect='auto', cmap='gray')

    # print(new_boxes_idx.shape)
    # testing_priors(positions, error_matrix[..., 0], boxes, cm, new_boxes_idx)

    # priors_trainingset_test()

    # if option == 'deeplabcut':
    #     positions = deepLabCut_positions(pose_path)
    # else:
    #     pose_data = dd.io.load(pose_path)
    #     positions = pose_data['positions']
    # vr = VideoReader(video_path)
    # # priors_normal_test(vr, positions, trackfixed_path, pose_path, option=option)
    # looping_priors_normal(vr, positions, trackfixed_path, pose_path, option=option)

    view_looping_priors_results()

    plt.show()


def load_fixed_tracks(trackfixed_path, pose_path, option: str = 'temp'):

    track_data = dd.io.load(trackfixed_path)
    tracks = track_data['lines']
    centers = track_data['centers']
    chbb = track_data['chambers_bounding_box'][:]
    box_centers = centers[:, 0, :, :]   # nframe, fly id, coordinates
    box_centers = box_centers + chbb[1][0][:]
    nflies = np.zeros(1, dtype=np.int)
    nflies = box_centers.shape[1]

    heads = tracks[:, 0, :, 0, ::-1]   # nframe, fly id, coordinates
    tails = tracks[:, 0, :, 1, ::-1]   # nframe, fly id, coordinates
    heads = heads + chbb[1][0][:]   # nframe, fly id, coordinates
    tails = tails + chbb[1][0][:]   # nframe, fly id, coordinates

    dataperfly = np.zeros(nflies, dtype=np.int)
    if option == 'deeplabcut':
        positions = deepLabCut_positions(pose_path)
        fly_id = np.zeros(positions.shape[0])
        for ifly in range(nflies):
            fly_id[ifly::nflies] = ifly
        fixed_angles = 180 + angles(heads, tails)
    else:
        pose_data = dd.io.load(pose_path)
        positions = pose_data['positions']
        fly_id = pose_data['fly_id']
        fixed_angles = unflatten(pose_data['fixed_angles'], nflies)

    for ii in range(nflies):
        dataperfly[ii] = np.sum(fly_id == ii).astype(np.int)

    return nflies, box_centers, dataperfly, fixed_angles, fly_id, positions


def deepLabCut_positions(dlc_pose_path, print_example: bool = False):
    data = dd.io.load(dlc_pose_path)
    A = data['df_with_missing']
    B = np.asarray(A.iloc[:, :])
    B = np.delete(B, np.arange(2, B.shape[1], 3), 1)
    positions = B.reshape((B.shape[0], 12, 2)).astype(np.int)

    # Example to test
    if print_example:
        logging.info(f"   original")
        print(A['DeepCut_resnet50_flyposeNov26shuffle1_600000']['head'])
        logging.info(f"   shape: {B.shape}")
        print(B[:5, 0:2])
        logging.info(f"   reshaped")
        print(positions[:5, 0, :])

    return positions


def cwt_pipeline(trackfixed_path, pose_path, cwt_path, min_scale: float = 0, max_scale: float = np.log2(50), nscale: int = 25,  option: str = 'temp'):

    logging.info(f"   ---- cwt pipeline ----")

    # Setting scales
    scales = 162.5/(2*2**np.linspace(min_scale, max_scale, nscale))
    scale_len = scales.shape[0]

    # Get data
    logging.info(f"   getting data.")
    nflies, _, _, _, fly_id, positions = load_fixed_tracks(trackfixed_path, pose_path, option=option)
    logging.info(f"   nflies: {nflies}, positions shape: {positions.shape}.")

    # Select data
    logging.info(f"   selecting data.")
    ref_part_id = 8     # thorax = 8, neck = 1, head = 0
    ref_part = np.zeros((positions.shape[0], 1, positions.shape[2]))
    ref_part[:, 0, :] = positions[:, ref_part_id, :]
    zRel_positions = np.delete(zscore(positions - ref_part), ref_part_id, 1)

    # Do CWT
    logging.info(f"   doing cwt.")
    cwt_array = np.zeros((0, scale_len*11*2))
    cwt_fly_id = np.empty(1)
    for sfly in range(nflies):
        ifly_cwt_array = np.zeros((np.sum(fly_id == sfly), cwt_array.shape[1]))
        n = 0
        for sCoor in range(2):
            for bodypart_idx in range(11):
                # Input
                frame_step = 1
                timeserie = zRel_positions[fly_id == sfly, bodypart_idx, sCoor]

                # Period
                time = np.arange(timeserie.shape[0])/(100/frame_step)  # seconds
                dt = time[1]-time[0]

                # CWT
                [cfs, frequencies] = pywt.cwt(timeserie, scales, 'morl', dt)
                power = (abs(cfs)) ** 2
                power[np.isnan(power)] = 0

                ifly_cwt_array[:, scale_len*n:(n+1)*scale_len] = np.swapaxes(power, 0, 1)
                n += 1

        cwt_array = np.concatenate((cwt_array, ifly_cwt_array))
        cwt_fly_id = np.concatenate((cwt_fly_id, np.ones(cwt_array.shape[0])*sfly))
        logging.info(f"   fly {sfly}: {ifly_cwt_array.shape}.")

    logging.info(f"   cwt_array shape: {cwt_array.shape}.")

    # Save results
    logging.info(f"   saving cwt in: {cwt_path}")
    if os.path.exists(cwt_path):
        logging.info(f"   deleting older file.")
        os.remove(cwt_path)

    cwt_data = {'cwt': cwt_array,
                'dt': dt,
                'frequencies': frequencies,
                'cwt_fly_id': cwt_fly_id,
                'scales': scales}

    dd.io.save(cwt_path, cwt_data)

    return cwt_data


def tsne_pipeline(cwt_data, tsne_path, overwrite: bool = False, prandom: int = 0.1, max_iter=5000, start_late_exag_iter=4000, perplexity=[15, 150], late_exag_coeff=3):

    logging.info(f"   ---- tsne pipeline ----")
    cwt_array = cwt_data['cwt']

    logging.info(f"   PCA.")
    X = cwt_array
    X = X - X.mean(axis=0)
    U, s, V = np.linalg.svd(X, full_matrices=False)
    all_X50 = np.dot(U, np.diag(s))[:, :50]

    logging.info(f"   random sampling of {prandom*100}% of data.")
    nrandom = int(prandom*cwt_array.shape[0])
    random_data_idx = np.random.choice(all_X50.shape[0], nrandom, replace=False)
    X50 = all_X50[random_data_idx, ...]

    # PCA initialisation
    PCAinit = X50[:, :2] / np.std(X50[:, 0]) * 0.0001

    # ---   fast_tsne
    logging.info(f"   doing tsne over {X50.shape[0]} samples from {cwt_array.shape[0]}.")
    initialization = PCAinit
    Z = fast_tsne(X50, perplexity_list=perplexity, initialization=initialization, max_iter=max_iter, late_exag_coeff=late_exag_coeff, start_late_exag_iter=start_late_exag_iter, verbose=False)

    logging.info(f"   saving tsne in: {tsne_path}")
    if os.path.exists(tsne_path):
        if overwrite:
            logging.info(f"   deleting older file.")
            os.remove(tsne_path)
        else:
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
    return tsne_data


def watershed_pipeline(tsne_data, ws_path, bw: int = 2, grid_size: int = 100, min_distance: int = 2, connectivity: int = 8):

    logging.info(f"   ---- watershed pipeline ----")

    from sklearn.neighbors import KernelDensity
    from scipy import ndimage as ndi
    from skimage.morphology import watershed

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
               'grid_size': grid_size}
    dd.io.save(ws_path, ws_data)

    return ws_data


def plot_watershed(tsne_data, ws_data):

    Z = tsne_data['tsne']
    image = ws_data['image']
    labels = ws_data['labels']

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


def do_kdtree(combined_arrays, points, k: int = 1):
    from scipy.spatial import cKDTree

    mytree = cKDTree(combined_arrays)
    dist, indexes = mytree.query(points, k=k, n_jobs=-1)
    return indexes


def labeling_pipeline(tsne_data, ws_data, cwt_data, all_labels_path):

    logging.info(f"   ---- labeling pipeline ----")
    logging.info(f"   loading data for labeling.")
    rlabels = ws_data['labels'].ravel()
    Z = tsne_data['tsne']
    grid_size = ws_data['grid_size']
    all_X50 = tsne_data['all_X50']
    random_data_idx = tsne_data['random_data_idx']

    # grid
    xx_d = np.linspace(np.min(Z[:, 0]), np.max(Z[:, 0]), grid_size)
    yy_d = np.linspace(np.min(Z[:, 1]), np.max(Z[:, 1]), grid_size)
    xx_dv, yy_dv = np.meshgrid(xx_d, yy_d)
    combined_x_y_arrays = np.dstack([yy_dv.ravel(), xx_dv.ravel()])[0]

    logging.info(f"   finding nearest neighbors for sample data to grid.")
    nearest_neighbor_index = do_kdtree(combined_x_y_arrays, list(Z))
    Z_labels = rlabels[nearest_neighbor_index]

    logging.info(f"   finding nearest neighbors for all data to sample data.")
    all_nearest_neighbor_index = do_kdtree(all_X50[random_data_idx, ...], list(all_X50), k=2)
    all_labels = Z_labels[all_nearest_neighbor_index[:, 0]]

    logging.info(f"   estimating new positions in tsne space by median to k = 2 neighbors.")
    all_Z = np.median(Z[all_nearest_neighbor_index, :], axis=1).astype(np.int)

    logging.info(f"   saving labels.")
    all_labels_data = {'Z_labels': Z_labels,
                       'all_labels': all_labels,
                       'all_nearest_neighbor_index': all_nearest_neighbor_index,
                       'all_Z': all_Z}
    dd.io.save(all_labels_path, all_labels_data)

    return all_labels_data


def plot_labeling(ws_data, tsne_data, all_labels_data):

    image = ws_data['image']
    labels = ws_data['labels']
    Z = tsne_data['tsne']
    all_Z = all_labels_data['all_Z']
    Z_labels = all_labels_data['Z_labels']
    all_labels = all_labels_data['all_labels']

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


def subgroup(X, parts):

    subgrouped = np.empty([0])
    for count, ipart in enumerate(parts):
        subgrouped = np.concatenate((subgrouped, X[:, ipart]), axis=0)

    return subgrouped


def error_distance(pred_path, saveit: bool = False):

    logging.info(f"   ---- error distance ----")

    from leap_utils.postprocessing import load_labels

    # Load labels
    label_pos, _, boxes = load_labels()
    label_pos = np.flip(np.swapaxes(label_pos, 1, 2), 2)

    # Get predictions
    pos = deepLabCut_positions(pred_path)

    # Euclidian distance
    eu_dist = np.sqrt(np.power(pos[:, :, 0]-label_pos[:, :, 0], 2) + np.power(pos[:, :, 1]-label_pos[:, :, 1], 2))

    # Subgrouping data
    body = subgroup(eu_dist, [0, 1, 8, 11])
    wings = subgroup(eu_dist, [9, 10])
    legs = subgroup(eu_dist, [2, 3, 4, 5, 6, 7])

    # Plot
    plt.hist([eu_dist, body, wings, legs], bins=np.linspace(0, 10, 21), density=True, histtype='bar', label=['all', 'body', 'wings', 'legs'])
    plt.legend()

    # Save plot (where the pred_path came from)
    if saveit:
        histFig = pred_path[:-2] + 'png'
        plt.savefig(histFig)


def confusion_mat(confmat_path, tsne_data, all_labels_data, k=1, p_test=0.5, saveit: bool = False):

    logging.info(f"   ---- confusion matrix ----")

    from scipy import stats
    from sklearn.metrics import confusion_matrix

    # load all required data
    all_X50 = tsne_data['all_X50']
    random_data_idx = tsne_data['random_data_idx']
    Z_labels = all_labels_data['Z_labels']

    test_meta_idx = np.random.choice(random_data_idx.shape[0], int(p_test*random_data_idx.shape[0]), replace=False)
    train_meta_idx = np.arange(random_data_idx.shape[0])[~test_meta_idx]
    test_idx = random_data_idx[test_meta_idx]
    train_idx = random_data_idx[train_meta_idx]
    ground_truth = Z_labels[test_meta_idx]

    conf = np.zeros(test_idx.shape[0], dtype=bool)

    # set nearest neighbor algorithm with train and embed test data
    test_neighbors = do_kdtree(all_X50[train_idx, ...], list(all_X50[test_idx, ...]), k=k)
    if k == 1:
        test_results = Z_labels[train_meta_idx[test_neighbors[:]]]  # Use closest neighbor
    elif k < 3:
        test_results = Z_labels[train_meta_idx[test_neighbors[:, 0]]]  # Use closest neighbor
    else:
        test_results, test_counts = stats.mode(Z_labels[train_meta_idx[test_neighbors]], axis=1)    # Use mode neighbor

    # compare true versus found labels in a confusion matrix
    conf_mat = confusion_matrix(ground_truth, test_results)

    for ii in range(test_results.shape[0]):
        conf[ii] = test_results[ii] == ground_truth[ii]
    confmat_data = {'conf': conf, 'k': k, 'p_test': p_test}

    logging.info(f"   k: {k}, p = {np.sum(conf)/conf.shape[0]}")
    plt.figure(k)
    plot_confusion_matrix(conf_mat, normalize=True, title='Normalized with k={}'.format(k))
    if saveit:
        plot_path = confmat_path[:-3] + '_k' + str(k) + '.png'
        plt.savefig(plot_path)
        dd.io.save(confmat_path, confmat_data)

    return confmat_data


def plot_confusion_matrix(cm, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def transition_matrix(transitions, tau: int = 1, itself: bool = False):

    states = np.unique(transitions)
    M = np.zeros((states[-1] - states[0] + 1, states[-1] - states[0] + 1))

    for (i, j) in zip(transitions - states[0], transitions[tau:] - states[0]):
        if i != j:
            M[i][j] += 1
        elif itself:
            M[i][j] += 1

    s = np.sum(M, axis=1)
    # M[s > 0, :] = np.divide(M[s > 0, :].T, s[s > 0]).T
    M[s.nonzero(), :] = np.divide(M[s.nonzero(), :].T, s[s.nonzero()]).T

    return M, states


def transitions_clustering(M, labels: np.array, option: str = 'row', n_clusters: int = 6, random_state: int = 0, n_jobs: int = 1):

    from sklearn.cluster.bicluster import SpectralCoclustering

    model = SpectralCoclustering(n_clusters=n_clusters, random_state=0, n_jobs=-1)
    model.fit(M)
    if option == 'row':
        fit_data = M[np.argsort(model.row_labels_)]
        fit_data = fit_data[:, np.argsort(model.row_labels_)]
        rowL = labels[np.argsort(model.row_labels_)]
        colL = labels[np.argsort(model.row_labels_)]
    elif option == 'col':
        fit_data = M[np.argsort(model.column_labels_)]
        fit_data = fit_data[:, np.argsort(model.column_labels_)]
        rowL = labels[np.argsort(model.column_labels_)]
        colL = labels[np.argsort(model.column_labels_)]
    elif option == 'both':
        fit_data = M[np.argsort(model.row_labels_)]
        fit_data = fit_data[:, np.argsort(model.column_labels_)]
        rowL = labels[np.argsort(model.row_labels_)]
        colL = labels[np.argsort(model.column_labels_)]
    else:
        logging.info(f"   option not found. Available options for transitions_clustering() are 'rol', 'col' or 'both'")
        pass

    return fit_data, rowL, colL


def transitions_pipeline(all_labels_data, taus=[1, 10, 100, 1000, 5000], clip_min=0.025, clip_max=0.15):

    logging.info(f"   ---- transitions pipeline ----")

    all_labels = all_labels_data['all_labels']

    plt.figure(figsize=[len(taus)*3, len(taus)*0.75])

    ws = np.zeros((len(taus), 5))

    for count, tau in enumerate(taus):
        Trans_Mat, real_labels = transition_matrix(all_labels, tau=tau, itself=False)

        to_delete = []
        for ii in range(Trans_Mat.shape[0]):
            if np.sum(Trans_Mat[ii, :]) == 0:
                to_delete.append(ii)
        Trans_Mat = np.delete(Trans_Mat, to_delete, 0)
        Trans_Mat = np.delete(Trans_Mat, to_delete, 1)

        fit, *_ = transitions_clustering(Trans_Mat, real_labels,  option='row', n_jobs=-1)

        w, v = np.linalg.eig(fit)

        ws[count, :] = abs(w[1:6])

        fit[fit > clip_max] = clip_max
        fit[fit < clip_min] = 0

        plt.subplot(1, len(taus), count+1)
        plt.imshow(fit, cmap=plt.cm.BuPu_r)
        plt.title(r'$\tau$ = {}'.format(tau))
        plt.axis('off')

    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9, wspace=0.1)
    cax = plt.axes([0.85, 0.2, 0.025, 0.6])
    plt.colorbar(cax=cax)

    plt.figure()
    plt.plot(taus, ws)
    plt.legend([r'$\mu=$ 2', r'$\mu=$3', r'$\mu=$4', r'$\mu=$5', r'$\mu=$6'])
    plt.gca().set_xscale('log')

    clusters, counts = np.unique(all_labels, return_counts=True)
    sorted_clusters = clusters[np.argsort(-counts)]

    mean_residency = np.zeros(clusters.shape)
    total_mean_residency = np.zeros(1)
    for zz in range(clusters.shape[0]):
        X = np.squeeze(np.asarray(np.where(all_labels == sorted_clusters[zz])))
        all_intervals_idx = np.split(X, np.where(np.diff(X) != 1)[0]+1)
        interval_idx = np.zeros((len(all_intervals_idx), 2), dtype=np.int)
        for count, y in enumerate(all_intervals_idx):
            interval_idx[count, 0] = np.amin(y).astype(int)
            interval_idx[count, 1] = np.amax(y).astype(int)
        mean_residency[zz] = np.mean(interval_idx[:, 1]-interval_idx[:, 0])*0.01
        total_mean_residency = np.concatenate((total_mean_residency, interval_idx[:, 1]-interval_idx[:, 0]), axis=0)

    logging.info(f"   total mean residency: {np.mean(total_mean_residency)*0.01}")


def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) > stepsize)[0] + 1)


def get_multi_movie(vr, icluster: int, labels, pose_path, trackfixed_path, option: str = 'temp'):

    logging.info(f"   ---- plotting movies for cluster {icluster} ----")

    nflies, box_centers, dataperfly, fixed_angles, fly_id, positions = load_fixed_tracks(trackfixed_path, pose_path, option)

    # Indexes intervals
    all_intervals_idx = consecutive(np.squeeze(np.asarray(np.where(labels == icluster))))
    interval_idx = np.zeros((len(all_intervals_idx), 2), dtype=np.int)
    for count, x in enumerate(all_intervals_idx):
        interval_idx[count, 0] = np.amin(x).astype(int)
        interval_idx[count, 1] = np.amax(x).astype(int)

    if interval_idx.shape[0] < 9:
        n_videos = interval_idx.shape[0]
    else:
        n_videos = 9

    idxs = np.argsort(interval_idx[:, 0]-interval_idx[:, 1])[:n_videos]

    videos_list = list()
    all_frames_list = list()
    for count, idx in enumerate(idxs):
        # get fly id
        previous = 0
        ifly = 0
        while ~np.isin(interval_idx[idx, 0], np.arange(previous, previous + dataperfly[ifly])):
            previous = dataperfly[ifly]
            ifly += 1

        # Frames
        frames = np.arange(interval_idx[idx, 0], interval_idx[idx, 1]+1) - np.sum(dataperfly[:ifly])

        frames_list = list(vr[frames.tolist()])

        boxes, *_ = export_boxes(frames_list, box_centers[frames, ...], box_size=np.array([120, 120]), box_angles=fixed_angles[frames, ...])

        videos_list.append(boxes[ifly::nflies, ...])

        boxes_idx = frames*nflies + ifly
        if boxes_idx[-1] == len(labels):
            boxes_idx[-1] -= 1

        all_frames_list.append(boxes_idx)

    multi_vplay(videos_list, all_frames_list, positions)


def multi_vplay(frames_list, all_frames_list, positions):

    import cv2
    import time

    if frames_list[0].shape[3] == 1:
        for frames in frames_list:
            frames = np.repeat(frames, 3, axis=2)

    for nwindows in range(len(frames_list)):
        cv2.namedWindow(str(nwindows), cv2.WINDOW_AUTOSIZE)
        x_shift = 200+(nwindows % 3)*200
        y_shift = 200+int(nwindows/3)*200
        cv2.moveWindow(str(nwindows), x_shift, y_shift)

    fcount = np.zeros(len(frames_list), dtype=np.int)
    while True:
        for zz in range(len(frames_list)):
            frame = frames_list[zz][fcount[zz], ...]
            if positions is not None:
                frame = annotate(frame, positions[all_frames_list[zz][fcount[zz]], ...])
            cv2.imshow(str(zz), frame)

            fcount[zz] += 1
            if fcount[zz] >= len(frames_list[zz]):
                fcount[zz] = 0
        time.sleep(0.01)    # to reproduce at 100 frames per second
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


def testing_poses(positions, epsilon=[0, 2, 5], plotit: bool = False, detailed: bool = False):

    if plotit:
        cmap = get_cmap('gist_ncar')
        norm = Normalize(vmin=0, vmax=12)
        colours = cmap(norm(np.arange(0, 12)))

        plt.figure(figsize=[5, 8])
        for ii in range(12):
            for jj in range(2):
                plt.subplot(12, 2, ii*2-jj+2)
                plt.hist(positions[:, ii, jj], color=colours[ii, ...])
                # plt.gca().set_yscale('log')
                plt.xlim(0, 120)
                plt.tick_params(axis='both', which='both', top=False, left=False, right=False, labelleft=False)
        plt.tight_layout()

        bps = ['head', 'neck', 'frontL', 'middleL', 'backL', 'frontR', 'middleR', 'backR', 'thorax', 'wingL', 'wingR', 'tail']
        plt.figure(figsize=[6, 8])
        for ii, b in enumerate(bps):
            plt.subplot(4, 3, ii+1)
            plt.title(b)
            plt.scatter(positions[:, ii, 1], positions[:, ii, 0], s=0.1, c=colours[ii:ii+1, ...])
            plt.tick_params(axis='y', which='both', bottom=False, top=False, left=False, right=False, labelleft=False, labelbottom=False)
            plt.ylim(120, 0)
            plt.xlim(0, 120)
        plt.tight_layout()

    bp_thresholds = np.zeros((12, 2, 2))    # bodypart, x/y, min/max

    # Original values before 29.12.2018:
    # bp_thresholds[:, 1, 0] = [50, 40, 20, 10, 20, 50, 60, 50, 50, 10, 50, 50]
    # bp_thresholds[:, 1, 1] = [80, 70, 70, 60, 70, 100, 110, 100, 80, 70, 110, 80]
    # bp_thresholds[:, 0, 0] = [10, 25, 5, 25, 50, 5, 25, 50, 40, 60, 60, 60]
    # bp_thresholds[:, 0, 1] = [60, 75, 60, 100, 100, 60, 100, 100, 90, 120, 120, 100]

    # x min
    bp_thresholds[:, 1, 0] = [50, 40, 20, 10, 20, 50, 60, 50, 50, 10, 50, 50]
    # x max
    bp_thresholds[:, 1, 1] = [80, 70, 70, 60, 70, 100, 110, 100, 80, 70, 110, 80]

    # y min
    bp_thresholds[:, 0, 0] = [10, 25, 5, 25, 50, 5, 25, 50, 40, 60, 60, 60]
    # y max
    bp_thresholds[:, 0, 1] = [60, 75, 60, 100, 100, 60, 100, 100, 90, 120, 120, 100]

    p_errors = np.zeros((12, 2, len(epsilon)))
    error_matrix = np.zeros((positions.shape[0], positions.shape[1], len(epsilon)), dtype=bool)
    nelements = positions.shape[0]
    all_error_boxes_idxs = list()
    for ii, eps in enumerate(epsilon):
        error_count = np.zeros((12, 2), dtype=int)
        error_boxes_idxs = np.empty((0, 1), dtype=int)
        for bp in range(12):
            for jj in range(2):
                cond1 = positions[:, bp, jj] > (bp_thresholds[bp, jj, 1] - eps)
                cond2 = positions[:, bp, jj] < (bp_thresholds[bp, jj, 0] + eps)
                error_matrix[:, bp, ii] = np.any([error_matrix[:, bp, ii], cond1, cond2], axis=0)
                error_count[bp, jj] += np.sum(cond1)
                error_count[bp, jj] += np.sum(cond2)
                ierror_boxes_idxs = np.asarray(cond1).nonzero()
                error_boxes_idxs = np.append(error_boxes_idxs, ierror_boxes_idxs)
                ierror_boxes_idxs = np.asarray(cond2).nonzero()
                error_boxes_idxs = np.append(error_boxes_idxs, ierror_boxes_idxs)
        error_boxes_idxs = np.unique(error_boxes_idxs)
        all_error_boxes_idxs.append(error_boxes_idxs)
        p_errors[..., ii] = 100*error_count/nelements
        if detailed:
            logging.info(f"    --- epsilon {eps} --- ")
            np.set_printoptions(precision=2, suppress=True)
            print("   y ----- x  ")
            print(p_errors[..., ii])
            logging.info(f"   number of errors: {error_boxes_idxs.shape[0]}")
            logging.info(f"   number of errors according to error_matrix: {np.sum(np.any(error_matrix, axis=1))}")
            # print('proportion: ', "{0:.2f}".format(100*error_boxes_idxs.shape[0]/nelements))
            logging.info(f"   proportion: {100*error_boxes_idxs.shape[0]/nelements}")

    return all_error_boxes_idxs, error_matrix, p_errors


def Jan_Script(vr, all_boxes_idx, ifly, trackfixed_path, pose_path, nshow: int = 20, option: str = 'temp'):

    nflies, box_centers, dataperfly, fixed_angles, fly_id, positions = load_fixed_tracks(trackfixed_path, pose_path, option='leap')

    boxes_idx = all_boxes_idx[all_boxes_idx % nflies == ifly]
    if nshow > boxes_idx.shape[0]:
        nshow = boxes_idx.shape[0]
    frames = np.floor_divide(boxes_idx[:nshow], nflies)

    if frames.size == 0:
        logging.info(f"   no frames match requirement (try another fly id?)")
    else:
        frames_list = list(vr[frames.tolist()])

        boxes, *_ = export_boxes(frames_list, box_centers[frames, ...], box_size=np.array([120, 120]), box_angles=fixed_angles[frames, ...])

        net_boxes = normalize_boxes(boxes[ifly::nflies, ...])
        network = load_network('Z:/#Common/chainingmic/leap/best_model.h5', image_size=[120, 120])
        cm = predict_confmaps(network, net_boxes[:, :, :, :1])
        net_positions, confidence = process_confmaps_simple(cm)

        for ii in np.linspace(0, net_boxes.shape[0]-1, num=2).astype(np.int):
            plt.figure(figsize=[9, 3])
            bp_to_plot = np.array([2, 3, 4])
            nplots = bp_to_plot.shape[0]+2
            plt.subplot(1, nplots, 1)
            boxpos(net_boxes[ii, ...], net_positions[ii, ...])
            plt.axis('off')
            for jj in range(1, nplots):
                plt.subplot(1, nplots, jj+1)
                plt.imshow(cm[ii, ..., jj])
                plt.axis('off')
            plt.subplot(1, nplots, nplots)
            confmaps(cm[ii, ...])
            plt.axis('off')
            plt.tight_layout()

        vplay(frames=boxes[ifly::nflies, ...], positions=positions[boxes_idx[:nshow], ...], moviemode=True)


def reget_boxes(vr, boxes_idx, trackfixed_path, pose_path, option: str = 'leap', ifly: int = 0):

    nflies, box_centers, dataperfly, fixed_angles, _, _ = load_fixed_tracks(trackfixed_path, pose_path, option='leap')

    frames = np.floor_divide(boxes_idx, nflies)
    specific_fly_id = boxes_idx % nflies

    if frames.size == 0:
        logging.info(f"   no frames match requirement (try another fly id?)")
    else:
        frames_list = list(vr[frames.tolist()])
        boxes, *_ = export_boxes(frames_list, box_centers[frames, ...], box_size=np.array([120, 120]), box_angles=fixed_angles[frames, ...])

        boxes = boxes[ifly::nflies, ...]
        boxes = boxes[specific_fly_id == ifly]
        new_boxes_idx = boxes_idx[specific_fly_id == ifly]

        if option == 'leap':
            net_boxes = normalize_boxes(boxes)
            network = load_network('Z:/#Common/chainingmic/leap/best_model.h5', image_size=[120, 120])
            cm = predict_confmaps(network, net_boxes[:, :, :, :1])
            net_positions, confidence = process_confmaps_simple(cm)
        else:
            logging.info(f"   not yet implemented to get conf maps from options other than leap")
            cm = None

    return boxes, cm, new_boxes_idx


def max2d_multi(mask: np.ndarray, num_peaks: int, axis: int = 0, smooth: float = None,
                exclude_border: bool = True, min_distance: int = 10) -> (np.ndarray, np.ndarray):
    """Detect one or multiple peaks in each channel of an image."""

    import skimage.filters
    from leap_utils.utils import it

    maxima = np.ndarray((num_peaks, 2, mask.shape[axis]), dtype=int)
    for idx, plane in enumerate(it(mask, axis=axis)):
        if smooth:
            plane = skimage.filters.gaussian(plane, smooth)
        tmp = peak_local_max(plane, num_peaks=num_peaks, exclude_border=exclude_border, min_distance=min_distance)
        if tmp.shape[0] == maxima.shape[0]:
            maxima[..., idx] = tmp
        else:
            maxima[:tmp.shape[0], :, idx] = tmp
            maxima[tmp.shape[0]:, :, idx] = 0
    return maxima


def testing_priors(positions, error_matrix, boxes, cm, boxes_idxs, priors_path: str = '/#Common/adrian/Workspace/temp/priors.h5', plotit_bybp: bool = False, plotit_summary: bool = False):

    from leap_utils.postprocessing import max_simple

    priors_data = dd.io.load(priors_path)
    priors = priors_data['priors']

    new_positions = np.copy(positions).astype(np.int)

    ref_thorax, _ = max_simple(priors[:, :, 8:9])

    # logging.info(f"   boxes_idxs shape: {boxes_idxs.shape}, boxes shape {boxes.shape}")

    cmap = get_cmap('OrRd')
    bp_names = ['head', 'neck', 'frontL', 'middleL', 'backL', 'frontR', 'middleR', 'backR', 'thorax', 'wingL', 'wingR', 'tail']

    for ii, box_idx in enumerate(boxes_idxs):

        row_shift = int(ref_thorax[0, 1]-positions[box_idx, 8, 1])
        col_shift = int(ref_thorax[0, 0]-positions[box_idx, 8, 0])

        bp_to_prior = np.where(error_matrix[box_idx, :])[0]

        # Get peaks of confidence map
        maxima = max2d_multi(cm[box_idx, :, :, bp_to_prior], num_peaks=5)
        maxima_val = cm[box_idx, maxima[:, 0, :], maxima[:, 1, :], bp_to_prior]

        for jj, bp in enumerate(bp_to_prior):
            # logging.info(f"   {ii+1} of {boxes_idxs.shape[0]}: bp {bp_to_prior}, {bp}")
            col_in_prior = maxima[:, 0, jj] - col_shift
            col_in_prior[col_in_prior > 119] = 119
            col_in_prior[col_in_prior < 0] = 0
            row_in_prior = maxima[:, 1, jj] - row_shift
            row_in_prior[row_in_prior > 119] = 119
            row_in_prior[row_in_prior < 0] = 0
            bay_score = np.multiply(maxima_val[:, jj], priors[col_in_prior, row_in_prior, bp])
            new_positions[box_idx, bp, :] = maxima[np.argmax(bay_score), :, jj]
            # logging.info(f"   old: {positions[box_idx, bp, :]}, new: {new_positions[box_idx, bp, :]}, from: {maxima[np.argmax(bay_score), :, jj]}")

            if plotit_bybp:
                norm = Normalize(vmin=0, vmax=1.2*np.amax(bay_score))
                plt.figure(figsize=[8, 2])
                plt.subplot(141)
                plt.title(box_idx)
                plt.imshow(boxes[box_idx, :, :, 0], aspect='auto', cmap='gray')
                plt.scatter(maxima[:, 1, jj], maxima[:, 0, jj], facecolors='none', edgecolors=cmap(norm(bay_score)))
                plt.scatter(positions[box_idx, 8, 1], positions[box_idx, 8, 0], marker='D', c='yellow')
                plt.scatter(positions[box_idx, bp, 1], positions[box_idx, bp, 0], s=80, marker='s', facecolors='none', edgecolors='yellow')
                plt.scatter(maxima[np.argmax(bay_score), 1, jj], maxima[np.argmax(bay_score), 0, jj], s=80, marker='s', facecolors='none', edgecolors='red')
                plt.subplot(142)
                plt.title(bp_names[bp])
                plt.imshow(cm[box_idx, :, :, bp], aspect='auto')
                plt.scatter(maxima[:, 1, jj], maxima[:, 0, jj], facecolors='none', edgecolors=cmap(norm(bay_score)))
                plt.scatter(positions[box_idx, bp, 1], positions[box_idx, bp, 0], s=80, marker='s', facecolors='none', edgecolors='yellow')
                plt.scatter(maxima[np.argmax(bay_score), 1, jj], maxima[np.argmax(bay_score), 0, jj], s=80, marker='s', facecolors='none', edgecolors='red')
                plt.subplot(143)
                plt.title('score: {:.6f}'.format(np.amax(bay_score)))
                plt.imshow(priors[:, :, bp], aspect='auto')
                plt.scatter(maxima[:, 1, jj], maxima[:, 0, jj], facecolors='none', edgecolors=cmap(norm(bay_score)))
                plt.scatter(positions[box_idx, bp, 1], positions[box_idx, bp, 0], s=80, marker='s', facecolors='none', edgecolors='yellow')
                plt.scatter(maxima[np.argmax(bay_score), 1, jj], maxima[np.argmax(bay_score), 0, jj], s=80, marker='s', facecolors='none', edgecolors='red')
                plt.subplot(144)
                plt.title("cm x prior")
                plt.imshow(np.multiply(priors[:, :, bp], cm[box_idx, :, :, bp]), aspect='auto')
                plt.scatter(maxima[:, 1, jj], maxima[:, 0, jj], facecolors='none', edgecolors=cmap(norm(bay_score)))
                plt.scatter(positions[box_idx, bp, 1], positions[box_idx, bp, 0], s=80, marker='s', facecolors='none', edgecolors='yellow')
                plt.scatter(maxima[np.argmax(bay_score), 1, jj], maxima[np.argmax(bay_score), 0, jj], s=80, marker='s', facecolors='none', edgecolors='red')
                plt.show()

        if plotit_summary:
            plt.figure()
            plt.title(box_idx)
            plt.imshow(boxes[box_idx, :, :, 0], aspect='auto', cmap='gray')
            plt.scatter(positions[box_idx, :, 1], positions[box_idx, :, 0], s=30, marker='s', c='red')
            plt.scatter(new_positions[box_idx, :, 1], new_positions[box_idx, :, 0], s=10, c='blue')
            plt.show()

    return new_positions


def training_dataset_priors(plotit: bool = True, network_path: str = 'Z:/#Common/chainingmic/leap/best_model.h5', temp_save_path: str = 'Z:/#Common/adrian/Workspace/temp/leap/temp_save.h5'):
    """ Runs the pipeline to test the prior application to the predictions on the training dataset, using the model specified by network_path.
        A temporary save file of the confidence maps and boxes will be saved after the first time in the temp_save_path to speed up repeated calls.
        Make sure to change temp_save_path or it will use my old temporary save from 24.12.2018 with leap network."""

    # Get positions and confidence maps
    logging.info(f"   getting positions and confidence maps")
    if not os.path.exists(temp_save_path):
        # Load positions and boxes from training data set
        from leap_utils.postprocessing import load_labels
        label_pos, _, boxes = load_labels()
        label_pos = np.flip(np.swapaxes(label_pos, 1, 2), 2)

        # Prepare boxes to be processed
        net_boxes = normalize_boxes(boxes)

        # Find confidence maps and positions using specified model
        network = load_network(network_path, image_size=[120, 120])
        cm = predict_confmaps(network, net_boxes[:, :, :, :1])
        net_positions, confidence = process_confmaps_simple(cm)

        # Temporary save of results
        temp_save_data = {'cm': cm,
                          'label_pos': label_pos,
                          'net_positions': net_positions,
                          'net_boxes': net_boxes}
        dd.io.save(temp_save_path, temp_save_data)
    else:
        logging.info(f"   loading temp save")
        temp_save_data = dd.io.load(temp_save_path)
        net_boxes = temp_save_data['net_boxes']
        cm = temp_save_data['cm']
        net_positions = temp_save_data['net_positions']
        label_pos = temp_save_data['label_pos']

    # Find errors (error matrix)
    logging.info(f"   calculating first pass errors")
    all_error_boxes_idxs, error_matrix, p_errors = testing_poses(net_positions, epsilon=[0], plotit=False, detailed=False)
    eu_dist1 = np.sqrt(np.power(net_positions[:, :, 0]-label_pos[:, :, 0], 2) + np.power(net_positions[:, :, 1]-label_pos[:, :, 1], 2))
    logging.info("   mean error distance: {:02.4f} +- {:02.4f}.".format(np.mean(eu_dist1), np.std(eu_dist1)))

    # Apply priors to errors
    logging.info(f"   applying priors")
    new_positions = testing_priors(net_positions, error_matrix[..., 0], net_boxes, cm, all_error_boxes_idxs[0])

    # Try to find errors
    logging.info(f"   calculating second pass errors")
    all_error_boxes_idxs2, error_matrix2, p_errors2 = testing_poses(new_positions, epsilon=[0], plotit=False, detailed=False)
    eu_dist2 = np.sqrt(np.power(new_positions[:, :, 0]-label_pos[:, :, 0], 2) + np.power(new_positions[:, :, 1]-label_pos[:, :, 1], 2))
    logging.info("   mean error distance: {:02.4f} +- {:02.4f}.".format(np.mean(eu_dist2), np.std(eu_dist2)))

    # Calculate the decrease of error distance
    eu_dist_change = eu_dist1 - eu_dist2
    eu_dist_change = eu_dist_change[eu_dist_change.nonzero()]

    # Compare error distance pre- and post- priors
    if plotit:
        plt.figure(figsize=[6, 8])

        plt.subplot(311)
        plt.title('Error distance, before prior')
        body1 = subgroup(eu_dist1, [0, 1, 8, 11])
        wings1 = subgroup(eu_dist1, [9, 10])
        legs1 = subgroup(eu_dist1, [2, 3, 4, 5, 6, 7])
        plt.hist([eu_dist1, body1, wings1, legs1], bins=np.linspace(0, 6, 10), density=True, histtype='bar', label=['all', 'body', 'wings', 'legs'])
        plt.legend()

        plt.subplot(312)
        plt.title('Error distance, after prior')
        body2 = subgroup(eu_dist2, [0, 1, 8, 11])
        wings2 = subgroup(eu_dist2, [9, 10])
        legs2 = subgroup(eu_dist2, [2, 3, 4, 5, 6, 7])
        plt.hist([eu_dist2, body2, wings2, legs2], bins=np.linspace(0, 6, 10), density=True, histtype='bar', label=['all', 'body', 'wings', 'legs'])
        plt.legend()

        plt.subplot(313)
        plt.title('Decrease of error distance (preprior - postprior)')
        plt.hist(eu_dist_change, bins=np.linspace(-60, 60, 10), histtype='bar', rwidth=0.8)
        plt.subplots_adjust(bottom=0.05, top=0.95, wspace=0.4)

    # Report result
    logging.info(f"   {100*(all_error_boxes_idxs[0].shape[0]-all_error_boxes_idxs2[0].shape[0])/all_error_boxes_idxs[0].shape[0]}% of errors fixed by priors, {all_error_boxes_idxs2[0].shape[0]} unfixed.")


def priors_trainingset_test(plotit: bool = True, network_path: str = 'Z:/#Common/chainingmic/leap/best_model.h5', temp_save_path: str = 'Z:/#Common/adrian/Workspace/temp/leap/train_temp_save.h5'):
    """ Runs the pipeline to test the prior application to the predictions on the training dataset, using the model specified by network_path.
        A temporary save file of the confidence maps and boxes will be saved after the first time in the temp_save_path to speed up repeated calls.
        Make sure to change temp_save_path or it will use my old temporary save from 24.12.2018 with leap network."""

    # Get positions and confidence maps
    logging.info(f"   getting positions and confidence maps")
    if not os.path.exists(temp_save_path):
        # Load positions and boxes from training data set
        from leap_utils.postprocessing import load_labels
        label_pos, _, boxes = load_labels()
        label_pos = np.flip(np.swapaxes(label_pos, 1, 2), 2)

        # Prepare boxes to be processed
        net_boxes = normalize_boxes(boxes)

        # Find confidence maps and positions using specified model
        network = load_network(network_path, image_size=[120, 120])
        cm = predict_confmaps(network, net_boxes[:, :, :, :1])
        net_positions, confidence = process_confmaps_simple(cm)

        # Temporary save of results
        temp_save_data = {'cm': cm,
                          'label_pos': label_pos,
                          'net_positions': net_positions,
                          'net_boxes': net_boxes}
        dd.io.save(temp_save_path, temp_save_data)
    else:
        logging.info(f"   loading temp save")
        temp_save_data = dd.io.load(temp_save_path)
        net_boxes = temp_save_data['net_boxes']
        cm = temp_save_data['cm']
        net_positions = temp_save_data['net_positions']
        label_pos = temp_save_data['label_pos']

    # Find errors (error matrix)
    logging.info(f"   calculating first pass errors")
    all_error_boxes_idxs, error_matrix, p_errors = testing_poses(net_positions, epsilon=[0], plotit=False, detailed=False)
    eu_dist1 = np.sqrt(np.power(net_positions[:, :, 0]-label_pos[:, :, 0], 2) + np.power(net_positions[:, :, 1]-label_pos[:, :, 1], 2))
    logging.info("   mean error distance: {:02.4f} +- {:02.4f}.".format(np.mean(eu_dist1), np.std(eu_dist1)))

    # Preparing inputs for priors (selection of error indexes)
    input_pos = net_positions[all_error_boxes_idxs[0], ...]
    input_errors = error_matrix[all_error_boxes_idxs[0], :, 0]
    input_boxes = net_boxes[all_error_boxes_idxs[0], ...]
    input_cm = cm[all_error_boxes_idxs[0], ...]

    # Apply priors to errors
    logging.info(f"   applying priors")
    output_pos = new_testing_priors(input_pos, input_errors, input_boxes, input_cm)

    # Replacing new positions
    net_positions[all_error_boxes_idxs[0], ...] = output_pos

    # Try to find errors
    logging.info(f"   calculating second pass errors")
    all_error_boxes_idxs2, error_matrix2, p_errors2 = testing_poses(net_positions, epsilon=[0], plotit=False, detailed=False)
    eu_dist2 = np.sqrt(np.power(net_positions[:, :, 0]-label_pos[:, :, 0], 2) + np.power(net_positions[:, :, 1]-label_pos[:, :, 1], 2))
    logging.info("   mean error distance: {:02.4f} +- {:02.4f}.".format(np.mean(eu_dist2), np.std(eu_dist2)))

    # Calculate the decrease of error distance
    eu_dist_change = eu_dist1 - eu_dist2
    eu_dist_change = eu_dist_change[eu_dist_change.nonzero()]

    # Compare error distance pre- and post- priors
    if plotit:
        plt.figure(figsize=[6, 8])

        plt.subplot(311)
        plt.title('Error distance, before prior')
        body1 = subgroup(eu_dist1, [0, 1, 8, 11])
        wings1 = subgroup(eu_dist1, [9, 10])
        legs1 = subgroup(eu_dist1, [2, 3, 4, 5, 6, 7])
        plt.hist([eu_dist1, body1, wings1, legs1], bins=np.linspace(0, 6, 10), density=True, histtype='bar', label=['all', 'body', 'wings', 'legs'])
        plt.legend()

        plt.subplot(312)
        plt.title('Error distance, after prior')
        body2 = subgroup(eu_dist2, [0, 1, 8, 11])
        wings2 = subgroup(eu_dist2, [9, 10])
        legs2 = subgroup(eu_dist2, [2, 3, 4, 5, 6, 7])
        plt.hist([eu_dist2, body2, wings2, legs2], bins=np.linspace(0, 6, 10), density=True, histtype='bar', label=['all', 'body', 'wings', 'legs'])
        plt.legend()

        plt.subplot(313)
        plt.title('Decrease of error distance (preprior - postprior)')
        plt.hist(eu_dist_change, bins=np.linspace(-60, 60, 10), histtype='bar', rwidth=0.8)
        plt.subplots_adjust(bottom=0.05, top=0.95, wspace=0.4)

    # Report result
    logging.info(f"   {100*(all_error_boxes_idxs[0].shape[0]-all_error_boxes_idxs2[0].shape[0])/all_error_boxes_idxs[0].shape[0]}% of errors fixed by priors, {all_error_boxes_idxs2[0].shape[0]} unfixed.")


def new_testing_priors(positions, error_matrix, boxes, cm, priors_path: str = '/#Common/adrian/Workspace/temp/priors.h5', num_peaks: int = 5, plotit_bybp: bool = False, plotit_summary: bool = False):

    from leap_utils.postprocessing import max_simple

    priors_data = dd.io.load(priors_path)
    priors = priors_data['priors']

    new_positions = np.copy(positions).astype(np.int)

    if positions.shape[0] != error_matrix.shape[0]:
        print('pos - error')
        print('conflict of input shapes: {}, {}'.format(positions.shape, error_matrix.shape))
        pass
    elif positions.shape[0] != cm.shape[0]:
        print('pos - cm')
        print('conflict of input shapes: {}, {}'.format(positions.shape, cm.shape))
        pass
    elif positions.shape[0] != boxes.shape[0]:
        print('pos - boxes')
        print('conflict of input shapes: {}, {}'.format(positions.shape, boxes.shape))
        pass

    ref_thorax, _ = max_simple(priors[:, :, 8:9])
    ref_neck, _ = max_simple(priors[:, :, 1:2])
    ref_tail, _ = max_simple(priors[:, :, 11:12])

    cmap = get_cmap('OrRd')
    bp_names = ['head', 'neck', 'frontL', 'middleL', 'backL', 'frontR', 'middleR', 'backR', 'thorax', 'wingL', 'wingR', 'tail']

    for ii in range(positions.shape[0]):

        # Use thorax, neck, tail or no reference, for alignment of the prior
        if error_matrix[ii, 8] == 0:
            # print('thorax')
            row_shift = int(ref_thorax[0, 1]-positions[ii, 8, 1])
            col_shift = int(ref_thorax[0, 0]-positions[ii, 8, 0])
        elif error_matrix[ii, 1] == 0:
            # print('neck')
            row_shift = int(ref_neck[0, 1]-positions[ii, 1, 1])
            col_shift = int(ref_neck[0, 0]-positions[ii, 1, 0])
        elif error_matrix[ii, 11] == 0:
            # print('tail')
            row_shift = int(ref_tail[0, 1]-positions[ii, 11, 1])
            col_shift = int(ref_tail[0, 0]-positions[ii, 11, 0])
        else:
            # print('none')
            row_shift = 0
            col_shift = 0

        bp_to_prior = np.where(error_matrix[ii, :])[0]

        # Get peaks of confidence map
        maxima = max2d_multi(cm[ii, :, :, bp_to_prior], num_peaks=num_peaks)
        maxima_val = cm[ii, maxima[:, 0, :], maxima[:, 1, :], bp_to_prior]

        for jj, bp in enumerate(bp_to_prior):
            col_in_prior = maxima[:, 0, jj] - col_shift
            col_in_prior[col_in_prior > 119] = 119
            col_in_prior[col_in_prior < 0] = 0
            row_in_prior = maxima[:, 1, jj] - row_shift
            row_in_prior[row_in_prior > 119] = 119
            row_in_prior[row_in_prior < 0] = 0
            bay_score = np.multiply(maxima_val[:, jj], priors[col_in_prior, row_in_prior, bp])
            new_positions[ii, bp, :] = maxima[np.argmax(bay_score), :, jj]

            if plotit_bybp:
                norm = Normalize(vmin=0, vmax=1.2*np.amax(bay_score))
                plt.figure(figsize=[8, 2])
                plt.subplot(141)
                plt.title(ii)
                plt.imshow(boxes[ii, :, :, 0], aspect='auto', cmap='gray')
                plt.scatter(maxima[:, 1, jj], maxima[:, 0, jj], facecolors='none', edgecolors=cmap(norm(bay_score)))
                plt.scatter(positions[ii, 8, 1], positions[ii, 8, 0], marker='D', c='yellow')
                plt.scatter(positions[ii, bp, 1], positions[ii, bp, 0], s=80, marker='s', facecolors='none', edgecolors='yellow')
                plt.scatter(maxima[np.argmax(bay_score), 1, jj], maxima[np.argmax(bay_score), 0, jj], s=80, marker='s', facecolors='none', edgecolors='red')
                plt.subplot(142)
                plt.title(bp_names[bp])
                plt.imshow(cm[ii, :, :, bp], aspect='auto')
                plt.scatter(maxima[:, 1, jj], maxima[:, 0, jj], facecolors='none', edgecolors=cmap(norm(bay_score)))
                plt.scatter(positions[ii, bp, 1], positions[ii, bp, 0], s=80, marker='s', facecolors='none', edgecolors='yellow')
                plt.scatter(maxima[np.argmax(bay_score), 1, jj], maxima[np.argmax(bay_score), 0, jj], s=80, marker='s', facecolors='none', edgecolors='red')
                plt.subplot(143)
                plt.title('score: {:.6f}'.format(np.amax(bay_score)))
                plt.imshow(priors[:, :, bp], aspect='auto')
                plt.scatter(maxima[:, 1, jj], maxima[:, 0, jj], facecolors='none', edgecolors=cmap(norm(bay_score)))
                plt.scatter(positions[ii, bp, 1], positions[ii, bp, 0], s=80, marker='s', facecolors='none', edgecolors='yellow')
                plt.scatter(maxima[np.argmax(bay_score), 1, jj], maxima[np.argmax(bay_score), 0, jj], s=80, marker='s', facecolors='none', edgecolors='red')
                plt.subplot(144)
                plt.title("cm x prior")
                plt.imshow(np.multiply(priors[:, :, bp], cm[ii, :, :, bp]), aspect='auto')
                plt.scatter(maxima[:, 1, jj], maxima[:, 0, jj], facecolors='none', edgecolors=cmap(norm(bay_score)))
                plt.scatter(positions[ii, bp, 1], positions[ii, bp, 0], s=80, marker='s', facecolors='none', edgecolors='yellow')
                plt.scatter(maxima[np.argmax(bay_score), 1, jj], maxima[np.argmax(bay_score), 0, jj], s=80, marker='s', facecolors='none', edgecolors='red')
                plt.show()

        if plotit_summary:
            plt.figure()
            plt.title(ii)
            plt.imshow(boxes[ii, :, :, 0], aspect='auto', cmap='gray')
            plt.scatter(positions[ii, :, 1], positions[ii, :, 0], s=30, marker='s', c='red')
            plt.scatter(new_positions[ii, :, 1], new_positions[ii, :, 0], s=10, c='blue')
            plt.legend(['old', 'new'])
            plt.show()

    return new_positions


def priors_normal_test(vr, positions, trackfixed_path, pose_path, overwrite: bool = False, option: str = 'temp', network_path: str = 'Z:/#Common/chainingmic/leap/best_model.h5', temp_save_path: str = 'Z:/#Common/adrian/Workspace/temp/leap/normal_temp_save.h5'):
    """ Runs the pipeline to test the prior application to the predictions on the training dataset, using the model specified by network_path.
        A temporary save file of the confidence maps and boxes will be saved after the first time in the temp_save_path to speed up repeated calls.
        Make sure to change temp_save_path or it will use my old temporary save from 24.12.2018 with leap network."""

    nflies, box_centers, dataperfly, fixed_angles, _, _ = load_fixed_tracks(trackfixed_path, pose_path, option=option)

    first_frames = 10000

    # Find errors (error matrix)
    logging.info(f"   calculating first pass errors")
    all_error_boxes_idxs, error_matrix, p_errors = testing_poses(positions[:first_frames], epsilon=[0], plotit=False, detailed=True)

    # Get boxes and confidence maps
    logging.info(f"   get confidence maps")
    if overwrite or not os.path.exists(temp_save_path):

        # Get boxes
        boxes_idx = all_error_boxes_idxs[0]     # box indexing
        frame_idx, fly_ids = indexconvertion_box2frame(boxes_idx, nflies)   # frame indexing
        result_idx = fly_ids + np.arange(0, fly_ids.shape[0]*nflies, nflies)   # index to get specific box from export_boxes()

        logging.info(f"   ---getting frames from video")
        frames_list = list(vr[frame_idx.tolist()])

        logging.info(f"   ---exporting boxes from video")
        boxes, *_ = export_boxes(frames_list, box_centers[frame_idx, ...], box_size=np.array([120, 120]), box_angles=fixed_angles[frame_idx, ...])
        boxes = boxes[result_idx, ...]

        # Prepare boxes to be processed
        boxes = normalize_boxes(boxes)

        # Find confidence maps and positions using specified model
        logging.info(f"   ---processing boxes to get confidence maps")
        network = load_network(network_path, image_size=[120, 120])
        cm = predict_confmaps(network, boxes[:, :, :, :1])

        # Temporary save of results
        temp_save_data = {'cm': cm,
                          'boxes': boxes,
                          'boxes_idx': boxes_idx
                          }
        dd.io.save(temp_save_path, temp_save_data)
    else:
        logging.info(f"   ---loading temp save")
        temp_save_data = dd.io.load(temp_save_path)
        boxes = temp_save_data['boxes']
        cm = temp_save_data['cm']

    # Preparing inputs for priors (selection of error indexes)
    input_pos = positions[all_error_boxes_idxs[0], ...]
    input_errors = error_matrix[all_error_boxes_idxs[0], ...]
    input_boxes = boxes
    input_cm = cm

    # Apply priors to errors
    logging.info(f"   applying priors")
    output_pos = new_testing_priors(input_pos, input_errors, input_boxes, input_cm)

    # Replacing new positions
    positions[all_error_boxes_idxs[0], ...] = output_pos

    # Try to find errors
    logging.info(f"   calculating second pass errors")
    all_error_boxes_idxs2, error_matrix2, p_errors2 = testing_poses(positions[:first_frames], epsilon=[0], plotit=False, detailed=True)

    # Report result
    logging.info(f"   {100*(all_error_boxes_idxs[0].shape[0]-all_error_boxes_idxs2[0].shape[0])/all_error_boxes_idxs[0].shape[0]}% of errors fixed by priors, {all_error_boxes_idxs2[0].shape[0]} unfixed.")

    pass


def indexconvertion_data2frame(data_idx, dataperfly):
    """ converts a vector of indexes from cwt_data into frame and fly id according to dataperfly (amount of boxes per fly in the data) """

    # get cumulative dataperfly vector
    cdataperfly = np.concatenate((np.zeros(1), np.copy(dataperfly)))
    precdataperfly = 0
    for ii, idataperfly in enumerate(cdataperfly):
        precdataperfly += idataperfly
        cdataperfly[ii] = precdataperfly

    # find fly id
    f_fly_id = np.searchsorted(cdataperfly, data_idx, side='right')-1

    # convert to frame
    f = data_idx - cdataperfly[f_fly_id]

    return f, f_fly_id


def indexconvertion_frame2box(f, ifly, nflies: int = 2):
    b = f*nflies + ifly
    return b


def indexconvertion_box2frame(boxes_idx, nflies):
    f = np.floor_divide(boxes_idx, nflies)
    f_fly_id = boxes_idx % nflies
    return f, f_fly_id


def indexconvertion_frame2data(f, f_fly_id, dataperfly):

    # get cumulative dataperfly vector
    cdataperfly = np.concatenate((np.zeros(1), np.copy(dataperfly)))
    precdataperfly = 0
    for ii, idataperfly in enumerate(cdataperfly):
        precdataperfly += idataperfly
        cdataperfly[ii] = precdataperfly

    # convert to frame
    data_idx = f + cdataperfly[f_fly_id]

    return data_idx


def looping_priors_normal(vr, original_positions, trackfixed_path, pose_path, overwrite: bool = False, option: str = 'temp', network_path: str = 'Z:/#Common/chainingmic/leap/best_model.h5', temp_save_path: str = 'Z:/#Common/adrian/Workspace/temp/leap/normal_temp_save.h5', testing_priors_path: str = 'Z:/#Common/adrian/Workspace/temp/leap/normal_temp_save_priors.h5'):
    """ Runs the pipeline to test the prior application to the predictions on the training dataset, using the model specified by network_path.
        A temporary save file of the confidence maps and boxes will be saved after the first time in the temp_save_path to speed up repeated calls.
        Make sure to change temp_save_path or it will use my old temporary save from 24.12.2018 with leap network."""

    nflies, box_centers, dataperfly, fixed_angles, _, _ = load_fixed_tracks(trackfixed_path, pose_path, option=option)

    first_frames = 5000
    epsilon = np.arange(-12, 13, 1).tolist()
    print(epsilon)

    positions = np.copy(original_positions)

    # Find errors (error matrix)
    logging.info(f"   calculating first pass errors")
    all_error_boxes_idxs, error_matrix, p_errors = testing_poses(positions[:first_frames], epsilon=epsilon, plotit=False, detailed=False)

    # Get boxes and confidence maps
    logging.info(f"   get confidence maps")
    if overwrite or not os.path.exists(temp_save_path):

        # Get boxes
        boxes_idx = all_error_boxes_idxs[-1]  # box indexing
        frame_idx, fly_ids = indexconvertion_box2frame(boxes_idx, nflies)   # frame indexing
        result_idx = fly_ids + np.arange(0, fly_ids.shape[0]*nflies, nflies)   # index to get specific box from export_boxes()

        logging.info(f"   ---getting frames from video")
        frames_list = list(vr[frame_idx.tolist()])

        logging.info(f"   ---exporting boxes from video")
        boxes, *_ = export_boxes(frames_list, box_centers[frame_idx, ...], box_size=np.array([120, 120]), box_angles=fixed_angles[frame_idx, ...])
        boxes = boxes[result_idx, ...]

        # Prepare boxes to be processed
        boxes = normalize_boxes(boxes)

        # Find confidence maps and positions using specified model
        logging.info(f"   ---processing boxes to get confidence maps")
        network = load_network(network_path, image_size=[120, 120])
        cm = predict_confmaps(network, boxes[:, :, :, :1])

        # Temporary save of results
        temp_save_data = {'cm': cm,
                          'boxes': boxes,
                          'boxes_idx': boxes_idx,
                          'p_errors': p_errors,
                          'all_error_boxes_idxs': all_error_boxes_idxs
                          }
        dd.io.save(temp_save_path, temp_save_data)
    else:
        logging.info(f"   ---loading temp save")
        temp_save_data = dd.io.load(temp_save_path)
        boxes = temp_save_data['boxes']
        boxes_idx = temp_save_data['boxes_idx']
        cm = temp_save_data['cm']

    p_errors2 = np.zeros((12, 2, len(epsilon)))
    for ii in range(len(epsilon)):
        # Preparing inputs for priors (selection of error indexes)
        iboxes_idx = np.isin(boxes_idx, list(all_error_boxes_idxs[ii]))
        input_pos = positions[all_error_boxes_idxs[ii], ...]
        input_errors = error_matrix[all_error_boxes_idxs[ii], :, ii]
        input_boxes = boxes[iboxes_idx, ...]
        input_cm = cm[iboxes_idx, ...]

        # Apply priors to errors
        logging.info(f"   applying priors")
        output_pos = new_testing_priors(input_pos, input_errors, input_boxes, input_cm)

        # Replacing new positions
        positions[all_error_boxes_idxs[ii], ...] = output_pos

        # Try to find errors
        logging.info(f"   calculating second pass errors")
        all_error_boxes_idxs2, error_matrix2, ip_errors2 = testing_poses(positions[:first_frames], epsilon=[epsilon[ii]], plotit=False, detailed=False)

        # Report result
        logging.info(f"   {100*(all_error_boxes_idxs[ii].shape[0]-all_error_boxes_idxs2[0].shape[0])/all_error_boxes_idxs[ii].shape[0]}% of errors fixed by priors, {all_error_boxes_idxs2[0].shape[0]} unfixed.")

        # Save result
        p_errors2[..., ii] = np.squeeze(ip_errors2)

        # Re-initialise positions
        positions = np.copy(original_positions)

    testing_priors_data = {'p_errors': p_errors, 'p_errors2': p_errors2, 'epsilon': epsilon}
    dd.io.save(testing_priors_path, testing_priors_data)

    pass


def view_looping_priors_results(testing_priors_path: str = 'Z:/#Common/adrian/Workspace/temp/leap/normal_temp_save_priors.h5'):
    data = dd.io.load(testing_priors_path)
    p_errors = data['p_errors']
    p_errors2 = data['p_errors2']
    epsilon = data['epsilon']

    # Relative percent of proportion change
    d_errors = (p_errors - p_errors2) / p_errors

    # plt.figure()
    # for ii in range(12):
    #     plt.subplot(12, 2, ii*2+1)
    #     plt.bar(epsilon, d_errors[ii, 0, :])
    #     plt.ylim(-3, 3)
    #     plt.subplot(12, 2, ii*2+2)
    #     plt.bar(epsilon, d_errors[ii, 1, :])
    #     plt.ylim(-3, 3)

    bp_names = ['head', 'neck', 'frontL', 'middleL', 'backL', 'frontR', 'middleR', 'backR', 'thorax', 'wingL', 'wingR', 'tail']
    plt.figure(figsize=[10, 10])
    for ii in range(12):
        plt.subplot(12, 1, ii+1)
        width = 0.35
        plt.bar(np.arange(len(epsilon)), d_errors[ii, 0, :], width, label='y', zorder=3)
        plt.bar(np.arange(len(epsilon)) + width, d_errors[ii, 1, :], width, label='x', zorder=3)
        plt.legend(loc='upper right', prop={'size': 8})
        plt.ylim(-2, 2)
        plt.xlim(-0.5, len(epsilon)+2)
        plt.ylabel(bp_names[ii])
        plt.gca().grid(b=True, which='major', axis='y', zorder=0)
        if ii == 11:
            plt.xticks(np.arange(len(epsilon)) + width / 2, map(str, epsilon))
            plt.xlabel('Epsilon (pixels reducing the error detection thresholds per body part, + = narrower accepted range)')
        else:
            plt.xticks([], [])
        if ii == 0:
            plt.title('Proportion of Errors fixed (Pre-prior - Post-prior / Pre-prior)')
        plt.tight_layout()

    bp_names = ['head', 'neck', 'frontL', 'middleL', 'backL', 'frontR', 'middleR', 'backR', 'thorax', 'wingL', 'wingR', 'tail']
    plt.figure(figsize=[10, 10])
    for ii in range(12):
        plt.subplot(12, 1, ii+1)
        width = 0.35
        plt.bar(np.arange(len(epsilon)), p_errors[ii, 0, :], width, label='y', zorder=3)
        plt.bar(np.arange(len(epsilon)) + width, p_errors[ii, 1, :], width, label='x', zorder=3)
        plt.legend(loc='upper right', prop={'size': 8})
        plt.ylim(0, 25)
        plt.xlim(-0.5, len(epsilon)+2)
        plt.ylabel(bp_names[ii])
        plt.gca().grid(b=True, which='major', axis='y', zorder=0)
        if ii == 11:
            plt.xticks(np.arange(len(epsilon)) + width / 2, map(str, epsilon))
            plt.xlabel('Epsilon (pixels reducing the error detection thresholds per body part, + = narrower accepted range)')
        else:
            plt.xticks([], [])
        if ii == 0:
            plt.title('Proportion of Errors Pre-prior')
        plt.tight_layout()

    # all-x, all-y, body, ..., wings, ..., legs, ...

    # # Subgrouping data
    # body = subgroup(eu_dist, [0, 1, 8, 11])
    # wings = subgroup(eu_dist, [9, 10])
    # legs = subgroup(eu_dist, [2, 3, 4, 5, 6, 7])
    #
    # # Plot
    # plt.hist([eu_dist, body, wings, legs], bins=np.arange(-13, 13, 1), density=True, histtype='bar')
    # plt.legend()

    pass


def error_subgroup(X, parts, xory, epsilon):

    subgrouped = np.empty([0])
    for count, ipart in enumerate(parts):
        for count3, eps in enumerate(epsilon):
            for count2, ixory in enumerate(xory):
                subgrouped = np.concatenate((subgrouped, X[ipart, xory, eps]), axis=0)

    return subgrouped


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    defopt.run(main)
