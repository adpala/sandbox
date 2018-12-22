import numpy as np
import logging
import os
import deepdish as dd
import matplotlib.pyplot as plt
from leap_utils.utils import iswin, ismac, unflatten
from leap_utils.plot import vplay, annotate
from videoreader import VideoReader
from leap_utils.preprocessing import export_boxes
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
from sklearn.neighbors import KernelDensity
from matplotlib import cm
# from scipy.stats import gaussian_kde as kde

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
video_path = f"{data_path}/{expID}/{expID}.mp4"
if not os.path.exists(video_path):
    data_path = root+'chainingmic/dat.processed'
    video_path = f"{data_path}/{expID}/{expID}.mp4"
vr = VideoReader(video_path)
trackfixed_path = f"{res_path}/{expID}/{expID}_tracks_fixed.h5"
labels_path = f"{res_path}/{expID}/{expID}_all_labels.h5"
cwt_path = f"{res_path}/{expID}/{expID}_cwt.h5"
pose_path = f"{res_path}/{expID}/{expID}_poses.h5"
freq_path = f"{res_path}/{expID}/cluster_freq"
densities_path = f"{res_path}/{expID}/cluster_dens"

# Load data
labels_data = dd.io.load(labels_path)  # {'all_labels', 'all_nearest_neighbor_index', 'all_Z', 'expID', 'tsne_path'}
labels = labels_data['all_labels']
Z = labels_data['all_Z']

pose_data = dd.io.load(pose_path)
fly_id = pose_data['fly_id']
fixed_angles = pose_data['fixed_angles']
positions = pose_data['positions']

data = dd.io.load(trackfixed_path)
centers = data['centers']
chbb = data['chambers_bounding_box'][:]
box_centers = centers[:, 0, :, :]   # nframe, fly id, coordinates
box_centers = box_centers + chbb[1][0][:]
nflies = box_centers.shape[1]
dataperfly = np.zeros(nflies, dtype=np.int)
for ii in range(nflies):
    dataperfly[ii] = np.sum(fly_id == ii).astype(np.int)


def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) > stepsize)[0] + 1)


def save_vplay(frames: np.array, positions: np.array = None):

    import cv2

    if frames.shape[3] == 1:
        frames = np.repeat(frames, 3, axis=3)

    output = 'C:/Users/apalaci/Dropbox/Master_Thesis/Presentation/video.mp4'

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output, fourcc, 20.0, (120, 120))

    for ii in range(frames.shape[0]):
        frame = frames[ii, ...]
        if positions is not None:
            frame = annotate(frame, positions[ii, ...])

        out.write(frames[ii, ...])

        cv2.imshow('video', frames[ii, ...])
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break
    # Release everything if job is finished
    out.release()
    cv2.destroyAllWindows()
    print("The output video is {}".format(output))


def multi_vplay(frames_list):

    import cv2
    import time

    if frames_list[0].shape[3] == 1:
        for frames in frames_list:
            frames = np.repeat(frames, 3, axis=2)

    for nwindows in range(len(frames_list)):
        cv2.namedWindow(str(nwindows), cv2.WINDOW_AUTOSIZE)
        x_shift = (nwindows % 3)*200
        y_shift = int(nwindows/3)*200
        cv2.moveWindow(str(nwindows), x_shift, y_shift)

    fcount = np.zeros(len(frames_list), dtype=np.int)
    while True:
        for zz in range(len(frames_list)):
            frame = frames_list[zz][fcount[zz], ...]
            cv2.imshow(str(zz), frame)

            fcount[zz] += 1
            if fcount[zz] >= len(frames_list[zz]):
                fcount[zz] = 0
        time.sleep(0.01)    # to reproduce at 100 frames per second
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


def get_intervals_idx(idata_idx):
    all_intervals_idx = consecutive(idata_idx)
    interval_idx = np.zeros((len(all_intervals_idx), 2), dtype=np.int)
    for count, x in enumerate(all_intervals_idx):
        interval_idx[count, 0] = np.amin(x).astype(int)
        interval_idx[count, 1] = np.amax(x).astype(int)
    return interval_idx


def get_longest(interval_idx):
    return np.argmax(interval_idx[:, 1]-interval_idx[:, 0])


def get_fly_id(interval_start):
    previous = 0
    ii = 0
    while ~np.isin(interval_start, np.arange(previous, previous + dataperfly[ii])):
        previous = dataperfly[ii]
        ii += 1
    return ii


def idx2frames(interval_idx, choosen_interval, ifly):
    return np.arange(interval_idx[choosen_interval, 0], interval_idx[choosen_interval, 1]+1) - np.sum(dataperfly[:ifly])


def frames2boxes(frames, nflies, ifly):
    boxes = frames*nflies + ifly
    if boxes[-1] == len(labels):
        boxes[-1] -= 1
    return boxes


def get_movie(icluster: int, chosen_interval: int = 0, saveit: bool = False):

    # Indexes intervals
    interval_idx = get_intervals_idx(np.squeeze(np.asarray(np.where(labels == icluster))))
    idx = np.argsort(interval_idx[:, 0]-interval_idx[:, 1])[chosen_interval]

    # Fly id
    ifly = get_fly_id(interval_idx[idx, 0])

    # Frames
    frames = idx2frames(interval_idx, idx, ifly)

    # Frames to boxes
    boxes_idx = frames2boxes(frames, nflies, ifly)

    frames_list = list(vr[frames.tolist()])

    boxes, *_ = export_boxes(frames_list, box_centers[frames, ...], box_size=np.array([120, 120]), box_angles=unflatten(fixed_angles, 2)[frames, ...])

    if saveit:
        save_vplay(frames=boxes[ifly::nflies, ...], positions=positions[boxes_idx, ...])
    else:
        vplay(frames=boxes[ifly::nflies, ...], positions=positions[boxes_idx, ...], moviemode=True)


def get_multi_movie(icluster: int, saveit: bool = False):

    # Indexes intervals
    interval_idx = get_intervals_idx(np.squeeze(np.asarray(np.where(labels == icluster))))

    if interval_idx.shape[0] < 9:
        n_videos = interval_idx.shape[0]
    else:
        n_videos = 9

    idxs = np.argsort(interval_idx[:, 0]-interval_idx[:, 1])[:n_videos]

    # Fly id
    ifly = np.zeros(n_videos)
    videos_list = list()
    for count, idx in enumerate(idxs):
        ifly = get_fly_id(interval_idx[idx, 0])

        # Frames
        frames = idx2frames(interval_idx, idx, ifly)

        frames_list = list(vr[frames.tolist()])
        boxes, *_ = export_boxes(frames_list, box_centers[frames, ...], box_size=np.array([120, 120]), box_angles=unflatten(fixed_angles, 2)[frames, ...])

        videos_list.append(boxes[ifly::nflies, ...])

    multi_vplay(videos_list)


def get_centroid(points, points_labels: np.array = None):
    """ Either use get_centroid(Z[labels == icluster, :]) for individual cluster, or cluster_center = get_centroid(Z, labels) for all clusters. """

    if points_labels is None:
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        centroid = np.asarray([sum(x) / len(points), sum(y) / len(points)])
    else:
        clusters, counts = np.unique(points_labels, return_counts=True)
        clusters = clusters[np.argsort(-counts)]
        centroid = np.zeros((clusters.shape[0], 2))
        for zz in range(clusters.shape[0]):
            ipoints = points[points_labels == clusters[zz], :]
            x = [p[0] for p in ipoints]
            y = [p[1] for p in ipoints]
            centroid[zz, :] = np.asarray([sum(x) / len(ipoints), sum(y) / len(ipoints)])
    return centroid


def select_cluster(icluster, labels, option, chosen_interval: int = 0):
    """
    interval = select_cluster(icluster, labels, 'interval')
    frames = select_cluster(icluster, labels, 'frames')
    boxes_idx = select_cluster(icluster, labels, 'boxes')
    """

    # Indexes intervals
    interval_idx = get_intervals_idx(np.squeeze(np.asarray(np.where(labels == icluster))))
    idx = np.argsort(interval_idx[:, 0]-interval_idx[:, 1])[chosen_interval]

    if option == 'interval':
        return interval_idx[idx, :]
    else:
        # Fly id
        ifly = get_fly_id(interval_idx[idx, 0])
        frames = idx2frames(interval_idx, idx, ifly)

    if option == 'frames':
        return frames
    elif option == 'boxes':
        boxes_idx = frames2boxes(frames, nflies, ifly)
        return boxes_idx
    else:
        print("wrong type of option ('intervals', 'frames' or 'boxes')")
        return None


def residency_times(labels, sorted_clusters):
    mean_residency = np.zeros(sorted_clusters.shape)
    total_mean_residency = np.zeros(1)
    for zz in range(sorted_clusters.shape[0]):
        X = np.squeeze(np.asarray(np.where(labels == sorted_clusters[zz])))
        all_intervals_idx = np.split(X, np.where(np.diff(X) != 1)[0]+1)
        interval_idx = np.zeros((len(all_intervals_idx), 2), dtype=np.int)
        for count, y in enumerate(all_intervals_idx):
            interval_idx[count, 0] = np.amin(y).astype(int)
            interval_idx[count, 1] = np.amax(y).astype(int)
        mean_residency[zz] = np.mean(interval_idx[:, 1]-interval_idx[:, 0])*0.01
        total_mean_residency = np.concatenate((total_mean_residency, interval_idx[:, 1]-interval_idx[:, 0]), axis=0)

    print('mean residency time: {}'.format(np.mean(total_mean_residency)*0.01))

    return mean_residency, total_mean_residency


def poses_dist():
    """ Will plot the collection of poses that are clustered together (no time encoded) for a set of 20 (or better 9?) clusters """
    # select cluster
    clusters, cluster_counts = np.unique(labels, return_counts=True)
    sorted_clusters = clusters[np.argsort(-cluster_counts)]
    cluster_counts = cluster_counts[np.argsort(-cluster_counts)]

    cmap = get_cmap('gist_ncar')
    norm = Normalize(vmin=0, vmax=12)
    plt.figure()
    for count, zz in enumerate(sorted_clusters[:16]):
        plt.subplot(4, 4, count+1)
        plt.axis('off')
        boxes_idx = select_cluster(zz, labels, 'boxes')
        plt.title('c {}, n {}, I {}'.format(zz, cluster_counts[count], boxes_idx.shape[0]))
        bodyparts = range(12)
        for jj in bodyparts:
            plt.scatter(positions[boxes_idx, jj, 1], positions[boxes_idx, jj, 0], s=0.1, cmap=cmap(norm(jj)))
        plt.ylim(120, 0)
        plt.xlim(0, 120)


def poses_withTime():
    """ Will plot the poses that are clustered together for a given cluster, for the top 9 longest intervals in it, encoding time """
    # collect poses
    # plot poses encoding time per interval
    pass


def poses_animated():
    """ Will plot the animation of poses that are clustered together for a given cluster, for the top 9 longest intervals in it """
    # collect poses
    # create stack of images per interval
    # play videos of the stacks simultaneously (with multi_vplay or something similar?)
    pass


def old_freq_analysis(cwt_path, icluster, labels):
    cwtdata = dd.io.load(cwt_path)
    cwt = cwtdata['cwt']    # [nboxes, (X-parts-scales y-parts-scales)]
    freq = cwtdata['frequencies']
    boxes_idx = select_cluster(icluster, labels, 'boxes')
    half_dim = int(cwt.shape[1]/2)
    im = np.zeros((22, int(cwt.shape[1]/22)))
    for xoy in range(2):
        for jj in range(11):
            im[jj*2 + xoy, :] = np.mean(cwt[boxes_idx, jj*25+xoy*half_dim: (jj+1)*25+xoy*half_dim], axis=0)
    plt.matshow(cwt[boxes_idx, :])
    plt.matshow(im)
    bp = ['head', 'neck', 'frontL', 'middleL', 'backL', 'frontR', 'middleR', 'backR', 'wingL', 'wingR', 'tail']
    plt.gca().set_xticks(np.arange(len(freq), step=4))
    plt.gca().set_yticks(np.arange(2*len(bp), step=2))
    plt.gca().set_xticklabels(["{0:.2f}".format(f) for f in freq[np.arange(len(freq), step=4)]])
    plt.gca().set_yticklabels(bp)


def freq_analysis(cwt_path, icluster, labels):
    cwtdata = dd.io.load(cwt_path)
    cwt = cwtdata['cwt']    # [nboxes, (X-parts-scales y-parts-scales)]
    freq = cwtdata['frequencies']

    # Indexes intervals
    interval_idx = get_intervals_idx(np.squeeze(np.asarray(np.where(labels == icluster))))
    idxs = np.argsort(interval_idx[:, 0]-interval_idx[:, 1])[:]

    # Fly id
    ifly = []
    frames = []
    boxes_idx = np.empty(1, dtype=np.int)
    for n, idx in enumerate(idxs):
        ifly = get_fly_id(interval_idx[idx, 0])
        frames = idx2frames(interval_idx, idx, ifly)
        boxes_idx = np.concatenate((boxes_idx, frames2boxes(frames, nflies, ifly)))
    half_dim = int(cwt.shape[1]/2)
    im = np.zeros((22, int(cwt.shape[1]/22)))
    for xoy in range(2):
        for jj in range(11):
            im[jj*2 + xoy, :] = np.mean(cwt[boxes_idx, jj*25+xoy*half_dim: (jj+1)*25+xoy*half_dim], axis=0)
    plt.matshow(im)
    bp = ['head', 'neck', 'frontL', 'middleL', 'backL', 'frontR', 'middleR', 'backR', 'wingL', 'wingR', 'tail']
    plt.gca().set_xticks(np.arange(len(freq), step=4))
    plt.gca().set_yticks(np.arange(2*len(bp), step=2))
    plt.gca().set_xticklabels(["{0:.2f}".format(f) for f in freq[np.arange(len(freq), step=4)]])
    plt.gca().set_yticklabels(bp)


clusters, cluster_counts = np.unique(labels, return_counts=True)
sorted_clusters = clusters[np.argsort(-cluster_counts)]
poses_dist()


def makeColours(vals):
    colours = np.zeros((len(vals), 3))
    norm = Normalize(vmin=vals.min(), vmax=vals.max())
    colours = [cm.ScalarMappable(norm=norm, cmap='plasma').to_rgba(val) for val in vals]
    return colours


def cluster_densities(c):
    bps = ['head', 'neck', 'frontL', 'middleL', 'backL', 'frontR', 'middleR', 'backR', 'thorax', 'wingL', 'wingR', 'tail']
    xx_d = np.linspace(0, 120, 120)
    yy_d = np.linspace(0, 120, 120)
    xx_dv, yy_dv = np.meshgrid(xx_d, yy_d)
    coor = np.array([xx_dv.flatten(), yy_dv.flatten()]).swapaxes(0, 1)

    interval_idx = get_intervals_idx(np.squeeze(np.asarray(np.where(labels == c))))
    idxs = np.argsort(interval_idx[:, 0]-interval_idx[:, 1])[:]
    ifly = []
    frames = []
    boxes_idx = np.empty(1, dtype=np.int)
    for n, idx in enumerate(idxs):
        ifly = get_fly_id(interval_idx[idx, 0])
        frames = idx2frames(interval_idx, idx, ifly)
        boxes_idx = np.concatenate((boxes_idx, frames2boxes(frames, nflies, ifly)))
    plt.figure(c)
    for jj, b in enumerate(bps):
        # plt.scatter(positions[boxes_idx, jj, 1], positions[boxes_idx, jj, 0], s=2, cmap=cmap(norm(jj)), edgecolors='none', alpha=0.5)
        plt.subplot(3, 4, jj+1)
        plt.title(b)
        kde = KernelDensity(bandwidth=5).fit(positions[boxes_idx, jj, :])
        image = np.exp(kde.score_samples(coor)).reshape((120, 120)).swapaxes(0, 1)
        plt.imshow(image)
        # densObj = kde(positions[boxes_idx, jj, :].T)
        # colours = makeColours(densObj.evaluate(positions[boxes_idx, jj, :].T))
        # plt.scatter(positions[boxes_idx, jj, 1], positions[boxes_idx, jj, 0], color=colours)
        plt.axis('off')
        plt.ylim(120, 0)
        plt.xlim(0, 120)


# Run density of body part positions per cluster on all clusters

if not os.path.exists(densities_path):
    os.mkdir(densities_path)
for ii in sorted_clusters:
    print('doing analysis on: {}'.format(ii))
    cluster_densities(ii)
    figname = densities_path + f"/cluster_{ii}.png"
    print(figname)
    plt.savefig(figname)
    plt.close()

# cluster_densities(sorted_clusters)

# Run frequency analysis per cluster on all clusters

# if not os.path.exists(freq_path):
#     os.mkdir(freq_path)
# for ii in sorted_clusters:
#     print('doing analysis on: {}'.format(ii))
#     freq_analysis(cwt_path, ii, labels)
#     figname = freq_path + f"/cluster_{ii}.png"
#     print(figname)
#     plt.savefig(figname)
#     plt.close()


# print('individual cluster centroids')
# for ii in range(5):
#     icluster = sorted_clusters[ii]
#     cluster_center = get_centroid(Z[labels == icluster, :])
#     print('cluster {}: {}'.format(icluster, cluster_center))
#
# print('all cluster centroids')
# cluster_center = get_centroid(Z, labels)
# for ii in range(5):
#     icluster = sorted_clusters[ii]
#     print('cluster {}: {}'.format(icluster, cluster_center[ii, :]))
#
# plt.figure()
# cluster_center = get_centroid(Z, labels)
# plt.scatter(cluster_center[:5, 0], cluster_center[:5, 1])
# plt.ylim(-90, 90)
# plt.xlim(-90, 90)

# Go through multiple clusters to inspect videos
# for n in range(2, 10):
#     print('cluster {}'.format(sorted_clusters[n]))
#     get_multi_movie(sorted_clusters[n])


# Get pose distribution or movie from pose distribution
# for bp in [0, 11]:
#     plt.scatter(positions[iframes_idx, bp, 1], positions[iframes_idx, bp, 0], s=1)

# Get average gesture (variation within frames of datapoints in cluster)

# Get average freq. analysis from each body part for a cluster

plt.show()
