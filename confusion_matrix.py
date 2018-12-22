import numpy as np
import logging
import deepdish as dd
from leap_utils.utils import iswin, ismac
from sklearn.metrics import confusion_matrix
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from scipy import stats


def do_kdtree(combined_arrays, points, k: int = 1):
    mytree = cKDTree(combined_arrays)
    dist, indexes = mytree.query(points, k=k, n_jobs=-1)
    return indexes


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

    # fmt = '.2f' if normalize else 'd'
    # thresh = cm.max() / 2.
    # for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #     plt.text(j, i, format(cm[i, j], fmt),
    #              horizontalalignment="center",
    #              color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# General stuff
np.seterr(divide='ignore', invalid='ignore')
logging.basicConfig(level=logging.INFO)
if iswin():
    root = 'Z:/#Common/'
elif ismac():
    root = '/Volumes/ukme04/#Common/'
else:
    root = '/scratch/clemens10/'

doit = True
expID = 'localhost-20180720_182837'
res_path = root+'chainingmic/res'
ws_path = f"{res_path}/{expID}/{expID}_ws.h5"
cwt_path = f"{res_path}/{expID}/{expID}_cwt.h5"
tsne_path = f"{res_path}/{expID}/{expID}_temptsne.h5"
clabels_path = f"{res_path}/{expID}/{expID}_clabels.h5"
testlabels_path = f"{res_path}/{expID}/{expID}_testlabels.h5"

if doit:
    # load all required data
    tdata = dd.io.load(tsne_path)
    clabels_data = dd.io.load(clabels_path)
    all_X50 = tdata['all_X50']
    random_data_idx = tdata['random_data_idx']
    Z_labels = clabels_data['Z_labels']

    # divide random sample labeled data into test and train
    p_test = 0.5
    test_meta_idx = np.random.choice(random_data_idx.shape[0], int(p_test*random_data_idx.shape[0]), replace=False)
    train_meta_idx = np.arange(random_data_idx.shape[0])[~test_meta_idx]
    test_idx = random_data_idx[test_meta_idx]
    train_idx = random_data_idx[train_meta_idx]

    # References
    ground_truth = Z_labels[test_meta_idx]

    ks = range(1, 2)
    conf = np.zeros((len(ks), test_idx.shape[0]), dtype=bool)
    for k in ks:
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

        # testlabels_data = {'test_results': test_results,
        #                    'ground_truth': ground_truth,
        #                    'test_neighbors': test_neighbors,
        #                    'Z_labels': Z_labels,
        #                    'test_meta_idx': test_meta_idx,
        #                    'train_meta_idx': train_meta_idx,
        #                    'test_idx': test_idx,
        #                    'train_idx': train_idx,
        #                    'random_data_idx': random_data_idx,
        #                    'conf_mat': conf_mat,
        #                    'k': k}
        # dd.io.save(testlabels_path, testlabels_data)

        for ii in range(test_results.shape[0]):
            conf[k-np.min(ks), ii] = test_results[ii] == ground_truth[ii]
        print(k, np.sum(conf[k-np.min(ks), :])/conf.shape[1])
        plt.figure(k)
        plot_confusion_matrix(conf_mat, normalize=True, title='Normalized with k={}'.format(k))
        plot_path = testlabels_path[:-3] + '_k_' + str(k) + '.png'
        plt.savefig(plot_path)
        plt.close()

    testlabels_data = {'conf': conf}
    dd.io.save(testlabels_path, testlabels_data)

else:
    data = dd.io.load(testlabels_path)
    conf_mat = data['conf_mat']
    test_results = data['test_results']
    ground_truth = data['ground_truth']
    k = data['k']
    conf = np.zeros(test_results.shape, dtype=bool)
    for ii in range(test_results.shape[0]):
        conf[ii] = test_results[ii] == ground_truth[ii]
    print(np.sum(conf)/conf.shape[0])
    plot_confusion_matrix(conf_mat, normalize=True, title='Normalized with k={}'.format(k))
