from leap_utils.postprocessing import load_labels
import numpy as np
import deepdish as dd
import matplotlib.pyplot as plt
from leap_utils.utils import iswin, ismac
import defopt
import logging


if iswin():
    root = 'Z:/#Common/'
elif ismac():
    root = '/Volumes/ukme04/#Common/'
else:
    root = '/scratch/clemens10/'


def subgroup(X, parts):

    subgrouped = np.empty([0])
    for count, ipart in enumerate(parts):
        subgrouped = np.concatenate((subgrouped, X[:, ipart]), axis=0)

    return subgrouped


def plot_errordistance(Data, DataLabels, bins):
    print(np.linspace(0, 10, 21))
    plt.hist(Data, bins=bins, density=True, histtype='bar', label=DataLabels)
    plt.legend()


def main():

    # Inputs
    tempPredsPath = root+'chainingmic/leap/label_predictions.h5'
    networkPath = root+'chainingmic/leap/best_model.h5'
    redo_pred = False

    # Other paths
    histFig = tempPredsPath[:-2] + 'png'

    # Load labels
    label_pos, _, boxes = load_labels()
    label_pos = np.flip(np.swapaxes(label_pos, 1, 2), 2)

    # Get predictions
    if redo_pred:
        # In case that we need to re-do the predictions from the labeled boxes:
        from leap_utils.preprocessing import normalize_boxes
        from leap_utils.postprocessing import process_confmaps_simple
        from leap_utils.predict import predict_confmaps, load_network
        box_size = [120, 120]
        network = load_network(networkPath, image_size=box_size)
        confmaps = predict_confmaps(network, normalize_boxes(boxes))
        positions, _ = process_confmaps_simple(confmaps)
        dd.io.save(tempPredsPath, positions)
    else:
        pos = dd.io.load(tempPredsPath)

    # Euclidian distance
    eu_dist = np.sqrt(np.power(pos[:, :, 0]-label_pos[:, :, 0], 2) + np.power(pos[:, :, 1]-label_pos[:, :, 1], 2))

    # Subgrouping data
    body = subgroup(eu_dist, [0, 1, 8, 11])
    wings = subgroup(eu_dist, [9, 10])
    legs = subgroup(eu_dist, [2, 3, 4, 5, 6, 7])

    plot_errordistance(Data=[eu_dist, body, wings, legs], DataLabels=['all', 'body', 'wings', 'legs'], bins=np.linspace(0, 10, 21))

    # plt.savefig(histFig)
    plt.show()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    defopt.run(main)
