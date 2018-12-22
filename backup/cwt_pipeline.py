import numpy as np
import deepdish as dd
from scipy.stats import zscore
import pywt
import os
import time as tm
import logging
import defopt
from leap_utils.utils import iswin, ismac

# General stuff
np.seterr(divide='ignore', invalid='ignore')
logging.basicConfig(level=logging.INFO)
if iswin():
    root = 'Z:/#Common/'
elif ismac():
    root = '/Volumes/ukme04/#Common/'
else:
    root = '/scratch/clemens10/'
res_path = root+'chainingmic/res'


def cwt_pipeline(expID: str = 'localhost-20180720_143054', min_scale: float = 0, max_scale: float = np.log2(50), nscale: int = 25):

    t0 = tm.perf_counter()

    # Paths
    pose_path = f"{res_path}/{expID}/{expID}_poses.h5"
    cwt_path = f"{res_path}/{expID}/{expID}_cwt.h5"
    # pose_path = 'Z:/#Common/adrian/Workspace/temp/'+expID+'_poses.h5'
    # cwt_path = 'Z:/#Common/adrian/Workspace/temp/'+expID+'_cwt.h5'

    # Setting scales
    t = tm.perf_counter()
    scales = 162.5/(2*2**np.linspace(min_scale, max_scale, nscale))
    scale_len = scales.shape[0]
    logging.info(f"   scales: {scales}.")

    # Get data
    t = tm.perf_counter()
    data = dd.io.load(pose_path)
    positions = data['positions']   # [boxes, bodyparts, coordinates]
    fly_id = data['fly_id']
    nfly = int(np.max(fly_id)+1)
    logging.info(f"   nflies: {nfly}, positions shape: {positions.shape}.")
    logging.info(f"   get data: {tm.perf_counter()-t}.")

    # Select data
    t = tm.perf_counter()
    thorax_id = 8
    thorax = np.zeros((positions.shape[0], 1, positions.shape[2]))
    thorax[:, 0, :] = positions[:, thorax_id, :]
    zRel_positions = np.delete(zscore(positions - thorax), thorax_id, 1)
    logging.info(f"   select data: {tm.perf_counter()-t}.")

    # Do CWT
    t = tm.perf_counter()
    cwt_array = np.zeros((0, scale_len*nfly*11*2))
    for sfly in range(nfly):
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
        logging.info(f"   fly {sfly}: {ifly_cwt_array.shape[0]}.")

    logging.info(f"   cwt_array shape: {cwt_array.shape}.")
    logging.info(f"   do cwt: {tm.perf_counter()-t}.")

    # Save results
    t = tm.perf_counter()
    print('saving!')
    if os.path.exists(cwt_path):
        print("{} exists - deleting to overwrite.".format(cwt_path))
        os.remove(cwt_path)
    logging.info(f"   deleted file: {tm.perf_counter()-t}.")

    t = tm.perf_counter()
    cwt_data = {'expID': expID,
                'cwt': cwt_array,
                'frequencies': frequencies,
                'scales': scales}
    logging.info(f"   created cwt_data: {tm.perf_counter()-t}.")

    t = tm.perf_counter()
    dd.io.save(cwt_path, cwt_data)
    logging.info(f"   saved cwt_data: {tm.perf_counter()-t}.")
    logging.info(f"   Total: {tm.perf_counter()-t0}.")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    defopt.run(cwt_pipeline)
