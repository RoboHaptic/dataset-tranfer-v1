import scipy.io as scio
import numpy as np
from scipy.signal import decimate
from utils import channel_alignment_position, channel_selection


CHANNELS = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'TP9',
            'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10', 'FC3',
            'FC4', 'C5', 'C1', 'C2', 'C6', 'CP3', 'CPz', 'CP4', 'P1', 'P2', 'POz', 'FT9', 'FTT9h', 'TTP7h', 'TP7',
            'TPP9h', 'FT10', 'FTT10h', 'TPP8h', 'TP8', 'TPP10h', 'F9', 'F10', 'AF7', 'AF3', 'AF4', 'AF8', 'PO3', 'PO4']

INTERSECTION = ['Fz', 'FC3', 'FC1', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz',
                'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz']


def sub54():

    path = './data/sub_54/'

    x, y = np.load(path + 'sub_x.npy'), np.load(path + 'sub_y.npy')
    x = np.array(x, dtype=np.float32)
    train = [x, y]

    # pos = channel_alignment_position(CHANNELS, INTERSECTION)
    #
    # train = None
    # for sub in range(1, 55):
    #
    #     x = np.load(path + 'sub{:02d}_x.npy'.format(sub))
    #     y = np.load(path + 'sub{:02d}_y.npy'.format(sub))
    #
    #     if train is None:
    #         train = [x, y]
    #     else:
    #         train[0], train[1] = np.vstack((train[0], x)), np.vstack((train[1], y))
    #
    # train = channel_selection(pos, train, INTERSECTION)
    # np.save('../data/sub_x.npy', train[0])
    # np.save('../data/sub_y.npy', train[1])

    # Type: List[data, label]
    # Dimension: data: Sub x Trial x Channel x Length
    return train


# Extract from *.mat files and store in *.npy files
def subject54():

    root = '/home/yk/data/sub54/'
    path = root

    for sub in range(1, 55):
        train = None
        temp_x, temp_y = None, None
        for session in range(1, 3):
            data = scio.loadmat(path + 'sess{:02d}_subj{:02d}_EEG_MI.mat'.format(session, sub))
            x1 = np.moveaxis(data['EEG_MI_train']['smt'][0][0], 0, -1)
            x1 = decimate(x1, 4)
            x2 = np.moveaxis(data['EEG_MI_test']['smt'][0][0], 0, -1)
            x2 = decimate(x2, 4)
            x = np.concatenate((x1, x2), axis=0)
            # Obtain target: 0 -> right, 1 -> left
            y1 = (data['EEG_MI_train']['y_dec'][0][0][0] - 1)
            y2 = (data['EEG_MI_test']['y_dec'][0][0][0] - 1)
            y = np.concatenate((y1, y2), axis=0)[:, np.newaxis]

            if temp_x is None:
                temp_x, temp_y = x, y
            else:
                temp_x, temp_y = np.vstack((temp_x, x)), np.vstack((temp_y, y))

        if train is None:
            train = [temp_x[np.newaxis, :], temp_y[np.newaxis, :]]
        else:
            train[0], train[1] = np.vstack((train[0], temp_x[np.newaxis, :])), np.vstack((train[1], temp_y[np.newaxis, :]))

        # print(sub*2-2+session)
        # Type: List[data, label]
        # Dimension = Trial x Channel x Length
        np.save('../data/sub{:02d}_x.npy'.format(sub), train[0])
        np.save('../data/sub{:02d}_y.npy'.format(sub), train[1])

    # return train


def shorten_sub54():

    path = '/media/yk/ac72589c-5358-4228-bed2-b44d0e56ca00/home/yk/yk/Dataset/Old/sub54/npy_trialxchannelxlength/'
    for sub in range(1, 55):

        x = np.load(path + 'sub{:02d}_x.npy'.format(sub))
        y = np.load(path + 'sub{:02d}_y.npy'.format(sub))

# sub54()


# pos = channel_alignment_position(CHANNELS, INTERSECTION)
# print(pos)
