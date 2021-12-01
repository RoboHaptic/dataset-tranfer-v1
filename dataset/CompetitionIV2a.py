import scipy.io as scio
import numpy as np
from utils import channel_position


CHANNELS = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz',
            'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz']

INTERSECTION = ['Fz', 'FC3', 'FC1', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz',
                'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz']


def competition_iv_2a():

    root = './data/BCICompetition-IV2a/'
    p_train, p_test = root + 'train/', root + 'test/'

    train, test = None, None
    for sub in range(1, 10):

        data_train = scio.loadmat(p_train + '{}.mat'.format(sub))
        data_test = scio.loadmat(p_test + '{}.mat'.format(sub))

        x1 = np.transpose(data_train.get('trainX'), (2, 1, 0))
        x2 = np.transpose(data_test.get('testX'), (2, 1, 0))
        x = np.vstack((x1, x2))[np.newaxis, :]

        y1 = np.transpose(data_train.get('trainY'), (1, 0))
        y2 = np.transpose(data_test.get('testY'), (1, 0))
        y = np.vstack((y1, y2))[np.newaxis, :]

        for i in range(y.shape[1]-1, 0, -1):
            if (y[0][i][0] != 0) and (y[0][i][0] != 1):
                x, y = np.delete(x, i, 1), np.delete(y, i, 1)

        if train is None:
            train = [x, y]

        else:
            train[0], train[1] = np.vstack((train[0], x)), np.vstack((train[1], y))

    # Intersection channel
    pos = channel_position(CHANNELS, INTERSECTION)
    for i in range(pos.shape[0]-1, -1, -1):
        if pos[i] == 0:
            train[0] = np.delete(train[0], i, 2)

    # Length cut from 1001 to 1000
    train[0] = np.delete(train[0], 0, 3)

    # Type: List[data, label]
    # Dimension: data: Sub x Trial x Channel x Length
    return train


competition_iv_2a()
