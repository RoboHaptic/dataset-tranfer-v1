from torch.utils.data import Dataset
from torchvision import transforms, utils
import numpy as np
from dataset.CompetitionIV2a import competition_iv_2a
from dataset.Subject54 import sub54


class OmniSet(Dataset):
    def __init__(self, dataset):

        # List: [data, label]
        if dataset == 'sub54':
            self.data = sub54()
        elif dataset == 'iv_2a':
            self.data = competition_iv_2a()

        [self.x, self.y] = self.data
        self.y = self.y.astype(np.int64)

        self.num_trial = 0
        self.sub, self.idx = 0, 0

    def __len__(self):
        return self.num_trial * self.y.shape[0]

    def __getitem__(self, index):

        # sub, idx = int(index / self.num_trial), index % self.num_trial
        temp = self.sub
        signal, label = self.x[self.sub][self.idx], self.y[self.sub][self.idx]
        if (self.idx + 1) == self.num_trial:
            self.sub += 1
            self.idx = 0
        else:
            self.idx += 1

        if (index + 1) == self.__len__():
            self.sub, self.idx = 0, 0

        return temp, signal[np.newaxis, :], label


class TrainingSet(OmniSet):
    def __init__(self, dataset, test_sub=None, valid_sub=None):
        OmniSet.__init__(self, dataset)

        if valid_sub is not None:
            if valid_sub < test_sub:
                self.x = np.delete(self.x, test_sub, axis=0)
                self.y = np.delete(self.y, test_sub, axis=0)
                self.x = np.delete(self.x, valid_sub, axis=0)
                self.y = np.delete(self.y, valid_sub, axis=0)
            else:
                self.x = np.delete(self.x, valid_sub, axis=0)
                self.y = np.delete(self.y, valid_sub, axis=0)
                self.x = np.delete(self.x, test_sub, axis=0)
                self.y = np.delete(self.y, test_sub, axis=0)

        self.num_trial = self.y.shape[1]
        # self.x, self.y = self.to_tensor(self.x, self.y)


class TestSet(OmniSet):
    def __init__(self, dataset, test_sub):
        OmniSet.__init__(self, dataset)

        self.x, self.y = self.x[test_sub][np.newaxis, :], self.y[test_sub][np.newaxis, :]
        # self.x, self.y = self.to_tensor(self.x, self.y)
        # self.x = self.x.unsqueeze(0)

        self.num_trial = self.y.shape[1]


class ValidSet(OmniSet):
    def __init__(self, dataset, valid_sub):
        OmniSet.__init__(self, dataset)

        self.x, self.y = self.x[valid_sub][np.newaxis, :], self.y[valid_sub][np.newaxis, :]
        # self.x, self.y = self.to_tensor(self.x, self.y)
        # self.x = self.x.unsqueeze(0)

        self.num_trial = self.y.shape[1]
