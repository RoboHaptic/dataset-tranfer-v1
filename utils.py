import torch
from args import args
import numpy as np
# from tsnecuda import TSNE
# import matplotlib.pyplot as plt


def visualizer(data, label, batch=None):

    print('T-SNE starts!')

    data, label = data.cpu().data.numpy(), label
    x_embedded = TSNE(n_components=2, perplexity=15, learning_rate=10).fit_transform(data)
    plt.figure(figsize=(12, 8))

    label = list(label.squeeze(1))
    scatter = plt.scatter(x_embedded[:, 0], x_embedded[:, 1], s=10, c=label, cmap='rainbow')
    plt.legend(handles=scatter.legend_elements()[0], labels=[str(i) for i in range(label[len(label)-1]+1)])
    plt.show()
    # if batch is not None:
    #     plt.savefig('visualization/{}.png'.format(batch), bbox_inches='tight')


def to_tensor(data, label):

    return torch.tensor(data, dtype=torch.float, device=torch.device('cuda:{}'.format(args.gpu))), \
           torch.tensor(label, device=torch.device('cuda:{}'.format(args.gpu)))


def acc(pred, label):

    count = torch.tensor(0)
    for i in range(label.shape[0]):
        if torch.argmax(pred[i]) == label[i]:
            count += 1

    return count / label.shape[0]


def channel_position(channel, intersection):

    pos = np.zeros((len(channel)))
    count, i = 0, 0
    while count < len(intersection):
        if channel[i] == intersection[count]:
            pos[i] = 1
            count += 1
        i += 1
    return pos


def channel_alignment_position(channel, intersection):

    pos = np.zeros((len(channel)))-1

    for i in range(len(intersection)):
        for j in range(len(channel)):
            if channel[j] == intersection[i]:
                pos[j] = i

    return pos


def channel_selection(pos, train, intersection):

    train[0] = np.transpose(train[0], (2, 1, 0, 3))

    data = None
    for i in range(len(intersection)):
        for j in range(len(pos)):
            if pos[j] == i:
                if data is None:
                    data = train[0][j][np.newaxis, :]
                else:
                    data = np.vstack((data, train[0][j][np.newaxis, :]))

    data = np.transpose(data, (2, 1, 0, 3))
    return [data, train[1]]


class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        # self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1, 3, 1, 1)
        # self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1, 3, 1, 1)
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            self.next_input = self.next_input.float()
            # self.next_input = self.next_input.sub_(self.mean).div_(self.std)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target


