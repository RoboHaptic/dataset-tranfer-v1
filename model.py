import torch
import torch.nn as nn


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(Conv2dWithConstraint, self).forward(x)


class EEGNet(nn.Module):
    def __init__(self, n_classes=4, n_subjects=9, channels=60, samples=151,
                 dropoutRate=0.5, kernelLength=64, kernelLength2=16, F1=8,
                 D=2, F2=16):
        super(EEGNet, self).__init__()
        self.F1 = F1
        self.F2 = F2
        self.D = D
        self.samples = samples
        self.n_classes = n_classes
        self.channels = channels
        self.kernelLength = kernelLength
        self.kernelLength2 = kernelLength2
        self.dropoutRate = dropoutRate

        self.blocks = self.InitialBlocks(dropoutRate)
        self.blockOutputSize = self.CalculateOutSize(self.blocks, channels, samples)
        self.linear = self.LinearBlock(self.F2 * self.blockOutputSize[1])
        self.cls = self.ClassBlock(n_classes)
        self.subject = self.SubjectBlock(n_subjects)

    def InitialBlocks(self, dropoutRate, *args, **kwargs):
        block1 = nn.Sequential(
            nn.Conv2d(1, self.F1, (1, self.kernelLength), stride=1, padding=(0, self.kernelLength // 2), bias=False),
            nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3),

            # DepthwiseConv2D =======================
            Conv2dWithConstraint(self.F1, self.F1 * self.D, (self.channels, 1), max_norm=1, stride=1, padding=(0, 0),
                                 groups=self.F1, bias=False),
            # ========================================

            nn.BatchNorm2d(self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d((1, 4), stride=4),
            nn.Dropout(p=dropoutRate))
        block2 = nn.Sequential(
            # SeparableConv2D =======================
            nn.Conv2d(self.F1 * self.D, self.F1 * self.D, (1, self.kernelLength2), stride=1,
                      padding=(0, self.kernelLength2 // 2), bias=False, groups=self.F1 * self.D),
            nn.Conv2d(self.F1 * self.D, self.F2, 1, padding=(0, 0), groups=1, bias=False, stride=1),
            # ========================================

            nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d((1, 8), stride=8),
            nn.Dropout(p=dropoutRate))
        return nn.Sequential(block1, block2)

    def LinearBlock(self, inputSize):
        return nn.Sequential(
            nn.Linear(inputSize, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.ReLU())

    def ClassBlock(self, n_classes):
        return nn.Sequential(
            nn.Linear(128, n_classes),
            nn.Softmax(dim=1))

    def SubjectBlock(self, n_classes):
        return nn.Sequential(
            nn.Linear(128, n_classes),
            nn.Softmax(dim=1))

    def CalculateOutSize(self, model, channels, samples):
        '''
        Calculate the output based on input size. -
        model is from nn.Module and inputSize is a array.
        '''
        data = torch.rand(1, 1, channels, samples)
        model.eval()
        out = model(data).shape
        return out[2:]

    def forward(self, x):
        x = self.blocks(x)
        x = x.view(x.size()[0], -1)  # Flatten
        x = self.linear(x)
        feature = x.detach()
        cls = self.cls(x)
        sub = self.subject(x)
        return cls, feature, sub
