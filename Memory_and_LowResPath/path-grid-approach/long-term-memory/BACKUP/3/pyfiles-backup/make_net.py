from torch import nn
import torch.nn.functional as F
import torch

def make_simple_feedforward(infeatures, linearsDim = [32,16]):
        # build the feed-forward network
        return nn.ModuleList([nn.Linear(infeatures, linearsDim[0]), # input layer
                            *[nn.Linear(linearsDim[i-1], linearsDim[i]) 
                                for i in range(1,len(linearsDim))]
                            ])

def make_simple_convnet(inchannel:int, channels:list, kernels:list, strides:list, paddings:list, maxpkernels:list):
    """ odd layers are max-pools """
    conv = nn.ModuleList([nn.Conv2d(inchannel, channels[0], kernels[0], strides[0], padding=paddings[0])])
    for i in range(1, len(channels)):
        conv.extend([
            nn.MaxPool2d(maxpkernels[i-1]),
            nn.Conv2d(channels[i-1], channels[i], kernels[i], strides[i], padding=paddings[i]),
        ])
    if len(maxpkernels) == len(channels):
        conv.append(nn.MaxPool2d(maxpkernels[-1]))
    return conv