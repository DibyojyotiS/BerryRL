from torch import nn
import torch.nn.functional as F
import torch

def make_simple_feedforward(infeatures, linearsDim = [32,16]):
    """ layer -> relu -> layer"""
    # build the feed-forward network
    input_layer = nn.Linear(infeatures, linearsDim[0])
    ffn = nn.ModuleList([input_layer])
    for i in range(1,len(linearsDim)):
        ffn.extend([nn.ReLU(),nn.Linear(linearsDim[i-1], linearsDim[i])])
    return ffn

def make_simple_conv1dnet(inchannel:int, channels:list, kernels:list, strides:list, 
                        paddings:list, maxpkernels:list, padding_mode='zeros'):
    """ odd layers are max-pools {conv,maxpool,relu,conv, [?maxpool?]}"""
    conv = nn.ModuleList([nn.Conv1d(inchannel, channels[0], kernels[0], strides[0], 
                                    paddings[0], padding_mode=padding_mode)])
    for i in range(1, len(channels)):
        if len(maxpkernels) > i-1 and maxpkernels[i-1]:
            conv.append(nn.MaxPool1d(maxpkernels[i-1]))
        conv.extend([
            nn.ReLU(),
            nn.Conv1d(channels[i-1], channels[i], kernels[i], 
                    strides[i], paddings[i], padding_mode=padding_mode)
        ])
    if len(maxpkernels) == len(channels): conv.append(nn.MaxPool1d(maxpkernels[-1]))
    return conv

def make_simple_conv2dnet(inchannel:int, channels:list, kernels:list, strides:list, 
                        paddings:list, maxpkernels:list, padding_mode='zeros'):
    """ odd layers are max-pools {conv,maxpool,relu,conv, [?maxpool?]}"""
    conv = nn.ModuleList([nn.Conv2d(inchannel, channels[0], kernels[0], strides[0], 
                                    paddings[0], padding_mode=padding_mode)])
    for i in range(1, len(channels)):
        if len(maxpkernels) > i-1 and maxpkernels[i-1]:
            conv.append(nn.MaxPool2d(maxpkernels[i-1]))
        conv.extend([ 
            nn.ReLU(), 
            nn.Conv2d(channels[i-1], channels[i], kernels[i], 
                    strides[i], paddings[i], padding_mode=padding_mode)
        ])
    if len(maxpkernels) == len(channels): conv.append(nn.MaxPool2d(maxpkernels[-1]))
    return conv