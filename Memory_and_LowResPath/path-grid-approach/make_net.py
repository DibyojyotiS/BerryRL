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


def example_net(inDim, outDim, hDim, output_probs=False, TORCH_DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    class net(nn.Module):
        def __init__(self, inDim, outDim, hDim, activation = F.relu):
            super(net, self).__init__()
            self.outDim = outDim
            self.inputlayer = nn.Linear(inDim, hDim[0])
            self.hiddenlayers = nn.ModuleList([nn.Linear(hDim[i], hDim[i+1]) for i in range(len(hDim)-1)])
            self.outputlayer = nn.Linear(hDim[-1], outDim)
            self.activation = activation
            if outDim > 1 and not output_probs:
                self.actadvs = nn.Linear(hDim[-1], outDim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            t = self.activation(self.inputlayer(x))
            for layer in self.hiddenlayers:
                t = self.activation(layer(t))
            o = self.outputlayer(t) # value or probs
            if output_probs: o = F.log_softmax(o, -1)
            elif self.outDim > 1:
                advs = self.actadvs(t)
                o = o + (advs - advs.mean())
            return o
    
    netw = net(inDim, outDim, hDim)
    netw.to(TORCH_DEVICE)
    return netw

# example
if __name__ == "__main__":
    nnet = example_net(
        inDim = 4,
        outDim = 2,
        hDim = [8,8]
    )