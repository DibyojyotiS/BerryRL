from typing import List
from torch import nn, Tensor
import torch


class SimpleDuelingNet(nn.Module):
    def __init__(
        self, in_features:int, n_actions:int, layers:List[int]=[32,16,16],
        lrelu_negative_slope=-0.01, noise=0.01
    ) -> None:
        super(SimpleDuelingNet, self).__init__()
        assert lrelu_negative_slope <= 0 # must be -ve
        self.n_input_feats = in_features
        self.n_outputs = n_actions
        self.noise_scale = 2*noise

        self.feedforward = self.make_simple_feedforward(
            infeatures= in_features,
            layers=layers,
            lreluslope=lrelu_negative_slope
        )

        self.valuelayer = nn.Linear(layers[-1], 1)
        self.actadv = nn.Linear(layers[-1], n_actions)

    @staticmethod
    def make_simple_feedforward(infeatures, layers:List[int], lreluslope=-1e-2):
        """ layer -> relu -> layer -> relu"""
        # build the feed-forward network
        input_layer = nn.Linear(infeatures, layers[0])
        ffn = nn.ModuleList(
            [input_layer, nn.LeakyReLU(negative_slope=lreluslope)]
        )
        for i in range(1,len(layers)):
            ffn.extend([nn.Linear(layers[i-1], layers[i]), 
                        nn.LeakyReLU(negative_slope=lreluslope)])
        return ffn

    def forward(self, features:Tensor):
        # process feedforward_part
        features = self._add_uniform_noise(features)
        for layer in self.feedforward: 
            features = layer(features)         

        value = self.valuelayer(features)
        advs = self.actadv(features)
        qvalues = value + (advs - advs.mean())

        return qvalues

    def _add_uniform_noise(self, feat:Tensor):
        random_noise = torch.rand(size=feat.shape, device=feat.device)
        return feat + self.noise_scale * (random_noise - 0.5)