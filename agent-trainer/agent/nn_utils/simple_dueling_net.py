from typing import List
from torch import nn, Tensor


class SimpleDuelingNet(nn.Module):
    def __init__(
        self, in_features:int, n_actions:int, layers:List[int]=[32,16,16],
        lrelu_negative_slope=-0.01
    ) -> None:
        super(SimpleDuelingNet, self).__init__()
        self.n_input_feats = in_features
        self.n_outputs = n_actions

        self.feedforward = self.make_simple_feedforward(
            infeatures= in_features,
            layers=layers,
            lreluslope=lrelu_negative_slope
        )

        self.valuelayer = nn.Linear(layers[-1], 1)
        self.actadv = nn.Linear(layers[-1], n_actions)

    @staticmethod
    def make_simple_feedforward(infeatures, layers:List[int], lreluslope=1e-2):
        """ layer -> relu -> layer -> relu"""
        # build the feed-forward network
        input_layer = nn.Linear(infeatures, layers[0])
        ffn = nn.ModuleList(
            [input_layer, nn.LeakyReLU(negative_slope=-lreluslope)]
        )
        for i in range(1,len(layers)):
            ffn.extend([nn.Linear(layers[i-1], layers[i]), 
                        nn.LeakyReLU(negative_slope=lreluslope)])
        return ffn

    def forward(self, fetures:Tensor):
        # process feedforward_part
        for layer in self.feedforward: 
            fetures = layer(fetures)         

        value = self.valuelayer(fetures)
        advs = self.actadv(fetures)
        qvalues = value + (advs - advs.mean())

        return qvalues