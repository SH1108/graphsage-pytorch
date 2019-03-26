import torch
import torch.nn as nn
import torch.nn.functional as F

class MeanAggregator(nn.Module):
    def __init__(self, in_dim, out_dim, activation):
        super(MeanAggregator, self).__init__()

        ## use for defining next aggregator
        self.out_dim = out_dim

        self.fc_self = nn.Linear(in_dim, out_dim)
        self.fc_neighbors = nn.Linear(in_dim, out_dim)

        self.fc = nn.Linear(out_dim*2, out_dim)

        # self.out_dim_ = out_dim
        self.activation = activation

    def forward(self, x, neighbors):
        """
        x: (n_sampled_node, n_dim)
        neighbors: (n_sampled_node*n_sample, n_dim)
        """
        agg_neighbors = neighbors.view(x.size(0), -1, neighbors.size(1)) # !! Careful
        agg_neighbors = agg_neighbors.mean(dim=1) # Careful

        out = torch.cat((self.fc_self(x), self.fc_neighbors(agg_neighbors)), dim=1)
        out = self.fc(out)
        # if self.activation:
        out = self.activation(out)
        out = F.normalize(out, dim=1)
        return out

aggregator_lookup = {
    "mean" : MeanAggregator,
    # "max_pool" : MaxPoolAggregator,
    # "mean_pool" : MeanPoolAggregator,
    # "lstm" : LSTMAggregator,
    # "attention" : AttentionAggregator,
}
