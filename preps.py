import torch
from torch import nn
# from torch.nn import functional as F
# from torch.autograd import Variable

class LinearPrep(nn.Module):
    def __init__(self, in_dim, n_node, out_dim=32):
        """ adds node embedding """
        super(LinearPrep, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.out_dim = out_dim

    def forward(self, ids, features, layer_idx=0):
        return self.fc(features)

prep_lookup = {
    # "identity" : IdentityPrep,
    # "node_embedding" : NodeEmbeddingPrep,
    "linear" : LinearPrep,
}
