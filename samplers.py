import torch

class UniformNeighborSampler(object):
    """
        Samples from a "dense 2D edgelist", which looks like

            [
                [1, 2, 3, ..., 1],
                [1, 3, 3, ..., 3],
                ...
            ]

        stored as torch.LongTensor.

        This relies on a preprocessing step where we sample _exactly_ K neighbors
        for each node -- if the node has less than K neighbors, we upsample w/ replacement
        and if the node has more than K neighbors, we downsample w/o replacement.

        This seems like a "definitely wrong" thing to do -- but it runs pretty fast, and
        I don't know what kind of degradation it causes in practice.
    """

    def __init__(self, adj):
        self.adj = adj

    def __call__(self, ids, n_sample=-1):
        tmp = self.adj[ids]
        perm = torch.randperm(tmp.size(1))
        if ids.is_cuda:
            perm = perm.cuda()
        tmp = tmp[:,perm]
        return tmp[:,:n_sample]
