
from aggregators import MeanAggregator, aggregator_lookup
# , MaxPoolingAggregator, MeanPoolingAggregator, SeqAggregator, GCNAggregator

from functools import partial
import torch
import torch.nn as nn


class SupervisedGraphsage(torch.nn.Module):
    """Implementation of supervised GraphSAGE with PyTorch"""

    def __init__(self, n_class, train_adj, val_adj, in_dim, n_node, train_sampler, val_sampler, lr, layer_infos, aggregator, prep, depth=2, aggregator_type="mean",
            model_size="small", sigmoid_loss=False, identity_dim=0):

        super(SupervisedGraphsage, self).__init__()

        # Prep
        self.prep = prep(in_dim=in_dim, n_node=n_node)
        in_dim = self.prep.out_dim

        ## Network definition
        agg_layers = []
        for info in layer_infos:
            agg = aggregator(
                in_dim=in_dim,
                out_dim=info['out_dim'],
                activation=info['activation'],
            )
            agg_layers.append(agg)
            in_dim = agg.out_dim
        self.agg_layers = nn.Sequential(*agg_layers)
        self.fc = nn.Linear(in_dim, n_class, bias=True)

        self.train_sampler = train_sampler
        self.val_sampler = val_sampler

        self.train_sampler_call = [partial(self.train_sampler, n_sample=s['n_train_samples']) for s in layer_infos]
        self.val_sampler_call = [partial(self.val_sampler, n_sample=s['n_val_samples']) for s in layer_infos]

        self.lr = lr
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, ids, features, train=True):
        sampler = self.train_sampler_call if train else self.val_sampler_call
        temp_features = features[ids]
        all_features = [self.prep(ids, temp_features, layer_idx=0)]
        for layer_idx, sampler in enumerate(sampler):
            ids = sampler(ids=ids).contiguous().view(-1)
            temp_features = features[ids]
            all_features.append(self.prep(ids, temp_features, layer_idx=layer_idx + 1))

        for agg_layer in self.agg_layers.children():
            all_features = [agg_layer(all_features[k], all_features[k + 1]) for k in range(len(all_features) - 1)]

        assert len(all_features) == 1, "len(all_feats) != 1"

        # out = F.normalize(all_feats[0], dim=1) # ?? Do we actually want this? ... Sometimes ...
        out = all_features[0]
        return self.fc(out)

    def predict(self, ids, features, labels, loss_fun):
        self.optimizer.zero_grad()
        preds = self(ids, features, train=True)
        loss = loss_fun(preds, labels.squeeze())
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.parameters(), 5)
        self.optimizer.step()
        return preds
