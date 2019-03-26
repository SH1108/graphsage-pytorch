import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import functional as F
from scipy import sparse
from sklearn import metrics
from scipy.sparse import csr_matrix

from utils import load_data


class ProblemLosses:
    @staticmethod
    def multilabel_classification(preds, targets):
        return F.multilabel_soft_margin_loss(preds, targets)

    @staticmethod
    def classification(preds, targets):
        return F.cross_entropy(preds, targets)


    @staticmethod
    def regression_mae(preds, targets):
        return F.l1_loss(preds, targets)


class ProblemMetrics:
    @staticmethod
    def multilabel_classification(y_true, y_pred):
        y_pred = (y_pred > 0).astype(int)
        return {
            "micro" : float(metrics.f1_score(y_true, y_pred, average="micro")),
            "macro" : float(metrics.f1_score(y_true, y_pred, average="macro")),
        }

    @staticmethod
    def classification(y_true, y_pred):
        y_pred = np.argmax(y_pred, axis=1)
        return {
            "micro" : float(metrics.f1_score(y_true, y_pred, average="micro")),
            "macro" : float(metrics.f1_score(y_true, y_pred, average="macro")),
        }

    @staticmethod
    def regression_mae(y_true, y_pred):
        return float(np.abs(y_true - y_pred).mean())



class Problem(object):
    def __init__(self, file_path, task, cuda=True, batch_size=100, max_degree=25):

        self.task = task
        self.cuda = False
        self.loss_fn = getattr(ProblemLosses, self.task)
        self.metric_fn = getattr(ProblemMetrics, self.task)

        print('Problem: loading started')
        self.max_degree = max_degree
        self.batch_size = batch_size

    def construct_adj(self):
        raise NotImplementedError()

    def construct_test_adj(self):
        adj = len(self.id2idx)*np.ones((len(self.id2idx)+1, self.max_degree))
        for nodeid in self.G.nodes():
            neighbors = np.array([self.id2idx[neighbor]
                for neighbor in self.G.neighbors(nodeid)])
            if len(neighbors) == 0:
                continue
            if len(neighbors) > self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
            elif len(neighbors) < self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
            adj[self.id2idx[nodeid], :] = neighbors
        return adj

    def _to_torch(self):
        if not sparse.issparse(self.val_adj):
            self.val_adj = Variable(torch.LongTensor(self.val_adj))
            self.train_adj = Variable(torch.LongTensor(self.train_adj))
            if self.cuda:
                self.adj = self.val_adj.cuda()
                self.train_adj = self.train_adj.cuda()

        if self.features is not None:
            self.features = Variable(torch.FloatTensor(self.features))
            if self.cuda:
                self.features = self.features.cuda()

    def _batch_to_torch(self, mids, targets):
        """ convert batch to torch """
        mids = Variable(torch.LongTensor(mids))
        # print(targets)
        if self.task == 'multilabel_classification':
            targets = Variable(torch.FloatTensor(targets))
        elif self.task == 'classification':
            targets = Variable(torch.LongTensor(targets))
        elif 'regression' in self.task:
            targets = Variable(torch.FloatTensor(targets))
        else:
            raise Exception('NodeDataLoader: unknown task: %s' % self.task)

        if self.cuda:
            mids, targets = mids.cuda(), targets.cuda()

        return mids, targets

    def iterate(self, mode, batch_size=512, shuffle=False):
        nodes = self.nodes[mode]
        idx = np.arange(nodes.shape[0])
        if shuffle:
            idx = np.random.permutation(idx)

        n_chunks = idx.shape[0] // batch_size + 1
        """
        chunk: batch id
        chunkd id: 1, 2,..., n
        """
        for chunk_id, chunk in enumerate(np.array_split(idx, n_chunks)):
            mids = nodes[chunk]
            targets = np.concatenate(self.targets[mids]).reshape(len(mids), -1)
            mids, targets = self._batch_to_torch(mids, targets)
            yield mids, targets, chunk_id / n_chunks

class NodeProblem(Problem):
    def __init__(self, file_path, task, cuda=True, batch_size=100, max_degree=25):
        self.G, self.id2idx, self.id2target, self.features = load_data(file_path)
        self.max_degree = max_degree
        self.batch_num = 0

        Problem.__init__(self, file_path, task, cuda=True, batch_size=100, max_degree=25)


        self.targets = self.id2target.values

        self.n_dim = self.features.shape[1] if self.features is not None else None


        if isinstance(list(self.id2target.values[0]), list):
            self.n_class = len(list(self.id2target.values)[0])
        else:
            self.n_class = len(set(self.id2target.values))

        self.train_adj, self.deg = self.construct_adj()
        self.val_adj = self.construct_test_adj()

        self.n_node   = self.train_adj.shape[0]
        self.val_nodes = [n for n in self.G.nodes() if self.G.node[n]['val']]
        self.test_nodes = [n for n in self.G.nodes() if self.G.node[n]['test']]

        self.no_train_nodes_set = set(self.val_nodes + self.test_nodes)
        self.train_nodes = set(self.G.nodes()).difference(self.no_train_nodes_set)

        # don't train on nodes that only have edges to test set
        self.train_nodes = [n for n in self.train_nodes if self.deg[self.id2idx[n]] > 0]

        self.nodes = {
            'train': np.array(self.train_nodes),
            'val': np.array(self.val_nodes),
            'test': np.array(self.test_nodes),
        }

        print(self.nodes['train'].shape)
        print(self.nodes['val'].shape)
        print(self.nodes['test'].shape)

        self._to_torch()

        print('NodeProblem: loading finished')

    def construct_adj(self):
        adj = len(self.id2idx)*np.ones((len(self.id2idx)+1, self.max_degree))
        deg = np.zeros((len(self.id2idx),))

        for nodeid in self.G.nodes():
            if self.G.node[nodeid]['test'] or self.G.node[nodeid]['val']:
                continue
            neighbors = np.array([self.id2idx[neighbor]
                for neighbor in self.G.neighbors(nodeid)
                if (not self.G[nodeid][neighbor]['train_removed'])])
            deg[self.id2idx[nodeid]] = len(neighbors)
            if len(neighbors) == 0:
                continue
            if len(neighbors) > self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
            elif len(neighbors) < self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
            adj[self.id2idx[nodeid], :] = neighbors
        return adj, deg
