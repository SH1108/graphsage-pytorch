#!/usr/bin/env python

"""
    train.py
"""

from __future__ import division
from __future__ import print_function

import sys
sys.setrecursionlimit(10000)
import argparse
import ujson as json
import numpy as np
from time import time
from tqdm import tqdm

import torch
from torch.autograd import Variable
from torch.nn import functional as F

from supervisedgraphsage import SupervisedGraphsage
from problem import NodeProblem
from aggregators import aggregator_lookup
from preps import prep_lookup
from samplers import UniformNeighborSampler

def set_seeds(seed=0):
    np.random.seed(seed)
    _ = torch.manual_seed(seed)
    if torch.cuda.is_available():
        _ = torch.cuda.manual_seed(seed)

def to_numpy(x):
    if isinstance(x, Variable):
        x = x.data

    return x.cpu().numpy() if x.is_cuda else x.numpy()

def evaluate(model, problem, mode='val'):
    assert mode in ['test', 'val']
    preds, acts = [], []
    for (ids, targets, _) in problem.iterate(mode=mode, shuffle=False):
        preds.append(to_numpy(model(ids, problem.feats, train=False)))
        acts.append(to_numpy(targets))

    return problem.metric_fn(np.vstack(acts), np.vstack(preds))

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--file_path', type=str, required=True)
    parser.add_argument('--no-cuda', action="store_true")

    # parser.add_argument('--task', type=str, default='classification')
    parser.add_argument('--task', type=str, default='multilabel_classification')

    # Optimization params
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lr-schedule', type=str, default='constant')
    parser.add_argument('--weight-decay', type=float, default=0.0)

    # Architecture params
    parser.add_argument('--sampler-class', type=str, default='uniform_neighbor_sampler')
    parser.add_argument('--aggregator-class', type=str, default='mean')
    parser.add_argument('--prep-class', type=str, default='linear')

    parser.add_argument('--n-train-samples', type=str, default='25,10')
    parser.add_argument('--n-val-samples', type=str, default='25,10')
    parser.add_argument('--output-dims', type=str, default='256,128')

    parser.add_argument('--max_degree', type=int, default='10')

    # Logging
    parser.add_argument('--log-interval', default=10, type=int)
    parser.add_argument('--seed', default=123, type=int)
    parser.add_argument('--show-test', action="store_true")

    # --
    # Validate args

    args = parser.parse_args()
    args.cuda = not args.no_cuda
    # assert args.prep_class in prep_lookup.keys(), 'parse_args: prep_class not in %s' % str(prep_lookup.keys())
    assert args.aggregator_class in aggregator_lookup.keys(), 'parse_args: aggregator_class not in %s' % str(aggregator_lookup.keys())
    assert args.batch_size > 1, 'parse_args: batch_size must be > 1'
    return args


if __name__ == "__main__":
    args = parse_args()
    set_seeds(args.seed)

    problem = NodeProblem(args.file_path, cuda=True, batch_size=args.batch_size, max_degree=args.max_degree, task=args.task)
    train_sampler = UniformNeighborSampler(problem.train_adj)
    val_sampler = UniformNeighborSampler(problem.val_adj)

    n_train_samples = list(map(int, args.n_train_samples.split(',')))
    n_val_samples = list(map(int, args.n_val_samples.split(',')))
    out_dims = list(map(int, args.output_dims.split(',')))

    in_dim = problem.n_dim
    n_node = problem.n_node

    layer_infos = [
        {
            "n_train_samples" : n_train_samples[0],
            "n_val_samples" : n_val_samples[0],
            "out_dim" : out_dims[0],
            "activation" : F.relu,
        },
        {
            "n_train_samples" : n_train_samples[1],
            "n_val_samples" : n_val_samples[1],
            "out_dim" : out_dims[1],
            "activation" : lambda x: x,
        },
    ]

    model = SupervisedGraphsage(problem.n_class,
                                 problem.train_adj,
                                 problem.val_adj,
                                 lr=args.lr,
                                 in_dim=in_dim,
                                 n_node=n_node,
                                 aggregator=aggregator_lookup[args.aggregator_class],
                                 prep=prep_lookup[args.prep_class],
                                 train_sampler=train_sampler,
                                 val_sampler=val_sampler,
                                 layer_infos=layer_infos,
                                 sigmoid_loss = False)

    if args.cuda:
        model = model.cuda()
    # import IPython
    # IPython.embed()
    print(model, file=sys.stderr)

    set_seeds(args.seed ** 2)

    start_time = time()
    val_metric = None
    for epoch in range(args.epoch):

        # Train
        model.train()
        allloss = 0
        for ids1, ids2, labels, epoch_progress in tqdm(problem.iterate(mode='train', shuffle=True, batch_size=args.batch_size)):
            preds = model.predict(
                ids=ids,
                features=problem.features,
                labels=labels,
                loss_fun=problem.loss_fn,
            )
            ## Eval
            model.eval()

            train_metric = problem.metric_fn(to_numpy(labels), to_numpy(preds))
            # Evaluate
            model.eval()
        val_metric = evaluate(model, problem, mode='val')
        print(json.dumps({
            "epoch" : epoch,
            "epoch_progress" : epoch_progress,
            "train_metric" : train_metric,
            "val_metric" : val_metric,
            "time" : time() - start_time,
        }, double_precision=5))
        sys.stdout.flush()
