import numpy as np
import pandas as pd
import os
import json
from networkx.readwrite import json_graph
import networkx as nx
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm as tqdm


def load_data(file_path):
    ## node id to node idx
    id2idx = pd.Series(json.load(open(os.path.join(file_path, 'toy-ppi-id_map.json'))))

    ## node id to node target
    id2target = pd.Series(json.load(open(os.path.join(file_path, 'toy-ppi-class_map.json'))))

    ## node features
    features = np.load(os.path.join(file_path, 'toy-ppi-feats.npy'))

    ## construct graph
    G = json_graph.node_link_graph(json.load(open(os.path.join(file_path, 'toy-ppi-G.json'))))
    for edge in G.edges():
        if (G.node[edge[0]]['val'] or G.node[edge[1]]['val'] or
            G.node[edge[0]]['test'] or G.node[edge[1]]['test']):
            G[edge[0]][edge[1]]['train_removed'] = True
        else:
            G[edge[0]][edge[1]]['train_removed'] = False

    ## normalize
    train_ids = np.array([id2idx[n] for n in G.nodes() if not G.node[n]['val'] and not G.node[n]['test']])
    train_feats = features[train_ids]
    train_ids = np.array([id2idx[n] for n in G.nodes() if not G.node[n]['val'] and not G.node[n]['test']])
    train_features = features[train_ids]
    scaler = StandardScaler()
    scaler.fit(train_features)
    features = scaler.transform(features)

    return G, id2idx, id2target, features
