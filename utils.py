#!/usr/bin/env python3

from collections import defaultdict
from functools import partial, reduce
import json
import math
import multiprocessing
import random

import dgl
import networkx as nx
import numpy as np
import torch

from dataset import GraphDataset


def load_linkpred_dataset(dataset, holdout_split=None):
    if holdout_split is not None:
        g_name = '10layer'
        train_G, val_edges, test_edges = load_cv_split_data(dataset, g_name, holdout_split)
    else:
        train_G, _ = dgl.load_graphs(f'data/{dataset}/train_G.bin')
        train_G = train_G[0]

        with open(f'data/{dataset}/val_edges.json') as f:
            val_edges = json.load(f)
        with open(f'data/{dataset}/test_edges.json') as f:
            test_edges = json.load(f)

    return train_G, val_edges, test_edges


def load_graphpred_dataset(dataset, cv_fold=None, bidirected=False):
    assert dataset in ['wget', 'streamspot'], 'Invalid dataset.'

    data_path = f'data/{dataset}'
    graph_path = f'{data_path}/{dataset}_dgl_graphs.bin'

    graphs, graph_attr = dgl.load_graphs(graph_path)
    with open(f'{data_path}/{dataset}_relations.txt') as f:
        relations = [line.strip() for line in f]

    feat_dim = graphs[0].ndata['feat'].size(-1)

    if bidirected:
        graphs = [
            dgl.to_bidirected(g, copy_ndata=True)
            for g in graphs
        ]

    if cv_fold is None:
        train_dataset = GraphDataset(dataset, graphs, graph_attr['label'].float())
        test_dataset = None
    else:
        train_graphs, test_graphs = [], []
        train_labels, test_labels = [], []
        for g, cv, lbl in zip(graphs, graph_attr['cv_fold'], graph_attr['label']):
            if cv == cv_fold:
                test_graphs.append(g)
                test_labels.append(lbl)
            else:
                train_graphs.append(g)
                train_labels.append(lbl)

        train_dataset = GraphDataset(dataset, train_graphs, torch.FloatTensor(train_labels))
        test_dataset = GraphDataset(dataset, test_graphs, torch.FloatTensor(test_labels))

    return train_dataset, test_dataset, feat_dim, relations


def load_cv_split_data(dataset, g_name, holdout_split):
    G = nx.read_gpickle(f'data/{dataset}/{g_name}_network.gpickle')

    graph_data = defaultdict(lambda: ([], []))
    val_edges = defaultdict(lambda: ([], [], []))
    test_edges = defaultdict(lambda: ([], [], []))
    for src, dst, data in G.edges(data=True):
        edge_schema = ('node', data['etype'], 'node')
        # Avoid removing the only edge for src/dst when creating train graph
        if data['split'] == holdout_split and not (G.degree(src) == 1 or G.degree(dst) == 1):
            if random.random() < 0.3:
                val_edges[data['etype']][0].append(src)
                val_edges[data['etype']][1].append(dst)
                val_edges[data['etype']][2].append(1)  # edge label
            else:
                test_edges[data['etype']][0].append(src)
                test_edges[data['etype']][1].append(dst)
                test_edges[data['etype']][2].append(1)  # edge label

        else:
            graph_data[edge_schema][0].append(src)
            graph_data[edge_schema][1].append(dst)

    with open(f'data/{dataset}/feat_dict.json') as f:
        feat_dict = json.load(f)
        feat_dict = {int(k): v for k, v in feat_dict.items()}

    train_G = dgl.heterograph(graph_data)
    train_G = dgl.add_reverse_edges(train_G)
    train_G = dgl.to_simple(train_G)

    train_G.ndata['feat'] = torch.FloatTensor(
        [feat_dict[node.item()] for node in train_G.nodes()]
    )

    with open(f'data/{dataset}/neg_edges.json') as f:
        neg_edges = json.load(f)
        neg_edges = {int(k): v for k, v in neg_edges.items()}

    neg_edges = neg_edges[holdout_split]
    for etype in neg_edges:
        layer_neg_edges = neg_edges[etype]

        for src, dst in layer_neg_edges:
            if random.random() < 0.3:
                val_edges[etype][0].append(src)
                val_edges[etype][1].append(dst)
                val_edges[etype][2].append(0)  # edge label
            else:
                test_edges[etype][0].append(src)
                test_edges[etype][1].append(dst)
                test_edges[etype][2].append(0)  # edge label

    return train_G, val_edges, test_edges


def load_inductive_data():
    dgl_graphs, _ = dgl.load_graphs('data/tissue_ppi/inductive_graphs.bin')

    with open('data/tissue_ppi/inductive_val_edges.json') as f:
        val_edges = json.load(f)
    with open('data/tissue_ppi/inductive_test_edges.json') as f:
        test_edges = json.load(f)

    return dgl_graphs, val_edges, test_edges


def generate_walks(G, num_walks=20, walk_length=10):
    nodes_by_etype = defaultdict(set)
    for etype in G.etypes:
        src, dst = G.edges(etype=etype)
        nodes_by_etype[etype].update(src.tolist())
        nodes_by_etype[etype].update(dst.tolist())

    all_walks = []
    for etype in G.etypes:
        nodes = torch.LongTensor(list(nodes_by_etype[etype]) * num_walks)
        traces, types = dgl.sampling.random_walk(
            G, nodes,
            metapath=[etype] * (walk_length-1)
        )
        all_walks.append(traces)

    return all_walks


def generate_pairs_parallel(walks, skip_window=None, etype_idx=None):
    pairs = []
    for walk in walks:
        walk = walk.tolist()
        for i in range(len(walk)):
            for j in range(1, skip_window + 1):
                if i - j >= 0:
                    pairs.append((walk[i], etype_idx, walk[i - j]))
                if i + j < len(walk):
                    pairs.append((walk[i], etype_idx, walk[i + j]))
    return pairs


def generate_pairs(walks, window_size=5, num_workers=1):
    # For each node, create pairs with its adjacent neighbors (window_size // 2)
    # (<node>, <node + 1>, <etype>), (<node>, <node + 2>, <etype>)
    pool = multiprocessing.Pool(processes=num_workers)

    pairs = []
    skip_window = window_size // 2
    for etype_idx, walks in enumerate(walks):
        block_num = math.ceil(len(walks) / num_workers)
        if block_num > 0:
            walks_list = [
                walks[i * block_num: min((i + 1) * block_num, len(walks))]
                for i in range(num_workers)
            ]
        else:
            walks_list = [walks]

        tmp_result = pool.map(
            partial(
                generate_pairs_parallel, skip_window=skip_window, etype_idx=etype_idx
            ),
            walks_list,
        )

        pairs += reduce(lambda x, y: x + y, tmp_result)

    pool.close()

    # Train on each unique tuple: (node, node, etype)
    return np.array([list(pair) for pair in set(pairs) if -1 not in pair])


def get_score(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
