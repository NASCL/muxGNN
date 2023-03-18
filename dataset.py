#!/usr/bin/env python3

from dgl.data import DGLDataset


class GraphDataset(DGLDataset):
    def __init__(self, name, graphs, labels):
        super(GraphDataset, self).__init__(name=name)
        self.graphs = graphs
        self.labels = labels

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

    def get_graphs(self):
        return self.graphs

    def get_labels(self):
        return self.labels

    def process(self):
        pass
