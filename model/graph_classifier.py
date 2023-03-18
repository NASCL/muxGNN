
import os
import logging
from time import time

import dgl
from dgl.nn.pytorch import GATConv, GINConv, GraphConv, SumPooling
from dgl.dataloading import GraphDataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
from torch import nn, optim
import torch.nn.functional as F

from .mux_gnn import SemanticAttentionBatched


class GraphClassifier(nn.Module):
    def __init__(
            self,
            gnn_type,
            num_gnn_layers,
            relations,
            feat_dim,
            embed_dim,
            dim_a,
            dropout=0.,
            activation=None
    ):
        super(GraphClassifier, self).__init__()
        self.gnn_type = gnn_type
        self.num_gnn_layers = num_gnn_layers
        self.relations = relations
        self.num_relations = len(relations)

        self.feat_dim = feat_dim
        self.embed_dim = embed_dim
        self.dim_a = dim_a

        self.dropout = dropout
        self.activation = activation.casefold()

        self.embedder = MuxGNNGraph(
            gnn_type=self.gnn_type,
            num_gnn_layers=self.num_gnn_layers,
            relations=self.relations,
            feat_dim=self.feat_dim,
            embed_dim=self.embed_dim,
            dim_a=self.dim_a,
            dropout=self.dropout,
            activation=self.activation,
        )
        self.classifier = BinaryClassifier(self.embed_dim)

    def forward(self, graph):
        feat = graph.ndata['feat'].float()  # message passing only supports float dtypes
        embed = self.embedder(graph, feat)
        return self.classifier(embed)

    def train_model(
            self,
            train_dataset,
            batch_size=16,
            EPOCHS=50,
            lr=1e-3,
            weight_decay=0.01,
            accum_steps=1,
            num_workers=2,
            pos_class_weight=1.,
            device='cpu',
            model_dir='saved_models/model'
    ):
        self.to(device)

        os.makedirs(model_dir, exist_ok=True)

        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

        train_loader = GraphDataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )

        start_train = time()
        train_acc, train_p, train_r, train_f1 = [], [], [], []
        train_score = 0
        for epoch in range(EPOCHS):
            self.train()
            self.to(device)

            data_iter = tqdm(
                train_loader,
                desc=f'Epoch: {epoch:02}',
                total=len(train_loader),
                position=0
            )

            loss, avg_loss = None, 0.
            for i, (batch_graph, labels) in enumerate(data_iter):
                batch_graph = batch_graph.to(device)
                labels = labels.squeeze().to(device)

                logits = self(batch_graph).squeeze()
                loss = F.binary_cross_entropy_with_logits(logits, labels)

                # Apply weight for positive class
                loss *= pos_class_weight
                # Normalize for batch accumulation
                loss /= accum_steps

                loss.backward()

                if ((i+1) % accum_steps == 0) or ((i+1) == len(data_iter)):
                    optimizer.step()
                    optimizer.zero_grad()

                avg_loss += loss.item()

                data_iter.set_postfix({
                    'train_score': train_score,
                    'avg_loss': avg_loss / (i+1)
                })

            if epoch % 2 == 0:
                acc, p, r, f1, = self.eval_model(
                    train_dataset, batch_size=batch_size, num_workers=num_workers, device=device
                )
                train_score = acc

                logging.info(f'{epoch:02}: Acc: {acc:.4f} | Prec: {p:.4f} | Recall: {r:.4f} | F1: {f1:.4f} ')

                train_acc.append(acc)
                train_p.append(p)
                train_r.append(r)
                train_f1.append(f1)

                # Save checkpoint
                torch.save({
                    'epoch': epoch,
                    'loss': loss,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, f'{model_dir}/checkpoint.pt')

        end_train = time()
        logging.info(f'Total training time... {end_train - start_train:.2f}s')

        # Save final model separately
        model_name = os.path.normpath(model_dir).split(os.sep)[-1]
        torch.save(self.state_dict(), f'{model_dir}/{model_name}.pt')

        return train_acc, train_p, train_r, train_f1

    def eval_model(
            self,
            eval_dataset,
            batch_size=16,
            num_workers=2,
            device='cpu'
    ):
        self.eval()
        self.to(device)

        eval_loader = GraphDataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        pred_probs, y_true = self.predict(eval_loader)

        num_true = int(sum(y_true))
        sorted_pred = sorted(pred_probs)
        threshold = sorted_pred[-num_true]

        y_pred = [
            1 if pred > threshold else 0
            for pred in pred_probs
        ]

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        return accuracy, precision, recall, f1

    def predict(self, graph_loader, device='cpu'):
        self.eval()
        self.to(device)

        data_iter = tqdm(
            graph_loader,
            desc=f'',
            total=len(graph_loader),
            position=0
        )

        with torch.no_grad():
            preds, labels = [], []
            for batch_graph, batch_labels in data_iter:
                batch_graph = batch_graph.to(device)

                batch_preds = torch.sigmoid(self(batch_graph))

                preds.extend(batch_preds.cpu())
                labels.extend(batch_labels)

        return preds, labels


class MuxGNNGraph(nn.Module):
    def __init__(
            self,
            gnn_type,
            num_gnn_layers,
            relations,
            feat_dim,
            embed_dim,
            dim_a,
            dropout=0.,
            activation=None
    ):
        super(MuxGNNGraph, self).__init__()
        self.gnn_type = gnn_type
        self.num_gnn_layers = num_gnn_layers
        self.relations = relations
        self.num_relations = len(self.relations)
        self.feat_dim = feat_dim
        self.embed_dim = embed_dim
        self.dim_a = dim_a
        self.activation = activation
        self.dropout = dropout

        self.layers = nn.ModuleList([
            MuxGNNLayer(
                gnn_type=self.gnn_type,
                relations=self.relations,
                in_dim=self.feat_dim,
                out_dim=self.embed_dim,
                dim_a=self.dim_a,
                dropout=self.dropout,
                activation=self.activation,
            )
        ])
        for _ in range(1, self.num_gnn_layers):
            self.layers.append(
                MuxGNNLayer(
                    gnn_type=self.gnn_type,
                    relations=self.relations,
                    in_dim=self.embed_dim,
                    out_dim=self.embed_dim,
                    dim_a=self.dim_a,
                    dropout=self.dropout,
                    activation=self.activation,
                )
            )

        self.readout_fn = SumPooling()

    @staticmethod
    def _get_activation_fn(activation):
        if activation is None:
            act_fn = None
        elif activation == 'relu':
            act_fn = nn.ReLU()
        elif activation == 'elu':
            act_fn = nn.ELU()
        elif activation == 'gelu':
            act_fn = nn.GELU()
        else:
            raise ValueError('Invalid activation function.')

        return act_fn

    def forward(self, graph, feat):
        h = feat
        for i, layer in enumerate(self.layers):
            h = layer(graph, h)

        return self.readout_fn(graph, h)


class MuxGNNLayer(nn.Module):
    def __init__(
            self,
            gnn_type,
            relations,
            in_dim,
            out_dim,
            dim_a,
            dropout=0.,
            activation=None
    ):
        super(MuxGNNLayer, self).__init__()
        self.gnn_type = gnn_type
        self.relations = relations
        self.num_relations = len(self.relations)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dim_a = dim_a
        self.act_str = activation

        self.dropout = nn.Dropout(dropout)
        self.activation = self._get_activation_fn(self.act_str)

        if self.gnn_type == 'gcn':
            self.gnn = GraphConv(
                in_feats=self.in_dim,
                out_feats=self.out_dim,
                norm='both',
                weight=True,
                bias=True,
                activation=self.activation,
                allow_zero_in_degree=True
            )
        elif self.gnn_type == 'gat':
            self.gnn = GATConv(
                in_feats=self.in_dim,
                out_feats=self.out_dim,
                num_heads=2,
                feat_drop=dropout,
                residual=False,
                activation=self.activation,
                allow_zero_in_degree=True
            )
        elif self.gnn_type == 'gin':
            self.gnn = GINConv(
                apply_func=nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    self.dropout,
                    self.activation,
                    nn.Linear(out_dim, out_dim),
                    self.dropout,
                    self.activation,
                ),
                aggregator_type='sum',
            )
        else:
            raise ValueError('Invalid GNN type.')

        self.attention = SemanticAttentionBatched(self.num_relations, self.out_dim, self.dim_a)

        self.norm = None
        # self.norm = nn.LayerNorm(self.out_dim, elementwise_affine=True)
        # self.norm = nn.BatchNorm1d(self.out_dim)

    @staticmethod
    def _get_activation_fn(activation):
        if activation is None:
            act_fn = None
        elif activation == 'relu':
            act_fn = nn.ReLU()
        elif activation == 'elu':
            act_fn = nn.ELU()
        elif activation == 'gelu':
            act_fn = nn.GELU()
        else:
            raise ValueError('Invalid activation function.')

        return act_fn

    def forward(self, graph, feat):
        h = torch.zeros(self.num_relations, graph.num_nodes(), self.out_dim, device=graph.device)
        with graph.local_scope():
            for i, graph_layer in enumerate(self.relations):
                try:
                    rel_graph = graph[dgl.NTYPE, graph_layer, dgl.NTYPE]

                    h_out = self.gnn(rel_graph, feat).squeeze()
                    if self.gnn_type == 'gat':
                        h_out = h_out.sum(dim=1)

                    h[i] = h_out
                except dgl.DGLError:
                    pass

        if self.norm:
            h = self.norm(h)

        h = self.attention(graph, h)
        return h


class BinaryClassifier(nn.Module):
    def __init__(self, embed_dim):
        super(BinaryClassifier, self).__init__()
        self.embed_dim = embed_dim

        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1)
        )

    def forward(self, x):
        return self.classifier(x)
