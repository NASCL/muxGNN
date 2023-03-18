#!/usr/bin/env python3

import argparse
import logging
import os
from time import time

import numpy as np
import torch

from model.mux_gnn import MuxGNN
from model.sample import NegativeSamplingLoss, NeighborSampler
import utils


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    suffix = ''
    if args.inductive:
        assert args.dataset == 'tissue_ppi'
        suffix = '_inductive'

    model_fname = f'muxgnn_{args.gnn}'

    out_model_dir = f'saved_models/muxgnn/{args.dataset}{suffix}/{model_fname}'
    os.makedirs(out_model_dir, exist_ok=True)

    log_path = f'logs/{args.dataset}{suffix}/{args.model_name}'
    log_fname = f'{log_path}/{model_fname}_log.out'
    os.makedirs(log_path, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        filename=log_fname,
        filemode='a'
    )
    logging.info(args)

    start = time()
    if args.inductive:
        (train_G, test_G), val_edges, test_edges = utils.load_inductive_data()
    else:
        test_G = None
        train_G, val_edges, test_edges = utils.load_linkpred_dataset(args.dataset, holdout_split=args.cv_split)

    end = time()
    logging.info(f'Loading graph data... {end - start:.2f}s')

    feat_dim = train_G.ndata['feat'].shape[-1]

    fanouts = args.neigh_samples * args.num_layers if len(args.neigh_samples) == 1 else args.neigh_samples
    assert len(fanouts) == args.num_layers

    start = time()
    model = MuxGNN(
        gnn_type=args.gnn,
        num_gnn_layers=args.num_layers,
        relations=train_G.etypes,
        feat_dim=feat_dim,
        embed_dim=args.embed_dim,
        dim_a=args.dim_a,
        dropout=args.dropout,
        activation=args.activation,
    )

    neigh_sampler = NeighborSampler(fanouts)
    nsloss = NegativeSamplingLoss(
        train_G, num_neg_samples=args.neg_samples, embedding_dim=args.embed_dim, dist=args.neg_dist
    )
    end = time()
    logging.info(f'Initializing model... {end - start:.2f}s')

    val_aucs, val_f1s, val_prs = model.train_model(
        train_G=train_G,
        val_edges=val_edges,
        neigh_sampler=neigh_sampler,
        loss_module=nsloss,
        num_walks=args.num_walks,
        walk_length=args.walk_length,
        window_size=args.window_size,
        batch_size=args.batch_size,
        EPOCHS=args.epochs,
        patience_limit=args.patience,
        num_workers=args.num_workers,
        device=device,
        model_dir=out_model_dir
    )
    with open(log_fname, 'a') as f:
        f.write(
            '\n'.join(
                ('-' * 25,
                 f'Validation metrics:',
                 f'ROC-AUCs: {val_aucs}',
                 f'F1s: {val_f1s}',
                 f'PR-AUCs: {val_prs}\n')
            )
        )

    test_aucs, test_f1s, test_prs = model.eval_model(
        test_G if args.inductive else train_G,
        test_edges,
        neigh_sampler,
        batch_size=args.batch_size,
        device=device
    )
    with open(log_fname, 'a') as f:
        f.write(
            '\n'.join(
                ('-' * 25,
                 'Test metrics:',
                 f'Mean ROC-AUC: {np.mean(test_aucs):.4f}',
                 f'\t{test_aucs}',
                 f'Mean F1: {np.mean(test_f1s):.4f}',
                 f'\t{test_f1s}',
                 f'Mean: PR-AUC: {np.mean(test_prs):.4f}',
                 f'\t{test_prs}\n')
            )
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('dataset', type=str,
                        help='Dataset name.')
    parser.add_argument('model_name', type=str,
                        help='Name to save the trained model.')
    parser.add_argument('--cv-split', type=int, default=None,
                        help='Cross-validation split to hold out.')
    parser.add_argument('--gnn', type=str, default='gin',
                        help='GNN layer to use with muxGNN. "gcn", "gat", or "gin". Default is "gin".')
    parser.add_argument('--inductive', action='store_true', default=False,
                        help='Whether to run the inductive tissue-ppi experiment.')

    parser.add_argument('--num-layers', type=int, default=2,
                        help='Number of k-hop neighbor aggregations to perform.')
    parser.add_argument('--neigh-samples', type=int, default=[10], nargs='+',
                        help='Number of neighbors to sample for aggregation.')
    parser.add_argument('--embed-dim', type=int, default=200,
                        help='Size of output embedding dimension.')
    parser.add_argument('--dim-a', type=int, default=20,
                        help='Dimension of attention.')
    parser.add_argument('--activation', type=str, default='elu',
                        help='Activation function.')
    parser.add_argument('--no-bias', action='store_true', default=False,
                        help='Whether to add a bias vector to model transformation matrices.')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate during training.')

    parser.add_argument('--num-walks', type=int, default=20,
                        help='Number of random walks to perform.')
    parser.add_argument('--walk-length', type=int, default=10,
                        help='Length of random walks.')
    parser.add_argument('--window-size', type=int, default=5,
                        help='Size of sliding window.')
    parser.add_argument('--neg-samples', type=int, default=5,
                        help='Number of negative samples.')
    parser.add_argument('--neg-dist', type=str, default='uniform',
                        help='Distribution from which to draw negative samples. Either "uniform" or "log-uniform"')
    parser.add_argument('--patience', type=int, default=3,
                        help='Number of epochs to wait for improvement before early stopping.')
    parser.add_argument('--num-workers', type=int, default=2,
                        help='Number of worker processes.')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size during training')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Maximum limit on training epochs.')
    parser.add_argument('--learn-rate', type=float, default=1e-3,
                        help='Learning rate for optimizer.')

    args = parser.parse_args()
    main(args)
