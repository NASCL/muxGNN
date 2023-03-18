#!/usr/bin/env python3

import os
import argparse
import logging

import numpy as np
import torch
import utils
from model.graph_classifier import GraphClassifier


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = args.dataset.lower()

    out_model_dir = f'saved_models/{dataset}/muxgnn/{args.model_name}'
    os.makedirs(out_model_dir, exist_ok=True)

    log_path = f'logs/{dataset}/muxgnn/{args.model_name}'
    log_fname = f'{log_path}/log.out'
    os.makedirs(log_path, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        filename=log_fname,
        filemode='a'
    )
    logging.info(args)

    CV_FOLDS = 5
    all_acc, all_p, all_r, all_f1 = [], [], [], []
    for cv_fold in range(CV_FOLDS):
        (train_dataset,
         test_dataset,
         feat_dim,
         relations) = utils.load_graphpred_dataset(dataset, cv_fold=cv_fold, bidirected=args.bidirected)

        model = GraphClassifier(
            gnn_type=args.gnn,
            num_gnn_layers=args.num_gnn_layer,
            relations=relations,
            feat_dim=feat_dim,
            embed_dim=args.embed_dim,
            dim_a=args.dim_a,
            dropout=args.dropout,
            activation=args.activation
        )

        val_acc, val_p, val_r, val_f1 = model.train_model(
            train_dataset,
            batch_size=args.batch_size,
            EPOCHS=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            accum_steps=args.accum_steps,
            num_workers=args.num_workers,
            pos_class_weight=args.pos_class_weight,
            device=device,
            model_dir=out_model_dir
        )
        with open(log_fname, 'a') as f:
            f.write(
                '\n'.join(
                    ('-' * 25,
                     f'Validation metrics:',
                     f'Accuracies: {val_acc}',
                     f'Precisions: {val_p}',
                     f'Recalls: {val_r}',
                     f'F1s: {val_f1}\n')
                )
            )

        test_acc, test_p, test_r, test_f1 = model.eval_model(
            test_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device
        )
        with open(log_fname, 'a') as f:
            f.write(
                '\n'.join(
                    ('-' * 25,
                     f'Test metrics: CV-fold: {cv_fold}',
                     f'Accuracy: {test_acc:.4f}',
                     f'Precision: {test_p:.4f}',
                     f'Recall: {test_r:.4f}',
                     f'F1: {test_f1:.4f}',
                     '-' * 25,
                     '-' * 25 + '\n')
                )
            )

        all_acc.append(test_acc)
        all_p.append(test_p)
        all_r.append(test_r)
        all_f1.append(test_f1)

    with open(log_fname, 'a') as f:
        f.write(
            '\n'.join(
                ('-' * 25,
                 'Test Metrics: Mean and St Dev',
                 f'Accuracy: {np.mean(all_acc):.4f} {np.std(all_acc):.4f}',
                 f'Precision: {np.mean(all_p):.4f} {np.std(all_p):.4f}',
                 f'Recall: {np.mean(all_r):.4f} {np.std(all_r):.4f}',
                 f'F1: {np.mean(all_f1):.4f} {np.std(all_f1):.4f}')
            )
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('dataset', type=str,
                        help='Name of the dataset. Options are "wget" or "streamspot".')
    parser.add_argument('model_name', type=str,
                        help='Name to save the trained model.')
    parser.add_argument('--gnn', type=str, default='gin',
                        help='GNN layer to use with muxGNN. "gcn", "gat", or "gin". Default is "gin".')
    parser.add_argument('--num-gnn-layer', type=int, default=1,
                        help='Number of GNN layers in the embedding module.')
    parser.add_argument('--pos-class-weight', type=float, default=1.,
                        help='Additional weight to apply to loss contribution of positive class.')
    parser.add_argument('--embed-dim', type=int, default=64,
                        help='Size of output embedding dimension.')
    parser.add_argument('--dim-a', type=int, default=16,
                        help='Dimension of attention.')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate during training.')
    parser.add_argument('--activation', type=str, default='elu',
                        help='Activation function. Options are "relu", "elu", or "gelu".')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size during training and inference.')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Maximum limit on training epochs.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate for optimizer.')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='L2 regularization penalty.')
    parser.add_argument('--accum-steps', type=int, default=4,
                        help='Number of gradient accumulation steps to take before weight update.')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of worker processes.')
    parser.add_argument('--bidirected', action='store_true', default=False,
                        help='Use a bidirectional version of the input graphs.')

    args = parser.parse_args()
    main(args)
