#!/usr/bin/env python3

import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import dgl
from dgl.dataloading import Collator, BlockSampler
# noinspection PyProtectedMember
from dgl.dataloading.pytorch import _pop_blocks_storage, _restore_blocks_storage

"""
Negative Loss Module adapted from: https://github.com/dmlc/dgl/tree/master/examples/pytorch/GATNE-T
Original author code repo: https://github.com/THUDM/GATNE

hrt-tuple data loader adapted from:
    https://docs.dgl.ai/_modules/dgl/dataloading/dataloader.html
    https://docs.dgl.ai/_modules/dgl/dataloading/pytorch.html

"""


class NeighborSampler(BlockSampler):
    def __init__(self, fanouts, add_self_loops=False):
        super(NeighborSampler, self).__init__(len(fanouts), return_eids=False)
        self.fanouts = fanouts
        self.add_self_loops = add_self_loops

    def sample_frontier(self, block_id, g, seed_nodes):
        # Temp fix to handle single node type
        # DGL converts seed nodes to dict: {node_type: tensor(node_ids)}
        if isinstance(seed_nodes, dict):
            assert len(seed_nodes) == 1
            seed_nodes = next(iter(seed_nodes.values()))

        fanout = self.fanouts[block_id]
        frontier = dgl.sampling.sample_neighbors(g, seed_nodes, fanout)

        return frontier


class HrtDataLoader:
    def __init__(self, g, hrt_tuples, block_sampler, device='cpu', **kwargs):
        self.collator = _HrtCollator(g, hrt_tuples, block_sampler)
        self.dataloader = DataLoader(
            self.collator.dataset,
            collate_fn=self.collator.collate,
            **kwargs
        )
        self.device = device

        if kwargs.get('num_workers', 0) > 0:
            g.create_formats_()

    def __iter__(self):
        return _HrtDataLoaderIter(self)

    def __len__(self):
        return len(self.dataloader)


class HrtCollator(Collator):
    def __init__(self, g, hrt_tuples, block_sampler):
        self.g = g
        self.block_sampler = block_sampler
        self._dataset = hrt_tuples

    @property
    def dataset(self):
        return self._dataset

    def collate(self, items):
        heads, rel_idx, tails = (
            torch.LongTensor(x)
            for x in zip(*items)
        )

        # Same head node may appear in batch multiple times. Only sample neighbors for each node once
        seeds, head_invmap = torch.unique(heads, return_inverse=True)

        blocks = self.block_sampler.sample_blocks(self.g, seeds.type(self.g.idtype))

        return blocks, head_invmap, rel_idx, tails


class _HrtCollator(HrtCollator):
    def collate(self, items):
        # blocks, heads, rel_idx, tails
        result = super().collate(items)
        _pop_blocks_storage(result[0], self.g)
        return result


class _HrtDataLoaderIter:
    def __init__(self, hrt_dataloader):
        self.device = hrt_dataloader.device
        self.hrt_dataloader = hrt_dataloader
        self.iter_ = iter(hrt_dataloader.dataloader)

    def __next__(self):
        # blocks, heads, rel_idx, tails
        result_ = next(self.iter_)
        _restore_blocks_storage(result_[0], self.hrt_dataloader.collator.g)

        result = [_to_device(data, self.device) for data in result_]
        return result


class NegativeSamplingLoss(nn.Module):
    def __init__(self, g, num_neg_samples, embedding_dim, dist='uniform'):
        super(NegativeSamplingLoss, self).__init__()
        self.num_neg_samples = num_neg_samples
        self.embedding_dim = embedding_dim
        self.num_nodes = g.num_nodes()
        self.dist = dist  # Can be 'uniform' or 'log-uniform'

        self.weights = nn.Parameter(
            torch.FloatTensor(self.num_nodes, self.embedding_dim)
        )

        # Draw negative samples from Uniform distribution or Log-Uniform distribution (ordered by node degree)
        sample_weights = None if self.dist == 'uniform' else self._compute_sample_weights(g)
        self.register_buffer('sample_weights', sample_weights)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weights, std=1.0 / math.sqrt(self.embedding_dim))

    def _compute_sample_weights(self, g):
        assert g.num_nodes() == self.num_nodes

        g = dgl.to_homogeneous(g)
        _, sorted_idx = torch.sort(g.in_degrees(), descending=True)

        sample_weights = torch.empty_like(g.nodes(), dtype=torch.float32)
        for k, n_id in enumerate(sorted_idx):
            sample_weights[n_id] = (math.log(k+2) - math.log(k+1)) / math.log(self.num_nodes+1)

        return F.normalize(sample_weights, dim=0)

    def forward(self, heads, embeds, tails):
        num_heads = heads.shape[0]

        tail_weights = self.weights[tails]
        log_target = torch.log(
            torch.sigmoid(
                torch.sum(
                    torch.mul(embeds, tail_weights),
                    dim=1
                )
            )
        )

        if self.dist == 'uniform':
            neg_tails = torch.randint(
                0, self.num_nodes,
                (num_heads * self.num_neg_samples,),
                device=embeds.device
            )
        else:
            neg_tails = torch.multinomial(
                self.sample_weights,
                self.num_neg_samples * num_heads,
                replacement=True
            )
        neg_tails = neg_tails.view(num_heads, self.num_neg_samples)

        noise = torch.neg(self.weights[neg_tails])

        sum_log_targets = torch.sum(
            torch.log(
                torch.sigmoid(
                    torch.bmm(noise, embeds.unsqueeze(-1))
                )
            ),
            dim=1
        ).squeeze()

        loss = log_target + sum_log_targets

        return -loss.sum() / num_heads


def _to_device(data, device):
    if isinstance(data, dict):
        for k, v in data.items():
            data[k] = v.to(device)
    elif isinstance(data, list):
        data = [item.to(device) for item in data]
    else:
        data = data.to(device)
    return data
