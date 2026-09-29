"""Accumulate code assignments, then update one shared codebook per optimizer step."""

import torch
import torch.distributed as dist
from torch.nn import functional as F


class CodebookEMA:
    """Transient sufficient statistics; persistent EMA buffers stay on the quantizer.

    These are deliberately not module buffers: DDP must not broadcast or replace
    a rank's pending local assignments before they are reduced at step().
    Checkpoints are saved after step(), when no pending statistics remain.
    """

    def __init__(self):
        self.counts = None
        self.sums = None

    @torch.no_grad()
    def accumulate(self, values, indices, codebook_size):
        with torch.autocast(device_type=values.device.type, enabled=False):
            assignments = F.one_hot(indices, codebook_size).float()
            counts = assignments.sum(0)
            sums = assignments.t() @ values.detach().float()
        if self.counts is None:
            self.counts, self.sums = counts, sums
        else:
            self.counts.add_(counts)
            self.sums.add_(sums)

    @torch.no_grad()
    def update(self, embedding, cluster_size, weight_sum, decay, epsilon):
        if self.counts is None:
            return
        counts, sums = self.counts, self.sums
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(counts)
            dist.all_reduce(sums)
        cluster_size.mul_(decay).add_(counts, alpha=1 - decay)
        weight_sum.mul_(decay).add_(sums, alpha=1 - decay)
        total = cluster_size.sum()
        normalized_size = (cluster_size + epsilon) / (
            total + embedding.shape[0] * epsilon
        ) * total
        embedding.copy_(weight_sum / normalized_size.unsqueeze(1))
        self.counts = self.sums = None
