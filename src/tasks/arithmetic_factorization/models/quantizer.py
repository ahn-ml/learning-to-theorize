"""Vector quantizer.

Copyright (2024) Bytedance Ltd. and/or its affiliates
Includes modifications by the Learning to Theorize authors.

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

    http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License.

Reference: 
    https://github.com/CompVis/taming-transformers/blob/master/taming/modules/vqvae/quantize.py
    https://github.com/google-research/magvit/blob/main/videogvt/models/vqvae.py
"""
import math
import torch
import torch.nn.functional as F

from typing import Mapping, Text, Tuple

class VectorQuantizer(torch.nn.Module):
    def __init__(
            self,
            codebook_size: int = 1024,
            embedding_dim: int = 256,
            commitment_cost: float = 0.25,
            tau_start: float = 2.0,
            tau_end: float = 0.1,
            tau_steps: int = 80000,
            use_ema: bool = False,
            ema_decay: float = 0.99,
            ema_epsilon: float = 1e-5,
        ):
        super().__init__()
        self.commitment_cost = commitment_cost
        self.current_step = 0
        self.tau_start = tau_start
        self.tau_end = tau_end
        self.tau_steps = tau_steps

        # EMA related parameters
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.ema_epsilon = ema_epsilon

        self.embedding = torch.nn.Embedding(codebook_size, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)

        if self.use_ema:
            # Register buffers for EMA updates
            self.register_buffer('ema_cluster_size', torch.zeros(codebook_size))
            self.register_buffer('ema_weight', self.embedding.weight.data.clone())
            # EMA supplies the centroids, so the codebook stays out of the
            # optimizer. Reconstruction still uses the identity STE.
            self.embedding.requires_grad_(False)

    def get_temperature(self):
        """Cosine annealing schedule for temperature"""
        if self.current_step >= self.tau_steps:
            return self.tau_end

        # Cosine schedule from tau_start to tau_end
        cos_factor = math.cos(math.pi * self.current_step / self.tau_steps)
        temp = self.tau_end + 0.5 * (self.tau_start - self.tau_end) * (1 + cos_factor)
        return temp

    def step(self):
        """Advance the temperature schedule by one optimizer step."""
        self.current_step += 1

    @torch.no_grad()
    def _update_codebook_now(self, z_flattened, min_encoding_indices):
        """Update the EMA centroids from this call's code assignments.

        Codes are assigned with the codebook as it was before the call; the
        quantized values are then read from the updated codebook, so later
        rollout steps see it. Single-rank only: ranks would update their
        codebooks independently.
        """
        if torch.distributed.is_available() and torch.distributed.is_initialized() \
                and torch.distributed.get_world_size() > 1:
            raise RuntimeError("per-quantization codebook updates require a single rank")
        one_hot = F.one_hot(min_encoding_indices, self.embedding.weight.shape[0]).float()
        encodings_sum = one_hot.sum(0)
        self.ema_cluster_size.copy_(self.ema_cluster_size * self.ema_decay + (1 - self.ema_decay) * encodings_sum)
        dw = torch.matmul(one_hot.t(), z_flattened)
        self.ema_weight.copy_(self.ema_weight * self.ema_decay + (1 - self.ema_decay) * dw)
        n = torch.sum(self.ema_cluster_size.view(-1))
        updated_cluster_size = (self.ema_cluster_size + self.ema_epsilon) / \
            (n + self.embedding.weight.shape[0] * self.ema_epsilon) * n
        self.embedding.weight.data.copy_(self.ema_weight / updated_cluster_size.unsqueeze(1))

    # Ensure quantization is performed using f32
    @torch.autocast("cuda", enabled=False)
    def forward(self, z: torch.Tensor, training_mode: bool=True) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        """Sample a code from softmax(-distance / tau) in training; take the nearest code otherwise."""
        is_eval = not (self.training and training_mode)
        z_flattened = torch.nn.functional.normalize(z.flatten(end_dim=-2).float(), dim=-1)
        embedding = torch.nn.functional.normalize(self.embedding.weight, dim=-1)

        # Euclidean Distance of Embedding Vectors
        d = (torch.sum(z_flattened**2, dim=1, keepdim=True) +
                torch.sum(embedding**2, dim=1) -
                2 * torch.einsum('bd,dn->bn', z_flattened, embedding.T))

        current_temp = self.get_temperature()
        if is_eval:
            logits = -d
            min_encoding_indices = torch.argmin(d, dim=-1)
        else:
            logits = -d / current_temp
            probs = torch.nn.functional.softmax(logits, dim=-1)
            min_encoding_indices = torch.multinomial(probs, 1).squeeze(-1)

        if self.use_ema and not is_eval:
            self._update_codebook_now(z_flattened, min_encoding_indices)

        z_quantized = self.get_codebook_entry(min_encoding_indices).view(z.shape)
        z_quantized = torch.nn.functional.normalize(z_quantized, dim=-1)
        z = torch.nn.functional.normalize(z, dim=-1)

        # compute loss for embedding
        commitment_loss = self.commitment_cost * torch.mean((z_quantized.detach() - z) **2)
        commitment_loss_ = self.commitment_cost * torch.mean((z_quantized.detach() - z) **2, dim=(1,2))
        codebook_loss = torch.mean((z_quantized - z.detach()) **2)
        codebook_loss_ = torch.mean((z_quantized - z.detach()) **2, dim=(1,2))

        # If using EMA, we don't need codebook loss (embeddings are updated via EMA)
        if self.use_ema:
            codebook_loss = 0.0 * codebook_loss
            codebook_loss_ = 0.0 * codebook_loss_

        loss = commitment_loss + codebook_loss
        loss_ = commitment_loss_ + codebook_loss_

        # preserve gradients
        z_quantized = z + (z_quantized - z).detach()

        result_dict = {
            'quantizer_loss': loss,
            'quantizer_loss_per_sample': loss_,
            'commitment_loss': commitment_loss,
            'codebook_loss': codebook_loss,
            'temperature': torch.tensor(current_temp),
            'min_encoding_indices': min_encoding_indices,
            'logit': logits,
        }
        return z_quantized, result_dict

    def get_codebook_entry(self, indices: torch.Tensor) -> torch.Tensor:
        """Convert code indices to normalized codebook vectors."""
        return torch.nn.functional.normalize(self.embedding(indices), dim=-1)
