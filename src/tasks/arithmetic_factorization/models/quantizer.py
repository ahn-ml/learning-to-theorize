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

from models.codebook_ema import CodebookEMA

from torch.cuda.amp import autocast
from typing import Mapping, Text, Tuple

class VectorQuantizer(torch.nn.Module):
    def __init__(
            self,
            codebook_size: int = 1024,
            embedding_dim: int = 256,
            commitment_cost: float = 0.25,
            use_l2_norm: bool = True,
            stochastic: bool = False,
            tau_start: float = 2.0,
            tau_end: float = 0.1,
            tau_steps: int = 80000,
            entropy_loss_weight: float = 0.0,
            entropy_temperature: float = 0.01,
            use_ema: bool = False,
            ema_decay: float = 0.99,
            ema_epsilon: float = 1e-5,
            orthogonal_reg_weight: float = 0.0,
            orthogonal_reg_max_codes: int | None = None,
        ):
        super().__init__()
        self.commitment_cost = commitment_cost
        self.use_l2_norm = use_l2_norm
        self.stochastic = stochastic
        self.current_step = 0
        self._ema = CodebookEMA()
        self.tau_start = tau_start
        self.tau_end = tau_end
        self.tau_steps = tau_steps
        self.entropy_loss_weight = entropy_loss_weight
        self.entropy_temperature = entropy_temperature
        
        # EMA related parameters
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.ema_epsilon = ema_epsilon
        self.orthogonal_reg_weight = orthogonal_reg_weight
        self.orthogonal_reg_max_codes = orthogonal_reg_max_codes
        self._embedding_before_optimizer = None
        
        self.embedding = torch.nn.Embedding(codebook_size, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)
        
        if self.use_ema:
            # Register buffers for EMA updates
            self.register_buffer('ema_cluster_size', torch.zeros(codebook_size))
            self.register_buffer('ema_weight', self.embedding.weight.data.clone())
            # EMA supplies centroids; only orthogonal regularization supplies
            # embedding gradients. Reconstruction still uses the identity STE.
            self.embedding.requires_grad_(orthogonal_reg_weight > 0)
        
    def get_temperature(self):
        """Cosine annealing schedule for temperature"""
        if self.current_step >= self.tau_steps:
            return self.tau_end
        
        # Cosine schedule from tau_start to tau_end
        cos_factor = math.cos(math.pi * self.current_step / self.tau_steps)
        temp = self.tau_end + 0.5 * (self.tau_start - self.tau_end) * (1 + cos_factor)
        return temp

    def step(self):
        """Commit EMA plus the optimizer's regularization correction once.

        With orthogonal regularization, save E before optimizer.step(), then
        combine EMA_centroid + (E_after_optimizer - E). Carry that correction
        into ema_weight so the following EMA step does not simply erase it.
        """
        if self.use_ema:
            correction = None
            if self._embedding_before_optimizer is not None:
                correction = self.embedding.weight.detach() - self._embedding_before_optimizer
            self._ema.update(self.embedding.weight, self.ema_cluster_size,
                             self.ema_weight, self.ema_decay, self.ema_epsilon)
            if correction is not None:
                with torch.no_grad():
                    self.embedding.weight.add_(correction)
                    total = self.ema_cluster_size.sum()
                    size = (self.ema_cluster_size + self.ema_epsilon) / (
                        total + self.embedding.num_embeddings * self.ema_epsilon
                    ) * total
                    self.ema_weight.copy_(self.embedding.weight * size.unsqueeze(1))
            self._embedding_before_optimizer = None
        self.current_step += 1

    def orthogonal_loss(self, *, sample_codes: bool = True):
        """Mean squared off-diagonal cosine, using the published normalization."""
        codes = self.embedding.weight
        limit = self.orthogonal_reg_max_codes
        if sample_codes and limit is not None and len(codes) > limit:
            codes = codes[torch.randperm(len(codes), device=codes.device)[:limit]]
        codes = F.normalize(codes.float(), dim=-1)
        gram = codes @ codes.T
        return gram.square().mean() - 1.0 / len(codes)
        
    def entropy_loss(self, affinity, temperature=0.01):
        """Calculate entropy loss for codebook usage"""
        flat_affinity = affinity.view(-1, affinity.shape[-1])
        flat_affinity = flat_affinity / temperature
        probs = F.softmax(flat_affinity, dim=-1)
        log_probs = F.log_softmax(flat_affinity + 1e-5, dim=-1)
        
        avg_probs = torch.mean(probs, dim=0)    
        avg_entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-5))
        sample_entropy = -torch.mean(torch.sum(probs * log_probs, dim=-1))
        # loss = sample_entropy - avg_entropy
        loss = avg_entropy # to minimize entropy
        
        return loss, sample_entropy, avg_entropy
    
    # Ensure quantization is performed using f32
    @autocast(enabled=False)
    def forward(self, z: torch.Tensor, training_mode: bool=True, return_raw: bool=False) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        is_eval = not (self.training and training_mode)
        z_flattened = z.flatten(end_dim=-2).float() #rearrange(z, 'b n d -> (b n) d')

        if self.use_l2_norm:
            z_flattened = torch.nn.functional.normalize(z_flattened, dim=-1)
            embedding = torch.nn.functional.normalize(self.embedding.weight, dim=-1)
        else:
            embedding = self.embedding.weight
            
        # Euclidean Distance of Embedding Vectors
        d = (torch.sum(z_flattened**2, dim=1, keepdim=True) + 
                torch.sum(embedding**2, dim=1) - 
                2 * torch.einsum('bd,dn->bn', z_flattened, embedding.T))
        
        if self.stochastic:
            # Get current temperature from schedule
            current_temp = self.get_temperature()
            
            if is_eval:
                logits = -d
                min_encoding_indices = torch.argmin(d, dim=-1)
            else:
                # For stochastic mode with affinity (negative distance)
                logits = -d / current_temp
                probs = torch.nn.functional.softmax(logits, dim=-1)
                min_encoding_indices = torch.multinomial(probs, 1).squeeze(-1)
        else:
            current_temp = 0.0
            min_encoding_indices = torch.argmin(d, dim=1)

        # Keep the codebook fixed throughout the rollout and its backward pass.
        # step() aggregates all ranks and commits once after optimizer.step().
        if self.use_ema and self.training and training_mode:
            if self.orthogonal_reg_weight > 0 and self._embedding_before_optimizer is None:
                self._embedding_before_optimizer = self.embedding.weight.detach().clone()
            self._ema.accumulate(z_flattened, min_encoding_indices,
                                 self.embedding.num_embeddings)
            
        z_quantized = self.get_codebook_entry(min_encoding_indices).view(z.shape)

        if self.use_l2_norm:
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

        # Calculate entropy loss if enabled
        if self.entropy_loss_weight != 0:
            loss_entropy, sample_entropy, avg_entropy = self.entropy_loss(
                -d,  # Note: we negate distances to convert to affinities
                temperature=self.entropy_temperature
            )
            entropy_loss_term = self.entropy_loss_weight * loss_entropy
        else:
            entropy_loss_term = torch.tensor(0.0, device=z.device, dtype=z.dtype)
            sample_entropy = torch.tensor(0.0, device=z.device, dtype=z.dtype)
            avg_entropy = torch.tensor(0.0, device=z.device, dtype=z.dtype)

        orthogonal_loss_term = (
            self.orthogonal_reg_weight * self.orthogonal_loss(sample_codes=not is_eval)
            if self.orthogonal_reg_weight > 0
            else z.new_zeros(())
        )
        # The same global regularizer enters each per-sample candidate loss.
        # NEO selects one candidate per sample and averages, so its effective
        # weight is independent of batch size and selected program length.
        loss = commitment_loss + codebook_loss + entropy_loss_term + orthogonal_loss_term
        loss_ = commitment_loss_ + codebook_loss_ + entropy_loss_term + orthogonal_loss_term

        # preserve gradients
        z_quantized = z + (z_quantized - z).detach()

        # if is_eval:
        #     min_encoding_indices_for_analysis = min_encoding_indices.reshape(z.shape[0], z.shape[1], -1)

        result_dict = {
            'quantizer_loss': loss,
            'quantizer_loss_per_sample': loss_,
            'commitment_loss': commitment_loss,
            'codebook_loss': codebook_loss,
            'entropy_loss': entropy_loss_term,
            'orthogonal_loss': orthogonal_loss_term,
            'sample_entropy': sample_entropy,
            'codebook_entropy': avg_entropy,
            'temperature': torch.tensor(current_temp),
            'min_encoding_indices': min_encoding_indices,
            'logit': 1/d if not self.stochastic else logits,
            # 'min_encoding_indices_for_analysis': min_encoding_indices_for_analysis[::2] # only for analysis
        }

        if return_raw:
            raw_vals = {
                'z_quantized_raw': z_quantized,
                'z_raw': z,
                'commitment_cost': self.commitment_cost,
            }
            return z_quantized, result_dict, raw_vals
            
        return z_quantized, result_dict
    
    def get_codebook_entry(self, indices: torch.Tensor) -> torch.Tensor:
        """Convert indices to codebook vectors."""
        if len(indices.shape) == 1:
            z_quantized = self.embedding(indices)
        elif len(indices.shape) == 2:
            z_quantized = torch.einsum('bd,dn->bn', indices, self.embedding.weight)
        else:
            raise NotImplementedError("Indices must be 1D or 2D")

        if self.use_l2_norm:
            z_quantized = torch.nn.functional.normalize(z_quantized, dim=-1)
            
        return z_quantized
