import torch
import torch.nn as nn

from tasks.arithmetic_factorization.models.model_utils import Encoder, EncoderLayer


class PolicyNetwork(nn.Module):
    """Transformer programmer mapping a support state pair to latent actions."""
    def __init__(self, params):
        super().__init__()
        self.state_dim = params.state_dim
        self.action_dim = params.action_dim
        self.num_state_tokens = params.num_state_tokens
        self.num_action_tokens = params.num_action_tokens

        self.d_model = params.policy.d_model
        self.num_heads = params.policy.num_heads
        self.d_ff = params.policy.d_ff
        self.dropout = params.policy.dropout
        self.num_layers = params.policy.num_layers

        scale = self.state_dim ** -0.5

        self.action_tokens = nn.Parameter(scale * torch.randn(self.num_action_tokens, self.d_model))
        self.action_tokens_positional_embedding = nn.Parameter(scale * torch.randn(self.num_action_tokens, self.d_model))

        # Learnable positional embeddings for state tokens
        self.state_positional_embedding = nn.Parameter(scale * torch.randn(self.num_state_tokens, self.d_model // 4))
        self.state_pair_positional_embedding = nn.Embedding(2, self.d_model // 4)
        self.combined_state_positional_embedding = nn.Parameter(scale * torch.randn(self.num_state_tokens * 2, self.d_model // 2))

        encoder_layer = EncoderLayer(
            self.d_model, self.num_heads, self.d_ff, self.dropout
        )
        self.encoder = Encoder(encoder_layer, self.num_layers)

        self.to_action = nn.Linear(self.d_model, self.action_dim) if self.d_model != self.action_dim else nn.Identity()
        self.state_to_hidden = nn.Linear(self.state_dim, self.d_model) if self.state_dim != self.d_model else nn.Identity()

    def _generate_actions(self, combined_state: torch.Tensor) -> torch.Tensor:
        """
        Generate action tokens from the encoded support state pair.

        Args:
            combined_state: (batch_size, seq_len, d_model) - concatenated state context
        """
        batch_size = combined_state.shape[0]

        action_tokens_expanded = self.action_tokens.unsqueeze(0).expand(batch_size, -1, -1).to(combined_state.dtype)
        action_tokens_positional_embedding = self.action_tokens_positional_embedding.unsqueeze(0).expand(batch_size, -1, -1).to(combined_state.dtype)
        action_tokens_expanded = action_tokens_expanded + action_tokens_positional_embedding

        combined_input = torch.cat([action_tokens_expanded, combined_state], dim=1)

        output = self.encoder(combined_input)

        action_tokens = output[:, :self.num_action_tokens, :]
        actions = self.to_action(action_tokens)

        return actions

    def forward(self, state_x: torch.Tensor, state_y: torch.Tensor) -> torch.Tensor:
        """
        Map source and target states to action tokens.

        Args:
            state_x: (batch_size, num_state_tokens, state_dim) - state vector for input x
            state_y: (batch_size, num_state_tokens, state_dim) - state vector for output y

        Returns:
            actions: (batch_size, num_action_tokens, action_dim) - fixed length actions
        """
        batch_size = state_x.shape[0]
        state_x = self.state_to_hidden(state_x)
        state_y = self.state_to_hidden(state_y)

        # Prepare state context
        combined_state = torch.cat([state_x, state_y], dim=1)  # (batch, 2*num_state_tokens, d_model)

        state_pos_emb = (
            self.state_positional_embedding
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(batch_size, 2, state_x.shape[1], -1)
        ).reshape(batch_size, 2*state_x.shape[1], -1)

        is_output = torch.arange(2, device=state_x.device)
        io_emb = (
            self.state_pair_positional_embedding(is_output)
            .unsqueeze(0)
            .unsqueeze(2)
            .expand(batch_size, 2, state_x.shape[1], -1)
        ).reshape(batch_size, 2*state_x.shape[1], -1)

        combined_state_pos_emb = (
            self.combined_state_positional_embedding
            .to(combined_state.dtype)
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
        )
        pos_emb = torch.cat([state_pos_emb, io_emb, combined_state_pos_emb], dim=2)
        combined_state = combined_state + pos_emb

        return self._generate_actions(combined_state)
