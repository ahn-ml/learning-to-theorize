import torch
import torch.nn as nn

from tasks.arithmetic_factorization.models.model_utils import (
    Encoder,
    EncoderLayer,
    sinusoidal_embeddings,
)


class CrossAttentionTransition(nn.Module):
    """
    Cross-attention based transition network using TransformerDecoderLayer.

    Architecture:
    - State s_x acts as query (what we want to transform)
    - Action acts as memory/key-value (what conditions the transformation)
    - Cross-attention allows state to attend to action for transition

    s_y = s_x + TransformerDecoder(query=s_x, memory=action)
    """
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.state_dim = params.state_dim
        self.num_state_tokens = params.num_state_tokens
        self.action_dim = params.action_dim
        self.num_action_tokens = params.num_action_tokens

        # Config
        self.d_model = getattr(params.transition, 'd_model', 128)
        num_layers = getattr(params.transition, 'num_layers', 2)
        num_heads = getattr(params.transition, 'num_heads', 4)
        d_ff = getattr(params.transition, 'd_ff', 512)
        dropout = getattr(params.transition, 'dropout', 0.1)
        self.use_residual = getattr(params.transition, 'use_residual', True)

        # Project state and action to d_model
        self.state_to_hidden = nn.Linear(self.state_dim, self.d_model) if self.state_dim != self.d_model else nn.Identity()
        self.action_to_hidden = nn.Linear(self.action_dim, self.d_model) if self.action_dim != self.d_model else nn.Identity()
        self.hidden_to_state = nn.Linear(self.d_model, self.state_dim) if self.d_model != self.state_dim else nn.Identity()

        # Positional embeddings
        scale = self.d_model ** -0.5
        self.use_sinusoidal_state_pos = getattr(params.transition, 'use_sinusoidal_state_pos', True)

        if self.use_sinusoidal_state_pos:
            # Fixed sinusoidal positional embeddings for state tokens
            self.register_buffer('state_positional_embedding',
                                sinusoidal_embeddings(self.num_state_tokens, self.d_model))
        else:
            # Learnable positional embeddings for state tokens
            self.state_positional_embedding = nn.Parameter(scale * torch.randn(self.num_state_tokens, self.d_model))

        self.action_positional_embedding = nn.Parameter(scale * torch.randn(self.num_action_tokens, self.d_model))

        # TransformerDecoderLayer: cross-attention between state (query) and action (memory)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='relu',
            batch_first=True,  # (batch, seq, feature)
            norm_first=False,
        )

        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
        )

    def forward(self, state_x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Apply cross-attention transition: s_y = s_x + Decoder(s_x, action)

        Args:
            state_x: (batch_size, num_state_tokens, state_dim) - query
            action: (batch_size, num_action_tokens, action_dim) - memory

        Returns:
            s_y: (batch_size, num_state_tokens, state_dim) - output state
        """
        batch_size = state_x.shape[0]

        # Project to hidden dimension
        state_hidden = self.state_to_hidden(state_x)  # (batch, num_state_tokens, d_model)
        action_hidden = self.action_to_hidden(action)  # (batch, num_action_tokens, d_model)

        # Add positional embeddings
        state_pos = self.state_positional_embedding.unsqueeze(0).expand(batch_size, -1, -1)
        action_pos = self.action_positional_embedding.unsqueeze(0).expand(batch_size, -1, -1)

        state_hidden = state_hidden + state_pos
        action_hidden = action_hidden + action_pos

        # TransformerDecoder: state as query, action as memory
        # Self-attention on state, cross-attention between state and action
        output = self.transformer_decoder(
            tgt=state_hidden,      # query: state we want to transform
            memory=action_hidden,  # key/value: action that conditions transformation
        )  # (batch, num_state_tokens, d_model)

        # Project back to state dimension
        delta = self.hidden_to_state(output)  # (batch, num_state_tokens, state_dim)

        # Residual connection
        if self.use_residual:
            s_y = state_x + delta
        else:
            s_y = delta

        return s_y


class TransitionNetwork(nn.Module):
    """
    State transition network that implements s_x + u -> s_y
    Supports multiple transition types: additive, mlp, transformer
    """
    def __init__(self, params):
        super().__init__()
        self.params = params

        # State space transition parameters
        self.state_dim = params.state_dim
        self.num_state_tokens = params.num_state_tokens
        self.action_dim = params.action_dim
        self.sequential_transition = params.transition.get('sequential_transition', False)
        self.halting_prediction = params.policy.get('halting_prediction', False)
        self.num_action_tokens = params.num_action_tokens if not self.sequential_transition else 1

        # Initialize transition function based on type
        self.linear_transition = params.get('linear_transition', False)
        self._init_transition_function()

    def _init_transition_function(self):
        """Initialize transition function based on transition_type"""
        self.d_model = self.params.transition.d_model
        if not self.linear_transition:
            scale = self.state_dim ** -0.5
            self.s_y_tokens = nn.Parameter(scale * torch.randn(self.num_state_tokens, self.d_model))
            self.s_y_tokens_positional_embedding = nn.Parameter(scale * torch.randn(self.num_state_tokens, self.d_model))

            # State positional embedding - optional sinusoidal (default) or learnable
            self.use_sinusoidal_state_pos = self.params.transition.get('use_sinusoidal_state_pos', True)

            if self.use_sinusoidal_state_pos:
                # Fixed sinusoidal positional embeddings for state tokens
                self.register_buffer('state_positional_embedding',
                                    sinusoidal_embeddings(self.num_state_tokens, self.d_model // 2))
            else:
                # Learnable positional embeddings for state tokens
                self.state_positional_embedding = nn.Parameter(scale * torch.randn(self.num_state_tokens, self.d_model // 2))

            self.action_positional_embedding = nn.Parameter(scale * torch.randn(self.num_action_tokens, self.d_model // 2))
            self.combined_positional_embedding = nn.Parameter(scale * torch.randn(self.num_state_tokens + self.num_action_tokens, self.d_model // 2))

            encoder_layer = EncoderLayer(
                self.d_model,
                self.params.transition.num_heads,
                self.params.transition.d_ff,
                self.params.transition.dropout
            )
            self.transition_encoder = Encoder(
                encoder_layer,
                self.params.transition.num_layers
            )
            # Projection for latent action to hidden dimension
            self.u_proj = nn.Linear(self.action_dim, self.d_model) if self.action_dim != self.d_model else nn.Identity()
            self.state_to_hidden = nn.Linear(self.state_dim, self.d_model) if self.state_dim != self.d_model else nn.Identity()
            self.hidden_to_state = nn.Linear(self.d_model, self.state_dim) if self.d_model != self.state_dim else nn.Identity()
        else:
            # Linear transition: s_y = s_x + W * u
            self.u_proj = nn.Linear(self.action_dim, self.state_dim) if self.action_dim != self.state_dim else nn.Identity()

    def forward(self, state_x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Apply transition: s_x + u -> s_y

        Args:
            state_x: (batch_size, num_state_tokens, state_dim) - input state
            action: (batch_size, num_action_tokens, action_dim) - latent action

        Returns:
            s_y: (batch_size, num_state_tokens, state_dim) - output state
        """

        assert action.dim() == 3, "Latent vector must be 3D for transformer transition"
        batch_size = state_x.shape[0]
        if not self.linear_transition:
            state_x = self.state_to_hidden(state_x)  # (batch_size, num_state_tokens, d_model)

            # Prepare input for transformer encoder
            s_y_tokens = self.s_y_tokens.unsqueeze(0).expand(batch_size, -1, -1).to(state_x.dtype)
            s_y_tokens_positional_embedding = self.s_y_tokens_positional_embedding.unsqueeze(0).expand(batch_size, -1, -1).to(state_x.dtype)
            s_y_tokens = s_y_tokens + s_y_tokens_positional_embedding
            if self.action_dim != self.d_model:
                action = self.u_proj(action)  # (batch_size, num_action_tokens, state_dim)

            combined = torch.cat([state_x, action], dim=1) # (batch_size, num_state_tokens + num_action_tokens, state_dim)

            state_pos_emb = (
                self.state_positional_embedding
                .unsqueeze(0)
                .expand(batch_size, state_x.shape[1], -1)
            )
            action_pos_emb = (
                self.action_positional_embedding[:action.shape[1]]
                .unsqueeze(0)
                .expand(batch_size, action.shape[1], -1)
            )
            pos_emb = torch.cat([state_pos_emb, action_pos_emb], dim=1) # (batch_size, num_state_tokens + num_action_tokens, state_dim // 2)
            combined_positional_embedding = self.combined_positional_embedding[:self.num_state_tokens + action.shape[1]].unsqueeze(0).expand(batch_size, -1, -1).to(combined.dtype)
            pos_emb_ = torch.cat([pos_emb, combined_positional_embedding], dim=2)

            combined = combined + pos_emb_

            tokens = torch.cat([combined, s_y_tokens], dim=1) # (batch_size, num_state_tokens + num_action_tokens + num_state_tokens, state_dim)
            output = self.transition_encoder(tokens)

            sy_pred = output[:, :self.num_state_tokens, :]  # (batch, num_state_tokens, state_dim)
            sy_pred = self.hidden_to_state(sy_pred)
        else:
            # Linear transition
            # Action: [batch_size, num_action_tokens=1, action_dim]
            action = self.u_proj(action)  # [batch_size, num_action_tokens=1, state_dim]
            sy_pred = state_x + action  # [batch_size, num_state_tokens, state_dim]


        return sy_pred
