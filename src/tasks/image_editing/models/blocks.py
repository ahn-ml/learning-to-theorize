"""Shared building block for the Image Editing latent-program modules."""

from __future__ import annotations

from torch import Tensor, nn


class FiLMBlock(nn.Module):
    """Transform a hidden vector, then modulate it with FiLM conditioning.

    ``h' = gamma(condition) * transform(h) + beta(condition)``

    The gamma/beta layers are initialised to the identity modulation so a freshly
    built stack starts out as a plain MLP.  :class:`~tasks.image_editing.models.
    theory_programmer.TheoryProgrammer` never supplies a condition and therefore
    only ever calls :attr:`transform`; see that module for why the unused
    parameters are still constructed.
    """

    def __init__(self, hidden_dim: int, condition_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.film_gamma = nn.Linear(condition_dim, hidden_dim)
        self.film_beta = nn.Linear(condition_dim, hidden_dim)

        # Match the paper-producing FiLM initialization exactly.
        nn.init.ones_(
            self.film_gamma.weight.data[:hidden_dim, :condition_dim].diagonal()
        )
        nn.init.zeros_(self.film_gamma.bias.data)
        nn.init.zeros_(self.film_beta.weight.data)
        nn.init.zeros_(self.film_beta.bias.data)

    def forward(self, hidden: Tensor, condition: Tensor) -> Tensor:
        hidden = self.transform(hidden)
        return self.film_gamma(condition) * hidden + self.film_beta(condition)


__all__ = ["FiLMBlock"]
