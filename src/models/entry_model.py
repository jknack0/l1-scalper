"""CNN-LSTM entry model (DeepLOB-inspired).

Input: [batch, 30, 9] — 30 one-second timesteps × 9 raw microstructure features
Output:
    - direction_prob: sigmoid probability of favorable move (0=short, 1=long)
    - expected_magnitude: linear, expected return in ticks

Architecture:
    Conv1D blocks → Inception module → LSTM → dual heads

Target: ~60K params, single-digit ms GPU inference, <50ms CPU.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class InceptionBlock(nn.Module):
    """Inception-style module with parallel conv branches + concatenation."""

    def __init__(self, in_channels: int, out_per_branch: int = 16) -> None:
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, out_per_branch, kernel_size=1),
            nn.BatchNorm1d(out_per_branch),
            nn.LeakyReLU(0.01),
        )
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, out_per_branch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_per_branch),
            nn.LeakyReLU(0.01),
        )
        self.branch5 = nn.Sequential(
            nn.Conv1d(in_channels, out_per_branch, kernel_size=5, padding=2),
            nn.BatchNorm1d(out_per_branch),
            nn.LeakyReLU(0.01),
        )
        self.out_channels = out_per_branch * 3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.branch1(x), self.branch3(x), self.branch5(x)], dim=1)


class EntryModel(nn.Module):
    """CNN-LSTM entry model for MES scalping.

    Args:
        n_features: number of input features per timestep (default 9)
        seq_len: number of timesteps (default 30)
        conv_channels: channels after initial conv block (default 32)
        inception_per_branch: channels per inception branch (default 16)
        lstm_hidden: LSTM hidden size (default 32)
        lstm_layers: number of LSTM layers (default 1)
        dropout: dropout rate (default 0.25)
    """

    def __init__(
        self,
        n_features: int = 9,
        seq_len: int = 30,
        conv_channels: int = 32,
        inception_per_branch: int = 16,
        lstm_hidden: int = 32,
        lstm_layers: int = 1,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()

        # Conv1D feature extractor: [batch, features, seq] -> [batch, conv_channels, seq]
        self.conv_block = nn.Sequential(
            nn.Conv1d(n_features, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels),
            nn.LeakyReLU(0.01),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels),
            nn.LeakyReLU(0.01),
        )

        # Inception module
        self.inception = InceptionBlock(conv_channels, inception_per_branch)
        inception_out = self.inception.out_channels  # 48

        # LSTM: [batch, seq, inception_out] -> [batch, lstm_hidden]
        self.lstm = nn.LSTM(
            input_size=inception_out,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)

        # Dual heads
        self.direction_head = nn.Linear(lstm_hidden, 1)  # sigmoid -> P(long)
        self.magnitude_head = nn.Linear(lstm_hidden, 1)  # linear -> expected ticks

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: [batch, seq_len, n_features]

        Returns:
            direction_logits: [batch, 1] — raw logits (apply sigmoid for P(long))
            expected_magnitude: [batch, 1] — expected return in ticks
        """
        # Conv1D expects [batch, channels, seq]
        x = x.transpose(1, 2)
        x = self.conv_block(x)
        x = self.inception(x)

        # Back to [batch, seq, channels] for LSTM
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)

        # Take last timestep
        x = lstm_out[:, -1, :]
        x = self.dropout(x)

        direction_logits = self.direction_head(x)
        magnitude = self.magnitude_head(x)

        return direction_logits, magnitude

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
