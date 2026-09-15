import torch
import torch.nn as nn


# ============================================================
# settings
# ============================================================

N_POINTS = 256
N_CLASSES = 4


class TransformerModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.embedding = nn.Linear(1,64)

        self.position = nn.Parameter(torch.zeros(1, N_POINTS, 64))

        layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True,
        )

        self.transformer = (nn.TransformerEncoder(layer, num_layers=3))

        self.output = nn.Linear(64, N_CLASSES)

    def forward(self, x):

        x = x.unsqueeze(-1)

        x = self.embedding(x)

        x = x + self.position

        x = self.transformer(x)

        x = x.mean(dim=1)

        return self.output(x)