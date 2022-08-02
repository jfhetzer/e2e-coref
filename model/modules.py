import torch
from torch import nn


class Scorer(nn.Sequential):

    def __init__(self, input_size, hidden_size=-1, hidden_depth=0, dropout=1):
        modules = []
        for i in range(hidden_depth):
            modules.extend([
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            input_size = hidden_size
        modules.append(nn.Linear(input_size, 1))
        super().__init__(*modules)


def init_weights(m):
    if isinstance(m, nn.Linear):
        m.weight.data = truncate_normal(m.weight.data.size())
        m.bias.fill_(0)
    elif isinstance(m, nn.Embedding):
        m.weight.data = truncate_normal(m.weight.data.size())


def truncate_normal(size):
    # truncated normal distribution (stddev: 0.02)
    draw = torch.randn(size)
    while (trunc := draw.abs() > 2).any():
        redraw = torch.randn(size)
        draw[trunc] = redraw[trunc]
    return draw * 0.02
