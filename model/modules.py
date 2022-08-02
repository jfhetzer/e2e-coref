import torch
from torch import nn


class ElmoAggregator(nn.Module):

    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(3))
        self.scale = nn.Parameter(torch.tensor(1.))

    def forward(self, input):
        output = input @ torch.softmax(self.weights, dim=-1)
        return output * self.scale


class HighwayLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, dropout):
        super().__init__()
        self.dropout = dropout
        self.lstms = nn.ModuleList([
            CustomLSTM(input_size, hidden_size, dropout),
            CustomLSTM(2*hidden_size, hidden_size, dropout),
            CustomLSTM(2*hidden_size, hidden_size, dropout)
        ])
        self.highway_gates = nn.ModuleList([
            nn.Linear(2*hidden_size, 2*hidden_size),
            nn.Linear(2*hidden_size, 2*hidden_size)
        ])

    def forward(self, input, sent_len):
        for i in range(len(self.lstms)):
            output = self.lstms[i](input, sent_len)
            output = torch.dropout(output, self.dropout, self.training)
            if i > 0:
                g = torch.sigmoid(self.highway_gates[i - 1](output))
                output = g * output + (1 - g) * input
            input = output
        return output


class CustomLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, dropout):
        super().__init__()
        self.fwd_cell = CustomLSTMCell(input_size, hidden_size, dropout)
        self.bwd_cell = CustomLSTMCell(input_size, hidden_size, dropout)

    def forward(self, input, sent_len):
        fwd_output = self.fwd_cell(input)
        rev_input = self.reverse(input.clone(), sent_len)
        bwd_output = self.bwd_cell(rev_input)
        bwd_output = self.reverse(bwd_output, sent_len)
        return torch.cat((fwd_output, bwd_output), dim=2)

    def reverse(self, input, sent_len):
        for i in range(input.shape[0]):
            input[i, :sent_len[i]] = input[i, :sent_len[i]].flip(dims=[0])
        return input


class CustomLSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size, dropout):
        super().__init__()
        self.hidden_size, self.dropout = hidden_size, dropout
        # initialize init-states by sampling from a uniform xavier distribution
        # (default initialization for tensorflow's get_variable)
        self.init_hidden_state = nn.Parameter(nn.init.xavier_uniform_(torch.empty(1, hidden_size)))
        self.init_cell_state = nn.Parameter(nn.init.xavier_uniform_(torch.empty(1, hidden_size)))
        # forget and input gate are combined -> 3 gates left
        self.hidden_layer = nn.Linear(input_size + hidden_size, 3 * hidden_size)
        # initialize hidden state layer with random orthonormal matrix
        nn.init.orthogonal_(self.hidden_layer.weight)

    def forward(self, input: torch.Tensor):
        batch_size, seq_len, _ = input.shape
        dropout_mask = torch.dropout(torch.ones(batch_size, self.hidden_size, device=input.device), self.dropout, self.training)

        # initialize hidden and cell state
        hidden_state = self.init_hidden_state.repeat(batch_size, 1)
        cell_state = self.init_cell_state.repeat(batch_size, 1)
        # list to store hidden states
        hidden_states = []

        for t in range(seq_len):
            hidden_state *= dropout_mask
            input_t = input[:, t, :]
            concat = self.hidden_layer(torch.cat((input_t, hidden_state), dim=1))
            i, j, o = torch.split(concat, self.hidden_size, dim=1)
            i = torch.sigmoid(i)
            cell_state = (1-i) * cell_state + i * torch.tanh(j)
            hidden_state = torch.tanh(cell_state) * torch.sigmoid(o)
            hidden_states.append(hidden_state)

        # stack in sequence dimension
        return torch.stack(hidden_states, dim=1)


class Scorer(nn.Sequential):

    def __init__(self, input_size, hidden_size, dropout):
        super().__init__(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )
