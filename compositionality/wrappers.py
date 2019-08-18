import torch
import torch.nn as nn
import torch.nn.functional as F

from egg import core
from egg.core.rnn import RnnEncoder


class RnnReceiverGS(core.RnnReceiverGS):

    def forward(self, message, input=None):
        outputs1, outputs2 = [], []

        emb = self.embedding(message)

        prev_hidden = None
        prev_c = None

        # to get an access to the hidden states, we have to unroll the cell ourselves
        for step in range(message.size(1)):
            e_t = emb[:, step, ...]
            if isinstance(self.cell, nn.LSTMCell):
                h_t, prev_c = self.cell(e_t, (prev_hidden, prev_c)) if prev_hidden is not None else \
                    self.cell(e_t)
            else:
                h_t = self.cell(e_t, prev_hidden)
            output1, output2 = self.agent(h_t, input)
            outputs1.append(output1)
            outputs2.append(output2)

            prev_hidden = h_t

        outputs1 = torch.stack(outputs1).permute(1, 0, 2)
        outputs2 = torch.stack(outputs2).permute(1, 0, 2)

        return outputs1, outputs2


class RnnReceiverDeterministic(nn.Module):

    def __init__(self, agent, vocab_size, embed_dim, hidden_size, cell='rnn', num_layers=1):
        super(RnnReceiverDeterministic, self).__init__()
        self.agent = agent
        self.encoder = RnnEncoder(vocab_size, embed_dim, hidden_size, cell, num_layers)

    def forward(self, message):
        encoded = self.encoder(message)
        return self.agent(encoded, None)
