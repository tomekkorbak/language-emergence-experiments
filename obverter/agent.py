from itertools import product

import torch
from egg.core.rnn import RnnEncoder
from torch import nn

from visual_compositionality.pretrain import Vision

class Agent(nn.Module):
    def __init__(self, n_hidden, n_features):
        super(Agent, self).__init__()
        self.fc1 = nn.Linear(n_hidden, n_features * 2)
        self.fc2_1 = nn.Linear(n_features * 2, n_features)
        self.fc2_2 = nn.Linear(n_features * 2, n_features)

    def forward(self, message, _):
        hidden = torch.nn.functional.leaky_relu(self.fc1(message))
        return self.fc2_1(hidden).squeeze(dim=0), self.fc2_2(hidden).squeeze(dim=0)


class AgentWrapper(nn.Module):

    def __init__(self, agent, vocab_size, embed_dim, hidden_size, cell='rnn', num_layers=1, obverter_loss=None):
        super(AgentWrapper, self).__init__()
        self.agent = agent
        self.encoder = RnnEncoder(vocab_size, embed_dim, hidden_size, cell, num_layers)
        self.vocab_size = vocab_size
        self.obverter_loss = obverter_loss
        self.max_len = 2

        self.vision = Vision()
        self.vision.load_state_dict(torch.load('visual_compositionality/vision_model.pth'))

    def forward(self, message):
        encoded = self.encoder(message)
        return self.agent(encoded, None)

    def decode(self, sender_input):
        outputs_1, outputs_2 = self.vision(sender_input)

        message_dimensions = [range(self.vocab_size)] * self.max_len
        all_possible_messages = torch.LongTensor(list(product(*message_dimensions)))
        prediction_1, prediction_2 = self(all_possible_messages)
        messages_to_send = []
        for output_1, output_2 in zip(outputs_1, outputs_2):
            proxy_target = torch.stack([output_1.argmax(dim=0), output_2.argmax(dim=0)], dim=0).repeat(all_possible_messages.size(0), 1)
            proxy_loss, _ = self.obverter_loss(proxy_target, prediction_1, prediction_2)
            best_message = all_possible_messages[proxy_loss.argmin()]
            messages_to_send.append(best_message)
        return torch.stack(messages_to_send, dim=0)