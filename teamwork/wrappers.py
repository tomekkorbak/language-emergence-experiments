from typing import List

import torch
import torch.nn as nn
from egg import core


class Sender(nn.Module):
    def __init__(self, n_hidden, n_features):
        super(Sender, self).__init__()
        self.fc1 = nn.Linear(n_features, n_hidden)

    def forward(self, x):
        return self.fc1(x)


class ReinforceMultiAgentWrapper(nn.Module):

    def __init__(self, agents: List[core.ReinforceWrapper]):
        super(ReinforceMultiAgentWrapper, self).__init__()
        self.agents = agents

    def forward(self, input, agent_indices, **kwargs):
        agent_indices = agent_indices.unsqueeze(dim=0)
        samples, log_probs, entropies = zip(*(agent(input, **kwargs) for agent in self.agents))
        samples = torch.stack(samples, dim=0).gather(dim=0, index=agent_indices)
        log_probs = torch.stack(log_probs, dim=0).gather(dim=0, index=agent_indices)
        entropies = torch.stack(entropies, dim=0).gather(dim=0, index=agent_indices)
        return samples.squeeze(dim=0), log_probs.squeeze(dim=0), entropies.squeeze(dim=0)


class GumbelSoftmaxMultiAgentWrapper(nn.Module):

    def __init__(self, agents: List[core.ReinforceWrapper]):
        super(GumbelSoftmaxMultiAgentWrapper, self).__init__()
        self.agents = agents

    def forward(self, input, agent_indices, **kwargs):
        samples = [agent(input, **kwargs) for agent in self.agents]
        agent_indices = agent_indices.reshape(1, agent_indices.size(0), 1).expand(1, agent_indices.size(0), samples[0].size(1))
        samples = torch.stack(samples, dim=0).gather(dim=0, index=agent_indices)
        return samples.squeeze(dim=0)


if __name__ == "__main__":
    BATCH_SIZE, INPUT_SIZE, OUTPUT_SIZE = 8, 10, 5
    AGENT_INDICES = torch.LongTensor([0, 1, 0, 1, 0, 1, 0, 1])
    multi_agent = ReinforceMultiAgentWrapper(agents=[core.ReinforceWrapper(Sender(OUTPUT_SIZE, INPUT_SIZE)),
                                                     core.ReinforceWrapper(Sender(OUTPUT_SIZE, INPUT_SIZE))])
    samples, log_probs, entropies = multi_agent(torch.Tensor(BATCH_SIZE, INPUT_SIZE),
                                                agent_indices=AGENT_INDICES)
    assert samples.shape == log_probs.shape == entropies.shape == torch.Size([BATCH_SIZE])

    multi_agent = GumbelSoftmaxMultiAgentWrapper(agents=[core.GumbelSoftmaxWrapper(Sender(OUTPUT_SIZE, INPUT_SIZE)),
                                                         core.GumbelSoftmaxWrapper(Sender(OUTPUT_SIZE, INPUT_SIZE))])
    samples = multi_agent(torch.Tensor(BATCH_SIZE, INPUT_SIZE), agent_indices=AGENT_INDICES)
    assert samples.shape == torch.Size([BATCH_SIZE, OUTPUT_SIZE])
