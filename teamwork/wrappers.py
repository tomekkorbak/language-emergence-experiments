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

    # def forward(self, input, agent_indices, **kwargs):
    #     batch_size = input.size(0)
    #     samples, log_probs, entropies = [], [], []
    #     for agent_index, agent in enumerate(self.agents):
    #         sample, log_prob, entropy = agent(input, **kwargs)
    #         mask = (torch.LongTensor([agent_index]).expand(batch_size) == agent_indices)
    #         samples.append(sample * mask.long())
    #         log_probs.append(log_prob * mask.float())
    #         entropies.append(entropy * mask.float())
    #     return torch.stack(samples).sum(dim=0), torch.stack(log_probs).sum(dim=0), torch.stack(entropies).sum(dim=0)

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
        batch_size = input.size(0)
        samples = []
        for agent_index, agent in enumerate(self.agents):
            sample = agent(input, **kwargs)
            mask = (torch.LongTensor([agent_index]).expand(batch_size) == agent_indices)
            if self.training:
                mask = mask.float()
            else:
                mask = mask.long()
            samples.append(sample)
        return torch.stack(samples).sum(dim=0)

    def forward(self, input, agent_indices, **kwargs):
        samples = torch.stack([agent(input, **kwargs) for agent in self.agents], dim=1)

        print(samples.shape)
        print(agent_indices.shape)

        samples = samples.gather(dim=0, index=agent_indices)
        return samples

if __name__ == "__main__":
    multi_agent = ReinforceMultiAgentWrapper(agents=[core.ReinforceWrapper(Sender(10, 10)),
                                                     core.ReinforceWrapper(Sender(10, 10))])
    samples, log_probs, entropies = multi_agent(torch.Tensor(8, 10),
                                                agent_indices=torch.LongTensor([0, 1, 0, 1, 0, 1, 0, 1]))
    print(samples, log_probs, entropies)
    # assert samples.shape == log_probs.shape == entropies.shape == torch.Size([8])
    #
    # multi_agent = GumbelSoftmaxMultiAgentWrapper(agents=[core.GumbelSoftmaxWrapper(Sender(10, 10)),
    #                                                      core.GumbelSoftmaxWrapper(Sender(10, 10))])
    # samples = multi_agent(torch.Tensor(8, 10), agent_indices=torch.LongTensor([0, 1, 0, 1, 0, 1, 0, 1]))
    # assert samples.shape == torch.Size([8, 10])
