from typing import List

import torch
import torch.nn as nn
from egg import core


class GSSequentialTeamworkGame(nn.Module):
    def __init__(
            self,
            sender_ensemble: 'GumbelSoftmaxMultiAgentEnsemble',
            receiver_ensemble: 'GumbelSoftmaxMultiAgentEnsemble',
            executive_sender: nn.Module,
            loss,
            executive_sender_entropy_coeff=0.1
    ):
        super(GSSequentialTeamworkGame, self).__init__()
        self.sender_ensemble = sender_ensemble
        self.receiver_ensemble = receiver_ensemble
        self.executive_sender = executive_sender
        self.loss = loss
        self.executive_sender_entropy_coeff = executive_sender_entropy_coeff

        self.mean_baseline = 0.0
        self.n_points = 0.0

    def forward(self, sender_input, _):
        indices, executive_sender_log_prob, executive_sender_entropy = self.executive_sender(sender_input)
        message = self.sender_ensemble(sender_input, agent_indices=indices)
        receiver_output = self.receiver_ensemble(message, agent_indices=indices)[:, -1, ...]

        loss, rest_info = self.loss(sender_input, message, None, receiver_output, None)
        advantage = (loss.detach() - self.mean_baseline)
        policy_loss = (advantage * executive_sender_log_prob).mean()
        entropy_loss = -executive_sender_entropy.mean() * self.executive_sender_entropy_coeff
        if self.training:
            self.n_points += 1.0
            self.mean_baseline += (loss.detach().mean().item() -
                                   self.mean_baseline) / self.n_points
        full_loss = policy_loss + entropy_loss + loss.mean()
        rest_info['baseline'] = self.mean_baseline
        rest_info['loss'] = loss.mean().item()
        rest_info['executive_sender_entropy'] = executive_sender_entropy.mean()
        return full_loss, rest_info


class ReinforceMultiAgentEnsemble(nn.Module):

    def __init__(self, agents: List[core.ReinforceWrapper]):
        super(ReinforceMultiAgentEnsemble, self).__init__()
        self.agents = nn.ModuleList(agents)

    def forward(self, input, agent_indices, **kwargs):
        samples, log_probs, entropies = zip(*(agent(input) for agent in self.agents))
        samples = torch.stack(samples, dim=0).gather(dim=0, index=agent_indices)
        log_probs = torch.stack(log_probs, dim=0).gather(dim=0, index=agent_indices)
        entropies = torch.stack(entropies, dim=0).gather(dim=0, index=agent_indices)
        return samples.squeeze(dim=0), log_probs.squeeze(dim=0), entropies.squeeze(dim=0)


class GumbelSoftmaxMultiAgentEnsemble(nn.Module):

    def __init__(self, agents: List[core.ReinforceWrapper]):
        super(GumbelSoftmaxMultiAgentEnsemble, self).__init__()
        self.agents = nn.ModuleList(agents)

    def forward(self, input, agent_indices, **kwargs):
        samples = [agent(input) for agent in self.agents]
        if samples[0].dim() > 2:  # RNN
            agent_indices = agent_indices.reshape(1, agent_indices.size(0), 1, 1).expand(1, agent_indices.size(0), samples[0].size(1), samples[0].size(2))
        else:
            agent_indices = agent_indices.reshape(1, agent_indices.size(0), 1).expand(1, agent_indices.size(0), samples[0].size(2))
        samples = torch.stack(samples, dim=0).gather(dim=0, index=agent_indices)
        return samples.squeeze(dim=0)


if __name__ == "__main__":
    from teamwork.agents import Sender
    BATCH_SIZE, INPUT_SIZE, OUTPUT_SIZE = 8, 10, 5
    AGENT_INDICES = torch.LongTensor([0, 1, 0, 1, 0, 1, 0, 1])
    multi_agent = ReinforceMultiAgentEnsemble(agents=[core.ReinforceWrapper(Sender(OUTPUT_SIZE, INPUT_SIZE)),
                                                      core.ReinforceWrapper(Sender(OUTPUT_SIZE, INPUT_SIZE))])
    samples, log_probs, entropies = multi_agent(torch.Tensor(BATCH_SIZE, INPUT_SIZE),
                                                agent_indices=AGENT_INDICES)
    assert samples.shape == log_probs.shape == entropies.shape == torch.Size([BATCH_SIZE])

    multi_agent = GumbelSoftmaxMultiAgentEnsemble(agents=[core.GumbelSoftmaxWrapper(Sender(OUTPUT_SIZE, INPUT_SIZE)),
                                                         core.GumbelSoftmaxWrapper(Sender(OUTPUT_SIZE, INPUT_SIZE))])
    samples = multi_agent(torch.Tensor(BATCH_SIZE, INPUT_SIZE), agent_indices=AGENT_INDICES)
    assert samples.shape == torch.Size([BATCH_SIZE, OUTPUT_SIZE])
