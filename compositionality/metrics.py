import collections

import neptune
import torch


def compute_context_independence(concept_symbol_matrix, opts, exclude_indices=None):
    v_cs = concept_symbol_matrix.argmax(dim=1)
    context_independence_scores = torch.zeros(opts.n_features * opts.n_attributes)
    for concept in range(concept_symbol_matrix.size(0)):
        v_c = v_cs[concept]
        p_vc_c = concept_symbol_matrix[concept, v_c] / concept_symbol_matrix[concept, :].sum(dim=0)
        p_c_vc = concept_symbol_matrix[concept, v_c] / concept_symbol_matrix[:, v_c].sum(dim=0)
        context_independence_scores[concept] = p_vc_c * p_c_vc

    return context_independence_scores, v_cs


def compute_concept_symbol_matrix(input_to_message, opts, epsilon=1e-4):
    concept_to_message = collections.defaultdict(list)
    for (concept1, concept2), messages in input_to_message.items():
            concept_to_message[concept1] += messages
            concept_to_message[opts.n_features + concept2] += messages
    concept_symbol_matrix = torch.FloatTensor(opts.n_features * opts.n_attributes,
                                              opts.vocab_size).fill_(epsilon)
    for concept, messages in concept_to_message.items():
        for message in messages:
            for symbol in message:
                concept_symbol_matrix[concept, symbol] += 1
    return concept_symbol_matrix
