from collections import OrderedDict, defaultdict
from functools import reduce
from operator import mul

import numpy as np
from scipy.stats import spearmanr


def compute_concept_symbol_matrix(input_to_message, input_dimensions=[3, 5], vocab_size=10, epsilon=1e-5):
    number_of_inputs, number_of_concepts = reduce(mul, input_dimensions), sum(input_dimensions)
    concept_to_message = defaultdict(list)
    concepts = [None] * sum(input_dimensions)
    for (concept1, concept2), messages in input_to_message.items():
        concept_to_message['1_' + str(concept1)] += messages
        concept_to_message['2_' + str(concept2)] += messages
    concept_to_message = OrderedDict(sorted(concept_to_message.items()))
    assert len(concept_to_message.keys()) == number_of_concepts, \
        f'Out of {number_of_concepts} concepts {len(concept_to_message.keys())} are instantiated'
    concept_symbol_matrix = np.ndarray((sum(input_dimensions), vocab_size))
    concept_symbol_matrix.fill(epsilon)
    for i, (concept, messages) in enumerate(concept_to_message.items()):
        for message in messages:
            for symbol in message:
                concept_symbol_matrix[i, symbol] += 1
                concepts[i] = concept
    return concept_symbol_matrix, concepts


def compute_context_independence(concept_symbol_matrix, input_dimensions=[3, 5], vocab_size=10, exclude_indices=None):
    number_of_inputs, number_of_concepts = reduce(mul, input_dimensions), sum(input_dimensions)
    v_cs = concept_symbol_matrix.argmax(axis=1)
    assert v_cs.shape == (number_of_concepts,)
    context_independence_scores = np.zeros(number_of_concepts)
    for concept in range(number_of_concepts):
        v_c = v_cs[concept]
        p_vc_c = concept_symbol_matrix[concept, v_c] / concept_symbol_matrix[concept, :].sum(axis=0)
        p_c_vc = concept_symbol_matrix[concept, v_c] / concept_symbol_matrix[:, v_c].sum(axis=0)
        context_independence_scores[concept] = p_vc_c * p_c_vc
    return context_independence_scores, v_cs


def distance(input1, input2):
    max_len = max(len(input1), len(input2))
    input_1_padded = list(input1) + [-1] * (max_len - len(input1))
    input_2_padded = list(input2) + [-1] * (max_len - len(input2))
    return sum([a != b for a, b in (zip(input_1_padded, input_2_padded))])


def compute_input_similarity_matrix(input_to_message, input_dimensions):
    number_of_inputs, number_of_concepts = reduce(mul, input_dimensions), sum(input_dimensions)
    matrix = np.ndarray((number_of_inputs, number_of_inputs))
    for i, input1 in enumerate(input_to_message.keys()):
        for j, input2 in enumerate(input_to_message.keys()):
            matrix[i, j] = distance(input1, input2)
    return matrix


def compute_message_similarity_matrix(input_to_message, input_dimensions):
    number_of_inputs, number_of_concepts = reduce(mul, input_dimensions), sum(input_dimensions)
    matrix = np.ndarray((number_of_inputs, number_of_inputs))
    for i, (input1, messages1) in enumerate(input_to_message.items()):
        for j, (input2, messages2) in enumerate(input_to_message.items()):
            matrix[i, j] = distance(messages1[0], messages2[0])
    return matrix


def compute_representation_similarity(input_to_message, input_dimensions=[3, 5]):
    input_similarity = get_upper_triangular_matrix(compute_input_similarity_matrix(input_to_message, input_dimensions))
    message_similarity = get_upper_triangular_matrix(compute_message_similarity_matrix(input_to_message, input_dimensions))
    correlation_coeff, p_value = spearmanr(input_similarity, message_similarity)
    return correlation_coeff, p_value


def get_upper_triangular_matrix(matrix):
    m = matrix.shape[0]
    r, c = np.triu_indices(m, 1)
    return matrix[r, c]


if __name__ == "__main__":
    from collections import namedtuple
    opts = namedtuple('opts', ['n_features', 'n_attributes', 'vocab_size'])
    opts = opts(n_features=6, n_attributes=2, vocab_size=10)
    input_to_message = {
        (1, 1): [[7, 6]],
        (1, 2): [[2, 6]],
        (1, 3): [[4, 6]],
        (1, 4): [[5, 6]],
        (1, 5): [[8, 6]],
        (2, 1): [[7, 0]],
        (2, 2): [[2, 0]],
        (2, 3): [[4, 0]],
        (2, 4): [[5, 0]],
        (2, 5): [[8]],
        (3, 1): [[7, 3]],
        (3, 2): [[2, 3]],
        (3, 3): [[4, 3]],
        (3, 4): [[1]],
        (3, 5): [[9]],
    }

    np.set_printoptions(precision=2, suppress=True)
    concept_symbol_matrix, concepts = compute_concept_symbol_matrix(input_to_message)
    context_independence_scores, v_cs = compute_context_independence(concept_symbol_matrix)
    print(context_independence_scores.mean(axis=0))

    input_dimensions = [3, 5]
    correlation_coeff, p_value = compute_representation_similarity(input_to_message, input_dimensions)
    print(correlation_coeff)