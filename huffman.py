import numpy as np
from copy import deepcopy


def entropy_given_probs_categorical(probabilities):
    # assume independent probabilities
    return -np.sum(probabilities * np.log2(probabilities))


def get_power_01(size):
    """Returns all possible binary vectors of size `size`."""
    return [[int(d) for d in np.binary_repr(i, width=size)] for i in range(2**size)]


def expected_length(huffman_node):
    if not huffman_node.children and len(huffman_node.indices) == 1:
        return 0
    elength = 1
    probs = np.array([huffman_node.prob for huffman_node in huffman_node.children])
    for child, prob in zip(huffman_node.children, probs):
        elength += prob * expected_length(child) / np.sum(probs)
    return elength


def huffman_node_entropy(predictions, huffman_node):
    if len(huffman_node.indices) == 1:
        return 0
    # the probability assigned to the predictions given a state is p(Y=y|s) = p(s|Y=y) p(Y=y) / p(s) = 1(y in s) p(Y=y) / p(s)
    new_probabilities = np.array([predictions[index] for index in huffman_node.indices])
    new_probabilities /= new_probabilities.sum()
    return entropy_given_probs_categorical(new_probabilities)


###### Huffman encoding #######
class HuffmanNode:
    def __init__(self, indices, prob, children=None):
        self.indices = indices
        self.prob = prob
        self.children = children
        self.parent = None  # because we build the tree from the bottom up

def get_pred_for_vectors(predictions):
    N = len(predictions)
    binary_vectors = np.array(get_power_01(N))
    # for each one of the binary vectors we compute its probability as prod p_i^x_i (1-p_i)^(1-x_i) where x_i is the i-th element of the binary vector
    preds = predictions.reshape(1, -1)
    log_preds_for_vectors = np.sum(
        np.log(preds**binary_vectors * (1 - preds) ** (1 - binary_vectors)), axis=1
    )
    preds_for_vectors = np.exp(log_preds_for_vectors)
    return preds_for_vectors

def huffman_encoding(predictions):
    # bla bla from copilot:
    # Huffman encoding is a way to encode a binary vector of length N
    # such that the expected number of bits to encode it is minimized
    # the expected number of bits is given by the entropy of the distribution
    # the entropy of a binary distribution is given by:
    # H(p) = -p log(p) - (1-p) log(1-p)
    # we can use this to find the optimal encoding
    # https://en.wikipedia.org/wiki/Huffman_coding
    # https://www.youtube.com/watch?v=dM6us854Jk0
    # https://www.youtube.com/watch?v=JsTptu56GM8
    # https://www.youtube.com/watch?v=ZdooBTdW5bM
    # https://www.youtube.com/watch?v=umTbivyJoiI
    preds_for_vectors = get_pred_for_vectors(predictions)
    assert np.allclose(preds_for_vectors.sum(), 1)
    # build the huffman tree
    nodes = [HuffmanNode([i], pred) for i, pred in enumerate(preds_for_vectors)]
    nodes_without_parent = deepcopy(nodes)
    nodes_without_parent.sort(key=lambda x: x.prob)  # increasing
    while len(nodes_without_parent) > 1:
        child1, child2 = nodes_without_parent[:2]
        new_parent = HuffmanNode(
            child1.indices + child2.indices, child1.prob + child2.prob, [child1, child2]
        )
        child1.parent = new_parent
        child2.parent = new_parent
        nodes.append(new_parent)
        nodes_without_parent = [new_parent] + nodes_without_parent[2:]
        nodes_without_parent.sort(key=lambda x: x.prob)  # increasing

    root_node = nodes_without_parent[-1]
    return root_node, preds_for_vectors


####### huffman analysis ########
def analyse_huffman_tree(root_node, node_predictions):
    """Here we compute for each node the expected length, the entropy and whether the action chosen is the most likely."""
    sorted_indices = np.argsort(node_predictions)[::-1]  # from larger to smaller
    results = {"expected_lengths": [], "entropies": [], "is_most_likely": []}

    # Define a helper function to traverse the tree recursively
    def traverse(node):
        # Create a node object with the state or guess as the label
        exp_length = expected_length(node)
        node_entropy = huffman_node_entropy(node_predictions, node)
        # if node has a parent, node is an action and we can see if it contains (or not) the most likely indices
        if node.parent:
            if (
                len(node.indices) == 1
                or len(set(sorted_indices[: len(node.indices)]) - set(node.indices))
                == 1
            ):
                is_most_likely = True  # we assume n=1 doesn't count
            else:
                is_most_likely = (
                    set(sorted_indices[: len(node.indices)]) == set(node.indices)
                ) or (
                    len(
                        set(
                            sorted_indices[: len(sorted_indices) - len(node.indices)]
                        ).intersection(set(node.indices))
                    )
                    == 0
                )
        else:
            is_most_likely = None
        results["expected_lengths"].append(exp_length)
        results["entropies"].append(node_entropy)
        results["is_most_likely"].append(is_most_likely)

        # If the node has children, traverse them recursively
        if node.children:
            for child in node.children:
                traverse(child)

    # Start the traversal from the root node
    traverse(root_node)

    return results
