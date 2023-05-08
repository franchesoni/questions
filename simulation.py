"""This script simulates some simple annotation problems. 
These problems will be solved exactly using Huffman encoding and exhaustive tree search.
Then we will truncate the tree and solve the problem approximately using our methods.
Lastly we will add noise to the predictor.
"""
import os
import numpy as np
import tqdm
import json
from copy import deepcopy
import pygraphviz as pgv
import matplotlib.pyplot as plt
import seaborn as sns


###### utilities #######

def entropy_given_probs(probabilities):
    # assume independent probabilities
    return - np.sum(probabilities * np.log2(probabilities) + (1 - probabilities) * np.log2(1 - probabilities))

def get_power_01(size):
    """Returns all possible binary vectors of size `size`."""
    return [[int(d) for d in np.binary_repr(i, width=size)] for i in range(2**size)]

#### visualization #########
def box_and_whisker_plot(n_questions, entropies, name):
    # create two plots, one with two columns, one with the differences
    sns.set_theme()

    fig, ax = plt.subplots()
    plot = sns.boxplot(data=[n_questions, entropies], ax=ax)
    sns.stripplot(data=[n_questions, entropies], ax=ax)
    plot.set_xticklabels(['n_questions', 'entropy'])
    fig.savefig(f'{name}.png')
    plt.close()

    differences = [nq - entropy for nq, entropy in zip(n_questions, entropies)]
    fig, ax = plt.subplots()
    plot = sns.boxplot(data=[differences], ax=ax)
    sns.stripplot(data=[differences], ax=ax)
    plot.set_xticklabels(['n_questions - entropy'])
    plot.set_ylabel('difference')
    fig.savefig(f'{name}_diffs.png')
    plt.close()

def violin_plot(n_questions, entropies, name):
    differences = [nq - entropy for nq, entropy in zip(n_questions, entropies)]
    sns.set_theme()
    plt.figure()
    plot = sns.violinplot(data=[n_questions, entropies], inner="quartile")
    plot = sns.stripplot(data=[n_questions, entropies], alpha=0.3)
    plot.set_xticklabels(['n_questions', 'entropy'])
    # save fig using name
    plt.savefig(f'{name}.png')
    plt.close()




def visualize_huffman_tree(root_node, name, node_predictions=None):
    # pass predictions if you want extra information

    # Create an empty graph object
    graph = pgv.AGraph(directed=True, strict=True)

    # Define a helper function to traverse the tree recursively
    def traverse(node):
        # Create a node object with the state or guess as the label
        node_label = str(node.indices)
        if node_predictions is not None:
            label = node_label + "\n" + f"value = {expected_length(node)}\nentropy = {huffman_node_entropy(node_predictions, node)}\nprob = {node.prob}"
            graph.add_node(node_label, label=label) 
        graph.add_node(node_label) 
        # If the node has a parent, create an edge object with or without a label
        if node.parent:
            parent_label = str(node.parent.indices)
            prob = node.prob / np.sum([child.prob for child in node.parent.children])
            edge_label = f"prob = {prob:.2f}"
            graph.add_edge(parent_label, node_label, label=edge_label)
        # If the node has children, traverse them recursively
        if node.children:
            for child in node.children:
                traverse(child)

    # Start the traversal from the root node
    traverse(root_node)

    # Write the graph to a DOT file and a PNG file
    graph.write(f"{name}.dot")
    graph.layout(prog=['neato','dot','twopi','circo','fdp','nop'][1])
    graph.draw(f"{name}.png")


#### data ####
def generate_2d_gaussians_and_predictor(N, pos_center, pos_ratio):
    # get samples
    pos_ratio = int(N*pos_ratio) / N  # make exact
    neg_samples = np.random.multivariate_normal(
        [0, 0], [[1, 0], [0, 1]], int(N * (1 - pos_ratio))
    )
    pos_center = [2, 2]
    pos_samples = np.random.multivariate_normal(
        pos_center, [[1, 0], [0, 1]], int(N * pos_ratio)
    )
    assert len(pos_samples) + len(neg_samples) == N

    # our predictor is given by the same gaussians that generate the data
    def prob_neg_fn(x):
        # the probability of belonging to the negative class is given by the probability of the negative 2d gaussian
        return 1 / (2 * np.pi) * np.exp(-0.5 * (x[:, 0] ** 2 + x[:, 1] ** 2))

    def prob_pos_fn(x):
        # the probability of belonging to the positive class is given by the probability of the positive 2d gaussian
        return 1 / (2 * np.pi) * np.exp(-0.5 * ((x[:, 0] - pos_center[0]) ** 2 + (x[:, 1] - pos_center[1]) ** 2))

    def prob_of_being_positive(x):
        # P(y=1|x) = P(x|y=1)P(y=1) / (P(x, y=1) + P(x, y=0)) = P(x|y=1)P(y=1) / (P(x|y=1)P(y=1) + P(x|y=0)P(y=0))
        px_given_y1_times_py1 = prob_pos_fn(x) * pos_ratio
        py1_given_x = px_given_y1_times_py1 / (px_given_y1_times_py1 + prob_neg_fn(x) * (1 - pos_ratio))
        return py1_given_x

    return pos_samples, neg_samples, prob_of_being_positive

###### Huffman encoding #######


class HuffmanNode:
    def __init__(self, indices, prob, children=None):
        self.indices = indices
        self.prob = prob
        self.children = children
        self.parent = None  # because we build the tree from the bottom up


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
    N = len(predictions)
    binary_vectors = np.array(get_power_01(N))
    # for each one of the binary vectors we compute its probability as prod p_i^x_i (1-p_i)^(1-x_i) where x_i is the i-th element of the binary vector
    preds = predictions.reshape(1, -1)
    log_preds_for_vectors = np.sum(np.log(preds ** binary_vectors * (1 - preds) ** (1 - binary_vectors)), axis=1)
    preds_for_vectors = np.exp(log_preds_for_vectors)
    assert np.allclose(preds_for_vectors.sum(), 1)
    # build the huffman tree
    nodes = [HuffmanNode([i], pred) for i, pred in enumerate(preds_for_vectors)]
    nodes_without_parent = deepcopy(nodes)
    nodes_without_parent.sort(key=lambda x: x.prob)  # increasing
    while len(nodes_without_parent) > 1:
        child1, child2 = nodes_without_parent[:2]
        new_parent = HuffmanNode(
            child1.indices + child2.indices,
            child1.prob + child2.prob,
            [child1, child2]
        )
        child1.parent = new_parent
        child2.parent = new_parent
        nodes.append(new_parent)
        nodes_without_parent = [new_parent] + nodes_without_parent[2:]
        nodes_without_parent.sort(key=lambda x: x.prob)  # increasing

    root_node = nodes_without_parent[-1]
    return root_node, preds_for_vectors
    
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
    return entropy_given_probs(new_probabilities)



def main():
    results = {'entropies': [], 'n_questions_huffman': []}
    n_seeds, N, pos_center, pos_ratio = 1000, 10, [1,1], 0.5
    dstfile = f"simulation_results_{n_seeds}_{N}_{pos_center}_{pos_ratio}.pkl"
    if not os.path.exists(dstfile):
        for seed in tqdm.tqdm(range(n_seeds)):
            np.random.seed(seed)
            pos_samples, neg_samples, prob_of_being_positive = generate_2d_gaussians_and_predictor(N, pos_center, pos_ratio)
            predictions = prob_of_being_positive(np.concatenate((neg_samples, pos_samples), axis=0))
            labels = np.array([0] * len(neg_samples) + [1] * len(pos_samples))
            entropy = entropy_given_probs(predictions)

            # huffman
            huffman_tree, preds_for_vectors = huffman_encoding(predictions)
            # visualize_huffman_tree(huffman_tree, 'huffman_tree', preds_for_vectors)

            # run huffman questioning
            label_as_index = get_power_01(N).index(labels.tolist())
            assert label_as_index in huffman_tree.indices
            node = huffman_tree
            n_questions = 0
            while node.indices != [label_as_index]:
                n_questions += 1  # ask one question and move down the latter
                response = label_as_index in node.children[0].indices
                if response:
                    node = node.children[0]
                else:
                    node = node.children[1]
            results['entropies'].append(entropy)
            results['n_questions_huffman'].append(n_questions)
        with open(dstfile, 'w') as f:
            json.dump(results)
    with open(dstfile, 'r') as f:
        data = json.load(f)
        entropies = data['entropies']
        n_questions_huffman = data['n_questions_huffman']
    box_and_whisker_plot(n_questions_huffman, entropies, 'box_and_whisker')
    violin_plot(n_questions_huffman, entropies, 'violin')

            

if __name__ == '__main__':
    import cProfile
    cProfile.run('main()', sort='cumtime', filename='simulation.profile')


