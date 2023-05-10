from pathlib import Path
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pygraphviz as pgv
from sklearn.calibration import calibration_curve
import cProfile
import textwrap

import simple_tree_search as sts
from simple_tree_search import StateNode, ActionNode
from simulation import generate_2d_gaussians_and_predictor
import visualization as vis 



def calibration_plot(probs_and_outcomes, bins=10, name="calibration_plot"):
    # split the list into two lists: probabilities and outcomes
    probabilities = [p[0] for p in probs_and_outcomes]
    outcomes = [p[1] for p in probs_and_outcomes]

    # compute the calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(outcomes, probabilities, n_bins=4)

    # create a calibration plot with probabilities on the x-axis and fraction of positives on the y-axis
    plt.plot(mean_predicted_value, fraction_of_positives, "s-")
    plt.plot([0, 1], [0, 1], "--")

    # add labels and title
    plt.xlabel("Predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration plot")

    # save the plot
    plt.savefig(name + ".png")




def visualize_tree(root_node, name):
    # Create an empty graph object
    graph = pgv.AGraph(directed=True, strict=True)

    # Define a helper function to traverse the tree recursively
    def traverse(node):
        # Create a node object with the state or guess as the label
        node_id = str(node.state) if isinstance(node, StateNode) else (str(node.guess) + " at " + str(node.parent.state))
        node_text = f"N={len(node.state['indices'])}, incorrect= {node.state['incorrect']}, cost = {node.cost:.2f}"  if isinstance(node, StateNode) else f"guess={node.guess}, cost={node.cost:.2f}"
        node_text = textwrap.fill(node_text, 2*int(np.sqrt(len(node_text)))) if len(node_text)>30 else node_text
        color = "red" if isinstance(node, ActionNode) else ("green" if node == root_node else "blue")
        graph.add_node(node_id, color=color, shape="box" if isinstance(node, ActionNode) else "ellipse", label=node_text)
        # If the node has a parent, create an edge object with or without a label
        if node.parent:
            parent_label = str(node.parent.state) if isinstance(node.parent, StateNode) else (str(node.parent.guess) + " at " + str(node.parent.parent.state))
            if isinstance(node.parent, ActionNode):
                state_prob = float(node.parent.children_probs[node.parent.state_children.index(node)])
                edge_label = f"prob = {state_prob:.2f}, cost = {float(node.cost):.2f}"
            else:
                edge_label = f"cost = {float(node.cost):.2f}"
            graph.add_edge(parent_label, node_id, label=edge_label)
        # If the node has children, traverse them recursively
        if isinstance(node, StateNode) and node.action_children:
            for child in node.action_children:
                traverse(child)
        elif isinstance(node, ActionNode) and node.state_children:
            for child in node.state_children:
                traverse(child)

    # Start the traversal from the root node
    traverse(root_node)

    # Write the graph to a DOT file and a PNG file
    graph.write(f"{name}.dot")
    graph.layout(prog=['neato','dot','twopi','circo','fdp','nop'][1])
    graph.draw(f"{name}.png")



def plot_model_outputs(model, pos_center, X, y, X_to_annotate, y_to_annotate, indices_to_ask, predictions, X_annotated, y_annotated, name):
  # generate the background using a grid
  x2s = np.linspace(-5, 5, 100)
  x1s = np.linspace(-5, 5, 100)
  x1, x2 = np.meshgrid(x1s, x2s)
  xs = np.stack([x1.reshape(-1), x2.reshape(-1)], axis=1)
  # predict the probability of each point in the grid
  ys = model(xs)
  # plot the background
  plt.figure()
  plt.contourf(x1, x2, ys.reshape(x1.shape))
  plt.colorbar()
  norm = colors.Normalize(vmin=0, vmax=1)
  if len(X_to_annotate) > 0:
    # plot the data
    plt.scatter(X_to_annotate[:, 0], X_to_annotate[:, 1], c=y_to_annotate, cmap=plt.cm.Paired, edgecolors="k", norm=norm)
    # draw a circle around those indices to ask 
    plt.scatter(X[indices_to_ask, 0], X[indices_to_ask, 1], c=np.round(np.array(predictions).squeeze()), cmap=plt.cm.Paired, marker="o", s=200, alpha=0.5, norm=norm)
  # plot the annotated data
  plt.scatter(X_annotated[:, 0], X_annotated[:, 1], c=y_annotated, cmap=plt.cm.Set3, edgecolors="k", marker="s", norm=norm)
  # draw the level lines of the two gaussians that generate the data
  plt.contour(x1, x2, 1 / np.sqrt(2 * np.pi) * np.exp(-(x1**2+x2**2) / 2), levels=3, colors="b")
  plt.contour(x1, x2, 1 / np.sqrt(2 * np.pi) * np.exp(-((x1 - pos_center[0]) ** 2+(x2-pos_center[1])**2) / 2), levels=3, colors="r")

  plt.savefig(name + ".png")
  plt.close()

def extend_results(results, name, value):
    if name not in results:
        results[name] = [value]
    else:
        results[name].append(value)
    return results




def example_2d_gaussians():
    # create a synthetic example with 2d gaussians
    # we want to visualize the points and the probability given by the predictor
    # also we want its decision boundary

    # create the point samples
    N = 100
    pos_ratio = 0.5
    pos_center = [2, 2]
    results = {}
    pos_samples, neg_samples, prob_of_being_positive = generate_2d_gaussians_and_predictor(N, pos_center, pos_ratio)

    inputs = np.concatenate((neg_samples, pos_samples), axis=0)
    labels = np.array([0] * len(neg_samples) + [1] * len(pos_samples))
    predictions = prob_of_being_positive(
            inputs
        )
    entropy = sts.entropy_given_probs_binary(predictions)
    extend_results(results, "entropies", entropy)
    # check that the function prob_of_being_positive works correctly
    vis.calibration_plot(predictions, labels, name="zcalibration_plot")
    alia = sts.STS(max_expansions=10, al_method=None, max_n=10, cost_fn="entropy")

    X, y = inputs, labels
    X_to_annotate = X
    y_to_annotate = y
    n_question = 0
    X_annotated = np.empty((0, 2))
    y_annotated = np.empty((0, 1))

    state = {"indices": list(range(N)), 'incorrect': None}
    root_node = sts.STS.get_root_node(state)
    new_predictions = (list(range(len(predictions))), list(predictions))
    alia.set_unlabeled_predictions(new_predictions)
    probs_and_outcomes = []
    estimated_costs = []
    labels_so_far = ([], [])
    n_question = 0

    while len(root_node.state["indices"]) > 0:
        # print and erase n question
        print(f"question {n_question}", end="\r")
        question, optimal_cost, correct_prob = alia.tree_search(root_node)
        # if n_question % 10 == 0:
        #     visualize_tree(root_node, name=f"alia/tree_step_{n_question}")
        estimated_costs.append((optimal_cost, n_question, sts.entropy_given_probs_binary(np.array(new_predictions[1]))))
        assert 0 < len(question[0]), "you should ask something"

        indices_to_ask, guess = question
        answer = (labels[indices_to_ask] == guess).all()
        probs_and_outcomes.append((correct_prob, answer))
        # if n_question % 10 < 3:
        plot_model_outputs(prob_of_being_positive, pos_center, X, y, X_to_annotate, y_to_annotate, indices_to_ask, guess, X_annotated, y_annotated, name=f"alia/report_step_{n_question}")
        state = sts.STS.update_state(state, question, answer)
        new_predictions = np.array(predictions)[state["indices"]]
        alia.set_unlabeled_predictions((state['indices'], new_predictions.tolist()))
        root_node = sts.STS.get_root_node(state)

        n_question += 1
        annotated_indices = sorted(list(set(range(N)) - set(state['indices'])))
        unlabeled_indices = sorted(state['indices'])
        X_annotated = X[annotated_indices]
        y_annotated = y[annotated_indices]
        X_to_annotate = X[unlabeled_indices]
        y_to_annotate = y[unlabeled_indices]
    estimated_costs = [{'expected length':cost, 'n questions':n_question-q, 'entropy':entropy} for cost, q, entropy in estimated_costs]
    extend_results(results, "estimated_costs", estimated_costs)

    vis.visualize_cost(estimated_costs, name="alia/estimated_costs")
    plot_model_outputs(prob_of_being_positive, pos_center, X, y, X_to_annotate, y_to_annotate, indices_to_ask, predictions, X_annotated, y_annotated, name=f"alia/report_step_{n_question}")
    calibration_plot(probs_and_outcomes, name=f"alia/calibration_plot")

    return True

def test():
    for max_expansions in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        for max_n in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]: 
            for seed in range(10):
                np.random.seed(seed)
                try:
                    if example_2d_gaussians(max_expansions, max_n):
                        print(f"success! max_expansions: {max_expansions}, max_n: {max_n}, seed: {seed}")
                except:
                    print("failed at max_expansions: {}, max_n: {}, seed: {}".format(max_expansions, max_n, seed))
                    exit()
    print('='*80)
    print("You're awesome!")





def main():
    shutil.rmtree('alia', ignore_errors=True)
    Path('alia').mkdir(parents=True, exist_ok=True)
    seed = 0
    np.random.seed(seed)
    example_2d_gaussians()

if __name__ == "__main__":
    cProfile.run('main()', 'profile.dat')


# max_expansions = 3
# max_n = 9
# seed = 0
# np.random.seed(seed)
# if example_2d_gaussians(max_expansions, max_n):
#     print(f"success! max_expansions: {max_expansions}, max_n: {max_n}, seed: {seed}")

# example_1d_gaussians()
