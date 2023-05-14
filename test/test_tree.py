from pathlib import Path
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pygraphviz as pgv
from sklearn.calibration import calibration_curve
import cProfile
import pstats


from tree_and_state import (
    get_node_with_state,
    initialize_tree,
    tree_search,
    update_state,
    ActionNode,
    StateNode,
)


def calibration_plot(probs_and_outcomes, bins=10, name="calibration_plot"):
    # split the list into two lists: probabilities and outcomes
    probabilities = [p[0] for p in probs_and_outcomes]
    outcomes = [p[1] for p in probs_and_outcomes]

    # compute the calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        outcomes, probabilities, n_bins=4
    )

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
    from tree_and_state import StateNode, ActionNode

    # Create an empty graph object
    graph = pgv.AGraph(directed=True, strict=True)

    # Define a helper function to traverse the tree recursively
    def traverse(node):
        # Create a node object with the state or guess as the label
        node_label = (
            str(node.state)
            if isinstance(node, StateNode)
            else (str(node.guess) + " at " + str(node.parent.state))
        )
        color = (
            "red"
            if isinstance(node, ActionNode)
            else ("green" if node == root_node else "blue")
        )
        graph.add_node(
            node_label,
            color=color,
            shape="box" if isinstance(node, ActionNode) else "ellipse",
        )
        # If the node has a parent, create an edge object with or without a label
        if node.parent:
            parent_label = (
                str(node.parent.state)
                if isinstance(node.parent, StateNode)
                else (str(node.parent.guess) + " at " + str(node.parent.parent.state))
            )
            if isinstance(node.parent, ActionNode):
                state_prob = float(
                    node.parent.children_probs[node.parent.state_children.index(node)][
                        0
                    ]
                )
                edge_label = f"prob = {state_prob:.2f}, cost = {float(node.cost):.2f}"
            else:
                edge_label = f"cost = {float(node.cost):.2f}"
            graph.add_edge(parent_label, node_label, label=edge_label)
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
    graph.layout(prog=["neato", "dot", "twopi", "circo", "fdp", "nop"][1])
    graph.draw(f"{name}.png")


def plot_model_outputs(
    model,
    pos_center,
    X,
    y,
    X_to_annotate,
    y_to_annotate,
    indices_to_ask,
    predictions,
    X_annotated,
    y_annotated,
    name,
):
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
        plt.scatter(
            X_to_annotate[:, 0],
            X_to_annotate[:, 1],
            c=y_to_annotate,
            cmap=plt.cm.Paired,
            edgecolors="k",
            norm=norm,
        )
        # draw a circle around those indices to ask
        plt.scatter(
            X[indices_to_ask, 0],
            X[indices_to_ask, 1],
            c=np.round(np.array(predictions).squeeze()),
            cmap=plt.cm.Paired,
            marker="o",
            s=200,
            alpha=0.5,
            norm=norm,
        )
    # plot the annotated data
    plt.scatter(
        X_annotated[:, 0],
        X_annotated[:, 1],
        c=y_annotated,
        cmap=plt.cm.Set3,
        edgecolors="k",
        marker="s",
        norm=norm,
    )
    # draw the level lines of the two gaussians that generate the data
    plt.contour(
        x1,
        x2,
        1 / np.sqrt(2 * np.pi) * np.exp(-(x1**2 + x2**2) / 2),
        levels=3,
        colors="b",
    )
    plt.contour(
        x1,
        x2,
        1
        / np.sqrt(2 * np.pi)
        * np.exp(-((x1 - pos_center[0]) ** 2 + (x2 - pos_center[1]) ** 2) / 2),
        levels=3,
        colors="r",
    )

    plt.savefig(name + ".png")
    plt.close()


def example_1d_gaussians():
    # create a synthetic example involving points sampled from two overlapped 1d gaussians
    # we want to see how the performance of the predictor changes as we add more labeled points
    # create the point samples
    N = 16
    pos_ratio = 0.5

    # get samples
    neg_samples = np.random.normal(0, 1, int(N * (1 - pos_ratio)))
    pos_samples = np.random.normal(1, 1, int(N * pos_ratio))

    # our predictor is given by the same gaussians that generate the data
    def prob_neg_fn(x):
        return 1 / np.sqrt(2 * np.pi) * np.exp(-(x**2) / 2)

    def prob_pos_fn(x):
        return 1 / np.sqrt(2 * np.pi) * np.exp(-((x - 1) ** 2) / 2)

    def prob_of_being_positive(x):
        return prob_pos_fn(x) / (prob_pos_fn(x) + prob_neg_fn(x))

    def prob_of_being_negative(x):
        return prob_neg_fn(x) / (prob_pos_fn(x) + prob_neg_fn(x))

    # make plot showing samples and probabilities
    x = np.linspace(-3, 3, 100)
    plt.plot(x, prob_of_being_positive(x), label="prob of being positive")
    plt.plot(x, prob_of_being_negative(x), label="prob of being negative")
    plt.scatter(
        neg_samples, np.zeros_like(neg_samples), color="r", label="negative samples"
    )
    plt.scatter(
        pos_samples, np.zeros_like(pos_samples), color="b", label="positive samples"
    )
    plt.legend()
    plt.show()


def example_2d_gaussians(max_expansions, max_n):
    # create a synthetic example with 2d gaussians
    # we want to visualize the points and the probability given by the predictor
    # also we want its decision boundary

    # create the point samples
    N = 100
    pos_ratio = 0.5

    # get samples
    neg_samples = np.random.multivariate_normal(
        [0, 0], [[1, 0], [0, 1]], int(N * (1 - pos_ratio))
    )
    pos_center = [2, 2]
    pos_samples = np.random.multivariate_normal(
        pos_center, [[1, 0], [0, 1]], int(N * pos_ratio)
    )

    # our predictor is given by the same gaussians that generate the data
    def prob_neg_fn(x):
        # the probability of belonging to the negative class is given by the probability of the negative 2d gaussian
        return 1 / (2 * np.pi) * np.exp(-0.5 * (x[:, 0] ** 2 + x[:, 1] ** 2))

    def prob_pos_fn(x):
        # the probability of belonging to the positive class is given by the probability of the positive 2d gaussian
        return (
            1
            / (2 * np.pi)
            * np.exp(
                -0.5 * ((x[:, 0] - pos_center[0]) ** 2 + (x[:, 1] - pos_center[1]) ** 2)
            )
        )

    def prob_of_being_positive(x):
        return prob_pos_fn(x) / (prob_pos_fn(x) + prob_neg_fn(x))

    state = {"labeled": [], "incorrect": []}
    inputs = np.concatenate((neg_samples, pos_samples), axis=0)
    labels = np.array([0] * len(neg_samples) + [1] * len(pos_samples))

    X, y = inputs, labels
    X_to_annotate = X
    y_to_annotate = y
    n_question = 0
    X_annotated = np.empty((0, 2))
    y_annotated = np.empty((0, 1))

    n_question = 0
    root_node = initialize_tree(state)
    probs_and_outcomes = []

    while len(state["labeled"]) < N:
        optimal_action, optimal_cost, correct_prob = tree_search(
            root_node,
            N=N,
            max_expansions=max_expansions,
            max_n=max_n,
            inputs=inputs,
            predictor=prob_of_being_positive,
        )
        # visualize_tree(root_node, name=f"aia/tree_step_{n_question}")
        indices_to_ask, predictions = optimal_action
        # print(state)
        answer = (labels[indices_to_ask] == predictions).all()
        probs_and_outcomes.append((correct_prob, answer))
        # print(indices_to_ask, predictions, answer)
        plot_model_outputs(
            prob_of_being_positive,
            pos_center,
            X,
            y,
            X_to_annotate,
            y_to_annotate,
            indices_to_ask,
            predictions,
            X_annotated,
            y_annotated,
            name=f"aia/report_step_{n_question}",
        )
        state = update_state(state, optimal_action, answer)
        root_node = get_node_with_state(root_node, state)
        root_node.priority = 1
        root_node.parent = None

        n_question += 1
        annotated_indices = [ind for ind, _ in state["labeled"]]
        unlabeled_indices = [
            ind for ind in range(len(X)) if ind not in annotated_indices
        ]
        X_annotated = X[annotated_indices]
        y_annotated = y[annotated_indices]
        X_to_annotate = X[unlabeled_indices]
        y_to_annotate = y[unlabeled_indices]
        # breakpoint()

    plot_model_outputs(
        prob_of_being_positive,
        pos_center,
        X,
        y,
        X_to_annotate,
        y_to_annotate,
        indices_to_ask,
        predictions,
        X_annotated,
        y_annotated,
        name=f"aia/report_step_{n_question}",
    )
    calibration_plot(probs_and_outcomes, name=f"aia/calibration_plot")

    return True


def test():
    for max_expansions in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        for max_n in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            for seed in range(10):
                np.random.seed(seed)
                try:
                    if example_2d_gaussians(max_expansions, max_n):
                        print(
                            f"success! max_expansions: {max_expansions}, max_n: {max_n}, seed: {seed}"
                        )
                except:
                    print(
                        "failed at max_expansions: {}, max_n: {}, seed: {}".format(
                            max_expansions, max_n, seed
                        )
                    )
                    exit()
    print("=" * 80)
    print("You're awesome!")


def main():
    shutil.rmtree("aia", ignore_errors=True)
    Path("aia").mkdir(parents=True, exist_ok=True)
    max_expansions = 10
    max_n = 10
    seed = 0
    np.random.seed(seed)
    example_2d_gaussians(max_expansions, max_n)


if __name__ == "__main__":
    cProfile.run("main()", "profile.dat")


# max_expansions = 3
# max_n = 9
# seed = 0
# np.random.seed(seed)
# if example_2d_gaussians(max_expansions, max_n):
#     print(f"success! max_expansions: {max_expansions}, max_n: {max_n}, seed: {seed}")

# example_1d_gaussians()
