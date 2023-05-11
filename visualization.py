import numpy as np
import os
import torchvision
import torch

# import pygraphviz as pgv
import matplotlib.pyplot as plt
import seaborn as sns

import huffman


def plot_curves(performances, experiment_name):
    """Plot performance curves by showing test loss and accuracy, recall, precision and specificity during training with respect to epochs. Also adds error bars for each metric.
    Args:
        performance (list): list of dicts containing ['loss', 'tp', 'fp', 'tn', 'fn'] as keys for each epoch
        experiment_name (str): name of the experiment
    """
    # compute accuracy, recall, precision and specificity from tp, fp, tn, fn
    formated_performances = []
    for performance in performances:
        for epoch in performance:
            tp = epoch["tp"]
            fp = epoch["fp"]
            tn = epoch["tn"]
            fn = epoch["fn"]
            epoch["accuracy"] = (tp + tn) / (tp + fp + tn + fn)
            epoch["recall"] = tp / (tp + fn)
            epoch["precision"] = tp / (tp + fp)
            epoch["specificity"] = tn / (tn + fp)

        # convert perfromance to dict of lists
        performance = {k: [epoch[k] for epoch in performance] for k in performance[0]}
        formated_performances.append(performance)

    # average each metric over all experiments
    mean_performance_curve = {
        k: [
            np.mean([performance[k][i] for performance in formated_performances])
            for i in range(len(performance[k]))
        ]
        for k in formated_performances[0]
    }
    std_of_performance_curve = {
        k: [
            np.std([performance[k][i] for performance in formated_performances])
            for i in range(len(performance[k]))
        ]
        for k in formated_performances[0]
    }

    # plot mean and std of performances for each metric shadowing the interval between 1 std
    fig, ax = plt.subplots(1, 5, figsize=(20, 5))

    for i, metric in enumerate(
        ["loss", "accuracy", "recall", "precision", "specificity"]
    ):
        ax[i].plot(mean_performance_curve[metric])
        ax[i].fill_between(
            range(len(mean_performance_curve[metric])),
            np.array(mean_performance_curve[metric])
            - np.array(std_of_performance_curve[metric]),
            np.array(mean_performance_curve[metric])
            + np.array(std_of_performance_curve[metric]),
            alpha=0.3,
        )
        ax[i].set_title(metric)
        ax[i].set_xlabel("epochs")
        ax[i].set_ylabel(metric)
    plt.suptitle(experiment_name)
    plt.show()


###### utilities #######


#### visualization #########
def box_and_whisker_plot(results, name):
    # create two plots, one with two columns, one with the differences
    sns.set_theme()
    n_questions_results = {
        key[12:]: results[key] for key in results if key.startswith("n_questions_")
    }
    entropies = results["entropies"]
    to_plot_names = list(n_questions_results.keys()) + ["entropies"]
    to_plot_values = list(n_questions_results.values()) + [entropies]

    fig, ax = plt.subplots()
    plot = sns.boxplot(data=to_plot_values, ax=ax)
    sns.stripplot(data=to_plot_values, ax=ax)
    plot.set_xticklabels(to_plot_names)
    fig.savefig(f"{name}.png")
    plt.close()

    # plot differences
    differences = {
        key: [nq - entropy for nq, entropy in zip(n_questions_results[key], entropies)]
        for key in n_questions_results
    }
    to_plot_names = list(differences.keys())
    to_plot_values = list(differences.values())
    fig, ax = plt.subplots()
    plot = sns.boxplot(data=to_plot_values, ax=ax)
    sns.stripplot(data=to_plot_values, ax=ax)
    plot.set_xticklabels(to_plot_names)
    plot.set_ylabel("n questions - entropy")
    fig.savefig(f"{name}_diffs.png")
    plt.close()


def violin_plot(n_questions, entropies, name):
    raise NotImplementedError
    differences = [nq - entropy for nq, entropy in zip(n_questions, entropies)]
    sns.set_theme()
    plt.figure()
    plot = sns.violinplot(data=[n_questions, entropies], inner="quartile")
    plot = sns.stripplot(data=[n_questions, entropies], alpha=0.3)
    plot.set_xticklabels(["n_questions", "entropy"])
    # save fig using name
    plt.savefig(f"{name}.png")
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
            label = (
                node_label
                + "\n"
                + f"value = {huffman.expected_length(node)}\nentropy = {huffman.huffman_node_entropy(node_predictions, node)}\nprob = {node.prob}"
            )
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
    graph.layout(prog=["neato", "dot", "twopi", "circo", "fdp", "nop"][1])
    graph.draw(f"{name}.png")
    plt.close()


def visualize_huffman_analysis(results, name):
    """We now create a scatter plot comparring expected length and entropy."""
    if isinstance(results, list):
        # join the lists under every key
        new_results = {}
        for analysis in results:
            for key, values in analysis.items():
                if key in new_results:
                    new_results[key].extend(values)
                else:
                    new_results[key] = values.copy()
        results = new_results

    sns.set_theme()
    fig, ax = plt.subplots()
    plt.title("Huffman expected number of questions")
    # make scatter
    plot = sns.scatterplot(data=results, x="entropies", y="expected_lengths", ax=ax)
    plot.set_ylabel("expected number of questions")
    # plot identity line
    limit = max(np.max(results["expected_lengths"]), np.max(results["entropies"]))
    sns.lineplot(x=[0, limit + 1], y=[0, limit + 1], ax=ax)
    fig.savefig(f"{name}.png")
    plt.close()


from sklearn.calibration import calibration_curve


def calibration_plot(probabilities, outcomes, bins=4, name="calibration_plot"):
    # compute the calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        outcomes, probabilities, n_bins=bins
    )

    plt.figure()
    # create a calibration plot with probabilities on the x-axis and fraction of positives on the y-axis
    plt.plot(mean_predicted_value, fraction_of_positives, "s-")
    plt.plot([0, 1], [0, 1], "--")

    # add labels and title
    plt.xlabel("Predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration plot")

    # save the plot
    plt.savefig(str(name) + ".png")
    plt.close()

def entropies_and_gold(entropies_and_gold, savename="testSTS_entropy"):
    # probs_and_outcomes is a dict of the form {name: [(prob, outcome), ...]}
    sns.set_theme()
    plt.figure()
    maximum = 0
    for name in entropies_and_gold:
        entropies = [p for p, o in entropies_and_gold[name]]
        gold = [o for p, o in entropies_and_gold[name]]
        sns.scatterplot(y=entropies, x=gold, label=name)
        maximum = max(maximum, max(max(entropies, gold)))
    maximum = np.ceil(maximum)
    plt.plot([0, maximum], [0, maximum], "--")
    plt.xlabel("Gold entropy")
    plt.ylabel("Computed entropy")
    plt.legend()
    plt.savefig(str(savename) + ".png")
    plt.close()



def calibration_plot2(probs_and_outcomes, bins=11, savename="testSTS"):
    # probs_and_outcomes is a dict of the form {name: [(prob, outcome), ...]}
    plt.figure()
    plt.plot([0, 1], [0, 1], "--")
    for name in probs_and_outcomes:
        probs = [p for p, o in probs_and_outcomes[name]]
        outcomes = [1*o for p, o in probs_and_outcomes[name]]
        # compute the calibration curve
        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                outcomes, probs, n_bins=bins
            )
        except:
            breakpoint()
            fraction_of_positives, mean_predicted_value = calibration_curve(
                outcomes, probs, n_bins=bins
            )
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label=name)
    plt.xlabel("Predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration plot")
    # save the plot
    plt.savefig(str(savename) + ".png")
    plt.close()




def cumulative_calibration_plot(
    probabilities, outcomes, name="cumulative_calibration_plot"
):
    probs = np.array(probabilities)
    outcomes = np.array(outcomes)

    # sort the probabilities and outcomes
    sorted_indices = np.argsort(probs)
    sorted_probs = probs[sorted_indices]
    sorted_outcomes = outcomes[sorted_indices]

    # compute the fraction of positives for each probability on the left and the right
    positives_from_the_left = np.cumsum(sorted_outcomes)
    total = len(sorted_outcomes)
    total_positives = np.sum(sorted_outcomes)
    fraction_of_positives_left = positives_from_the_left / (
        np.arange(1, len(sorted_outcomes) + 1)
    )
    fraction_of_positives_right = (total_positives - positives_from_the_left) / (
        total - np.arange(1, len(sorted_outcomes) + 1)
    )

    # plot the two curves and the identity
    plt.figure()
    plt.plot(sorted_probs, fraction_of_positives_left, "-", label="left")
    plt.plot(sorted_probs, fraction_of_positives_right, "-", label="right")
    # plt.plot(sorted_probs, fraction_of_positives_left+fraction_of_positives_right-1/2, "o", label="both")
    plt.plot([0, 1], [0, 1], "--")
    plt.legend()
    plt.savefig(str(name) + ".png")


def test_calibration():
    probs = np.random.rand(10000)
    more_probs = np.random.rand(100000)
    more_probs = more_probs[more_probs > 0.5]
    probs = np.concatenate((probs, more_probs))
    outcomes = np.array([np.random.rand() < prob for prob in probs])
    cumulative_calibration_plot(probs, outcomes, name="test_cumulative")
    calibration_plot(probs, outcomes, name="test_calibration")


def scatter_probs_and_outcomes(probs_and_outcomes, name):
    probs, outcomes = [x[0] for po in probs_and_outcomes for x in po], [
        x[1] for po in probs_and_outcomes for x in po
    ]
    calibration_plot(probs, outcomes, name=name)
    # cumulative_calibration_plot(probs, outcomes, name=str(name)+"_cumulative")


def visualize_cost(results, name):
    """We now create a scatter plot comparring expected length and entropy."""
    if isinstance(results, list):
        # join the lists under every key
        new_results = {}
        for entry in results:
            for key, value in entry.items():
                if key in new_results:
                    new_results[key].append(value)
                else:
                    new_results[key] = [value]
        results = new_results

    sns.set_theme()
    fig, ax = plt.subplots()
    plt.title("STS expected number of questions")
    # make scatter
    plot = sns.scatterplot(
        data=results, x="n questions", y="expected length", hue="entropy", ax=ax
    )
    limit = max(np.max(results["expected length"]), np.max(results["n questions"]))
    sns.lineplot(x=[0, limit + 1], y=[0, limit + 1], ax=ax)
    plot.set_ylabel("expected number of questions")
    fig.savefig(f"{name}_length_vs_nq.png")
    plt.close()
    fig, ax = plt.subplots()
    # make scatter
    plot = sns.scatterplot(
        data=results, x="entropy", y="expected length", hue="n questions", ax=ax
    )
    limit = max(np.max(results["expected length"]), np.max(results["entropy"]))
    sns.lineplot(x=[0, limit + 1], y=[0, limit + 1], ax=ax)
    plot.set_ylabel("expected number of questions")
    fig.savefig(f"{name}_length_vs_entropy.png")
    plt.close()


def plot_curves(curves, savename):
    # curves is a dict {'name1':{seed1:curve1}, 'name2':...}
    sns.set_theme()
    plt.figure()
    colors = sns.color_palette("husl", len(curves))
    for i, name in enumerate(curves):
        if name == 'coreset':
            continue
        accs = []
        for seed in curves[name]:
            losses = [res["loss"] for res in curves[name][seed]]
            tps = [res["tp"] for res in curves[name][seed]]
            fps = [res["fp"] for res in curves[name][seed]]
            fns = [res["fn"] for res in curves[name][seed]]
            tns = [res["tn"] for res in curves[name][seed]]
            accuracies = (np.array(tps) + np.array(tns)) / (
                np.array(tps) + np.array(fps) + np.array(fns) + np.array(tns)
            )
            sns.lineplot(x=np.arange(len(accuracies)), y=accuracies, linestyle='dashed', color=colors[i],
                         alpha=0.2,
                         )
            accs.append(accuracies)
        accs = np.array(accs)
        sns.lineplot(
            x=np.tile(np.arange(accs.shape[1]), accs.shape[0]),
            y=accs.flatten(),
            linewidth=5,
            color=colors[i],
            label=f"{name}",
            alpha=1,
            errorbar="sd",
        )
    plt.xlabel("number of questions")
    plt.ylabel("test accuracy")
    plt.legend(loc='lower center')
    plt.savefig(f"{savename}.png")
    plt.close()



def load_curves():
    filenames = os.listdir()
    curvefilenames = [
        file for file in filenames if file.startswith("curve") and file.endswith(".npy")
    ]
    curvenames = list(set([file.split("_")[1] for file in curvefilenames]))
    curves = {}
    for name in curvenames:
        curves[name] = {}
        for filename in curvefilenames:
            if filename.startswith("curve_" + name):
                seed = int(filename[:-4].split("_")[3])
                with open(filename, "rb") as file:
                    curve = np.load(file, allow_pickle=True)
                curves[name][seed] = curve
    return curves




def visualize_question(data, guess, n_queries):
    # data is a tensor of size (N, C, H, W)
    # we want to show those entries of data with guess = 1 on the plot of the right, those with guess = 0 on the plot of the left
    as_positive = data[torch.tensor(guess) == 1]
    as_negative = data[torch.tensor(guess) == 0]
    if len(as_positive) > 0:
        plt.subplot(1, 2, 1)
        # put the images in a grid
        grid = torchvision.utils.make_grid(as_positive, nrow=8, normalize=True)
        # show the grid on the first plot
        plt.imshow(grid.permute(1, 2, 0).cpu())
        plt.title("positive")
    if len(as_negative) > 0:
        plt.subplot(1, 2, 2)
        grid = torchvision.utils.make_grid(as_negative, nrow=8, normalize=True)
        plt.imshow(grid.permute(1, 2, 0).cpu())
        plt.title("negative")
    plt.suptitle(f"n_queries = {n_queries}")
    plt.savefig(f"question_{n_queries}.png")

    
