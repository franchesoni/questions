import tqdm
import numpy as np
import os
import torchvision
import torch

# import pygraphviz as pgv
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as colors
import huffman

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
    x2s = np.linspace(-4, 6, 100)
    x1s = np.linspace(-4, 6, 100)
    x1, x2 = np.meshgrid(x1s, x2s)
    xs = np.stack([x1.reshape(-1), x2.reshape(-1)], axis=1)
    # predict the probability of each point in the grid
    ys = model(xs)
    # plot the background
    plt.figure()
    sns.set_theme(context='paper', style='darkgrid', font_scale=1.5)
    levels = np.linspace(0, 1, 11)
    ctset = plt.contourf(x1, x2, ys.reshape(x1.shape), levels=levels, vmin=0, vmax=1)
    cbar = plt.colorbar(ctset, ticks=levels)
    cbar.set_label("Predicted probability")
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
            # set point size
            s=200,
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
    plt.savefig(name + ".png")
    plt.close()



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
    # change fontsize
    sns.set_theme('paper', font_scale=1.6)
    # plt.rcParams["font.size"] = "20"
    n_questions_results = {
        key[12:]: results[key] for key in results if key.startswith("n_questions_")
    }
    entropies = results["entropies"]
    to_plot_names = ["IA", "Huffman", "Entropy"]
    to_plot_values = list(n_questions_results.values()) + [entropies]

    print('='*20)
    print(name)
    print('mean entropy', f'{np.mean(to_plot_values[2]):.2f}')
    print('mean n questions Huffman', f'{np.mean(to_plot_values[1]):.2f}')
    print('mean n questions IA', f'{np.mean(to_plot_values[0]):.2f}')
    print('mean difference Huffman', f'{np.mean(np.array(to_plot_values[1]) - np.array(to_plot_values[2])):.2f}')
    print('mean difference IA', f'{np.mean(np.array(to_plot_values[0]) - np.array(to_plot_values[2])):.2f}')
    print('mean percentual difference Huffman', f'{np.mean(np.array(to_plot_values[1]) / np.array(to_plot_values[2])):.2f}')
    print('mean percentual difference IA', f'{np.mean(np.array(to_plot_values[0]) / np.array(to_plot_values[2])):.2f}')
    return


    fig, ax = plt.subplots()
    plot = sns.boxplot(data=to_plot_values, ax=ax)
    sns.stripplot(data=to_plot_values, ax=ax)
    plot.set_xticklabels(to_plot_names)
    plt.ylabel("Number of questions")
    fig.savefig(f"{name}.png")
    plt.close()

    # plot differences
    differences = {
        key: [nq - entropy for nq, entropy in zip(n_questions_results[key], entropies)]
        for key in n_questions_results
    }
    to_plot_names = ["IA", "Huffman"]  #list(differences.keys())
    to_plot_values = list(differences.values())
    fig, ax = plt.subplots()
    plot = sns.boxplot(data=to_plot_values, ax=ax)
    sns.stripplot(data=to_plot_values, ax=ax)
    plot.set_xticklabels(to_plot_names)
    plot.set_ylabel("Q - H(Y|X)")
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
    # plt.title("Huffman expected number of questions")
    # make scatter
    plot = sns.scatterplot(data=results, x="entropies", y="expected_lengths", ax=ax)
    plot.set_xlabel("Entropy")
    plot.set_ylabel("Expected number of questions")
    # plot identity line
    limit = max(np.max(results["expected_lengths"]), np.max(results["entropies"]))
    sns.lineplot(x=[0, limit + 1], y=[0, limit + 1], ax=ax)
    fig.savefig(f"{name}.png")
    plt.close()


def visualize_sts_estimated_cost(results, savename):
    costs = [res[0] for ress in results for res in ress]
    entropy1 = [res[1] for ress in results for res in ress]
    entropy2 = [res[2] for ress in results for res in ress]
    n_questions = [res[3] for ress in results for res in ress]
    results = {"cost": costs, "entropy1": entropy1, "entropy2": entropy2, "n_questions": n_questions}

    sns.set_theme()
    fig, ax = plt.subplots()
    plt.title("STS estimated cost")
    plot = sns.scatterplot(data=results, x="entropy1", y="cost", hue="n_questions", ax=ax)
    plot.set_ylabel("estimated cost")
    # plot identity line
    limit = max(np.max(results["cost"]), np.max(results["entropy1"]))
    sns.lineplot(x=[0, limit + 1], y=[0, limit + 1], ax=ax)
    fig.savefig(f"{savename}_with_inc.png")
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


def calibration_plot3(probs_and_outcomes, bins=10, name="calibration_plot"):
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


def plot_curves2(curves, savename):
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

def plot_curves3(curves, savename, max_queries=1000):
    # curves is a list [{'curve':curve, 'seed':seed, ...}]
    curves = [curve for curve in curves if 'queries' in curve and curve['queries']==str(max_queries)]
    datasets = set([curve['dataset'] for curve in curves])

    sns.set_theme('paper', font_scale=1.5)
    for dataset in sorted(datasets):

        print('='*20)
        print('dataset', dataset)
        # print('plotting dataset...', dataset)
        seeds_dict = {}
        for curve in curves:
            if curve['dataset'] != dataset:
                continue
            name = curve['type'] + '_' + curve['name'] + '_' + curve['maxn'] + '_' + curve['cost'] + '_' + curve['factor']
            if name not in seeds_dict:
                seeds_dict[name] = [curve]
                # print(name)
            else:
                same_seed = [(ind, oldcurve) for ind, oldcurve in enumerate(seeds_dict[name]) if oldcurve['seed'] == curve['seed']]
                if len(same_seed) > 0:
                    if same_seed[0][1]['timestamp'] < curve['timestamp']:
                        seeds_dict[name].pop(same_seed[0][0])
                    else:
                        continue
                seeds_dict[name].append(curve)
                # print(name)


        plt.figure()
        colors = sns.color_palette("dark", len(seeds_dict))
        for i, name in enumerate(sorted(list(seeds_dict.keys()))):
            accs = []
            n_labelss = []
            n_incorrects = []
            for seed_curve in seeds_dict[name]:
                losses = [res["loss"] for res in seed_curve['curve']]
                tps = np.array([res["tp"] for res in seed_curve['curve']])
                fps = np.array([res["fp"] for res in seed_curve['curve']])
                fns = np.array([res["fn"] for res in seed_curve['curve']])
                tns = np.array([res["tn"] for res in seed_curve['curve']])
                accuracies = (tps + tns) / (
                    tps + fps + fns + tns
                )
                # sns.lineplot(x=np.arange(len(accuracies)), y=accuracies, linestyle='dashed', color=colors[i],
                #             alpha=0.1,
                #             )
                if 'n_labels' in seed_curve:
                    n_labels = np.array(seed_curve['n_labels'])
                    sns.lineplot(x=np.arange(len(accuracies)), y=n_labels ,#/ np.arange(1, len(n_labels) + 1), 
                                 color=colors[i], alpha=0.2, linestyle='dotted')
                    n_incorrect = np.cumsum(n_labels[1:] - n_labels[:-1] == 0)
                    sns.lineplot(x=np.arange(len(n_incorrect)), y=n_incorrect,#/ np.arange(1, len(n_labels) + 1), 
                                 color=colors[i], alpha=0.2, linestyle='dashed')
                accs.append(accuracies)
                n_labelss.append(n_labels)
                n_incorrects.append(n_incorrect)

            print('-'*20)
            print('name', name)
            ql2500 = np.mean([np.argmax(n_labels > 2500) for n_labels in n_labelss])
            if ql2500 == 0:
                ql2500 = 2500
            print('Q_L2500',  ql2500, 'std', np.std([np.argmax(n_labels > 2500) for n_labels in n_labelss]))
            print('maxQ', np.mean([len(n_labels) for n_labels in n_labelss]), 'std', np.std([len(n_labels) for n_labels in n_labelss]))
            print('L_maxQ', np.mean([n_labels[-1] for n_labels in n_labelss]), 'std', np.std([n_labels[-1] for n_labels in n_labelss]))
            print('#Inc_maxQ', np.mean([n_incorrect[-1] for n_incorrect in n_incorrects]), 'std', np.std([n_incorrect[-1] for n_incorrect in n_incorrects]))
            print('Q_L2500',  ql2500/2500*100)

            # for i in range(len(n_labelss)):
            #     print('-'*20)
            #     print('seed', i)
            #     print('final n_incorrect', n_incorrects[i][-1])
            #     print('final n_labels', n_labelss[i][-1])
            #     # now we take the index for which n_labels is 2500
            #     print('n questions for 2500 labels',  np.argmax(n_labelss[i] > 2500))

            maxlen = max([len(nl) for nl in n_labelss])
            # fill in the rest with the last value
            for ind in range(len(n_labelss)):
                n_labelss[ind] = np.concatenate((n_labelss[ind],np.ones(maxlen - len(n_labelss[ind])) * n_labelss[ind][-1])) 
            n_labelss = np.array(n_labelss)
            label = {"ia_random_8_entropy_0.01": "Cost: H(s), AL: Random",
            "ia_random_8_length_0.05": "Cost: $log_2(|s|)$, AL: Random",
            "ia_uncertainty_8_entropy_0.01": "Cost: H(s), AL: Uncertainty",
            "ia_uncertainty_8_length_0.05": "Cost: $log_2(|s|)$, AL: Uncertainty",
                     }
            sns.lineplot(
                x=np.tile(np.arange(n_labelss.shape[1]), n_labelss.shape[0]),
                y=n_labelss.flatten(),
                linewidth=3,
                color=colors[i],
                label=label[name],#f"{name}",
                alpha=1,
                errorbar="sd",
            )
        sns.lineplot(
            x=np.arange(n_labelss.shape[1]),
            y=np.arange(n_labelss.shape[1]),
            linewidth=1,
            color='black',
            alpha=1,
            linestyle='dotted',
            label='Identity'
        )
        sns.lineplot(
            x=np.arange(1),
            y=np.arange(1),
            linewidth=1,
            color='black',
            alpha=0,
            linestyle='dashed',
            label='# of incorrect guesses'
        )
        plt.xlabel("Number of questions")
        plt.ylabel("Number of labeled samples")
        plt.legend(loc='upper left')#loc='lower center')
        plt.tight_layout()
        plt.savefig(f"{savename}_{dataset}_labels.png")
        plt.close()

        # plt.figure(figsize=(10,10))
        # plt.figure()
        # colors = sns.color_palette("dark", len(seeds_dict))
        # for i, name in enumerate(sorted(list(seeds_dict.keys()))):
        #     accs = []
        #     n_labelss = []
        #     for seed_curve in seeds_dict[name]:
        #         losses = [res["loss"] for res in seed_curve['curve']]
        #         tps = np.array([res["tp"] for res in seed_curve['curve']])
        #         fps = np.array([res["fp"] for res in seed_curve['curve']])
        #         fns = np.array([res["fn"] for res in seed_curve['curve']])
        #         tns = np.array([res["tn"] for res in seed_curve['curve']])
        #         accuracies = (tps + tns) / (
        #             tps + fps + fns + tns
        #         )
        #         sns.lineplot(x=np.arange(len(accuracies)), y=accuracies, linestyle='dashed', color=colors[i],
        #                     alpha=0.1,
        #                     )
        #         if 'guess_length' in seed_curve:
        #             # sns.lineplot(x=np.arange(len(accuracies)), y=np.array(seed_curve['guess_length'])/8,#/ np.arange(1, len(n_labels) + 1), 
        #             #              color=colors[i], alpha=0.4, linestyle='dotted')
        #             sns.scatterplot(x=np.arange(len(accuracies)), y=np.array(seed_curve['guess_length'])/8,#/ np.arange(1, len(n_labels) + 1), 
        #                          color=colors[i], alpha=0.1)
        #         accs.append(accuracies)

        #     maxlen = max([len(acc) for acc in accs])
        #     # fill in the rest with the last value
        #     for ind in range(len(accs)):
        #         accs[ind] = np.concatenate((accs[ind],np.ones(maxlen - len(accs[ind])) * accs[ind][-1])) 
        #     accs = np.array(accs)
        #     sns.lineplot(
        #         x=np.tile(np.arange(accs.shape[1]), accs.shape[0]),
        #         y=accs.flatten(),
        #         linewidth=3,
        #         color=colors[i],
        #         label=f"{name}",
        #         alpha=0.5,
        #         errorbar="sd",
        #     )
        # plt.xlabel("Number of questions")
        # plt.ylabel("Test accuracy")
        # plt.legend()#loc='lower center')
        # plt.savefig(f"{savename}_{dataset}.png")
        # plt.close()


def plot_curves4(curves1, curves2, savename):
    datasets = set([curve['dataset'] for curve in curves2])

    to_visualize = [
        'ia_uncertainty_4_entropy_0.1',
        'ia_uncertainty_4_entropy_0.5',
        'ia_uncertainty_8_entropy_0.5',
    ]

    sns.set_theme()
    for dataset in datasets:
        seeds_dict = {}
        for curve in curves1 + curves2:
            if curve['dataset'] != dataset:
                continue
            if curve['exp-type'] == 'al':
                name = curve['exp-type'] + '_' + curve['exp-name']
            elif curve['exp-type'] == 'ia':
                name = curve['exp-type'] + '_' + curve['exp-name'] + '_' + curve['max-n'] + '_' + curve['cost-fn'] + '_' + curve['reduce-certainty-factor']
            if 'max-n' in curve and name not in to_visualize:
                continue

            if name not in seeds_dict:
                seeds_dict[name] = [curve]
                print(name)
            else:
                same_seed = [(ind, oldcurve) for ind, oldcurve in enumerate(seeds_dict[name]) if oldcurve['seed'] == curve['seed']]
                if len(same_seed) > 0:
                    if same_seed[0][1]['timestamp'] < curve['timestamp']:
                        seeds_dict[name].pop(same_seed[0][0])
                    else:
                        continue
                seeds_dict[name].append(curve)
                print(name)


        plt.figure(figsize=(40,40))
        colors = sns.color_palette("dark", len(seeds_dict))
        for i, name in enumerate(sorted(list(seeds_dict.keys()))):
            accs = []
            n_labelss = []
            for seed_curve in seeds_dict[name]:
                losses = [res["loss"] for res in seed_curve['curve']]
                tps = np.array([res["tp"] for res in seed_curve['curve']])
                fps = np.array([res["fp"] for res in seed_curve['curve']])
                fns = np.array([res["fn"] for res in seed_curve['curve']])
                tns = np.array([res["tn"] for res in seed_curve['curve']])
                accuracies = (tps + tns) / (
                    tps + fps + fns + tns
                )
                # sns.lineplot(x=np.arange(len(accuracies)), y=accuracies, linestyle='dashed', color=colors[i],
                #             alpha=0.1,
                #             )
                accs.append(accuracies)

            # maxlen = max([len(acc) for acc in accs])
            maxlen = 500
            # fill in the rest with the last value
            for ind in range(len(accs)):
                if len(accs[ind]) > maxlen:
                    accs[ind] = accs[ind][:maxlen]
                else:
                    accs[ind] = np.concatenate((accs[ind], np.ones(maxlen - len(accs[ind])) * accs[ind][-1])) 
            accs = np.array(accs)
            sns.lineplot(
                x=np.tile(np.arange(accs.shape[1]), accs.shape[0]),
                y=accs.flatten(),
                linewidth=3,
                color=colors[i],
                label=f"{name}",
                alpha=0.5,
                errorbar="sd",
            )
        plt.xlabel("number of questions")
        plt.ylabel("test accuracy")
        plt.legend()#loc='lower center')
        plt.savefig(f"{savename}_al_ia_{dataset}.png")
        plt.close()




def load_curves(dir='results/curves/'):
    # curves have names
    # f"results/curve_{now}_seed_{seed}_exp_type_{exp_type}_exp_name_{experiment_name}_dataset_{dataset_name}_pretrained_{pretrained}_binarizer_{binarizer}_max_queries_{max_queries}_use_only_first_{use_only_first}_use_only_first_test_{use_only_first_test}.npy"
    # this function extracts the curves and returns them in a list of dicts
    # each dict on the list has {'curve':..., 'seed': ..., ...}
    filenames = os.listdir(dir)
    curvefilenames = [
        file for file in filenames if file.startswith("curve") and file.endswith(".npy")
    ]
    curves = []
    for curvefname in tqdm.tqdm(curvefilenames):
        new_curve = {}
        # only those in new version
        if not '-' in curvefname.split('_')[1]:
            continue
        curve = np.load(dir+curvefname, allow_pickle=True)
        if len(curve.shape) == 0:
            curve = curve.tolist()
            curve, n_labels, glen = curve['curve'], curve['n_labeled'], curve['guess_length']
            new_curve['n_labels'] = n_labels
            new_curve['guess_length'] = glen
        split_stem = curvefname[:-4].split("_")
        new_curve['curve'] = curve
        new_curve['timestamp'] = split_stem[1]
        for ind in range(3, len(split_stem), 2):
            new_curve[split_stem[ind-1]] = split_stem[ind]
        # new_curve['seed'] = int(curvefname[:-4].split("_")[3])
        # new_curve['exp_type'] = curvefname[:-4].split("_")[6]
        # new_curve['exp_name'] = curvefname[:-4].split("_")[9]
        # new_curve['dataset'] = curvefname[:-4].split("_")[11]
        # new_curve['pretrained'] = curvefname[:-4].split("_")[13]
        # new_curve['binarizer'] = curvefname[:-4].split("_")[15]
        # new_curve['max_queries'] = int(curvefname[:-4].split("_")[18])
        # new_curve['use_only_first'] = curvefname[:-4].split("_")[22]
        # new_curve['use_only_first_test'] = curvefname[:-4].split("_")[27]
        # # print("loading curve ", curvefname)
        # # print("loaded curve ", {k:v for k,v in new_curve.items() if k != 'curve'})
        curves.append(new_curve)
    return curves





def load_curves2():
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
    plt.savefig(f"questions/question_{n_queries}.png")
    plt.close()

    
if __name__ == '__main__':
    # curves = load_curves(dir='resultsjz/curves/')
    # plot_curves3(curves, savename='resultsjz/curves/plot', max_queries=500)
    curves = load_curves(dir='results/curves/')
    plot_curves3(curves, savename='results/curves/plot', max_queries=2500)

    # curves1 = load_curves(dir='results/curves/')
    # curves2 = load_curves(dir='resultsjz2/curves/')
    # plot_curves4(curves1, curves2, savename='results/curves/plot')



