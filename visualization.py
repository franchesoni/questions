import numpy as np
import matplotlib.pyplot as plt

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
            tp = epoch['tp']
            fp = epoch['fp']
            tn = epoch['tn']
            fn = epoch['fn']
            epoch['accuracy'] = (tp + tn) / (tp + fp + tn + fn)
            epoch['recall'] = tp / (tp + fn)
            epoch['precision'] = tp / (tp + fp)
            epoch['specificity'] = tn / (tn + fp)

        # convert perfromance to dict of lists
        performance = {k: [epoch[k] for epoch in performance] for k in performance[0]}
        formated_performances.append(performance)

    # average each metric over all experiments
    mean_performance_curve = {k: [np.mean([performance[k][i] for performance in formated_performances]) for i in range(len(performance[k]))] for k in formated_performances[0]}
    std_of_performance_curve = {k: [np.std([performance[k][i] for performance in formated_performances]) for i in range(len(performance[k]))] for k in formated_performances[0]}

    # plot mean and std of performances for each metric shadowing the interval between 1 std
    fig, ax = plt.subplots(1, 5, figsize=(20, 5))

    for i, metric in enumerate(['loss', 'accuracy', 'recall', 'precision', 'specificity']):
        ax[i].plot(mean_performance_curve[metric])
        ax[i].fill_between(range(len(mean_performance_curve[metric])), np.array(mean_performance_curve[metric]) - np.array(std_of_performance_curve[metric]), np.array(mean_performance_curve[metric]) + np.array(std_of_performance_curve[metric]), alpha=0.3)
        ax[i].set_title(metric)
        ax[i].set_xlabel('epochs')
        ax[i].set_ylabel(metric)
    plt.suptitle(experiment_name)
    plt.show()