import matplotlib.pyplot as plt

def plot_curves(performance, experiment_name):
    """Plot performance curves by showing test loss and accuracy, recall, precision and specificity during training with respect to epochs. Also adds error bars for each metric.
    Args:
        performance (list): list of dicts containing ['loss', 'tp', 'fp', 'tn', 'fn'] as keys for each epoch
        experiment_name (str): name of the experiment
    """
    # compute accuracy, recall, precision and specificity from tp, fp, tn, fn
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

    # plot everything
    fig, ax = plt.subplots(1, 5, figsize=(20, 5))
    for i, metric in enumerate(['loss', 'accuracy', 'recall', 'precision', 'specificity']):
        ax[i].plot(performance[metric], '-.', label=metric)
        ax[i].set_xlabel('epochs')
        ax[i].set_ylabel(metric)
    plt.suptitle(experiment_name)
    plt.show()