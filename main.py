import numpy as np

from data import get_dataset, DATASET_NAMES
from predictor import get_resnet
from active_learning import get_al_curve, RandomSampling
from intelligent_annotation import get_alia_curve   

def run_experiment(experiment_name='random', exp_type='al', dataset_index=1, binarizer='geq5', pretrained=False, seeds=[0], use_only_first=100):
    dataset_name = DATASET_NAMES[dataset_index]
    train_ds, test_ds = get_dataset(dataset_name, binarizer, use_only_first=use_only_first)
    n_channels = train_ds.data[0].shape[0]
    predictor = get_resnet(pretrained=pretrained, n_channels=n_channels, compile=True)
    if experiment_name == 'random':
        strategy = RandomSampling()
    elif experiment_name == 'alia':
        strategy = 'guess_rollout_length'
    curves = []
    for seed in seeds:
        if exp_type == 'al':
            curve = get_al_curve(seed, predictor, strategy, train_ds, test_ds, initial_labeled_indices=[0])
        elif exp_type == 'alia':
            curve = get_alia_curve(seed, predictor, strategy, train_ds, test_ds, initial_labeled_indices=[0])
        np.save(f'curve_{experiment_name}_seed_{seed}.npy', curve)
        curves.append(curve)
    # plot_curves(curves, experiment_name=experiment_name)


if __name__ == '__main__':
    # run_experiment('random', 'al', 1, 'geq5', False, [0], use_only_first=100)
    run_experiment('alia', 'alia', 1, 'geq5', False, [0], use_only_first=100)
