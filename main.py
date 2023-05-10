from engine import seed_everything
import numpy as np

from data import get_dataset, DATASET_NAMES
from predictor import get_resnet
from active_learning import CoreSet, UncertaintySampling, get_al_curve, RandomSampling
from intelligent_annotation import get_alia_curve   

def run_experiment(experiment_name='random', exp_type='al', dataset_index=1, binarizer='geq5', pretrained=False, seeds=[0], use_only_first=5000, use_only_first_test=1000, device=None):
    dataset_name = DATASET_NAMES[dataset_index]
    train_ds, test_ds = get_dataset(dataset_name, binarizer, use_only_first=use_only_first, use_only_first_test=use_only_first_test, device=device)
    n_channels = train_ds.data[0].shape[0]
    predictor_kwargs = {'pretrained': pretrained, 'n_channels': n_channels, 'compile': False}
    if experiment_name == 'random':
        strategy = RandomSampling()
    elif experiment_name == 'uncertainty':
        strategy = UncertaintySampling()
    elif experiment_name == 'coreset':
        strategy = CoreSet()
    elif experiment_name == 'alia':
        strategy = 'guess_rollout_length'
    curves = []
    for seed in seeds:
        seed_everything(seed)
        initial_labeled_indices = [int(np.random.randint(len(train_ds)))]
        predictor = get_resnet(pretrained=pretrained, n_channels=n_channels, compile=False, device=device)
        if exp_type == 'al':
            curve = get_al_curve(predictor, strategy, train_ds, test_ds, initial_labeled_indices=initial_labeled_indices)
        elif exp_type == 'alia':
            curve = get_alia_curve(predictor, strategy, train_ds, test_ds, initial_labeled_indices=initial_labeled_indices)
        np.save(f'curve_{experiment_name}_seed_{seed}.npy', curve)
        curves.append(curve)
    # plot_curves(curves, experiment_name=experiment_name)


if __name__ == '__main__':
    # import cProfile
    # cProfile.run('run_experiment("random", "al", 1, "geq5", False, [0], use_only_first=10, use_only_first_test=1000)', 'active_learning_random2.profile')

    # run_experiment("random", "al", 1, "geq5", False, [3, 4], use_only_first=500, use_only_first_test=1000, device='cuda:0')
    run_experiment("uncertainty", "al", 1, "geq5", False, [3], use_only_first=500, use_only_first_test=1000, device='cuda:0')
