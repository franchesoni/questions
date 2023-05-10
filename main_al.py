from engine import seed_everything
import numpy as np

from data import get_dataset, DATASET_NAMES
from predictor import get_resnet
from active_learning import CoreSet, UncertaintySampling, get_al_curve, RandomSampling
import visualization as vis


def run_experiment_al(
    experiment_name="random",
    dataset_index=1,
    binarizer="geq5",
    pretrained=False,
    seeds=[0],
    max_queries=500,
    use_only_first=5000,
    use_only_first_test=1000,
    device=None,
):
    dataset_name = DATASET_NAMES[dataset_index]
    train_ds, test_ds = get_dataset(
        dataset_name,
        binarizer,
        use_only_first=use_only_first,
        use_only_first_test=use_only_first_test,
        device=device,
    )
    n_channels = train_ds.data[0].shape[0]
    if experiment_name == "random":
        strategy = RandomSampling()
    elif experiment_name == "uncertainty":
        strategy = UncertaintySampling()
    elif experiment_name == "coreset":
        strategy = CoreSet()
    curves = []
    for seed in seeds:
        seed_everything(seed)
        initial_labeled_indices = [int(np.random.randint(len(train_ds)))]
        predictor = get_resnet(
            pretrained=pretrained, n_channels=n_channels, compile=False, device=device
        )
        curve = get_al_curve(
            max_queries,
            predictor,
            strategy,
            train_ds,
            test_ds,
            initial_labeled_indices=initial_labeled_indices,
        )
        np.save(f"curve_{experiment_name}_seed_{seed}.npy", curve)
        curves.append(curve)


if __name__ == "__main__":
    # import cProfile
    # cProfile.run('run_experiment("random", "al", 1, "geq5", False, [0], use_only_first=10, use_only_first_test=1000)', 'active_learning_random2.profile')

    # run_experiment("random", "al", 1, "geq5", False, [3, 4], use_only_first=500, use_only_first_test=1000, device='cuda:0')
    seeds = [0, 1, 2, 3, 4]
    for pretrained in [False, True]:
        for binarizer in ["geq5", "last", "odd"]:
            for dataset_index in [1, 2, 3]:
                for exp_name in ["random", "uncertainty", "coreset"]:
                    run_experiment_al(
                        experiment_name=exp_name,
                        dataset_index=dataset_index,
                        binarizer=binarizer,
                        pretrained=pretrained,
                        seeds=seeds,
                        max_queries=500,
                        use_only_first=5000,
                        use_only_first_test=1000,
                        device='cuda',
                    )


    # curves = vis.load_curves()
    # vis.plot_curves(curves, "curves")
