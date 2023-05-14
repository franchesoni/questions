from active_learning import get_al_curve
from engine import seed_everything
import datetime
import numpy as np

from data import get_dataset, DATASET_NAMES
from predictor import get_resnet
from intelligent_annotation import get_ia_curve
from active_learning import CoreSet, UncertaintySampling, get_al_curve, RandomSampling


def run_experiment(
    exp_type,
    experiment_name,
    dataset_index,
    binarizer,
    pretrained,
    seeds,
    max_queries,
    use_only_first,
    use_only_first_test,
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
    for seed in seeds:
        seed_everything(seed)
        initial_labeled_indices = [int(np.random.randint(len(train_ds)))]
        predictor = get_resnet(
            pretrained=pretrained, n_channels=n_channels, compile=False, device=device
        )
        if exp_type == 'al':
            curve = get_al_curve(
                max_queries,
                predictor,
                strategy,
                train_ds,
                test_ds,
                initial_labeled_indices=initial_labeled_indices,
            )   
        elif exp_type == 'ia':
            curve = get_ia_curve(
                max_queries,
                predictor,
                strategy,
                train_ds,
                test_ds,
                initial_labeled_indices=initial_labeled_indices,
            )
        now = str(datetime.datetime.utcnow()).replace(' ', '-').replace(':', '-').replace('.', '-')
        dstfilename = f"results/curve_{now}_seed_{seed}_exp_type_{exp_type}_exp_name_{experiment_name}_dataset_{dataset_name}_pretrained_{pretrained}_binarizer_{binarizer}_max_queries_{max_queries}_use_only_first_{use_only_first}_use_only_first_test_{use_only_first_test}.npy"
        np.save(dstfilename, curve)

def run_all_experiments(dev=False, profiler=None):
    for binarizer in ["geq5"]:#, "last", "odd"]:
        for pretrained in [False]:#, True]:  # not working because it needs 1000 classes
            for dataset_index in [1, 0, 2, 3, 4]:
                for exp_name in ["random", "uncertainty", "certainty"]:#, "coreset"]:  # coreset is broken
                    for exp_type in ['al']:#['ia', 'al']:  # does ia work?
                        if exp_type == 'ia' and exp_name == 'coreset':
                            continue  # not implemented
                        seeds = [0, 1, 2]
                        device='cuda:0'
                        # seeds = [3, 4, 5]
                        # device='cuda:1'
                        max_queries = 500
                        use_only_first = 5000
                        use_only_first_test = 1000
                        if dev:
                            max_queries = 50
                            use_only_first = 500
                            use_only_first_test = 100
                            seeds = [0]
                        run_experiment(
                            seeds=seeds,
                            exp_type=exp_type,
                            experiment_name=exp_name,
                            dataset_index=dataset_index,
                            pretrained=pretrained,
                            binarizer=binarizer,
                            max_queries=max_queries,
                            use_only_first=use_only_first,
                            use_only_first_test=use_only_first_test,
                            device=device,
                        )
                        print("_"*80)
                        print(f"Finished {exp_type} {exp_name} {dataset_index} {binarizer} {pretrained}")
                        print("_"*80)
                        if profiler and dev:
                            profiler.disable()
                            profiler.dump_stats('run_all_dev.profile')
                            profiler.enable()
        #                 break
        #             break
        #         break
        #     break
        # break



if __name__ == "__main__":
    run_all_experiments(dev=False, profiler=None)
    # import cProfile
    # pr = cProfile.Profile()
    # pr.enable()
    # run_all_experiments(dev=True, profiler=pr)
    # pr.disable()
    # pr.dump_stats('run_all_dev.profile')

    # run_experiment("random", "al", 1, "geq5", False, [3, 4], use_only_first=500, use_only_first_test=1000, device='cuda:0')


