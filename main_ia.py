from pathlib import Path
from engine import seed_everything
import datetime
import numpy as np

from data import get_dataset, DATASET_NAMES
from predictor import get_resnet
from intelligent_annotation import get_ia_curve


def run_ia_experiment(
    sts_kwargs,
    update_predictor_every,
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
    Path('results/curves').mkdir(parents=True, exist_ok=True)
    # assert exp_type == 'ia'
    # assert experiment_name in ['random', 'uncertainty']
    dataset_name = DATASET_NAMES[dataset_index]
    train_ds, test_ds = get_dataset(
        dataset_name,
        binarizer,
        use_only_first=use_only_first,
        use_only_first_test=use_only_first_test,
        device=device,
    )
    n_channels = train_ds.data[0].shape[0]


    for seed in seeds:
        seed_everything(seed)
        initial_labeled_indices = [int(np.random.randint(len(train_ds)))]
        predictor = get_resnet(
            pretrained=pretrained, n_channels=n_channels, compile=False, device=device
        )
        now = str(datetime.datetime.utcnow()).replace(' ', '-').replace(':', '-').replace('.', '-')
        dstfilename = f"results/curves/curve_{now}_seed_{seed}_exp-type_{exp_type}_exp-name_{experiment_name}_dataset_{dataset_name}_pretrained_{pretrained}_binarizer_{binarizer}_max-queries_{max_queries}_use-only-first_{use_only_first}_use-only-first-test_{use_only_first_test}_max-expansions_{sts_kwargs['max_expansions']}_max-n_{sts_kwargs['max_n']}_cost-fn_{sts_kwargs['cost_fn']}_reduce-certainty-factor_{sts_kwargs['reduce_certainty_factor']}.npy"
        curve = get_ia_curve(
            max_queries,
            predictor,
            update_predictor_every,
            train_ds,
            test_ds,
            initial_labeled_indices=initial_labeled_indices,
            sts_kwargs=sts_kwargs,
            dstfilename=dstfilename,
            )


def run_all_experiments(run=0, dev=False, profiler=None):
    for binarizer in ["geq5"]:
        for pretrained in [False]:#, True]:  # not working because it needs 1000 classes
            for max_n in [8, 16]:
                for exp_name in ["uncertainty", "random"]:
                    for dataset_index in [1, 0, 2, 3]:#, 4]:
                        for cost_fn in ["entropy", "length"]:
                            for reduce_uncertainty_factor in [0.01]:#, 0.5, 0.9, 0.1]:
                                update_predictor_every = 10
                                sts_kwargs = {
                                    "max_expansions": 8,
                                    "max_n": max_n,
                                    "al_method": exp_name,
                                    "cost_fn": cost_fn,
                                    "reduce_certainty_factor":reduce_uncertainty_factor,
                                    "reset_tree":update_predictor_every==1,
                                }
                                exp_type = 'ia'
                                max_queries = 2500
                                use_only_first = 6000
                                use_only_first_test = 1000
                                if run == 0:
                                    seeds = [0]#, 1, 2]
                                    device='cuda:0'
                                elif run == 1:
                                    seeds = [1]#, 4, 5]
                                    device='cuda:1'
                                    # device='cpu'
                                else:
                                    raise ValueError("run must be 0 or 1")

                                if dev:
                                    max_queries = 30
                                    use_only_first = 6000
                                    use_only_first_test = 1000
                                    seeds = [0]

                                run_ia_experiment(
                                    sts_kwargs=sts_kwargs,
                                    update_predictor_every=update_predictor_every,
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
                                print(f"Finished {exp_type} {exp_name} {dataset_index} {sts_kwargs}")
                                print("_"*80)
                                if profiler and dev:
                                    profiler.disable()
                                    profiler.dump_stats('run_all_dev.profile')
                                    profiler.enable()



if __name__ == "__main__":
    run_all_experiments(run=1, dev=False, profiler=None)
    # import cProfile
    # pr = cProfile.Profile()
    # pr.enable()
    # run_all_experiments(run=0, dev=True, profiler=pr)
    # pr.disable()
    # pr.dump_stats('run_all_dev.profile')




