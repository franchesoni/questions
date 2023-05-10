from abc import ABC, abstractmethod
import random

import tqdm
import torch
import gc

from engine import optimizer_setup, seed_everything, fit_predictor, evaluate_predictor
from data import get_labeled_unlabeled


class Strategy(ABC):
    def __call__(self, predictor, unlabeled_dataset):
        return self.get_index_to_label_next(predictor, unlabeled_dataset)

    @abstractmethod
    def get_index_to_label_next(self, predictor, unlabeled_dataset):
        """Should return index_to_label_next"""
        pass


class RandomSampling(Strategy):
    def get_index_to_label_next(self, predictor, unlabeled_dataset):
        index_to_label_next = int(random.choice(unlabeled_dataset.indices))
        return index_to_label_next


class UncertaintySampling(Strategy):
    def get_index_to_label_next(self, predictor, unlabeled_dataset):
        with torch.no_grad():
            output = predictor(unlabeled_dataset.data)[:, 0]
            certainty = torch.abs(output - 0.5)
            index_to_label_next = int(
                unlabeled_dataset.indices[torch.argmin(certainty)]
            )
        return index_to_label_next


def get_al_curve(
    predictor, strategy, train_dataset, test_dataset, initial_labeled_indices
):
    curve = [None] * (len(initial_labeled_indices) - 1)
    labeled_indices = initial_labeled_indices
    unlabeled_ds = "a value to start the loop"

    optimizer = optimizer_setup(predictor)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    eval_loss_fn = torch.nn.BCEWithLogitsLoss(reduction="sum")

    pbar = tqdm.tqdm(total=len(train_dataset) - len(initial_labeled_indices))
    while len(unlabeled_ds) > 0:  # fit, get point, label
        labeled_ds, unlabeled_ds = get_labeled_unlabeled(
            train_dataset, labeled_indices
        )  # unlabeled_ds also stores the unlabeled indices
        predictor = fit_predictor(
            predictor, loss_fn, optimizer, labeled_ds, verbose=False
        )
        performance = evaluate_predictor(
            predictor, eval_loss_fn, test_dataset, verbose=False
        )
        curve = curve + [performance]  # add performance to curve
        strategy = strategy.update(predictor, labeled_ds, unlabeled_ds)
        index_to_label_next = strategy(predictor, unlabeled_ds)
        labeled_indices = labeled_indices + [index_to_label_next]
        labeled_ds, unlabeled_ds = get_labeled_unlabeled(train_dataset, labeled_indices)
        pbar.update(1)
        pbar.set_description(f"Test performance: {performance}")
        gc.collect()
    return curve
