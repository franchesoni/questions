from abc import ABC, abstractmethod
import random

import tqdm
import torch
import gc

from engine import optimizer_setup, seed_everything, fit_predictor, evaluate_predictor
from data import InMemoryDataset, get_labeled_unlabeled
from predictor import compute_embedding


class Strategy(ABC):
    def __call__(self, predictor, unlabeled_dataset):
        return self.get_index_to_label_next(predictor, unlabeled_dataset)

    @abstractmethod
    def get_index_to_label_next(self, predictor, unlabeled_dataset):
        """Should return index_to_label_next"""
        pass

    def update(self, predictor, labeled_dataset, unlabeled_dataset):
        return self






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

class CoreSet(Strategy):

    """
    Implementation of CoreSet :footcite:`sener2018active` Strategy. A diversity-based
    approach using coreset selection. The embedding of each example is computed by the network’s
    penultimate layer and the samples at each round are selected using a greedy furthest-first
    traversal conditioned on all labeled examples.

    Stolen from distil library
    """

    def update(self, predictor, labeled_dataset: InMemoryDataset, unlabeled_dataset):
        predictor.eval()
        with torch.no_grad():
            self.labeled_embeddings = compute_embedding(predictor, labeled_dataset.data)
        return self
        
    def _furthest_first(self, unlabeled_embeddings, labeled_embeddings):
        if labeled_embeddings.shape[0] == 0:
            raise ValueError("labeled_embeddings must be non-empty")
        dist_ctr = torch.cdist(unlabeled_embeddings, labeled_embeddings, p=2)
        min_dist = torch.min(dist_ctr, dim=1)[0]
        return torch.argmax(min_dist)

    def get_index_to_label_next(self, predictor, unlabeled_dataset):
        predictor.eval()
        with torch.no_grad():
            unlabeled_embeddings = compute_embedding(predictor, unlabeled_dataset.data)
        furthest_unlabeled_index = self._furthest_first(unlabeled_embeddings, self.labeled_embeddings)
        index_to_label_next = unlabeled_dataset.indices[furthest_unlabeled_index]
        return int(index_to_label_next)



def get_al_curve(
    max_queries, predictor, strategy, train_dataset, test_dataset, initial_labeled_indices
):
    curve = [None] * (len(initial_labeled_indices) - 1)
    labeled_indices = initial_labeled_indices
    unlabeled_ds = "a value to start the loop"

    optimizer = optimizer_setup(predictor)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    eval_loss_fn = torch.nn.BCEWithLogitsLoss(reduction="sum")

    pbar = tqdm.tqdm(total=len(train_dataset) - len(initial_labeled_indices))
    n_queries = 0
    while n_queries < max_queries and len(unlabeled_ds) > 0:  # fit, get point, label
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
        n_queries += 1
        pbar.update(1)
        pbar.set_description(f"Test performance: {performance}")
        gc.collect()
    return curve
