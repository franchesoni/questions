from abc import ABC, abstractmethod
import random
import numpy as np
import torch
import gc

from engine import seed_everything, fit_predictor, evaluate_predictor
from data import DATASET_NAMES, get_dataset, get_labeled_unlabeled
from predictor import get_resnet
from visualization import plot_curves



class Strategy(ABC):
    def __call__(self, predictor, unlabeled_dataset):
        return self.get_index_to_label_next(predictor, unlabeled_dataset)

    @abstractmethod
    def get_index_to_label_next(self, predictor, unlabeled_dataset):
        """Should return index_to_label_next"""
        pass


class RandomSampling(Strategy):
    def get_index_to_label_next(self, predictor, unlabeled_dataset):
        index_to_label_next = random.choice(unlabeled_dataset.indices)
        return index_to_label_next

def get_al_curve(seed, predictor, strategy, train_dataset, test_dataset, initial_labeled_indices):
  seed_everything(seed)
  curve = [None] * (len(initial_labeled_indices) - 1)
  labeled_indices = initial_labeled_indices
  labeled_ds, unlabeled_ds = get_labeled_unlabeled(train_dataset, labeled_indices)  # unlabeled_ds also stores the unlabeled indices
  while True:  # fit, get point, label
    predictor = fit_predictor(predictor, labeled_ds)
    performance = evaluate_predictor(predictor, test_dataset)
    curve = curve + [performance]  # add performance to curve
    index_to_label_next = strategy(predictor, unlabeled_ds)
    labeled_indices = labeled_indices + [index_to_label_next]
    labeled_ds, unlabeled_ds = get_labeled_unlabeled(train_dataset, labeled_indices)
    if len(unlabeled_ds) == 0: 
      break
    gc.collect()
  return curve


def run_active_learning(experiment_name='random', dataset_index=1, binarizer='geq5', pretrained=False, seeds=[0], use_only_first=10):
    dataset_name = DATASET_NAMES[dataset_index]
    train_ds, test_ds = get_dataset(dataset_name, binarizer, use_only_first=use_only_first)
    n_channels = train_ds[0][0].shape[0]
    predictor = get_resnet(pretrained=pretrained, n_channels=n_channels, compile=True)
    if experiment_name == 'random':
        strategy = RandomSampling()
    curves = []
    for seed in seeds:
        curve = get_al_curve(seed, predictor, strategy, train_ds, test_ds, initial_labeled_indices=[0], experiment_name=experiment_name)
        np.save(f'curve_{experiment_name}_seed_{seed}.npy', curve)
        curves.append(curve)
    plot_curves(curves, experiment_name=experiment_name)
