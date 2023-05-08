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
        index_to_label_next = int(random.choice(unlabeled_dataset.indices))
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


