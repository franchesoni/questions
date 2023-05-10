import numpy as np
import torch
import gc
from data import get_initial_info
from abc import ABC, abstractmethod

from engine import seed_everything, fit_predictor, evaluate_predictor


####### IA STRATEGIES #######
class ALIAStrategy(ABC):
    def __call__(self, predictor, unlabeled_dataset, current_info):
        return self.get_new_question(predictor, unlabeled_dataset, current_info)

    @abstractmethod
    def get_new_question(self, predictor, train_dataset, current_info):
        """Should return new_question"""
        pass


class GuessRolloutLength(ALIAStrategy):
    def __init__(self, rollout_depth=1, n_max=10):
        self.rollout_length = rollout_depth

    def get_new_question(self, train_dataset, current_info):
        # we need current info to avoid redundancy
        # level 1 rollout is the same than a greedy method
        # in this case we care about the length (not the entropy)
        # minimizing expected length is checking the expected length for increasing $n$ where $n$ is the number of elements in the guess and are the most likely
        # if minimizer was already chosen we look for smaller and bigger ns
        train_dataset


########## ENGINE ###########


def get_alia_curve(
    seed, predictor, alia_strategy, train_dataset, test_dataset, initial_labeled_indices
):
    seed_everything(seed)
    curve = [None] * (len(initial_labeled_indices) - 1)
    current_info = get_initial_info(train_dataset, initial_labeled_indices)
    while True:  # fit, get point, label
        predictor = fit_predictor_to_info(predictor, current_info)
        performance = evaluate_predictor(predictor, test_dataset)
        curve = curve + [performance]  # add performance to curve

        new_question = alia_strategy(predictor, train_dataset, current_info)
        answer = ask_oracle(train_dataset, new_question)
        collected_info = update_collected_info(new_question, answer)

        if collected_info.is_complete:
            break
        gc.collect()
    return curve
