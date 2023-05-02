import numpy as np
import torch
import gc
from data import get_initial_info

from engine import seed_everything, fit_predictor, evaluate_predictor

def get_alia_curve(seed, predictor, alia_strategy, train_dataset, test_dataset, initial_labeled_indices):
  seed_everything(seed)
  curve = [None] * (len(initial_labeled_indices) - 1)
  current_info = get_initial_info(train_dataset, initial_labeled_indices)
  while True:  # fit, get point, label
    predictor = fit_predictor_to_info(predictor, current_info)
    performance = evaluate_predictor(predictor, test_dataset)
    curve = curve + [performance]  # add performance to curve

    new_question = alia_strategy(predictor, unlabeled_ds)
    answer = ask_oracle(train_dataset, new_question)
    collected_info = update_collected_info(new_question, answer)

    if collected_info.is_complete: 
      break
    gc.collect()
  return curve




