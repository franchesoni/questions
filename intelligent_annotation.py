import numpy as np
import torch
import shutil
from pathlib import Path
import gc

import tqdm

from data import get_labeled_unlabeled
import visualization as vis
from engine import optimizer_setup, fit_predictor_with_incorrect, evaluate_predictor
from simple_tree_search import STS


########## ENGINE ###########

def get_ia_curve(
    max_queries, predictor, strategy, train_dataset, test_dataset, initial_labeled_indices
):
    curve = [None] * (len(initial_labeled_indices) - 1)
    n_labeled = curve.copy()
    labeled_indices = initial_labeled_indices
    unlabeled_ds = "a value to start the loop"
    shutil.rmtree('questions')
    Path('questions').mkdir(exist_ok=True, parents=True)

    alia = STS(max_expansions=10, al_method=strategy.get_name(), max_n=8, cost_fn="length")   
    # state contains unlabeled indices and maybe an incorrect guess
    state = {"indices": [int(ind) for ind in train_dataset.indices if ind not in labeled_indices], "incorrect":None}
    root_node = STS.initialize_root_node(state)

    optimizer = optimizer_setup(predictor)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    eval_loss_fn = torch.nn.BCEWithLogitsLoss(reduction="sum")

    pbar = tqdm.tqdm(total=len(train_dataset) - len(initial_labeled_indices))
    n_queries = 0
    while n_queries < max_queries and len(unlabeled_ds) > 0:  # fit, get point, label
        ### data update ###
        labeled_ds, unlabeled_ds = get_labeled_unlabeled(
            train_dataset, labeled_indices
        )  # unlabeled_ds also stores the unlabeled indices
        if root_node.state['incorrect'] is None:
            incorrect_x_y_hat = None
        else:  # we have to build this tuple with the inputs and the incorrect guess
            inc_indices = root_node.state['incorrect'][0]
            x = train_dataset.data[inc_indices]
            y_hat = torch.tensor(root_node.state['incorrect'][1]).to(x.device)
            incorrect_x_y_hat = (x, y_hat)
        predictor = fit_predictor_with_incorrect(
            predictor, loss_fn, optimizer, labeled_ds, incorrect_x_y_hat, verbose=False
        )
        performance = evaluate_predictor(
            predictor, eval_loss_fn, test_dataset, verbose=False
        )
        curve = curve + [performance]  # add performance to curve
        n_labeled = n_labeled + [len(labeled_indices)]


        ### initialization of predictions ###
        predictor.eval()
        with torch.no_grad():
            pred_probs = torch.sigmoid(predictor(unlabeled_ds.data).flatten())
            new_predictions = (unlabeled_ds.indices.tolist(), pred_probs.cpu().numpy().tolist())
        alia.set_unlabeled_predictions(new_predictions)
        ### query ###
        best_question_node = alia.tree_search(root_node)
        question = best_question_node.guess
        indices_to_ask, guess = question
        vis.visualize_question(train_dataset.data[indices_to_ask], guess, n_queries)
        answer = train_dataset.targets[indices_to_ask].tolist() == guess
        # indices to remove from unlabeled_indices are the ones we annotate
        ### state update ###
        state, to_remove = STS.update_state(state, question, answer, ret_to_remove=True)
        root_node = STS.set_new_root_node(best_question_node.state_children[answer*1])
        assert root_node.state == state
        labeled_indices = labeled_indices + to_remove

        n_queries += 1
        pbar.update(1)
        pbar.set_description(f"Test performance: {performance}")
        gc.collect()
    return {'curve':curve, 'n_labeled':n_labeled}


