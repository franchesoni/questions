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
    max_queries, predictor, train_dataset, test_dataset, initial_labeled_indices, sts_kwargs={}, dstfilename=None
):
    curve = [None] * (len(initial_labeled_indices) - 1)
    n_labeled = curve.copy()
    labeled_indices = initial_labeled_indices
    unlabeled_ds = "a value to start the loop"
    if Path('questions').exists():
        shutil.rmtree('questions')
    Path('questions').mkdir(exist_ok=True, parents=True)

    alia = STS(**sts_kwargs)
    # state contains unlabeled indices and maybe an incorrect guess
    train_dataset_indices_list = train_dataset.indices.tolist()
    assert sorted(train_dataset_indices_list) == train_dataset_indices_list
    indices = sorted(list(set(train_dataset_indices_list) - set(initial_labeled_indices)))
    state = {"indices": np.array(indices), "incorrect":None}
    root_node = STS.initialize_root_node(state)

    loss_fn = torch.nn.BCEWithLogitsLoss()
    eval_loss_fn = torch.nn.BCEWithLogitsLoss(reduction="sum")

    pbar = tqdm.tqdm(total=len(train_dataset) - len(initial_labeled_indices))
    n_queries = 0
    while n_queries < max_queries and len(unlabeled_ds) > 0:  # fit, get point, label
        ### data update ###
        labeled_ds, unlabeled_ds, unlabeled_indices = get_labeled_unlabeled(
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
            predictor, loss_fn, labeled_ds, incorrect_x_y_hat, verbose=False
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
            new_predictions = (unlabeled_indices, pred_probs.cpu().numpy())
        alia.set_unlabeled_predictions(new_predictions)
        ### query ###
        best_question_node = alia.tree_search(root_node)
        question = best_question_node.guess
        indices_to_ask, guess = question
        # vis.visualize_question(train_dataset.data[indices_to_ask], guess, n_queries)
        answer = (train_dataset.targets[indices_to_ask].cpu().numpy() == guess).all()
        # indices to remove from unlabeled_indices are the ones we annotate
        ### state update ###
        state, to_remove = STS.update_state(state, question, answer, ret_to_remove=True)
        root_node = STS.set_new_root_node(best_question_node.state_children[answer*1])
        # check for equality
        check = True
        for key in root_node.state:
            if not np.array_equal(root_node.state[key], state[key]):
                check = False
                break
        assert check
        labeled_indices = labeled_indices + to_remove.tolist()

        n_queries += 1
        pbar.update(1)
        pbar.set_description(f"Test performance: {performance}")
        if dstfilename is not None:
            np.save(dstfilename, {'curve':curve, 'n_labeled':n_labeled})
    return {'curve':curve, 'n_labeled':n_labeled}


