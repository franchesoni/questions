import os
from pathlib import Path
import torch
import numpy as np

from engine import evaluate_predictor, fit_predictor_batched
from predictor import get_resnet
from data import get_dataset, DATASET_NAMES
import visualization as vis
"""Here we pretrain networks to CIFAR and SVHN to provide oracles for our annotation tool"""

def pretrain(dataset_index):
    """Check whether we can train a resnet on few data for cifar"""
    binarizer = "geq5"
    use_only_first = False  #6000
    use_only_first_test = 1000
    device = 'cuda:1'
    # device = 'cpu'
    pretrained = False

    dataset_name = DATASET_NAMES[dataset_index]
    train_ds, test_ds = get_dataset(
        dataset_name,
        binarizer,
        use_only_first=use_only_first,
        use_only_first_test=use_only_first_test,
        device=device,
    )
    train_ds = train_ds.get_new_ds_from_indices(np.arange(6000, len(train_ds)))
    n_channels = train_ds.data[0].shape[0]
    predictor = get_resnet(
        pretrained=pretrained, n_channels=n_channels, compile=False, device=device
    )
    if os.path.isdir('pretrained_models') and f'{dataset_name}.pt' in os.listdir('pretrained_models'):
        predictor.load_state_dict(torch.load(f'pretrained_models/{dataset_name}.pt'))
        print(f'Loaded pretrained model for {dataset_name}')
    else:
        loss_fn = torch.nn.BCEWithLogitsLoss()
        predictor = fit_predictor_batched(
            predictor, loss_fn, train_ds, 6000, verbose=True
        )
        # save_predictor 
        Path('pretrained_models').mkdir(exist_ok=True)
        torch.save(predictor.state_dict(), f'pretrained_models/{dataset_name}.pt')

    eval_loss_fn = torch.nn.BCEWithLogitsLoss(reduction="sum")
    performance = evaluate_predictor(
        predictor, eval_loss_fn, test_ds, verbose=False
    )
    with torch.no_grad():
        probs = torch.sigmoid(predictor(test_ds.data)[:, 0])
        outcomes = test_ds.targets
        vis.calibration_plot(probs.cpu().numpy(), outcomes.cpu().numpy(), bins=7)

    print(performance)
    print(cls_metrics(performance))

def cls_metrics(performance):
    tp = performance['tp']
    tn = performance['tn']
    fp = performance['fp']
    fn = performance['fn']
    acc = (tp + tn) / (tp + tn + fp + fn)
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    return {'acc': acc, 'prec': prec, 'rec': rec}


if __name__ == "__main__":
    # pretrain(1)
    # pretrain(0)
    pretrain(3)