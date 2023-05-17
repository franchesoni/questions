import torch

from engine import fit_predictor, evaluate_predictor
from predictor import get_resnet
from data import get_dataset, DATASET_NAMES

def mvp1():
    """Check whether we can train a resnet on few data for cifar"""
    dataset_index = 4
    binarizer = "geq5"
    use_only_first = 6000
    use_only_first_test = 6000
    device = 'cuda:1'
    pretrained = False

    dataset_name = DATASET_NAMES[dataset_index]
    train_ds, test_ds = get_dataset(
        dataset_name,
        binarizer,
        use_only_first=use_only_first,
        use_only_first_test=use_only_first_test,
        device=device,
    )
    n_channels = train_ds.data[0].shape[0]
    predictor = get_resnet(
        pretrained=pretrained, n_channels=n_channels, compile=False, device=device
    )


    loss_fn = torch.nn.BCEWithLogitsLoss()
    eval_loss_fn = torch.nn.BCEWithLogitsLoss(reduction="sum")

    predictor = fit_predictor(
        predictor, loss_fn, train_ds, verbose=False
    )
    performance = evaluate_predictor(
        predictor, eval_loss_fn, test_ds, verbose=False
    )
