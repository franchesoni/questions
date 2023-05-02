import torch
import psutil
import numpy as np
import random
import time

def seed_everything(seed):
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  np.random.seed(seed)
  random.seed(seed)



def cpu_count():
    return len(psutil.Process().cpu_affinity())

def fit_predictor(predictor, labeled_ds, max_visited_datapoints=60000, max_epochs=3, lr=0.001, optim='adam', batch_size=32, verbose=False):
    """Here we fit a pytorch model to the labeled dataset. We always allow the model to see visited_datapoints."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    predictor = predictor.to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    n_epochs = min(max(max_visited_datapoints // len(labeled_ds), 1), max_epochs)  # always do at least one epoch
    dl = torch.utils.data.DataLoader(labeled_ds, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=cpu_count(), pin_memory=device=='cuda')  # here comes the stochasticity
    optimizer = torch.optim.Adam(predictor.parameters(), lr=lr) if optim == 'adam' else torch.optim.SGD(predictor.parameters(), lr=lr)

    if verbose:
      print(f"Fitting predictor for {n_epochs} epochs")

    times = {'forward': 0, 'backward': 0, 'loss': 0, 'data': 0}
    for epoch in range(n_epochs):
        st = time.time()
        for batch in dl:
            input, target = batch
            input, target = input.to(device), target.to(device).float()
            times['data'] += time.time() - st
            st = time.time()
            output = predictor(input)[:, 0]  # select the only output and remove the channel dimension
            times['forward'] += time.time() - st
            st = time.time()
            loss = loss_fn(output, target)
            times['loss'] += time.time() - st
            st = time.time()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            times['backward'] += time.time() - st
            st = time.time()
        if verbose:
            print(f"Epoch {epoch} / {n_epochs}, loss {loss.item()}")
    if verbose:
        times = {k: v / n_epochs / len(dl) for k, v in times.items()}
        print(f"Average times per batch: {times}")
        print(f"Percentual time per batch: {dict(zip(times.keys(), [v / sum(times.values()) for v in times.values()]))}")
    return predictor


def fit_predictor_to_info(predictor, info, max_visited_datapoints=60000, max_epochs=3, lr=0.001, optim='adam', batch_size=32, verbose=False):
    """Here we fit a pytorch model to the labeled dataset. We always allow the model to see visited_datapoints."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    predictor = predictor.to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    labeled_ds = info['labeled']
    n_epochs = min(max(max_visited_datapoints // len(labeled_ds), 1), max_epochs)  # always do at least one epoch
    dl = torch.utils.data.DataLoader(labeled_ds, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=cpu_count(), pin_memory=device=='cuda')  # here comes the stochasticity
    optimizer = torch.optim.Adam(predictor.parameters(), lr=lr) if optim == 'adam' else torch.optim.SGD(predictor.parameters(), lr=lr)

    if verbose:
      print(f"Fitting predictor for {n_epochs} epochs")

    times = {'forward': 0, 'backward': 0, 'loss': 0, 'data': 0}
    for epoch in range(n_epochs):
        st = time.time()
        for batch in dl:
            input, target = batch
            input, target = input.to(device), target.to(device).float()
            times['data'] += time.time() - st
            st = time.time()
            output = predictor(input)[:, 0]  # select the only output and remove the channel dimension
            times['forward'] += time.time() - st
            st = time.time()
            loss = loss_fn(output, target)
            times['loss'] += time.time() - st
            st = time.time()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            times['backward'] += time.time() - st
            st = time.time()
        if verbose:
            print(f"Epoch {epoch} / {n_epochs}, loss {loss.item()}")
    if verbose:
        times = {k: v / n_epochs / len(dl) for k, v in times.items()}
        print(f"Average times per batch: {times}")
        print(f"Percentual time per batch: {dict(zip(times.keys(), [v / sum(times.values()) for v in times.values()]))}")
    return predictor


        
def evaluate_predictor(predictor, test_dataset, verbose=True):
    """Here we evaluate the predictor on the test dataset."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    predictor = predictor.to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction='sum')

    dl = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, drop_last=False, num_workers=cpu_count(), pin_memory=device=='cuda')  # here comes the stochasticity
    loss = 0
    tp, fp, tn, fn = 0, 0, 0, 0
    for batch in dl:
        input, target = batch
        input, target = input.to(device), target.to(device).float()
        output = predictor(input)[:, 0]  # select the only output and remove the channel dimension
        tp += ((output > 0) & (target == 1)).sum().item()
        fp += ((output > 0) & (target == 0)).sum().item()
        tn += ((output <= 0) & (target == 0)).sum().item()
        fn += ((output <= 0) & (target == 1)).sum().item()
        loss += loss_fn(output, target).item() 
    metrics = {'loss':loss / len(test_dataset), 'tp':tp, 'fp':fp, 'tn':tn, 'fn':fn}
    if verbose: 
        print(metrics)
    return metrics

