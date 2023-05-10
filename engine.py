import torch
import gc
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

def optimizer_setup(predictor, lr=0.01, optim='adam'):
    optimizer = torch.optim.Adam(predictor.parameters(), lr=lr) if optim == 'adam' else torch.optim.SGD(predictor.parameters(), lr=lr)
    return optimizer

def fit_predictor(predictor, loss_fn, optimizer, labeled_ds, n_epochs=24, verbose=True):
    """Here we fit a pytorch model to the labeled dataset. We always allow the model to see visited_datapoints."""
    # predictor.train()

    if verbose:
      print(f"Fitting predictor for {n_epochs} epochs")
    for epoch in range(n_epochs):
        output = predictor(labeled_ds.data)[:, 0]  # remove channel dimension
        loss = loss_fn(output, labeled_ds.targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if verbose:
            print(f"Epoch {epoch+1} / {n_epochs}, loss {loss.item()}", end='\r')
    if verbose:
        print("Done fitting predictor with loss", loss.item(), "       ")
    return predictor


        
def evaluate_predictor(predictor, loss_fn, test_dataset, batch_size=5000, verbose=True):
    """Here we evaluate the predictor on the test dataset."""
    # predictor = predictor.eval()

    loss = 0
    tp, fp, tn, fn = 0, 0, 0, 0
    batch_size = min(batch_size, len(test_dataset))
    n_batches = len(test_dataset) // batch_size 
    for batch in range(n_batches):
        if verbose:
            print('evaluating batch', batch+1, '/', n_batches, end='\r')
        ds = test_dataset.get_new_ds_from_indices(list(range(batch * batch_size, (batch + 1) * batch_size)))
        with torch.no_grad():
            output = predictor(ds.data)[:, 0]
        target = ds.targets
        tp += ((output > 0) & (target == 1)).sum().item()
        fp += ((output > 0) & (target == 0)).sum().item()
        tn += ((output <= 0) & (target == 0)).sum().item()
        fn += ((output <= 0) & (target == 1)).sum().item()
        loss += loss_fn(output, target).item() 
        gc.collect()
    metrics = {'loss':loss / len(test_dataset), 'tp':tp, 'fp':fp, 'tn':tn, 'fn':fn}
    if verbose: 
        print(metrics)
    return metrics

