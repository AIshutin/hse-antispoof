import torch

def getOneCycleScheduler(optimizer, max_lr, pct_start, anneal_strategy, 
                         steps_per_epoch, epochs, **kwargs):
    return torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=max_lr,
            pct_start=pct_start,
            anneal_strategy=anneal_strategy,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs
    )