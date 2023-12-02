import torch


def getExponentialScheduler(optimizer, gamma, **kwargs):
    return torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma
    )