"""
docstring
"""

def exp_lr(optimizer, epoch, init_lr=1e-3, lr_decay_epoch=30):
    """
    Decay learning rate as epochs progress.

    Parameters:
    ----------
    optimizer: torch.optim optimizer class
    epoch: int
    init_lr: float
        initial learning rate
    lr_decay_epoch: int
        number of epochs between decay steps

    Returns:
    --------
    optimizer
    """
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return optimizer

