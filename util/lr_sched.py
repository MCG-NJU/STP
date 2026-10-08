"""
Learning-rate schedule for STP pre-training: linear warmup, then cosine decay.
"""

import math


def adjust_learning_rate(optimizer, epoch, args):
    """Set the optimizer LR for the given (possibly fractional) epoch.

    Warmup ramps from 0 to `args.lr` over `args.warmup_epochs`, then a
    half-cycle cosine decays to `args.min_lr` at `args.epochs`.

    Returns the learning rate that was applied.
    """
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (1. + math.cos(math.pi * progress))

    for param_group in optimizer.param_groups:
        # per-group scaling (e.g. layer-wise lr decay) is applied on top
        param_group['lr'] = lr * param_group['lr_scale'] if 'lr_scale' in param_group else lr

    return lr
