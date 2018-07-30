import torch.nn as nn
import torch.optim as optim

from models.loss import *


def get_loss_fn(loss_name, args=None):
    """Get a loss function to evaluate a model.

    Args:
        loss_name: crps

    Returns:
        Differentiable criterion that can be applied to targets, logits.
    """

    # if loss_name == 'crps':
    return CRPSLoss()


def get_optimizer(parameters, args):
    """Get a PyTorch optimizer for params.

    Args:
        parameters: Iterator of network parameters to optimize (i.e., model.parameters()).
        args: Command line arguments.

    Returns:
        PyTorch optimizer specified by args_.
    """



    if args.optimizer == 'sgd':
        optimizer = optim.SGD(parameters, args.learning_rate,
                              momentum=args.sgd_momentum,
                              weight_decay=args.weight_decay,
                              dampening=args.sgd_dampening)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(parameters, args.learning_rate,
                               betas=(args.adam_beta_1, args.adam_beta_2), weight_decay=args.weight_decay)
    else:
        raise ValueError('Unsupported optimizer: {}'.format(args.optimizer))

    return optimizer


def get_scheduler(optimizer, args):
    """Get a learning rate scheduler.

    Args:
        optimizer: The optimizer whose learning rate is modified by the returned scheduler.
        args: Command line arguments.

    Returns:
        PyTorch scheduler that update the learning rate for `optimizer`.
    """
    if args.lr_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma)
    elif args.lr_scheduler == 'multi_step':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=args.lr_decay_gamma)
    elif args.lr_scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.lr_decay_gamma, patience=args.patience)
    else:
        raise ValueError('Invalid learning rate scheduler: {}.'.format(args.scheduler))

    return scheduler


def step_scheduler(lr_scheduler, metrics, epoch, best_ckpt_metric='val_loss'):
    """Step a LR scheduler."""
    if isinstance(lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
        if best_ckpt_metric in metrics:
            lr_scheduler.step(metrics[best_ckpt_metric], epoch=epoch)
    else:
        lr_scheduler.step(epoch=epoch)


def get_lr(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr.append(param_group['lr'])
    return lr
