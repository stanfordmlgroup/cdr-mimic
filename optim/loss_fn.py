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
