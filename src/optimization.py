import torch
import torch.nn as nn
import torch.optim


def get_loss():
    """
    Get an instance of the CrossEntropyLoss
    """
    loss  = nn.CrossEntropyLoss()

    return loss


def get_optimizer(
    model: nn.Module,
    optimizer: str = "SGD",
    learning_rate: float = 0.01,
    momentum: float = 0.5,
    weight_decay: float = 0,
):
    """
    Returns an optimizer instance

    Inputs:
        model - the model to optimize
        optimizer (str) - one of 'SGD' or 'Adam'
        learning_rate (float) - the learning rate
        momentum (float) - the momentum (if the optimizer uses it)
        weight_decay (float) - regularization coefficient
    """
    if optimizer.lower() == "sgd":
        # create an instance of the SGD optimizer
        opt = torch.optim.SGD(
            model.parameters(),
            lr = learning_rate,
            momentum = momentum,
            weight_decay = weight_decay
        )

    elif optimizer.lower() == "adam":
        # create an instance of the Adam optimizer
        opt = torch.optim.Adam(
            model.parameters(),
            lr = learning_rate,
            weight_decay = weight_decay
        )
    else:
        raise ValueError(f"Optimizer {optimizer} not supported")

    return opt


