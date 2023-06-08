import torch.nn as nn


def get_loss_fn(loss_name):
    if loss_name == 'xent':
        cel = nn.CrossEntropyLoss()

        def loss_fn(pred, y):
            if len(y.shape) == 2:
                y = y.squeeze(1)
            return cel(pred, y)
    elif loss_name == 'l1':
        loss_fn = nn.L1Loss()
    elif loss_name == 'l2':
        loss_fn = nn.MSELoss()
    elif loss_name == 'bce':
        loss_fn = nn.BCELoss()
    elif loss_name == 'bce_with_logits':
        bwl = nn.BCEWithLogitsLoss()

        def loss_fn(pred, y):
            if len(y.shape) == 1:
                y = y.unsqueeze(1)
            return bwl(pred, y.float())
    else:
        raise NotImplementedError
    return loss_fn
