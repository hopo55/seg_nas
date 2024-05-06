import torch.nn as nn
import torch


def binary_cross_entropy(pred, y, eps=1e-4):
    pred = pred.clamp(min=eps, max=1-eps)

    return -(pred.log()*y + (1-y)*(1-pred).log()).mean()


class DiceBCELoss(nn.Module):
    def __init__(self, weight=None):
        super(DiceBCELoss, self).__init__()

        self.weight = 0.5
        if weight is not None:
            self.weight = weight

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

        BCE = binary_cross_entropy(inputs, targets)
        Dice_BCE = self.weight * BCE + (1 - self.weight) * dice_loss

        return Dice_BCE
