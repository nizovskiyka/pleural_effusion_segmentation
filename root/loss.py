import torch
import segmentation_models_pytorch as smp

DiceLoss    = smp.losses.DiceLoss(mode='binary')

def criterion(y_pred: torch.tensor, y_true: torch.tensor) -> torch.tensor:
    '''The criterion to calculate loss'''
    return DiceLoss(y_pred, y_true)