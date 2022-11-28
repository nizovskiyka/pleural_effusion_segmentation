import gc
import os
import time
import copy
from collections import defaultdict
import numpy as np
import torch
import wandb
from torch import nn
from torch.cuda import amp
from torch.optim import lr_scheduler
from loss import criterion

def fetch_scheduler(optimizer: object, cfg: object) -> object:
    '''Create scheduler object'''

    if cfg.scheduler is None:
        return None
    elif cfg.scheduler == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=cfg.T_max, 
                                                   eta_min=cfg.min_lr)
    elif cfg.scheduler == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=cfg.T_0, 
                                                             eta_min=cfg.min_lr)
    elif cfg.scheduler == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                   mode='min',
                                                   factor=0.1,
                                                   patience=4,
                                                   threshold=0.0001,
                                                   min_lr=cfg.min_lr,)
    elif cfg.scheduer == 'ExponentialLR':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.85)

    return scheduler


def train_one_epoch(model: object,
                    optimizer: object,
                    dataloader: object,
                    cfg: object) -> float:

    '''Single epoch training loop'''

    model.train()
    scaler = amp.GradScaler()
    dataset_size = 0
    running_loss = 0.0

    for step, (images, masks) in enumerate(dataloader):
        images = images.to(cfg.device, dtype=torch.float)
        masks  = masks.to(cfg.device, dtype=torch.float)
        batch_size = images.size(0)

        with amp.autocast(enabled=True):
            y_pred = model(images)
            loss   = criterion(y_pred, masks)
            loss   = loss / cfg.n_accumulate

        scaler.scale(loss).backward()

        if (step + 1) % cfg.n_accumulate == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        epoch_loss = running_loss / dataset_size

    torch.cuda.empty_cache()
    gc.collect()
    return epoch_loss

@torch.no_grad()
def valid_one_epoch(model: object,
                    dataloader: object,
                    scheduler: object,
                    device: torch.cuda.device):
    '''Single epoch training loop'''

    model.eval()
    dataset_size = 0
    running_loss = 0.0
    val_scores = []

    for _, (images, masks) in enumerate(dataloader):
        images  = images.to(device, dtype=torch.float)
        masks   = masks.to(device, dtype=torch.float)
        batch_size = images.size(0)
        y_pred  = model(images)
        loss    = criterion(y_pred, masks)
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        epoch_loss = running_loss / dataset_size
        y_pred = nn.Sigmoid()(y_pred)

        val_scores.append(1-epoch_loss)

    if scheduler is not None:
        scheduler.step(epoch_loss)

    torch.cuda.empty_cache()
    gc.collect()

    return epoch_loss

def run_training(model: object,
                 train_loader: torch.utils.data.DataLoader,
                 valid_loader: torch.utils.data.DataLoader,
                 optimizer: object,
                 scheduler: object,
                 cfg: object,
                 run: object,
                 fold: int) -> tuple:
    '''Main training loop'''
    # To automatically log gradients
    # if cfg.use_wandb:
        # wandb.watch(model, log_freq=100)
    if not os.path.exists("/artefacts"):
        os.makedirs("/artefacts")
    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = np.inf
    best_epoch = -1
    history = defaultdict(list)

    for epoch in range(1, cfg.epochs + 1):
        gc.collect()
        print(f'Epoch {epoch}/{cfg.epochs}')

        train_loss = train_one_epoch(model,
                                     optimizer,
                                     train_loader,
                                     cfg)

        val_loss = valid_one_epoch(model,
                                   valid_loader,
                                   scheduler,
                                   device=cfg.device)

        history['Train Loss'].append(train_loss)
        history['Valid Loss'].append(val_loss)

        # Log the metrics
        if cfg.use_wandb:
            wandb.log({"Train Loss": train_loss,
                       "Valid Loss": val_loss,
                       "LR":scheduler.get_last_lr()[0]
                      })

        # deep copy the model
        if val_loss < best_loss:
            best_epoch = epoch
            run.summary["Best Epoch"]   = best_epoch
            best_model_wts = copy.deepcopy(model.state_dict())
            model_path = os.path.join("/artefacts", f"best_epoch-{fold:02d}.bin")
            torch.save(model.state_dict(), model_path)
            # Save a model file from the current directory
            wandb.save(model_path)
            best_loss = val_loss

        model_path = os.path.join("/artefacts", f"last_epoch-{fold:02d}.bin")
        torch.save(model.state_dict(), model_path)

    end = time.time()
    time_elapsed = end - start

    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, history, best_loss
