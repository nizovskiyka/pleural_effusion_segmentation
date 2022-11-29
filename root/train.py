import sys
import os
import glob
import gc
import json
import configparser
import torch
import wandb
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
import matplotlib.pyplot as plt
from model import build_model
from trainer import fetch_scheduler, run_training
from dataset import get_transforms, prepare_loaders
from utils import (set_seed,
                   get_mask_path,
                   mask_empty,
                   get_case,
                   get_id)

def parse_config(config_path: str) -> object:
    '''Parse config from .ini file'''
    config = configparser.ConfigParser()
    config.read(config_path)

    class CFG:
        '''Main config for training and logging'''
        dataset_path = config["TRAIN"]["dataset_path"]
        seed = int(config["TRAIN"]["seed"])
        debug = bool(config["TRAIN"]["debug"])
        exp_name = config["WANDB"]["exp_name"]
        comment = config["WANDB"]["comment"]
        model_name = config["TRAIN"]["model_name"]
        backbone = config["TRAIN"]["backbone"]
        train_bs = int(config["TRAIN"]["train_bs"])
        valid_bs = int(config["TRAIN"]["valid_bs"])
        img_size = json.loads(config["TRAIN"]["img_size"])
        epochs = int(config["TRAIN"]["epochs"])
        lr = float(config["TRAIN"]["lr"])
        scheduler = config["TRAIN"]["scheduler"]
        min_lr = float(config["TRAIN"]["min_lr"])
        T_max = int(30000/train_bs*epochs)+50
        T_0 = int(config["TRAIN"]["T_0"])
        warmup_epochs = int(config["TRAIN"]["warmup_epochs"])
        wd = float(config["TRAIN"]["wd"])
        n_accumulate = max(1, 32//train_bs)
        n_fold = int(config["TRAIN"]["n_fold"])
        num_classes = int(config["TRAIN"]["num_classes"])
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device = 'cpu'
        use_wandb = config["WANDB"]["use_wandb"]
        wandb_secret = config["WANDB"]["secret"]

    return CFG

if __name__ == "__main__":
    # CFG_PATH = sys.argv[1]
    CFG_PATH = "./default_params.ini"
    if not os.path.exists("/artefacts"):
        os.makedirs("/artefacts")
    cfg = parse_config(CFG_PATH)

    set_seed(cfg.seed)
    gc.collect()

    image_paths = glob.glob(os.path.join(cfg.dataset_path, "images", "*.npy"))

    mask_paths = [get_mask_path(x) for x in image_paths]
    is_empty = [mask_empty(x) for x in mask_paths]

    df = pd.DataFrame({"image_path":image_paths,
                    "mask_path":mask_paths,
                    "empty": is_empty})

    df["case"] = df["image_path"].apply(get_case)
    df["id"] = df["image_path"].apply(get_id)

    # K-fold split
    skf = StratifiedGroupKFold(n_splits=cfg.n_fold, shuffle=True, random_state=cfg.seed)
    for fold, (train_idx, val_idx) in enumerate(
        skf.split(df, df['empty'], groups = df["case"])):
        df.loc[val_idx, 'fold'] = fold

    model = build_model(cfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    scheduler = fetch_scheduler(optimizer, cfg)
    # Main Loop
    data_transforms = get_transforms(cfg)
    best_dice = -1
    best_fold = -1
    best_losslist = []
    for fold in range(cfg.n_fold):
        print(f'### Fold: {fold+1} of {cfg.n_fold}')
        if cfg.use_wandb:
            wandb.login(key=cfg.wandb_secret)
            run = wandb.init(project='pleural-effusion-2d-seg',
                config={k:v for k, v in dict(vars(cfg)).items() if '__' not in k},
                name=f"fold-{fold}|dim-{cfg.img_size[0]}x{cfg.img_size[1]}|model-{cfg.model_name}",
                group=cfg.comment)

        train_loader, valid_loader = prepare_loaders(df, fold, cfg)
        model = build_model(cfg)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
        scheduler = fetch_scheduler(optimizer, cfg)
        model, history, fold_best_dice = run_training(model, train_loader, valid_loader, 
                                                      optimizer, scheduler, cfg, run, fold)
        if fold_best_dice < best_dice:
            print(f"New best fold: {fold}")
            best_fold = fold
            PATH = os.path.join("/artefacts", "best_epoch-all-folds.bin")
            torch.save(model.state_dict(), PATH)
            wandb.save(PATH)
            best_dice = fold_best_dice
            best_losslist = history["Valid Loss"]

        fig, ax = plt.subplots( nrows=1, ncols=1 )
        ax.plot(best_losslist)
        ax.title.set_text('Dice loss')
        ax.title.set_text('Dice loss')
        ax.set_xlabel("Epochs")
        ax.set_ylabel("1 - Dice")
        fig.savefig("/artefacts/dice_plot.png")
        plt.close(fig)
        if cfg.use_wandb:
            run.finish()
