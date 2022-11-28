import torch
import cv2
import albumentations as A
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from utils import load_msk, load_img

class BuildDataset(torch.utils.data.Dataset):
    '''Npy 2D dataset'''

    def __init__(self,
                 dataset_df: pd.DataFrame,
                 label: bool = True,
                 transforms: dict = None):

        self.dataset_df = dataset_df
        self.label = label
        self.img_paths = dataset_df['image_path'].tolist()
        self.msk_paths = dataset_df['mask_path'].tolist()
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset_df)

    def __getitem__(self, index):

        img_path = self.img_paths[index]
        img = []
        img = load_img(img_path)

        if self.label:
            msk_path = self.msk_paths[index]
            msk = load_msk(msk_path)
            if self.transforms:
                data = self.transforms(image=img, mask=msk)
                img = data['image']
                msk = data['mask']
            img = np.transpose(img, (2, 0, 1))
            msk = np.transpose(msk, (2, 0, 1))
            return torch.tensor(img), torch.tensor(msk)

        if self.transforms:
            data = self.transforms(image=img)
            img = data['image']
        img = np.transpose(img, (2, 0, 1))

        return torch.tensor(img)


def get_transforms(cfg: object) -> dict:
    '''Generate transforms dict based on the cfg'''
    data_transforms = {
        "train": A.Compose([
            A.Resize(*cfg.img_size, interpolation=cv2.INTER_NEAREST),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625,
                               scale_limit=0.05,
                               rotate_limit=10, p=0.5),
            A.OneOf([
                A.GridDistortion(num_steps=5,
                                 distort_limit=0.05,
                                 p=1.0),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0)
            ], p=0.25),
            A.CoarseDropout(max_holes=8,
                            max_height=cfg.img_size[0]//20,
                            max_width=cfg.img_size[1]//20,
                            min_holes=5,
                            fill_value=0,
                            mask_fill_value=0,
                            p=0.5),
        ], p=1.0),

        "valid": A.Compose([
            A.Resize(*cfg.img_size, interpolation=cv2.INTER_NEAREST),
        ], p=1.0)
    }
    return data_transforms


def prepare_loaders(dataset_df: pd.DataFrame, fold: int, cfg: object) -> tuple:
    '''Create train and val dataloaders for current fold'''
    data_transforms = get_transforms(cfg)
    train_df = dataset_df.query("fold!=@fold").reset_index(drop=True)
    valid_df = dataset_df.query("fold==@fold").reset_index(drop=True)
    if cfg.debug:
        train_df = train_df.head(32*5).query("empty==0")
        valid_df = valid_df.head(32*3).query("empty==0")
    train_dataset = BuildDataset(train_df, transforms=data_transforms['train'])
    valid_dataset = BuildDataset(valid_df, transforms=data_transforms['valid'])

    train_loader = DataLoader(train_dataset, batch_size=cfg.train_bs if not cfg.debug else 20,
                              num_workers=4, shuffle=True, pin_memory=True, drop_last=False)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.valid_bs if not cfg.debug else 20,
                              num_workers=4, shuffle=False, pin_memory=True)
    return train_loader, valid_loader
