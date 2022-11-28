import random
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from segmentation_mask_overlay import overlay_masks

def load_img(img_path: str) -> np.array:
    '''Load single 2D image from np array'''
    img = np.load(img_path)
    img = np.tile(img[...,None], [1, 1, 1])
    img = img.astype('float32')
    max_num = np.max(img)
    if max_num:
        img/=max_num
    return img

def load_msk(msk_path: str) -> np.array:
    '''Load single 2D mask from np array'''
    msk = np.load(msk_path)
    msk = np.tile(msk[...,None], [1, 1, 1])
    return msk

def set_seed(seed: int = 42) -> None:
    '''Sets the seed so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_mask_path(image_path: str) -> str:
    '''Get path of the mask .nii.gz file corresponding to the image path'''
    path_list = image_path.split("/")
    path_list[-2] = "masks"
    return "/".join(path_list)

def mask_empty(mask_path: str) -> bool:
    '''Check if mask is empty (only 0s)'''
    mask = np.load(mask_path)
    return not np.any(mask)

def get_case(image_path: str) -> str:
    '''Get case from image path'''
    path_list = image_path.split("/")
    fname_list = path_list[-1].split("-")
    del fname_list[-1]
    return "-".join(fname_list)

def get_id(image_path: str) -> str:
    '''Get study id from image path'''
    path_list = image_path.split("/")
    fname_list = path_list[-1]
    return fname_list

def save_overlay(image: np.array, 
                 mask: np.array, 
                 predict: np.array = None, 
                 out_path: str = "./sample.png") -> None:
    '''Save image with mask overlay'''
    layers = []
    layer_labels = []
    mask = np.where(mask<0.5, 0, 1)
    bool_mask = np.array(mask, dtype=bool)
    layers.append(bool_mask)
    layer_labels.append("mask")
    if isinstance(predict, np.ndarray):
        predict = np.where(predict<0.5, 0, 1)
        bool_predict = np.array(predict, dtype=bool)
        layers.append(bool_predict)
        layer_labels.append("predict")
    cmap = np.array([[0., 0., 1., 1],[1., 0., 0., 1.,]])
    fig = overlay_masks(image, layers, labels=layer_labels, colors=cmap, mask_alpha=0.5)
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
