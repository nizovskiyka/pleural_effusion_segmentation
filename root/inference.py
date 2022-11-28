import numpy as np
import torch
import cv2
from torch import nn

@torch.no_grad()
def predict_image(model: object,
                  image: np.array,
                  device: torch.cuda.device):
    '''Single prediction of an image'''
    image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    image = cv2.resize(image, (224,224), interpolation = cv2.INTER_AREA)
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=0)
    image_tensor = torch.tensor(image).to(device)
    with torch.no_grad():
        y_pred  = model(image_tensor)
        y_pred = (nn.Sigmoid()(y_pred)>0.01).double()
    return y_pred.detach().cpu().numpy()[0][0]
