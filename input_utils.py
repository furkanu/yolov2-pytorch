from __future__ import division



import torch
import cv2
import numpy as np


def path_to_input(image_path, input_size, device):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (input_size, input_size))                              #Resize
    img = img[..., ::-1].transpose((2, 0, 1))                                    #BGR -> RGB and HxWxC -> CxHxW
    img = img[np.newaxis, ...] / 255.0                                           #Add a channel at 0, thus making it a batch
    img = torch.tensor(img, dtype=torch.float, device=device)                    #Convert to Tensor

    return img
def cv2cam_to_input(img, input_size, device):
    img = cv2.resize(img, (input_size, input_size))                              #Resize
    orig_img = img.copy()
    img = img[..., ::-1]
    img = img[..., ::-1].transpose((2, 0, 1))                                    #BGR -> RGB and HxWxC -> CxHxW
    img = img[np.newaxis, ...] / 255.0                                           #Add a channel at 0, thus making it a batch
    img = torch.tensor(img, dtype=torch.float, device=device)                    #Convert to Tensor

    return img, orig_img

