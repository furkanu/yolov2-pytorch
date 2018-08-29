from __future__ import division

import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import patches, patheffects
import cv2
import math


def to_np(arr):
    return arr.detach().cpu().numpy()
def centerwh_to_corners_pt(boxes):
    '''
    converts boxes in format (center_x, center_y, width, height)
    to corner format (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
    :param boxes: boxes with 4 coordinates of format (center_x, center_y, width, height)
    :return: boxes in format (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
    '''
    if len(boxes) == 0: #if there is no detection, return boxes
        return boxes

    return torch.cat([boxes[..., :2]-boxes[..., 2:]/2, boxes[..., :2]+boxes[... ,2:]/2], -1)
def centerwh_to_corners(boxes):
    '''
    converts boxes in format (center_x, center_y, width, height)
    to corner format (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
    :param boxes: boxes with 4 coordinates of format (center_x, center_y, width, height)
    :return: boxes in format (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
    '''
    if len(boxes) == 0: #if there is no detection, return boxes
        return boxes

    if isinstance(boxes, torch.Tensor):
        boxes = to_np(boxes)

    boxes = boxes.reshape(-1, 4) #just in case
    return np.concatenate([boxes[:, :2]-boxes[:, 2:]/2, boxes[:, :2]+boxes[:, 2:]/2], 1)
def draw_outline(o, lw):
    o.set_path_effects([
        patheffects.Stroke(linewidth=lw, foreground='black'),
        patheffects.Normal()
    ])
def draw_rect(ax, b, color='white'):
    patch = ax.add_patch(patches.Rectangle(b[:2], *b[2:], fill=False, edgecolor=color, lw=2))
    draw_outline(patch, 4)
def draw_text(ax, xy, text, sz=14, color='white'):
    text = ax.text(*xy, text, verticalalignment='top', color=color, fontsize=sz, weight='bold')
    draw_outline(text, 1)
def visualize_original_bboxes(data, num_images, figsize=(20, 20)):
    i = 0
    plot_size = np.sqrt(num_images)
    if not float.is_integer(plot_size):
        plot_size +=1
    plot_size = int(plot_size)
    fig, axs = plt.subplots(ncols=plot_size, nrows=plot_size, figsize=figsize)
    for x, y in data.trn_dl:
        for (im, bboxes) in zip(x, y):
            ax = axs.flat[i]
            ax.axis('off')
            im = data.denorm(im)
            im = im.clip(0, 1)

            bboxes = bboxes.view(-1, 5)
            for bbox in bboxes: #(center_x, center_y, width, height, class)
                bbox = np.concatenate([bbox[:2]-bbox[2:4]/2, bbox[2:4]]) #(top_left_x, top_left_y, width, height)
                draw_rect(ax, bbox[:4])

            ax.imshow(im)
            i+= 1
            if i == num_images:
                for j in range(i, plot_size*plot_size):
                    fig.delaxes(axs.flat[j])
                return
def get_categories(category_file):
    with open(category_file) as f:
        cats = f.read().splitlines() #cats = categories = classes
    return cats
def topleftwh_to_centerwh(boxes):
    return np.array([boxes[0]+boxes[2]/2, boxes[1]+boxes[3]/2, boxes[0], boxes[1]])

def topleft_to_centwerwh(boxes):
    w = boxes[2]-boxes[0]+1
    h = boxes[3]-boxes[1]+1
    return np.array([boxes[0]-1+math.ceil(w/2), boxes[1]-1+math.ceil(h/2),w, h])
def close_camera(cap):
    cap.release()
    cv2.destroyAllWindows()
    # I needed the loop below to properly close the window when running on jupyter notebooks
    # see https://stackoverflow.com/questions/6116564/destroywindow-does-not-close-window-on-mac-using-python-and-opencv
    for i in range(1, 5):
        cv2.waitKey(1)





