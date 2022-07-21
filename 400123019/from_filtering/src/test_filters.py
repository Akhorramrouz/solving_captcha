from calendar import c
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from zmq import get_library_dirs

# Read images from directory and convert them in grayscale, then save them in a list
def convert_images_to_gray():
    gray_images = []
    file_names = []
    for file_name in os.listdir("data/images_to_test_filters/"):
        if file_name.endswith('.jpg'):
            original_image = cv2.imread(f'data/images_to_test_filters/{file_name}',0)
            gray_images.append(original_image)
            file_names.append(file_name)
    return gray_images, file_names

# Plot images in a group and save them according to fig_name
def group_plot_images(images, file_names, fig_name, n_rows=5, n_cols=8):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(250, 250))
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, images)):
        ax.imshow(img,cmap = 'gray')
        ax.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_title(file_names[i],fontsize=25)
        ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.close('all')

# Test canny filter for differenet tresholds
def test_canny_filter(gray_images, file_names):
    tresholds = [[0, 255], [128, 255], [0, 128], [64, 200], [100, 200] ]
    for min_value, max_value in tresholds :
        canny_images = []
        for image in gray_images :
            canny_images.append(cv2.Canny(image, min_value, max_value))
        group_plot_images(canny_images, file_names, f"data/images_to_test_filters/results/canny/canny-[{min_value}, {max_value}]")


gray, names = convert_images_to_gray()

test_canny_filter(gray,names)