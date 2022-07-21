from calendar import c
from email.iterators import typed_subpart_iterator
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
    plt.savefig(f"{fig_name}.jpg")
    plt.close('all')

# function to plot a group of images in a list, with a title and a name for the figure
def save_images(images, file_names, fig_name,filter_name):
    os.mkdir(f"data/images_to_test_filters/results/{filter_name}/")
    for i,image in enumerate(images):
        plt.imshow(image,cmap = 'gray')
        plt.axis('off')
        plt.title(file_names[i],fontsize=25)
        plt.savefig(f"data/images_to_test_filters/results/{filter_name}/{fig_name}-{i}.jpg")
        plt.close('all')
    
# Test canny filter for differenet tresholds
def test_canny_filter(gray_images, file_names):
    tresholds = [[0, 255], [128, 255], [0, 128], [64, 200], [100, 200] ]
    for min_value, max_value in tresholds :
        canny_images = []
        for image in gray_images :
            canny_images.append(cv2.Canny(image, min_value, max_value))
        save_images(canny_images, file_names, f"canny",f"canny-(treshold = [{min_value}, {max_value}])")

# test_canny_filter(gray,names)

gray, names = convert_images_to_gray()

test_canny_filter(gray,names)

# Test Blur filter with different kernel sizes
def test_blur_filters(gray_images, file_names):
    kernel_sizes = [3, 5, 7]
    for kernel_size in kernel_sizes :
        blur_images = []
        for image in gray_images :
            blur_images.append(cv2.blur(image, (kernel_size, kernel_size)))
        save_images(blur_images, file_names, f"blur",f"blur-(kernel = {kernel_size})")
    
test_blur_filters(gray,names)

# Test Gaussian filter with different kernel sizes
def test_gaussian_filters(gray_images, file_names):
    kernel_sizes = [3, 5, 7]
    for kernel_size in kernel_sizes :
        gaussian_images = []
        for image in gray_images :
            gaussian_images.append(cv2.GaussianBlur(image, (kernel_size, kernel_size), 0))
        save_images(gaussian_images, file_names, f"gaussian",f"gaussian-(kernel = {kernel_size})")

test_gaussian_filters(gray,names)

# Test sobel filter with different kernel sizes
def test_sobel_filters(gray_images, file_names):
    kernel_sizes = [3, 5, 7]
    for kernel_size in kernel_sizes :
        sobel_images = []
        for image in gray_images :
            sobel_images.append(cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kernel_size))
        save_images(sobel_images, file_names, f"sobel",f"sobel-(kernel = {kernel_size})")

test_sobel_filters(gray,names)

# Test Errosion filter with different kernel sizes
def test_errosion_filters(gray_images, file_names):
    kernel_sizes = [3, 5, 7]
    for kernel_size in kernel_sizes :
        errosion_images = []
        for image in gray_images :
            errosion_images.append(cv2.erode(image, np.ones((kernel_size, kernel_size), np.uint8), iterations=1))
        save_images(errosion_images, file_names, f"errosion",f"errosion-(kernel = {kernel_size})")  

test_errosion_filters(gray, names)

# Test Dilation filter with different kernel sizes
def test_dilation_filters(gray_images, file_names):
    kernel_sizes = [3, 5, 7]
    for kernel_size in kernel_sizes :
        dilation_images = []
        for image in gray_images :
            dilation_images.append(cv2.dilate(image, np.ones((kernel_size, kernel_size), np.uint8), iterations=1))
        save_images(dilation_images, file_names, f"dilation",f"dilation-(kernel = {kernel_size})")

test_dilation_filters(gray, names)

# Test opening filter with different kernel sizes
def test_opening_filters(gray_images, file_names):
    kernel_sizes = [3, 5, 7]
    for kernel_size in kernel_sizes :
        opening_images = []
        for image in gray_images :
            opening_images.append(cv2.morphologyEx(image, cv2.MORPH_OPEN, np.ones((kernel_size, kernel_size), np.uint8)))
        save_images(opening_images, file_names, f"opening",f"opening-(kernel = {kernel_size})")

test_opening_filters(gray, names)

# Test closing filter with different kernel sizes
def test_closing_filters(gray_images, file_names):
    kernel_sizes = [3, 5, 7]
    for kernel_size in kernel_sizes :
        closing_images = []
        for image in gray_images :
            closing_images.append(cv2.morphologyEx(image, cv2.MORPH_CLOSE, np.ones((kernel_size, kernel_size), np.uint8)))
        save_images(closing_images, file_names, f"closing",f"closing-(kernel = {kernel_size})")

test_closing_filters(gray, names)

# Test gradient filter with different kernel sizes
def test_gradient_filters(gray_images, file_names):
    kernel_sizes = [3, 5, 7]
    for kernel_size in kernel_sizes :
        gradient_images = []
        for image in gray_images :
            gradient_images.append(cv2.morphologyEx(image, cv2.MORPH_GRADIENT, np.ones((kernel_size, kernel_size), np.uint8)))
        save_images(gradient_images, file_names, f"gradient",f"gradient-(kernel = {kernel_size})")
test_gradient_filters(gray, names)

# Test top hat filter with different kernel sizes
def test_top_hat_filters(gray_images, file_names):
    kernel_sizes = [3, 5, 7]
    for kernel_size in kernel_sizes :
        top_hat_images = []
        for image in gray_images :
            top_hat_images.append(cv2.morphologyEx(image, cv2.MORPH_TOPHAT, np.ones((kernel_size, kernel_size), np.uint8)))
        save_images(top_hat_images, file_names, f"top_hat",f"top_hat-(kernel = {kernel_size})")

test_top_hat_filters(gray, names)

# Test black hat filter with different kernel sizes
def test_black_hat_filters(gray_images, file_names):
    kernel_sizes = [3, 5, 7]
    for kernel_size in kernel_sizes :
        black_hat_images = []
        for image in gray_images :
            black_hat_images.append(cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, np.ones((kernel_size, kernel_size), np.uint8)))
        save_images(black_hat_images, file_names, f"black_hat",f"black_hat-(kernel = {kernel_size})")

test_black_hat_filters(gray, names)

# Test median filter with different kernel sizes
def test_median_filters(gray_images, file_names):
    kernel_sizes = [3, 5, 7]
    for kernel_size in kernel_sizes :
        median_images = []
        for image in gray_images :
            median_images.append(cv2.medianBlur(image, kernel_size))
        save_images(median_images, file_names, f"median",f"median-(kernel = {kernel_size})")

test_median_filters(gray, names)

# Test bilateral filter with different kernel sizes
def test_bilateral_filters(gray_images, file_names):
    kernel_sizes = [3, 5, 7]
    for kernel_size in kernel_sizes :
        bilaterial_images = []
        for image in gray_images :
            bilaterial_images.append(cv2.bilateralFilter(image, kernel_size, 75, 75))
        save_images(bilaterial_images, file_names, f"bilateral",f"bilateral-(kernel = {kernel_size})")

test_bilateral_filters(gray, names)

# Test histogram equalization with different kernel sizes
def test_histogram_equalization_filters(gray_images, file_names):
    kernel_sizes = []
    for kernel_size in kernel_sizes :
        hist_eq_images = []
        for image in gray_images :
            hist_eq_images.append(cv2.equalizeHist(image))
        save_images(hist_eq_images, file_names, f"hist_eq",f"hist_eq-(kernel = {kernel_size})")

test_histogram_equalization_filters(gray, names)

# Test CLAHE with different kernel sizes
def test_CLAHE_filters(gray_images, file_names):
    kernel_sizes = [3, 5, 7]
    for kernel_size in kernel_sizes :
        clahe_images = []
        for image in gray_images :
            clahe_images.append(cv2.createCLAHE(clipLimit=kernel_size, tileGridSize=(8,8)).apply(image))
        save_images(clahe_images, file_names, f"clahe",f"clahe-(kernel = {kernel_size})")
test_CLAHE_filters(gray, names)
