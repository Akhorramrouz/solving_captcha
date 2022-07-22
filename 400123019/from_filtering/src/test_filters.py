import os
from calendar import c
from email.iterators import typed_subpart_iterator

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from zmq import get_library_dirs


class Test_filters:

    def __init__(self):
        os.mkdir("src/data/results_of_filters")
        gray_images = []
        file_names = []
        original_images = []
        for file_name in os.listdir("src/data/images_to_test_filters/"):
            if file_name.endswith('.jpg'):
                original_image = cv2.imread(f'src/data/images_to_test_filters/{file_name}',0)
                gray_images.append(original_image)
                file_names.append(file_name)
        self.gray_images = gray_images
        self.file_names = file_names
    
    # Save images in a gird format for each filter
    def save_grid_images(self, images,title):
        fig = plt.figure(figsize=(100, 100))
        gs = GridSpec(8,5)
        for i, img in enumerate(images):
            ax = fig.add_subplot(gs[i])
            ax.imshow(img, cmap='gray')
            ax.axis('off')
        plt.suptitle(title, fontsize=200)
        
        plt.savefig(f"src/data/results_of_filters/{title}"+ '.jpg')
        plt.close('all')

    # Test canny filter for differenet tresholds
    def test_canny_filter(self, images):
        tresholds = [[0, 255], [128, 255], [0, 128], [64, 200], [100, 200] ]
        for min_value, max_value in tresholds :
            canny_images = []
            for image in images:
                canny_images.append(cv2.Canny(image, min_value, max_value))
            self.save_grid_images(canny_images,f"Canny: treshold = [{min_value}, {max_value}]")

    def test_blur_filter(self, images):
        kernel_sizes = [3, 5, 7]
        for kernel_size in kernel_sizes :
            blur_images = []
            for image in images:
                blur_images.append(cv2.blur(image, (kernel_size, kernel_size)))
            self.save_grid_images(blur_images,f"Blur: kernel_size = {kernel_size}")

    def test_gaussian_filter(self, images):
        kernel_sizes = [3, 5, 7]
        for kernel_size in kernel_sizes :
            gaussian_images = []
            for image in images:
                gaussian_images.append(cv2.GaussianBlur(image, (kernel_size, kernel_size), 0))
            self.save_grid_images(gaussian_images,f"Gaussian: kernel_size = {kernel_size}")
    
    def test_sobel_filter(self, images):
        kernel_sizes = [3, 5, 7]
        for kernel_size in kernel_sizes :
            sobel_images = []
            for image in images:
                sobel_images.append(cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kernel_size))
            self.save_grid_images(sobel_images,f"Sobel: kernel_size = {kernel_size}")

    def test_errosion_filter(self, images):
        kernel_sizes = [3, 5, 7]
        for kernel_size in kernel_sizes :
            errosion_images = []
            for image in images:
                errosion_images.append(cv2.erode(image, np.ones((kernel_size, kernel_size), np.uint8), iterations=1))
            self.save_grid_images(errosion_images,f"Erosion: kernel_size = {kernel_size}")

    def test_dilation_filter(self, images):
        kernel_sizes = [3, 5, 7]
        for kernel_size in kernel_sizes :
            dilation_images = []
            for image in images:
                dilation_images.append(cv2.dilate(image, np.ones((kernel_size, kernel_size), np.uint8), iterations=1))
            self.save_grid_images(dilation_images,f"Dilation: kernel_size = {kernel_size}")

    def test_opening_filter(self, images):
        kernel_sizes = [3, 5, 7]
        for kernel_size in kernel_sizes :
            opening_images = []
            for image in images:
                opening_images.append(cv2.morphologyEx(image, cv2.MORPH_OPEN, np.ones((kernel_size, kernel_size), np.uint8)))
            self.save_grid_images(opening_images,f"Opening: kernel_size = {kernel_size}")
    
    def test_closing_filter(self, images):
        kernel_sizes = [3, 5, 7]
        for kernel_size in kernel_sizes :
            closing_images = []
            for image in images:
                closing_images.append(cv2.morphologyEx(image, cv2.MORPH_CLOSE, np.ones((kernel_size, kernel_size), np.uint8)))
            self.save_grid_images(closing_images,f"Closing: kernel_size = {kernel_size}")
    
    def test_gradient_filter(self, images):
        kernel_sizes = [3, 5, 7]
        for kernel_size in kernel_sizes :
            gradient_images = []
            for image in images:
                gradient_images.append(cv2.morphologyEx(image, cv2.MORPH_GRADIENT, np.ones((kernel_size, kernel_size), np.uint8)))
            self.save_grid_images(gradient_images,f"Gradient: kernel_size = {kernel_size}")

    def test_blackhat_filter(self, images):
        kernel_sizes = [3, 5, 7]
        for kernel_size in kernel_sizes :
            blackhat_images = []
            for image in images:
                blackhat_images.append(cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, np.ones((kernel_size, kernel_size), np.uint8)))
            self.save_grid_images(blackhat_images,f"Blackhat: kernel_size = {kernel_size}")
    
    def test_tophat_filter(self, images):
        kernel_sizes = [3, 5, 7]
        for kernel_size in kernel_sizes :
            tophat_images = []
            for image in images:
                tophat_images.append(cv2.morphologyEx(image, cv2.MORPH_TOPHAT, np.ones((kernel_size, kernel_size), np.uint8)))
            self.save_grid_images(tophat_images,f"Tophat: kernel_size = {kernel_size}")
    
    def test_median_filter(self, images):
        kernel_sizes = [3, 5, 7]
        for kernel_size in kernel_sizes :
            median_images = []
            for image in images:
                median_images.append(cv2.medianBlur(image, kernel_size))
            self.save_grid_images(median_images,f"Median: kernel_size = {kernel_size}")
    
    def test_bilateral_filter(self, images):
        kernel_sizes = [3, 5, 7]
        for kernel_size in kernel_sizes :
            bilateral_images = []
            for image in images:
                bilateral_images.append(cv2.bilateralFilter(image, kernel_size, 75, 75))
            self.save_grid_images(bilateral_images,f"Bilateral: kernel_size = {kernel_size}")
    
    def test_histogram_equalization(self, images):
        histogram_equalized_images = []
        for image in images:
            histogram_equalized_images.append(cv2.equalizeHist(image))
        self.save_grid_images(histogram_equalized_images,f"Histogram Equalization")


if __name__ == "__main__":
    tester = Test_filters()
    tester.test_bilateral_filter(tester.gray_images)
    tester.test_blackhat_filter(tester.gray_images)
    tester.test_blur_filter(tester.gray_images)
    tester.test_canny_filter(tester.gray_images)
    tester.test_dilation_filter(tester.gray_images)
    tester.test_errosion_filter(tester.gray_images)
    tester.test_gradient_filter(tester.gray_images)
    tester.test_gaussian_filter(tester.gray_images)
    tester.test_median_filter(tester.gray_images)
    tester.test_opening_filter(tester.gray_images)
    tester.test_closing_filter(tester.gray_images)
    tester.test_tophat_filter(tester.gray_images)
    tester.test_histogram_equalization(tester.gray_images)
    tester.test_sobel_filter(tester.gray_images)
    