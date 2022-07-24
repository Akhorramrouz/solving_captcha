import os
import cv2
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Read original image
gray_images = []
file_names = []
for file_name in os.listdir("src/data/images_to_test_filters/"):
    if file_name.endswith('.jpg'):
        original_image = cv2.imread(f'src/data/images_to_test_filters/{file_name}',0)
        gray_images.append(original_image)
        file_names.append(file_name)

# Apply histogram equalization and blurring to the images
histeq_blur_images = []
for image in gray_images:
    histeq_blur_images.append(cv2.blur(cv2.equalizeHist(image), (3, 3)))

# save images in a gird format
fig = plt.figure(figsize=(100, 100))
gs = GridSpec(8,5)
for i, img in enumerate(histeq_blur_images):
    ax = fig.add_subplot(gs[i])
    ax.imshow(img, cmap='gray')
    ax.axis('off')
plt.suptitle('Hist.Eq & Blur', fontsize=200)
os.mkdir("src/data/results_of_testing_filters/final_filters")
# plt.show()
plt.savefig(f"src/data/results_of_testing_filters/final_filters/{'Hist.Eq & Blur'}"+ '.jpg')
plt.close('all')
