import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv2
import os
import re

from skimage import io
from PIL import Image

height, width = 256, 256


def labels_encoding(retina: np.ndarray, pigment: np.ndarray):
    new_labels = np.zeros((retina.shape[0], 256, 256, 3))
    for i in range(len(retina)):
        for j in range(256):
            for k in range(256):
                if retina[i][j][k] == 0.:
                    new_labels[i][j][k][0] = 1
                else:
                    if pigment[i][j][k] == 0:
                        new_labels[i][j][k][1] = 1
                    else:
                        new_labels[i][j][k][2] = 1
    return new_labels


def segmentation_result(image, area: str = 'all'):
    dic = {'retina': 1, 'pigment': 2}
    output = np.argmax(image, axis=2)

    final_image = np.zeros((256, 256, 3), dtype=np.uint8)

    if area == 'pigment' or area == 'retina':
        final_image[np.where(output == dic['pigment'])] = [255, 255, 255]
        if area == 'retina':
            final_image[np.where(output == dic[area])] = [255, 255, 255]
    else:
        final_image[np.where(output == dic['retina'])] = [255, 255, 255]
        final_image[np.where(output == dic['pigment'])] = [255, 0, 0]

    return final_image


def segmentation_result_2(image):
    output = np.argmax(image, axis=2)
    final_image = np.zeros((256, 256, 3), dtype=np.uint8)
    final_image[np.where(output == 1)] = [255, 255, 255]
    return final_image


def resize_file(src: str, dest: str):
    image = Image.open(src)
    image = image.crop((0, 200, 640, 840))
    image = image.resize((height, width))
    plt.imshow(image, cmap='gray')
    image.save(dest)


def preprocessing_files(src: str, dest: str):
    files = [src + n for n in os.listdir(src)]
    # files = sorted(files, key=lambda file: int(re.findall('\d+', file.split('_')[-1])[0]))
    files = sorted(files, key=lambda file: int((file.split('/')[-1]).split('.')[0]))
    for i, f in enumerate(files):
        resize_file(f, dest + str(i + 1) + '.png')


def filter_images(src: str):
    files = [src + str(i) + '.png' for i in range(1, 86)]
    for f in files:
        processed_image = cv2.medianBlur(cv2.imread(f), 5)
        cv2.imwrite(f, processed_image)


def resize_image(src: str):
    image = io.imread(src, as_gray=True)
    image = Image.fromarray(image)
    image = image.crop((0, 200, 640, 840))
    image = image.resize((height, width))
    image = np.asarray(image)
    image = image.reshape((1, height, width, 1))
    return image


def preprocessing_images(src: str):
    files = [src + n for n in os.listdir(src)]
    return [resize_image(f) for f in sorted(files, key=lambda file: int(re.findall('\d+', file.split('_')[-1])[0]))]
