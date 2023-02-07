import math

import cv2
import matplotlib.pyplot as plt
import numpy as np

from skimage import io


def imshow(image, cmap=None, figsize=None):
    height, width = image.shape[:2]
    if figsize is None:
        if height < width:
            plt.figure(figsize=(15, (height / width) * 15))
        else:
            plt.figure(figsize=((width / height) * 15, 15))
    else:
        plt.figure(figsize=figsize)

    if cmap is None:
        plt.imshow(image, cmap='gray', vmin=np.min(image), vmax=np.max(image))  # https://stackoverflow.com/a/3823822
    else:
        plt.imshow(image, cmap=cmap)

    plt.axis('off')
    plt.show()


def standardize(image):
    min_value = image.min()
    return (image - min_value).astype(float) / (image.max() - min_value)


def overlay(image1, image2):
    std_image1, std_image2 = standardize(image1), standardize(image2)
    return 0.5 * std_image1 + 0.5 * std_image2


# TODO: inspect all usages
def uint8_255(image):
    return np.uint8(255 * standardize(image))


def odd_round(value):
    floor = math.floor(value)
    if floor % 2 == 0:
        return floor + 1
    else:
        return floor


def make_odd(value):
    if value % 2 == 0:
        return value + 1
    else:
        return value


def imsave(image, file):
    io.imsave(file, image)


def visualize_keypoints(image, keypoints):
    return cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


def visualize_matches(image1, keypoints1, image2, keypoints2, matches):
    return cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches, None,
                           flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)


def plot_histograms(images):
    fig, ax = plt.subplots(1, len(images), sharey=True)
    for i, image in enumerate(images):
        ax[i].hist(image.flatten(), 256, (0, 256), density=True)
    plt.show()
