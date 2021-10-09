import numpy as np
import pandas as pd
import nibabel as nib
import os
import cv2

from util import *
import matplotlib.pyplot as plt


# 구조적 시각화
def visual(data):
    fig, ax = plt.subplots(1, 6, figsize=[18, 3])
    n = 0
    slice = 0
    for _ in range(6):
        ax[n].imshow(data[:, :, slice], 'gray')
        ax[n].set_xticks([])
        ax[n].set_yticks([])
        ax[n].set_title('Slice number: {}'.format(slice), color='r')
        n += 1
        slice += 10

    fig.subplots_adjust(wspace=0, hspace=0)
    plt.show()


def load_case(image_nifty_file, label_nifty_file):
    image = np.array(nib.load(image_nifty_file).get_fdata())
    label = np.array(nib.load(label_nifty_file).get_fdata())

    return image, label


def brain(DATA_DIR):
    image, label = load_case(DATA_DIR + "imagesTr/BRATS_003.nii.gz", DATA_DIR + "labelsTr/BRATS_003.nii.gz")

    img = nib.load(DATA_DIR + "imagesTr/BRATS_003.nii.gz")
    img_data = img.get_fdata()

    print(img_data.shape)
    visual(img_data)

# slice_0 = image[26, :, :, 0]
    # slice_1 = image[:, 30, :, 0]
    # slice_2 = image[:, :, 16, 0]
    # show_slices([slice_0, slice_1, slice_2])
    # plt.suptitle("Center slices for EPI image")


if __name__ == '__main__':
    DATA_DIR = '../BrainTumour/'
    brain(DATA_DIR)


