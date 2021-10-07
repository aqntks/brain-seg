import numpy as np
import pandas as pd
import nibabel as nib

import cv2

from util import *
import matplotlib.pyplot as plt


def load_case(image_nifty_file, label_nifty_file):
    image = np.array(nib.load(image_nifty_file).get_fdata())
    label = np.array(nib.load(label_nifty_file).get_fdata())

    return image, label


def brain(DATA_DIR):
    image, label = load_case(DATA_DIR + "imagesTr/BRATS_003.nii.gz", DATA_DIR + "labelsTr/BRATS_003.nii.gz")

    cv2.imshow('333')
    cv2.waitKey(0)
# slice_0 = image[26, :, :, 0]
    # slice_1 = image[:, 30, :, 0]
    # slice_2 = image[:, :, 16, 0]
    # show_slices([slice_0, slice_1, slice_2])
    # plt.suptitle("Center slices for EPI image")


if __name__ == '__main__':
    DATA_DIR = '../BrainTumour/'
    brain(DATA_DIR)


