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


def visual2(data):
    # 임의의 레이어 번호를 지정합니다.
    maxval = 154  # including depth
    i = np.random.randint(0, maxval)

    # 확인할 채널을 지정합니다.
    channel = 0

    print(f'Plotting layer Layer {i}, Channel {channel} of Image')
    plt.imshow(data[:, :, i, channel], cmap='gray')
    plt.axis('off')
    plt.show()


def explore_3d_image(layer):
    plt.figure(figsize=(10,5))
    channel = 1
    plt.imshow(image_data[:,:,layer,channel],cmap='gray')
    plt.title('Explore Layers of MRI', family = 'Arial', fontsize=18)
    plt.axis('off')
    plt.show()
    return layer


def load_case(image_nifty_file, label_nifty_file):
    image = np.array(nib.load(image_nifty_file).get_fdata())
    label = np.array(nib.load(label_nifty_file).get_fdata())

    return image, label


def brain(DATA_DIR):
    image, label = load_case(DATA_DIR + "imagesTr/BRATS_003.nii.gz", DATA_DIR + "labelsTr/BRATS_003.nii.gz")

    img = nib.load(DATA_DIR + "imagesTr/BRATS_003.nii.gz")
    img_data = img.get_fdata()

    print(img_data.shape)
    # visual2(img_data)


if __name__ == '__main__':
    DATA_DIR = '../BrainTumour/'
    brain(DATA_DIR)


