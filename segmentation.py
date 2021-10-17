import os
import torch

import matplotlib.pyplot as plt
from monai.inferers import sliding_window_inference
from monai.networks.nets import SegResNet, Unet
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    EnsureType,
)

from utils import cv2gif, show_brain_mri
from data_loader import brats_brain_val


def segmentation():
    # Check best model output with the input image and label ##################################################
    VAL_AMP = True

    val_ds, val_loader = brats_brain_val('data/')

    device = torch.device("cpu")
    # model = SegResNet(
    #     blocks_down=[1, 2, 2, 4],
    #     blocks_up=[1, 1, 1],
    #     init_filters=16,
    #     in_channels=4,
    #     out_channels=3,
    #     dropout_prob=0.2,
    # ).to(device)

    model = Unet(
        spatial_dims=3,
        in_channels=4,
        out_channels=3,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
    ).to(device)

    model.load_state_dict(
        torch.load(os.path.join('weights/', "UNet_sample.pth"), map_location=device)
    )
    model.eval()

    post_trans = Compose(
        [EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold_values=True)]
    )

    def inference(input):

        def _compute(input):
            return sliding_window_inference(
                inputs=input,
                roi_size=(240, 240, 160),
                sw_batch_size=1,
                predictor=model,
                overlap=0.5,
            )

        if VAL_AMP:
            with torch.cuda.amp.autocast():
                return _compute(input)
        else:
            return _compute(input)

    with torch.no_grad():
        # select one image to evaluate and visualize the model output
        val_input = val_ds[6]["image"].unsqueeze(0).to(device)

        roi_size = (128, 128, 64)
        sw_batch_size = 4
        val_output = inference(val_input)
        val_output = post_trans(val_output[0])

        # show_brain_mri(val_ds[6], val_output, index=70)
        cv2gif(val_ds[6]["image"], val_ds[6]["label"], val_output)


if __name__ == '__main__':
    segmentation()