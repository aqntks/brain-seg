import os
import shutil
import tempfile
import time
import matplotlib.pyplot as plt
import numpy as np
from monai.apps import DecathlonDataset
from monai.config import print_config
from monai.data import DataLoader, decollate_batch
from monai.handlers.utils import from_engine
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import SegResNet
from monai.transforms import (
    Activations,
    Activationsd,
    AsDiscrete,
    AsDiscreted,
    Compose,
    Invertd,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
    EnsureChannelFirstd,
    EnsureTyped,
    EnsureType,
)
from monai.utils import set_determinism

import torch

from data_loader import ConvertToMultiChannelBasedOnBratsClassesd


def segmentation(model, device, val_ds, inference, post_trans, dice_metric, dice_metric_batch, metric_batch, metric):
    # Check best model output with the input image and label ##################################################
    model.load_state_dict(
        torch.load(os.path.join('weights/', "best_metric_model.pth"))
    )
    model.eval()
    with torch.no_grad():
        # select one image to evaluate and visualize the model output
        val_input = val_ds[6]["image"].unsqueeze(0).to(device)
        roi_size = (128, 128, 64)
        sw_batch_size = 4
        val_output = inference(val_input)
        val_output = post_trans(val_output[0])
        plt.figure("image", (24, 6))
        for i in range(4):
            plt.subplot(1, 4, i + 1)
            plt.title(f"image channel {i}")
            plt.imshow(val_ds[6]["image"][i, :, :, 70].detach().cpu(), cmap="gray")
        plt.show()
        # visualize the 3 channels label corresponding to this image
        plt.figure("label", (18, 6))
        for i in range(3):
            plt.subplot(1, 3, i + 1)
            plt.title(f"label channel {i}")
            plt.imshow(val_ds[6]["label"][i, :, :, 70].detach().cpu())
        plt.show()
        # visualize the 3 channels model output corresponding to this image
        plt.figure("output", (18, 6))
        for i in range(3):
            plt.subplot(1, 3, i + 1)
            plt.title(f"output channel {i}")
            plt.imshow(val_output[i, :, :, 70].detach().cpu())
        plt.show()


    # Evaluation on original image spacings ####################################################
    val_org_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image"]),
            ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
            Orientationd(keys=["image"], axcodes="RAS"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            EnsureTyped(keys=["image", "label"]),
        ]
    )

    val_org_ds = DecathlonDataset(
        root_dir='data/',
        task="Task01_BrainTumour",
        transform=val_org_transforms,
        section="validation",
        download=False,
        num_workers=4,
        cache_num=0,
    )
    val_org_loader = DataLoader(val_org_ds, batch_size=1, shuffle=False, num_workers=4)

    post_transforms = Compose([
        EnsureTyped(keys="pred"),
        Invertd(
            keys="pred",
            transform=val_org_transforms,
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
        ),
        Activationsd(keys="pred", sigmoid=True),
        AsDiscreted(keys="pred", threshold_values=True),
    ])

    # #############################################################################################
    model.load_state_dict(torch.load(
        os.path.join('weights/', "best_metric_model.pth")))
    model.eval()

    with torch.no_grad():
        for val_data in val_org_loader:
            val_inputs = val_data["image"].to(device)
            val_data["pred"] = inference(val_inputs)
            val_data = [post_transforms(i) for i in decollate_batch(val_data)]
            val_outputs, val_labels = from_engine(["pred", "label"])(val_data)
            dice_metric(y_pred=val_outputs, y=val_labels)
            dice_metric_batch(y_pred=val_outputs, y=val_labels)

        metric_org = dice_metric.aggregate().item()
        metric_batch_org = dice_metric_batch.aggregate()

        dice_metric.reset()
        dice_metric_batch.reset()

    metric_tc, metric_wt, metric_et = metric_batch[0].item(), metric_batch[1].item(), metric_batch[2].item()

    print("Metric on original image spacing: ", metric)
    print(f"metric_tc: {metric_tc:.4f}")
    print(f"metric_wt: {metric_wt:.4f}")
    print(f"metric_et: {metric_et:.4f}")


if __name__ == '__main__':
    segmentation()