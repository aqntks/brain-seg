import matplotlib.pyplot as plt


def pre_visualize(val_ds):
    # pick one image from DecathlonDataset to visualize and check the 4 channels
    print(f"image shape: {val_ds[2]['image'].shape}")
    plt.figure("image", (24, 6))
    for i in range(4):
        plt.subplot(1, 4, i + 1)
        plt.title(f"image channel {i}")
        plt.imshow(val_ds[2]["image"][i, :, :, 60].detach().cpu(), cmap="gray")
    plt.show()
    # also visualize the 3 channels label corresponding to this image
    print(f"label shape: {val_ds[2]['label'].shape}")
    plt.figure("label", (18, 6))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(f"label channel {i}")
        plt.imshow(val_ds[2]["label"][i, :, :, 60].detach().cpu())
    plt.show()


def train_graph(epoch_loss_values, val_interval, metric_values, metric_values_tc, metric_values_wt, metric_values_et):
    # Plot the loss and metric ##########################################################################
    plt.figure("train", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Epoch Average Loss")
    x = [i + 1 for i in range(len(epoch_loss_values))]
    y = epoch_loss_values
    plt.xlabel("epoch")
    plt.plot(x, y, color="red")
    plt.subplot(1, 2, 2)
    plt.title("Val Mean Dice")
    x = [val_interval * (i + 1) for i in range(len(metric_values))]
    y = metric_values
    plt.xlabel("epoch")
    plt.plot(x, y, color="green")
    plt.show()

    plt.figure("train", (18, 6))
    plt.subplot(1, 3, 1)
    plt.title("Val Mean Dice TC")
    x = [val_interval * (i + 1) for i in range(len(metric_values_tc))]
    y = metric_values_tc
    plt.xlabel("epoch")
    plt.plot(x, y, color="blue")
    plt.subplot(1, 3, 2)
    plt.title("Val Mean Dice WT")
    x = [val_interval * (i + 1) for i in range(len(metric_values_wt))]
    y = metric_values_wt
    plt.xlabel("epoch")
    plt.plot(x, y, color="brown")
    plt.subplot(1, 3, 3)
    plt.title("Val Mean Dice ET")
    x = [val_interval * (i + 1) for i in range(len(metric_values_et))]
    y = metric_values_et
    plt.xlabel("epoch")
    plt.plot(x, y, color="purple")
    plt.show()


def temp(model, device, val_ds, inference, post_trans, dice_metric, dice_metric_batch, metric_batch, metric):
    # Check best model output with the input image and label ##################################################
    model.load_state_dict(
        torch.load(os.path.join('weights/', "SegResNet_sample.pth"))
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