import matplotlib.pyplot as plt


def visualize(val_ds):
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