import torch
import torch.nn as nn

from collections import OrderedDict


class NNUnet3d(nn.Module):
    def __init__(self, in_channels = 1, out_channels = 1, init_features=32):
        super(NNUnet3d, self).__init__()

        self.encoder1 = nn.Sequential(
            OrderedDict(
                [
                    ('enc1conv1', nn.Conv3d(in_channels=in_channels, out_channels=32, kernel_size=(3, 3, 3), stride=(1, 1, 1))),
                    ('enc1norm1', nn.BatchNorm3d(num_features=32)),
                    ('enc1lrelu1', nn.LeakyReLU(inplace=True)),
                    ('enc1conv2', nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(3, 3, 3),
                                            stride=(1, 1, 1))),
                    ('enc1norm2', nn.BatchNorm3d(num_features=32)),
                    ('enc1lrelu2', nn.LeakyReLU(inplace=True)),
                ]
            )
        )
        self.pool1 = nn.MaxPool3d((2, 2, 2))
        self.encoder2 = nn.Sequential(
            OrderedDict(
                [
                    ('enc2conv1', nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3, 3, 3),
                                            stride=(2, 2, 2))),
                    ('enc2norm1', nn.BatchNorm3d(num_features=64)),
                    ('enc2lrelu1', nn.LeakyReLU(inplace=True)),
                    ('enc2conv2', nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3, 3, 3),
                                            stride=(1, 1, 1))),
                    ('enc2norm2', nn.BatchNorm3d(num_features=64)),
                    ('enc2lrelu2', nn.LeakyReLU(inplace=True)),
                ]
            )
        )
        self.pool2 = nn.MaxPool3d((2, 2, 2))
        self.encoder3 = nn.Sequential(
            OrderedDict(
                [
                    ('enc3conv1', nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(3, 3, 3),
                                            stride=(2, 2, 2))),
                    ('enc3norm1', nn.BatchNorm3d(num_features=128)),
                    ('enc3lrelu1', nn.LeakyReLU(inplace=True)),
                    ('enc3conv2', nn.Conv3d(in_channels=128, out_channels=128, kernel_size=(3, 3, 3),
                                            stride=(1, 1, 1))),
                    ('enc3norm2', nn.BatchNorm3d(num_features=128)),
                    ('enc3lrelu2', nn.LeakyReLU(inplace=True)),
                ]
            )
        )
        self.pool2 = nn.MaxPool3d((2, 2, 2))
        self.encoder4 = nn.Sequential(
            OrderedDict(
                [
                    ('enc4conv1', nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(3, 3, 3),
                                            stride=(2, 2, 2))),
                    ('enc4norm1', nn.BatchNorm3d(num_features=256)),
                    ('enc4lrelu1', nn.LeakyReLU(inplace=True)),
                    ('enc4conv2', nn.Conv3d(in_channels=256, out_channels=256, kernel_size=(3, 3, 3),
                                            stride=(1, 1, 1))),
                    ('enc4norm2', nn.BatchNorm3d(num_features=256)),
                    ('enc4lrelu2', nn.LeakyReLU(inplace=True)),
                ]
            )
        )
        self.pool2 = nn.MaxPool3d((2, 2, 2))
        self.encoder5 = nn.Sequential(
            OrderedDict(
                [
                    ('enc5conv1', nn.Conv3d(in_channels=256, out_channels=320, kernel_size=(3, 3, 3),
                                            stride=(2, 2, 2))),
                    ('enc5norm1', nn.BatchNorm3d(num_features=320)),
                    ('enc5lrelu1', nn.LeakyReLU(inplace=True)),
                    ('enc5conv2', nn.Conv3d(in_channels=320, out_channels=320, kernel_size=(3, 3, 3),
                                            stride=(1, 1, 1))),
                    ('enc5norm2', nn.BatchNorm3d(num_features=320)),
                    ('enc5lrelu2', nn.LeakyReLU(inplace=True)),
                ]
            )
        )
        self.pool2 = nn.MaxPool3d((2, 2, 2))