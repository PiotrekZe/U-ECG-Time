import torch
import torch.nn as nn


class EncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation,
        pooling_size,
        stride=1,
        padding="same",
    ):
        super(EncoderBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm1d(out_channels),
            nn.Conv1d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm1d(out_channels),
            nn.MaxPool1d(kernel_size=pooling_size),
            nn.Dropout(p=0.1),
        )

    def forward(self, x):
        x = self.block(x)
        return x
