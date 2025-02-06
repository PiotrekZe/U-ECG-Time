import torch
import torch.nn as nn


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        out_length,
        kernel_size,
        dilation=1,
        stride=1,
        padding="same",
    ):
        super(DecoderBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Upsample(size=out_length),
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.BatchNorm1d(out_channels),
            nn.Dropout(p=0.1),
        )

    def forward(self, x):
        x = self.block(x)
        return x
