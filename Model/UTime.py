import torch
import torch.nn as nn
from DecoderBlock import DecoderBlock
from EncoderBlock import EncoderBlock


class UTime(nn.Module):
    def __init__(self, lengths, channels, num_classes):
        super(UTime, self).__init__()
        # to do: validate input

        self.enc1 = EncoderBlock(
            in_channels=2,
            out_channels=channels[0],
            kernel_size=5,
            dilation=9,
            pooling_size=2,
        )
        self.enc2 = EncoderBlock(
            in_channels=channels[0],
            out_channels=channels[1],
            kernel_size=5,
            dilation=9,
            pooling_size=2,
        )
        self.enc3 = EncoderBlock(
            in_channels=channels[1],
            out_channels=channels[2],
            kernel_size=5,
            dilation=9,
            pooling_size=2,
        )
        self.enc4 = EncoderBlock(
            in_channels=channels[2],
            out_channels=channels[3],
            kernel_size=5,
            dilation=9,
            pooling_size=2,
        )

        self.bottleneck = nn.Sequential(
            nn.Conv1d(
                in_channels=channels[3],
                out_channels=channels[3],
                kernel_size=5,
                padding="same",
            ),
            nn.BatchNorm1d(channels[3]),
            nn.Conv1d(
                in_channels=channels[3],
                out_channels=channels[3],
                kernel_size=5,
                padding="same",
            ),
            nn.BatchNorm1d(channels[3]),
            nn.Dropout(p=0.1),
        )

        self.dec1 = DecoderBlock(
            in_channels=channels[3],
            out_channels=channels[2],
            out_length=lengths[3],
            kernel_size=5,
        )
        self.dec2 = DecoderBlock(
            in_channels=channels[2] * 2,
            out_channels=channels[1],
            out_length=lengths[2],
            kernel_size=5,
        )
        self.dec3 = DecoderBlock(
            in_channels=channels[1] * 2,
            out_channels=channels[0],
            out_length=lengths[1],
            kernel_size=5,
        )
        self.dec4 = DecoderBlock(
            in_channels=channels[0] * 2,
            out_channels=num_classes,
            out_length=lengths[0],
            kernel_size=5,
        )

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)

        x_bottle = self.bottleneck(x4)

        x = self.dec1(x_bottle)
        x = torch.cat([x, x3], dim=1)
        x = self.dec2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.dec3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.dec4(x)

        return x
