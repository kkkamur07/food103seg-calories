import torch
import torch.nn as nn
import torch.nn.functional as F

# import segmentation_models_pytorch as smp


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UnetPlus(nn.Module):
    def __init__(self, n_classes=103, filters=[64, 128, 256, 512, 512]):
        super().__init__()

        # Encoder
        self.conv0_0 = ConvBlock(3, filters[0])
        self.pool0 = nn.MaxPool2d(2)
        self.conv1_0 = ConvBlock(filters[0], filters[1])
        self.pool1 = nn.MaxPool2d(2)
        self.conv2_0 = ConvBlock(filters[1], filters[2])
        self.pool2 = nn.MaxPool2d(2)
        self.conv3_0 = ConvBlock(filters[2], filters[3])
        self.pool3 = nn.MaxPool2d(2)
        self.conv4_0 = ConvBlock(filters[3], filters[4])

        # Decoder
        self.conv0_1 = ConvBlock(filters[0] + filters[1], filters[0])
        self.conv1_1 = ConvBlock(filters[1] + filters[2], filters[1])
        self.conv2_1 = ConvBlock(filters[2] + filters[3], filters[2])
        self.conv3_1 = ConvBlock(filters[3] + filters[4], filters[3])

        self.conv0_2 = ConvBlock(filters[0] * 2 + filters[1], filters[0])
        self.conv1_2 = ConvBlock(filters[1] * 2 + filters[2], filters[1])
        self.conv2_2 = ConvBlock(filters[2] * 2 + filters[3], filters[2])

        self.conv0_3 = ConvBlock(filters[0] * 3 + filters[1], filters[0])
        self.conv1_3 = ConvBlock(filters[1] * 3 + filters[2], filters[1])

        self.conv0_4 = ConvBlock(filters[0] * 4 + filters[1], filters[0])

        self.final = nn.Conv2d(filters[0], n_classes, 1)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool0(x0_0))
        x2_0 = self.conv2_0(self.pool1(x1_0))
        x3_0 = self.conv3_0(self.pool2(x2_0))
        x4_0 = self.conv4_0(self.pool3(x3_0))

        x0_1 = self.conv0_1(
            torch.cat(
                [
                    x0_0,
                    F.interpolate(
                        x1_0, x0_0.size()[2:], mode="bilinear", align_corners=False
                    ),
                ],
                1,
            )
        )
        x1_1 = self.conv1_1(
            torch.cat(
                [
                    x1_0,
                    F.interpolate(
                        x2_0, x1_0.size()[2:], mode="bilinear", align_corners=False
                    ),
                ],
                1,
            )
        )
        x2_1 = self.conv2_1(
            torch.cat(
                [
                    x2_0,
                    F.interpolate(
                        x3_0, x2_0.size()[2:], mode="bilinear", align_corners=False
                    ),
                ],
                1,
            )
        )
        x3_1 = self.conv3_1(
            torch.cat(
                [
                    x3_0,
                    F.interpolate(
                        x4_0, x3_0.size()[2:], mode="bilinear", align_corners=False
                    ),
                ],
                1,
            )
        )

        # Skip connectionss
        x0_2 = self.conv0_2(
            torch.cat(
                [
                    x0_0,
                    x0_1,
                    F.interpolate(
                        x1_1, x0_0.size()[2:], mode="bilinear", align_corners=False
                    ),
                ],
                1,
            )
        )

        x1_2 = self.conv1_2(
            torch.cat(
                [
                    x1_0,
                    x1_1,
                    F.interpolate(
                        x2_1, x1_0.size()[2:], mode="bilinear", align_corners=False
                    ),
                ],
                1,
            )
        )

        x2_2 = self.conv2_2(
            torch.cat(
                [
                    x2_0,
                    x2_1,
                    F.interpolate(
                        x3_1, x2_0.size()[2:], mode="bilinear", align_corners=False
                    ),
                ],
                1,
            )
        )

        x0_3 = self.conv0_3(
            torch.cat(
                [
                    x0_0,
                    x0_1,
                    x0_2,
                    F.interpolate(
                        x1_2, x0_0.size()[2:], mode="bilinear", align_corners=False
                    ),
                ],
                1,
            )
        )

        x1_3 = self.conv1_3(
            torch.cat(
                [
                    x1_0,
                    x1_1,
                    x1_2,
                    F.interpolate(
                        x2_2, x1_0.size()[2:], mode="bilinear", align_corners=False
                    ),
                ],
                1,
            )
        )

        x0_4 = self.conv0_4(
            torch.cat(
                [
                    x0_0,
                    x0_1,
                    x0_2,
                    x0_3,
                    F.interpolate(
                        x1_3, x0_0.size()[2:], mode="bilinear", align_corners=False
                    ),
                ],
                1,
            )
        )

        out = self.final(x0_4)
        out = F.softmax(out, dim=1)
        return out
