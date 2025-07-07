import torch
import torch.nn as nn
import torch.nn.init as init


class MiniUNet(nn.Module):
    def __init__(self):
        super(MiniUNet, self).__init__()

        # Encoder
        self.encoder1 = self.conv_block(3, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)

        # Bottleneck
        self.bottleneck = self.conv_block(256, 512)

        # Decoder
        self.decoder1 = self.conv_block(512, 256)
        self.decoder2 = self.conv_block(256, 128)
        self.decoder3 = self.conv_block(128, 64)

        # Pooling and upsampling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Upsampling
        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        # Final layer
        self.final = nn.Sequential(
            nn.Conv2d(64, 104, kernel_size=1),
        )

        # ✅ Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights using He initialization for ReLU networks"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
                # He initialization (Kaiming) - best for ReLU networks
                init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def conv_block(self, in_channels, out_channels):
        """Using two conv2d layers increases the receptive field"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc3))

        # Decoder with skip connections
        dec1 = self.decoder1(torch.cat([self.upconv1(bottleneck), enc3], dim=1))
        dec2 = self.decoder2(torch.cat([self.upconv2(dec1), enc2], dim=1))
        dec3 = self.decoder3(torch.cat([self.upconv3(dec2), enc1], dim=1))

        return self.final(dec3)
