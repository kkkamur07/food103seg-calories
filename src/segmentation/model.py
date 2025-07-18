import torch
import torch.nn as nn
import torch.nn.init as init


class MiniUNet(nn.Module):
    """
    Lightweight U-Net architecture for semantic segmentation.

    A simplified version of the U-Net architecture designed for food segmentation
    with 104 output classes. Features encoder-decoder structure with skip connections
    and proper weight initialization.

    Architecture:
        - Encoder: 3 conv blocks with max pooling (3→64→128→256 channels)
        - Bottleneck: 1 conv block (256→512 channels)
        - Decoder: 3 conv blocks with transpose convolutions (512→256→128→64 channels)
        - Output: 1x1 conv to 104 classes

    Attributes:
        encoder1, encoder2, encoder3: Encoder convolutional blocks
        bottleneck: Bottleneck convolutional block
        decoder1, decoder2, decoder3: Decoder convolutional blocks
        pool: Max pooling layer for downsampling
        upconv1, upconv2, upconv3: Transpose convolutions for upsampling
        final: Final 1x1 convolution to output classes

    Example:
        >>> model = MiniUNet()
        >>> x = torch.rand(1, 3, 224, 224)  # Batch of 1, RGB image
        >>> output = model(x)
        >>> print(output.shape)  # torch.Size([1, 104, 224, 224])
    """

    def __init__(self):
        """
        Initialize the MiniUNet model.

        Sets up the encoder-decoder architecture with skip connections,
        pooling/upsampling layers, and applies He weight initialization.
        """
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
        """
        Initialize model weights using He initialization for ReLU networks.

        Applies Kaiming normal initialization to all Conv2d and ConvTranspose2d
        layers, which is optimal for ReLU activation functions. Biases are
        initialized to zero.

        Note:
            He initialization helps prevent vanishing/exploding gradients
            in deep networks with ReLU activations.
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
                # He initialization (Kaiming) - best for ReLU networks
                init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def conv_block(self, in_channels, out_channels):
        """
        Create a convolutional block with two conv layers and ReLU activations.

        Each block consists of two 3x3 convolutions with padding, followed by
        ReLU activations. This design increases the receptive field while
        maintaining spatial dimensions.

        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels

        Returns:
            nn.Sequential: Sequential container with conv layers and ReLU activations

        Note:
            Using two conv layers increases the receptive field and adds
            non-linearity without increasing parameters significantly.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        Forward pass through the MiniUNet model.

        Implements the U-Net architecture with encoder-decoder structure
        and skip connections. The encoder progressively reduces spatial
        dimensions while increasing channel depth. The decoder upsamples
        and combines features using skip connections.

        Args:
            x (torch.Tensor): Input tensor with shape (B, 3, H, W) where:
                - B: batch size
                - 3: RGB channels
                - H, W: height and width

        Returns:
            torch.Tensor: Segmentation logits with shape (B, 104, H, W) where:
                - B: batch size
                - 104: number of food classes
                - H, W: same as input dimensions

        Architecture Flow:
            1. Encoder: x → enc1 → enc2 → enc3
            2. Bottleneck: enc3 → bottleneck
            3. Decoder: bottleneck + enc3 → dec1 + enc2 → dec2 + enc1 → dec3
            4. Output: dec3 → final (104 classes)
        """
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
