import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=16, use_transpose=True):
        super(UNet, self).__init__()
        features = [init_features * 2 ** i for i in range(5)]  # [16, 32, 64, 128, 256]
        self.use_transpose = use_transpose

        # Encoder
        self.enc1 = self._block(in_channels, features[0], name="enc1")
        self.enc2 = self._block(features[0], features[1], name="enc2")
        self.enc3 = self._block(features[1], features[2], name="enc3")
        self.enc4 = self._block(features[2], features[3], name="enc4")
        self.enc5 = self._block(features[3], features[4], name="enc5")

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dropout
        self.dropout = nn.Dropout2d(0.5)

        # Decoder
        self.upconv4 = self._up_block(features[4], features[3])
        self.dec4 = self._block(features[4], features[3], name="dec4")

        self.upconv3 = self._up_block(features[3], features[2])
        self.dec3 = self._block(features[3], features[2], name="dec3")

        self.upconv2 = self._up_block(features[2], features[1])
        self.dec2 = self._block(features[2], features[1], name="dec2")

        self.upconv1 = self._up_block(features[1], features[0])
        self.dec1 = self._block(features[1], features[0], name="dec1")

        # Final output
        self.final_conv = nn.Sequential(
            nn.Conv2d(features[0], features[0] // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[0] // 2, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def _block(self, in_channels, features, name):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=features),
            nn.ReLU(inplace=True)
        )

    def _up_block(self, in_channels, out_channels):
        if self.use_transpose:
            return nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2,
                stride=2
            )
        else:
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
            )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        enc4 = self.dropout(enc4)
        enc5 = self.enc5(self.pool(enc4))
        enc5 = self.dropout(enc5)

        # Decoder with skip connections
        dec4 = self.upconv4(enc5)
        dec4 = torch.cat((enc4, dec4), dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((enc3, dec3), dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((enc2, dec2), dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((enc1, dec1), dim=1)
        dec1 = self.dec1(dec1)

        return self.final_conv(dec1)


class ClassicalDiscriminator(nn.Module):
    def __init__(self, in_channels=1, filters=None, kernel_size=3, name="ClassicalDiscriminator"):
        super(ClassicalDiscriminator, self).__init__()
        if filters is None:
            filters = [64, 128, 256, 512, 512, 1, 196]
        self.filters = filters
        self.kernel_size = kernel_size
        self.name = name

        # Define the layers
        self.layers = nn.ModuleList()

        # Layer 1: Conv2d -> LeakyReLU
        self.layers.append(nn.Conv2d(in_channels=in_channels, out_channels=filters[0], kernel_size=kernel_size, stride=2,
                                     padding=kernel_size // 2))
        self.layers.append(nn.LeakyReLU(negative_slope=0.2))

        # Layer 2: Conv2d -> BatchNorm -> LeakyReLU
        self.layers.append(nn.Conv2d(in_channels=filters[0], out_channels=filters[1], kernel_size=kernel_size, stride=2,
                                     padding=kernel_size // 2))
        self.layers.append(nn.BatchNorm2d(filters[1]))
        self.layers.append(nn.LeakyReLU(negative_slope=0.2))

        # Layer 3: Conv2d -> BatchNorm -> LeakyReLU
        self.layers.append(nn.Conv2d(in_channels=filters[1], out_channels=filters[2], kernel_size=kernel_size, stride=2,
                                     padding=kernel_size // 2))
        self.layers.append(nn.BatchNorm2d(filters[2]))
        self.layers.append(nn.LeakyReLU(negative_slope=0.2))

        # Layer 4: Conv2d -> BatchNorm -> LeakyReLU
        self.layers.append(nn.Conv2d(in_channels=filters[2], out_channels=filters[3], kernel_size=kernel_size, stride=2,
                                     padding=kernel_size // 2))
        self.layers.append(nn.BatchNorm2d(filters[3]))
        self.layers.append(nn.LeakyReLU(negative_slope=0.2))

        # Layer 5: Conv2d -> BatchNorm -> LeakyReLU
        self.layers.append(nn.Conv2d(in_channels=filters[3], out_channels=filters[4], kernel_size=kernel_size, stride=1,
                                     padding=kernel_size // 2))
        self.layers.append(nn.BatchNorm2d(filters[4]))
        self.layers.append(nn.LeakyReLU(negative_slope=0.2))

        # Layer 6: Conv2d -> LeakyReLU (valid padding)
        self.layers.append(
            nn.Conv2d(in_channels=filters[4], out_channels=filters[5], kernel_size=kernel_size, stride=1, padding=0))
        self.layers.append(nn.LeakyReLU(negative_slope=0.2))

        # Flatten and Dense layers
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(filters[6], 1)  # Output size depends on input dimensions, adjusted dynamically
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.sigmoid(x)
        return x


# model = ClassicalDiscriminator(in_channels=3)
# x = torch.rand(size=(1, 3, 256, 256))
# print(model(x).shape)