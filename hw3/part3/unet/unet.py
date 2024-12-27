from torch.autograd import forward_ad
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.modules import padding
from torchvision.models.resnet import Weights, resnet18, ResNet18_Weights
import pdb

class SumUNet(nn.Module):
    def __init__(self, n_class: int, load_pretrained_encoder_layers: bool=False):
        super().__init__()
        # Using resnet18 model
        if load_pretrained_encoder_layers:
            weights = ResNet18_Weights.DEFAULT
        else:
            weights = None
        _encoder = resnet18(weights=weights)

        # Initializing layers to convert channels from 3 -> 16 to feed into resnet layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Encoder layers from resnet
        self.layer1 = _encoder.layer1
        self.layer2 = _encoder.layer2
        self.layer3 = _encoder.layer3
        self.layer4 = _encoder.layer4
        self.base_layers = [
          self.layer1, self.layer2, self.layer3, self.layer4
        ]

        # Decoder layers with skip connection as sum
        self.up_layer1 = DecoderBlock(512, 256, mode='sum')
        self.up_layer2 = DecoderBlock(256, 128, mode='sum')
        self.up_layer3 = DecoderBlock(128, 64, mode='sum')
        self.outc = nn.Conv2d(64, n_class, kernel_size=1)

    def forward(self, x: torch.TensorType) -> torch.TensorType:
        x = self.conv1(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x5 = self.up_layer1(x4, x3)
        x6 = self.up_layer2(x5, x2)
        x7 = self.up_layer3(x6, x1)
        return self.outc(x7)

class UNet(nn.Module):
    def __init__(self, n_class: int, load_pretrained_encoder_layers: bool=False):
        super().__init__()
        # Using resnet18 model
        if load_pretrained_encoder_layers:
            weights = ResNet18_Weights.DEFAULT
        else:
            weights = None
        _encoder = resnet18(weights=weights)

        # Initializing layers to convert channels from 3 -> 16 to feed into resnet layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Encoder layers from resnet
        self.layer1 = _encoder.layer1
        self.layer2 = _encoder.layer2
        self.layer3 = _encoder.layer3
        self.layer4 = _encoder.layer4
        self.base_layers = [
          self.layer1, self.layer2, self.layer3, self.layer4
        ]

        # Decoder layers with skip connection as concat
        self.up_layer1 = DecoderBlock(512, 256, mode='concat')
        self.up_layer2 = DecoderBlock(256, 128, mode='concat')
        self.up_layer3 = DecoderBlock(128, 64, mode='concat')
        self.outc = nn.Conv2d(64, n_class, kernel_size=1)

    def forward(self, x: torch.TensorType) -> torch.TensorType:
        x = self.conv1(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x5 = self.up_layer1(x4, x3)
        x6 = self.up_layer2(x5, x2)
        x7 = self.up_layer3(x6, x1)
        return self.outc(x7)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mode: bool):
        super().__init__()
        # Mode - 'sum' / 'concat'
        self.mode = mode
        if self.mode == 'concat':
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        elif self.mode == 'sum':
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            self.conv = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

    def forward(self, x: torch.TensorType, x_skip: torch.TensorType) -> torch.TensorType:
        x = self.up(x)

        # Padding for skip connection
        h_pad = x_skip.shape[2] - x.shape[2]
        w_pad = x_skip.shape[3] - x.shape[3]
        padding = (
          w_pad // 2, w_pad - w_pad // 2,
          h_pad // 2, h_pad - h_pad // 2,
        )
        x = F.pad(x, padding, 'reflect')

        # Skip connection
        if self.mode == 'concat':
            x = torch.concat([x, x_skip], dim=1)
        elif self.mode == 'sum':
            x = x + x_skip

        return self.conv(x)
