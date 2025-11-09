"""
ResNet50 Model Architecture
Contains the complete ResNet50 implementation with:
- Squeeze-and-Excitation blocks
- Stochastic depth (drop path)
- ResNet-D style stem
"""

import torch
import torch.nn as nn
from typing import Optional


def drop_path(x, drop_prob: float, training: bool) -> torch.Tensor:
    """Stochastic depth regularization."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor = random_tensor.floor()
    return x.div(keep_prob) * random_tensor


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block for channel attention"""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, reduced, kernel_size=1)
        self.fc2 = nn.Conv2d(reduced, channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.pool(x)
        scale = self.relu(self.fc1(scale))
        scale = self.sigmoid(self.fc2(scale))
        return x * scale


class Bottleneck(nn.Module):
    """ResNet Bottleneck block with SE and stochastic depth"""
    
    expansion = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        drop_prob: float = 0.0
    ):
        super(Bottleneck, self).__init__()
        width = out_channels
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)

        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)

        self.conv3 = nn.Conv2d(width, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.se = SqueezeExcitation(out_channels * self.expansion)
        self.drop_prob = drop_prob
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = drop_path(out, self.drop_prob, self.training)
        out += identity
        out = self.relu(out)

        return out


class ResNet50(nn.Module):
    """
    ResNet50 with modern improvements:
    - ResNet-D style stem
    - Squeeze-and-Excitation blocks
    - Stochastic depth
    - Label smoothing ready
    """
    
    def __init__(self, num_classes: int = 1000, drop_path_rate: float = 0.2, dropout: float = 0.2):
        super(ResNet50, self).__init__()

        self.inplanes = 64

        # ResNet-D style stem for better early feature extraction on smaller inputs
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.stem_pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        block_counts = [3, 4, 6, 3]
        total_blocks = sum(block_counts)
        if total_blocks > 1:
            drop_probs = [drop_path_rate * idx / (total_blocks - 1) for idx in range(total_blocks)]
        else:
            drop_probs = [drop_path_rate]

        start = 0
        self.layer1 = self._make_layer(
            Bottleneck, 64, block_counts[0], stride=1,
            drop_probs=drop_probs[start:start + block_counts[0]]
        )
        start += block_counts[0]
        self.layer2 = self._make_layer(
            Bottleneck, 128, block_counts[1], stride=2,
            drop_probs=drop_probs[start:start + block_counts[1]]
        )
        start += block_counts[1]
        self.layer3 = self._make_layer(
            Bottleneck, 256, block_counts[2], stride=2,
            drop_probs=drop_probs[start:start + block_counts[2]]
        )
        start += block_counts[2]
        self.layer4 = self._make_layer(
            Bottleneck, 512, block_counts[3], stride=2,
            drop_probs=drop_probs[start:start + block_counts[3]]
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

        self._initialize_weights()

    def _make_layer(self, block, out_channels, blocks, stride=1, drop_probs=None):
        downsample = None

        if stride != 1 or self.inplanes != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        drop_probs = drop_probs or [0.0] * blocks

        layers.append(block(self.inplanes, out_channels, stride, downsample, drop_prob=drop_probs[0]))
        self.inplanes = out_channels * block.expansion

        for idx in range(1, blocks):
            layers.append(block(self.inplanes, out_channels, drop_prob=drop_probs[idx]))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch
        for m in self.modules():
            if isinstance(m, Bottleneck):
                nn.init.constant_(m.bn3.weight, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.stem_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x
