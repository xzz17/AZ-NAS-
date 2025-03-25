import torch.nn as nn
import torch

class MobileNetBlock(nn.Module):
    """
    A single block used in MobileNet-like architecture, consisting of:
    - 1x1 pointwise conv (expansion)
    - depthwise conv (spatial feature learning)
    - 1x1 pointwise conv (projection)
    - optional Squeeze-and-Excitation (SE) module
    - optional skip connection if input and output channels match
    """
    def __init__(self, in_channels, out_channels, kernel_size, expand_ratio, activation, use_se):
        super().__init__()
        hidden_dim = int(in_channels * expand_ratio)
        self.use_se = use_se

        # Choose activation function
        act_fn = nn.ReLU(inplace=True) if activation == 'relu' else nn.SiLU(inplace=True)

        # Main convolution block
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            act_fn,
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride=1, padding=kernel_size // 2,
                      groups=hidden_dim, bias=False),  # depthwise convolution
            nn.BatchNorm2d(hidden_dim),
            act_fn,
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        # Optional Squeeze-and-Excitation module
        if use_se:
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(out_channels, out_channels // 4, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels // 4, out_channels, kernel_size=1),
                nn.Sigmoid()
            )

        # Shortcut: identity if same shape, otherwise 1x1 projection
        self.shortcut = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        out = self.conv(x)
        if self.use_se:
            se_weight = self.se(out)
            out = out * se_weight
        out += self.shortcut(x)
        return out


def build_mobilenet_from_config(config, num_classes=10):
    """
    Build a MobileNet-like architecture from a given config dictionary.

    Args:
        config (dict): contains keys:
            - 'activation': 'relu' or 'swish'
            - 'kernel_size': convolution kernel size (3 or 5)
            - 'expand_ratio': expansion ratio for hidden channels
            - 'prune_ratio': controls base channel width (e.g., 0.5, 0.75, 1.0)
            - 'use_se': whether to include SE module
            - 'num_blocks': number of MobileNet blocks
        num_classes (int): number of output classes (default 10 for CIFAR-10)

    Returns:
        nn.Sequential: the constructed model
    """
    layers = []
    in_channels = 3  # RGB input
    base_channels = int(32 * config['prune_ratio'])  # adjustable base width

    for i in range(config['num_blocks']):
        scale_factor = 2 ** (i // 4)  # progressive channel expansion
        out_channels = int(base_channels * scale_factor)
        layers.append(MobileNetBlock(
            in_channels, out_channels, config['kernel_size'],
            config['expand_ratio'], config['activation'], config['use_se']))
        in_channels = out_channels

    # Final classifier head
    layers.append(nn.AdaptiveAvgPool2d((1, 1)))
    layers.append(nn.Flatten())
    layers.append(nn.Linear(out_channels, num_classes))

    return nn.Sequential(*layers)
