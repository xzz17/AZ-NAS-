import torch.nn as nn
import torch

class MobileNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, expand_ratio, activation, use_se):
        super().__init__()
        hidden_dim = int(in_channels * expand_ratio)
        self.use_se = use_se
        act_fn = nn.ReLU(inplace=True) if activation == 'relu' else nn.SiLU(inplace=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            act_fn,
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride=1, padding=kernel_size//2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            act_fn,
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        if use_se:
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(out_channels, out_channels // 4, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels // 4, out_channels, 1),
                nn.Sigmoid()
            )
        self.shortcut = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        out = self.conv(x)
        if self.use_se:
            se_weight = self.se(out)
            out = out * se_weight
        out += self.shortcut(x)
        return out


def build_mobilenet_from_config(config, num_classes=10):
    layers = []
    in_channels = 3
    base_channels = int(32 * config['prune_ratio'])

    for i in range(config['num_blocks']):
        scale_factor = 2 ** (i // 4)
        out_channels = int(base_channels * scale_factor)
        layers.append(MobileNetBlock(
            in_channels, out_channels, config['kernel_size'],
            config['expand_ratio'], config['activation'], config['use_se']))
        in_channels = out_channels

    layers.append(nn.AdaptiveAvgPool2d((1,1)))
    layers.append(nn.Flatten())
    layers.append(nn.Linear(out_channels, num_classes))
    return nn.Sequential(*layers)
