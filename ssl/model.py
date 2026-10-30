"""Pure convolutional residual network (project spec: ReLU(S(x) + F(x)))
with a deep MLP classification head.

Competition constraint: <= 5,000,000 trainable parameters (final model).

Self-supervised pretraining (SimCLR/InfoNCE) uses a temporary ProjectionHead
that is discarded before supervised fine-tuning; it is NOT part of the final
model parameter count.
"""
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """ReLU(S(x) + F(x)), F = conv -> BN -> relu -> conv -> BN (per project PDF)."""

    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.short = nn.Sequential()
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride=stride, bias=False),
                nn.BatchNorm2d(cout),
            )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + self.short(x))


class ConvResNet(nn.Module):
    """Residual conv stages -> global average pool -> deep MLP head."""

    def __init__(self, widths=(64, 128, 256), num_blocks=(3, 3, 3),
                 num_classes=10, head_hidden=512):
        super().__init__()
        c1, c2, c3 = widths
        self.stem = nn.Sequential(
            nn.Conv2d(3, c1, 3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )
        self.stage1 = self._stage(c1, c1, num_blocks[0], stride=1)
        self.stage2 = self._stage(c1, c2, num_blocks[1], stride=2)
        self.stage3 = self._stage(c2, c3, num_blocks[2], stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        # deep MLP head: c3 -> head_hidden -> head_hidden -> num_classes
        self.head = nn.Sequential(
            nn.Linear(c3, head_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(head_hidden, head_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(head_hidden, num_classes),
        )

    @staticmethod
    def _stage(cin, cout, blocks, stride):
        layers = [ResidualBlock(cin, cout, stride=stride)]
        layers += [ResidualBlock(cout, cout, 1) for _ in range(blocks - 1)]
        return nn.Sequential(*layers)

    def forward_features(self, x):
        x = self.stem(x)          # 32x32 x c1
        x = self.stage1(x)        # 32x32 x c1
        x = self.stage2(x)        # 16x16 x c2
        x = self.stage3(x)        # 8x8   x c3
        return self.pool(x).flatten(1)   # (B, C)

    def forward(self, x):
        return self.head(self.forward_features(x))


class ProjectionHead(nn.Module):
    """SimCLR projection head (discarded after pretraining)."""

    def __init__(self, dim=256, hidden=512, out=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out),
        )

    def forward(self, x):
        return self.net(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("candidate configurations (with deep MLP head):")
    for widths, blocks in [
        ((64, 128, 256), (2, 2, 1)),
        ((64, 128, 256), (2, 2, 2)),
        ((64, 128, 256), (3, 3, 2)),
        ((64, 128, 256), (3, 3, 3)),
        ((80, 160, 320), (2, 2, 2)),
        ((96, 192, 384), (2, 2, 2)),
    ]:
        m = ConvResNet(widths=widths, num_blocks=blocks)
        n = count_params(m)
        ok = "OK " if n <= 5_000_000 else "OVER"
        print(f"  [{ok}] widths={widths} blocks={blocks}: {n:,}")

    model = ConvResNet()
    n = count_params(model)
    print(f"\ndefault config parameters: {n:,} (limit 5,000,000)")
    x = torch.randn(2, 3, 32, 32)
    print("forward pass:", tuple(x.shape), "->", tuple(model(x).shape))
