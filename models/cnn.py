import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """Baseline 3-block Convolutional Neural Network."""
    def __init__(self, num_classes=5):
        super().__init__()

        self.features = nn.Sequential(
            # Input: (3, 64, 64)
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> (16, 32, 32)

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> (32, 16, 16)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> (64, 8, 8)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class BetterCNN(nn.Module):
    """Optimized CNN architecture with Batch Normalization and Adaptive Avg Pooling."""
    def __init__(self, num_classes=5):
        super().__init__()

        def block(in_c, out_c, p_drop=0.0):
            layers = [
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            if p_drop > 0:
                layers.append(nn.Dropout2d(p_drop))
            return nn.Sequential(*layers)

        self.stem = block(3, 24, p_drop=0.05)    # 64 -> 32
        self.stage2 = block(24, 48, p_drop=0.08) # 32 -> 16
        self.stage3 = block(48, 96, p_drop=0.10) # 16 -> 8

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(96, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.head(x)
        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)