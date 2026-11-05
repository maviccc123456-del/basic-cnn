import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """Simple CNN for 6-class steel surface defect classification."""
    def __init__(self, num_classes=6):
        super().__init__()

        self.features = nn.Sequential(

            # 输入: [batch, 1, 200, 200]
            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # [batch, 32, 100, 100]
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # [batch, 64, 50, 50]
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # [batch, 128, 25, 25]
        # ↓
        # [batch, 128, 1, 1]
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 128个特征 → 6个类别
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):

        x = self.features(x)

        x = self.avg_pool(x)

        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x


if __name__ == "__main__":

    model = SimpleCNN()

    # 模拟一个 batch：16张灰度图
    x = torch.randn(16, 1, 200, 200)

    output = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)