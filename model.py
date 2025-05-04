import torch
import torch.nn as nn
import torch.nn.functional as F
class EEGMobileNet(nn.Module):
    def __init__(self, in_channels=64, num_classes=1, dropout=0.5):
        super().__init__()
        self.model = nn.Sequential(
            # Initial Conv
            nn.Conv1d(in_channels, 32, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(32, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),  # ← Insert dropout here

            # Depthwise
            nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1, groups=32),
            nn.BatchNorm1d(32, track_running_stats=False),
            nn.ReLU(inplace=True),
            # Pointwise
            nn.Conv1d(32, 64, kernel_size=1),
            nn.BatchNorm1d(64, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),  # ← Insert dropout here

            # Another Depthwise Separable block
            nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1, groups=64),
            nn.BatchNorm1d(64, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),  # ← Insert dropout here

            # Global Average Pool
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.model(x)#.squeeze(1)
