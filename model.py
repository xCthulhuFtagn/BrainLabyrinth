import torch
import torch.nn as nn
import torch.nn.functional as F

from EEGPT_mcae_finetune import EEGPTClassifier

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
        x = x.transpose(1, 2)  # your original transpose
        return self.model(x).squeeze(1)


class EEGPTWrapper(nn.Module):
    def __init__(
        self,
        pretrained_path,
        channel_list,       # e.g. 58 EEG channels
        num_classes=1
    ):
        super().__init__()
        
        self.eegpt = EEGPTClassifier(
            num_classes=num_classes,            
            in_channels=58,
            img_size=[58, 256*4],      # 62 channels, 4s * 256Hz
            patch_stride=64,
            desired_time_len=256*4,
            use_channels_names=channel_list,
        )

        # 1) Load the weights
        ckpt = torch.load(
            pretrained_path,
            map_location="cpu",
            weights_only=False  # <--- override the default True in PyTorch 2.6
        )

        self.eegpt.load_state_dict(ckpt['state_dict'], strict=False)

        # 2). Freeze everything if you want
        for param in self.eegpt.parameters():
            param.requires_grad = False
            
        # Unfreeze classifier head
        for param in self.eegpt.head.parameters():
            param.requires_grad = True
            
        # # Unfreeze last 2 transformer layers
        # for param in self.eegpt.target_encoder.blocks[-2:].parameters():
        #     param.requires_grad = True

    def forward(self, x):
        """
        x: shape = (batch_size, time, channels) or whatever
        that EEGPT expects. Make sure the dimension ordering
        is consistent with the original pretraining.
        """
        x = x.transpose(1, 2)  # now [batch, channels, time]
        return self.eegpt(x).float().squeeze(-1)
        # Because we froze the parameters above, they won't update
        # with gradient anyway.
        # features = self.eegpt(x)  # (batch_size, d_model) in Float16
        # features = features.float() 
        # out = self.classifier(features).squeeze(-1)
        # return out
