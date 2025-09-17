import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        return x1, x2, x3, x4

class HeatmapNet(nn.Module):
    def __init__(self, num_keypoints=5, skip=True, heatmap_size=64):
        """
        Initialize the heatmap regression network.
        
        Args:
            num_keypoints: Number of keypoints to detect
        """
        super().__init__()
        self.num_keypoints = num_keypoints
        
        # Encoder (downsampling path)
        # Input: [batch, 1, 128, 128]
        # Progressively downsample to extract features
        
        # Decoder (upsampling path)
        # Progressively upsample back to heatmap resolution
        # Output: [batch, num_keypoints, 64, 64]
        
        # Skip connections between encoder and decoder
        self.skip = skip
        self.heatmap_size = heatmap_size
        self.encoder = Encoder()
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2),  # 8→16
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(256 if skip else 128, 64, 2, stride=2),  # 16→32
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        if heatmap_size == 32:
            self.deconv2 = None
        else:
            self.deconv2 = nn.Sequential(
                nn.ConvTranspose2d(128 if skip else 64, 32, 2, stride=2),  # 32→64
                nn.BatchNorm2d(32),
                nn.ReLU()
            )

        if heatmap_size == 128:
            self.deconv1 = nn.Sequential(
                nn.ConvTranspose2d(64 if skip else 32, 16, 2, stride=2),  # 64→128
                nn.BatchNorm2d(16),
                nn.ReLU()
            )
            final_in = 16
        elif heatmap_size == 64:
            self.deconv1 = None
            final_in = 64 if skip else 32
        else:
            self.deconv1 = None
            final_in = 128 if skip else 64
        
        self.final = nn.Conv2d(final_in, num_keypoints, 1)

    def forward(self, x):
        x1, x2, x3, x4 = self.encoder(x)
        d4 = self.deconv4(x4)
        if self.skip:
            d4 = torch.cat([d4, x3], dim=1)

        d3 = self.deconv3(d4)
        if self.skip:
            d3 = torch.cat([d3, x2], dim=1)

        if self.heatmap_size >= 64:
            d2 = self.deconv2(d3)
            if self.skip:
                d2 = torch.cat([d2, x1], dim=1)
        else:
            d2 = d3

        if self.heatmap_size == 128:
            d1 = self.deconv1(d2)
            out = self.final(d1)
        else:
            out = self.final(d2)

        return out
        pass

class RegressionNet(nn.Module):
    def __init__(self, num_keypoints=5):
        """
        Initialize the direct regression network.
        
        Args:
            num_keypoints: Number of keypoints to detect
        """
        super().__init__()
        self.num_keypoints = num_keypoints
        
        # Use same encoder architecture as HeatmapNet
        # But add global pooling and fully connected layers
        # Output: [batch, num_keypoints * 2]
        self.encoder = Encoder()
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_keypoints*2)
        self.dropout = nn.Dropout(0.5)
        pass
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch, 1, 128, 128]
            
        Returns:
            coords: Tensor of shape [batch, num_keypoints * 2]
                   Values in range [0, 1] (normalized coordinates)
        """
        _, _, _, x4 = self.encoder(x)
        x_pool = F.adaptive_avg_pool2d(x4, (1,1)).view(x.size(0), -1)
        x_fc = F.relu(self.fc1(x_pool))
        x_fc = self.dropout(x_fc)
        x_fc = F.relu(self.fc2(x_fc))
        x_fc = self.dropout(x_fc)
        out = torch.sigmoid(self.fc3(x_fc))
        return out
        pass
