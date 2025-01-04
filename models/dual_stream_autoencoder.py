import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).view(x.size(0), -1))
        max_out = self.fc(self.max_pool(x).view(x.size(0), -1))
        out = avg_out + max_out
        return torch.sigmoid(out).view(x.size(0), x.size(1), 1, 1) * x

class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        # 添加1x1卷积保持通道数不变
        self.channel_conv = nn.Conv2d(1, in_channels, kernel_size=1)
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(attention)
        attention = self.channel_conv(attention)
        return torch.sigmoid(attention) * x

class DualStreamAutoencoder(nn.Module):
    def __init__(self):
        super(DualStreamAutoencoder, self).__init__()
        
        # RGB流编码器
        self.rgb_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            ChannelAttention(32),
            
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ChannelAttention(64),
            
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            ChannelAttention(128)
        )
        
        # 深度流编码器
        self.depth_conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.depth_attention1 = SpatialAttention(32)
        
        self.depth_conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.depth_attention2 = SpatialAttention(64)
        
        self.depth_conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.depth_attention3 = SpatialAttention(128)
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Tanh()
        )
        
    def forward(self, x):
        # 从RGB图像中提取深度信息
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        
        # RGB流特征提取
        rgb_features = self.rgb_encoder(x)
        
        # 深度流特征提取
        depth = self.depth_conv1(gray)
        depth = self.depth_attention1(depth)
        
        depth = self.depth_conv2(depth)
        depth = self.depth_attention2(depth)
        
        depth = self.depth_conv3(depth)
        depth_features = self.depth_attention3(depth)
        
        # 特征融合
        combined = torch.cat([rgb_features, depth_features], dim=1)
        fused = self.fusion(combined)
        
        # 解码重建
        output = self.decoder(fused)
        return output

    def get_features(self, x):
        """获取中间特征用于异常检测"""
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        
        rgb_features = self.rgb_encoder(x)
        
        depth = self.depth_conv1(gray)
        depth = self.depth_attention1(depth)
        
        depth = self.depth_conv2(depth)
        depth = self.depth_attention2(depth)
        
        depth = self.depth_conv3(depth)
        depth_features = self.depth_attention3(depth)
        
        combined = torch.cat([rgb_features, depth_features], dim=1)
        return self.fusion(combined) 