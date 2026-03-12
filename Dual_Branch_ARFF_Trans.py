import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, resnet34, resnet50
import math


class ConvBlock(nn.Module):
    """卷积块"""

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class TransformerEncoder(nn.Module):
    """Transformer编码器"""

    def __init__(self, dim, num_heads=8, num_layers=8, mlp_ratio=4.0, dropout=0.2):
        super(TransformerEncoder, self).__init__()
        self.dim = dim
        self.num_heads = num_heads

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=int(dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape

        # 将空间维度展平为序列
        x = x.flatten(2)  # [B, C, H*W]
        x = x.transpose(1, 2)  # [B, H*W, C]

        # 添加位置编码
        pos_encoding = self.positional_encoding(H * W, C, x.device)
        x = x + pos_encoding

        # Transformer编码
        x = self.transformer(x)  # [B, H*W, C]

        # 恢复空间维度
        x = x.transpose(1, 2)  # [B, C, H*W]
        x = x.view(B, C, H, W)  # [B, C, H, W]

        return x

    def positional_encoding(self, seq_len, d_model, device):
        """位置编码"""
        pos = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
        i = torch.arange(d_model, dtype=torch.float, device=device).unsqueeze(0)

        angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / d_model)
        angle_rads = pos * angle_rates

        # 对偶数位置应用sin，奇数位置应用cos
        pos_encoding = torch.zeros(seq_len, d_model, device=device)
        pos_encoding[:, 0::2] = torch.sin(angle_rads[:, 0::2])
        pos_encoding[:, 1::2] = torch.cos(angle_rads[:, 1::2])

        return pos_encoding.unsqueeze(0)  # [1, seq_len, d_model]


class RatioFusionBlock(nn.Module):
    """按比例特征融合模块 (3:1比例)"""

    def __init__(self, main_channels, aux_channels):
        super(RatioFusionBlock, self).__init__()
        self.main_channels = main_channels
        self.aux_channels = aux_channels

        # 计算按比例分配后的通道数
        self.total_channels = main_channels + aux_channels
        self.main_ratio_channels = int(main_channels * 0.75)  # 主分支占75%
        self.aux_ratio_channels = int(aux_channels * 0.25)  # 辅助分支占25%

        # 如果比例通道数之和小于总通道数，调整主分支通道数
        if self.main_ratio_channels + self.aux_ratio_channels < self.total_channels:
            self.main_ratio_channels = self.total_channels - self.aux_ratio_channels

        # 主分支通道调整卷积
        self.main_conv = nn.Sequential(
            nn.Conv2d(main_channels, self.main_ratio_channels, 1),
            nn.BatchNorm2d(self.main_ratio_channels),
            nn.ReLU(inplace=True)
        )

        # 辅助分支通道调整卷积
        self.aux_conv = nn.Sequential(
            nn.Conv2d(aux_channels, self.aux_ratio_channels, 1),
            nn.BatchNorm2d(self.aux_ratio_channels),
            nn.ReLU(inplace=True)
        )

        # 融合后的卷积
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(self.total_channels, main_channels, 3, padding=1),
            nn.BatchNorm2d(main_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, main_feat, aux_feat):
        # 调整主分支特征通道数
        main_adjusted = self.main_conv(main_feat)

        # 调整辅助特征
        aux_adjusted = self.aux_conv(aux_feat)

        # 确保空间尺寸一致（通过插值）
        if aux_adjusted.shape[-2:] != main_adjusted.shape[-2:]:
            aux_adjusted = F.interpolate(aux_adjusted, size=main_adjusted.shape[-2:], mode='bilinear',
                                         align_corners=False)

        # 按比例通道融合
        fused = torch.cat([main_adjusted, aux_adjusted], dim=1)
        fused = self.fusion_conv(fused)

        return fused


class AdaptiveRatioFusionBlock(nn.Module):
    """自适应比例特征融合模块 - 可学习权重"""

    def __init__(self, main_channels, aux_channels, init_ratio=0.75):
        super(AdaptiveRatioFusionBlock, self).__init__()
        self.main_channels = main_channels
        self.aux_channels = aux_channels

        # 可学习的融合权重
        self.ratio = nn.Parameter(torch.tensor(init_ratio))

        # 通道调整卷积
        self.main_conv = nn.Sequential(
            nn.Conv2d(main_channels, main_channels, 1),
            nn.BatchNorm2d(main_channels),
            nn.ReLU(inplace=True)
        )

        self.aux_conv = nn.Sequential(
            nn.Conv2d(aux_channels, main_channels, 1),  # 统一调整到main_channels
            nn.BatchNorm2d(main_channels),
            nn.ReLU(inplace=True)
        )

        # 融合后的卷积
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(main_channels, main_channels, 3, padding=1),
            nn.BatchNorm2d(main_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, main_feat, aux_feat):
        # 调整特征
        main_adjusted = self.main_conv(main_feat)
        aux_adjusted = self.aux_conv(aux_feat)

        # 确保空间尺寸一致
        if aux_adjusted.shape[-2:] != main_adjusted.shape[-2:]:
            aux_adjusted = F.interpolate(aux_adjusted, size=main_adjusted.shape[-2:], mode='bilinear',
                                         align_corners=False)

        # 按可学习比例融合
        ratio = torch.sigmoid(self.ratio)  # 限制在0-1之间
        fused = ratio * main_adjusted + (1 - ratio) * aux_adjusted
        fused = self.fusion_conv(fused)

        return fused


class ResidualBlock(nn.Module):
    """简化版残差块"""

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 捷径连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(residual)
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """Bottleneck残差块 - 用于ResNet50/101/152"""

    def __init__(self, in_channels, out_channels, stride=1, expansion=4):
        super(Bottleneck, self).__init__()
        self.expansion = expansion
        mid_channels = out_channels // expansion

        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)

        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        # 捷径连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += self.shortcut(residual)
        out = self.relu(out)

        return out


class DualInputResNetTransformer(nn.Module):
    """双输入ResNet-Transformer混合网络"""

    def __init__(self, num_classes=1, backbone='resnet18', fusion_type='ratio', is_transformer=True):
        super(DualInputResNetTransformer, self).__init__()

        # 选择ResNet骨干网络
        if backbone == 'resnet18':
            resnet = resnet18(pretrained=True)
            base_channels = [64, 128, 256, 512]
            blocks = [2, 2, 2, 2]  # resnet18每层的块数
            block_type = 'basic'
        elif backbone == 'resnet34':
            resnet = resnet34(pretrained=True)
            base_channels = [64, 128, 256, 512]
            blocks = [3, 4, 6, 3]  # resnet34每层的块数
            block_type = 'basic'
        elif backbone == 'resnet50':
            resnet = resnet50(pretrained=True)
            base_channels = [256, 512, 1024, 2048]  # ResNet50的通道数
            blocks = [3, 4, 6, 3]  # resnet50每层的块数
            block_type = 'bottleneck'
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.backbone = backbone
        self.block_type = block_type
        self.is_transformer = is_transformer

        # 主分支编码器 (RGB输入)
        self.main_encoder = nn.ModuleDict({
            'conv1': nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu
            ),
            'maxpool': resnet.maxpool,
            'layer1': resnet.layer1,  # 下采样2倍
            'layer2': resnet.layer2,  # 下采样4倍
            'layer3': resnet.layer3,  # 下采样8倍
            'layer4': resnet.layer4  # 下采样16倍
        })

        # 辅助分支编码器 (单波段输入)
        # 初始卷积层
        self.aux_conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.aux_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 辅助分支的残差层
        if block_type == 'basic':
            self.aux_layer1 = self._make_basic_layer(64, base_channels[0], blocks[0], stride=1)
            self.aux_layer2 = self._make_basic_layer(base_channels[0], base_channels[1], blocks[1], stride=2)
            self.aux_layer3 = self._make_basic_layer(base_channels[1], base_channels[2], blocks[2], stride=2)
            self.aux_layer4 = self._make_basic_layer(base_channels[2], base_channels[3], blocks[3], stride=2)
        else:  # bottleneck
            self.aux_layer1 = self._make_bottleneck_layer(64, base_channels[0], blocks[0], stride=1)
            self.aux_layer2 = self._make_bottleneck_layer(base_channels[0], base_channels[1], blocks[1], stride=2)
            self.aux_layer3 = self._make_bottleneck_layer(base_channels[1], base_channels[2], blocks[2], stride=2)
            self.aux_layer4 = self._make_bottleneck_layer(base_channels[2], base_channels[3], blocks[3], stride=2)

        # 特征融合模块 - 根据类型选择
        self.fusion_type = fusion_type
        if fusion_type == 'ratio':
            # 使用固定比例融合
            self.fusion1 = RatioFusionBlock(base_channels[0], base_channels[0])
            self.fusion2 = RatioFusionBlock(base_channels[1], base_channels[1])
            self.fusion3 = RatioFusionBlock(base_channels[2], base_channels[2])
            self.fusion4 = RatioFusionBlock(base_channels[3], base_channels[3])
        elif fusion_type == 'adaptive':
            # 使用自适应比例融合
            self.fusion1 = AdaptiveRatioFusionBlock(base_channels[0], base_channels[0])
            self.fusion2 = AdaptiveRatioFusionBlock(base_channels[1], base_channels[1])
            self.fusion3 = AdaptiveRatioFusionBlock(base_channels[2], base_channels[2])
            self.fusion4 = AdaptiveRatioFusionBlock(base_channels[3], base_channels[3])
        else:
            raise ValueError("fusion_type must be 'ratio' or 'adaptive'")

        # Transformer编码器
        if self.is_transformer:
            self.transformer = TransformerEncoder(dim=base_channels[3])

        # 解码器
        self.up1 = nn.ConvTranspose2d(base_channels[3], base_channels[2], 2, stride=2)
        self.dec1 = ConvBlock(base_channels[2] * 2, base_channels[2])

        self.up2 = nn.ConvTranspose2d(base_channels[2], base_channels[1], 2, stride=2)
        self.dec2 = ConvBlock(base_channels[1] * 2, base_channels[1])

        self.up3 = nn.ConvTranspose2d(base_channels[1], base_channels[0], 2, stride=2)
        self.dec3 = ConvBlock(base_channels[0] * 2, base_channels[0])

        # 根据backbone调整最终上采样
        if backbone == 'resnet50':
            # ResNet50需要更多的上采样步骤
            self.up4 = nn.ConvTranspose2d(base_channels[0], base_channels[0] // 4, 2, stride=2)
            self.dec4 = ConvBlock(base_channels[0] // 4, base_channels[0] // 4)
            final_in_channels = base_channels[0] // 4
        else:
            self.up4 = nn.ConvTranspose2d(base_channels[0], base_channels[0] // 2, 2, stride=2)
            self.dec4 = ConvBlock(base_channels[0] // 2, base_channels[0] // 2)
            final_in_channels = base_channels[0] // 2

        # 最终输出层
        self.final_conv = nn.Sequential(
            nn.Conv2d(final_in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, 1)
        )

    def _make_basic_layer(self, in_channels, out_channels, blocks, stride=1):
        """创建基本残差层"""
        layers = []
        # 第一个块可能有下采样
        layers.append(ResidualBlock(in_channels, out_channels, stride))

        # 剩余的块
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))

        return nn.Sequential(*layers)

    def _make_bottleneck_layer(self, in_channels, out_channels, blocks, stride=1):
        """创建Bottleneck残差层"""
        layers = []
        # 第一个块可能有下采样
        layers.append(Bottleneck(in_channels, out_channels, stride))

        # 剩余的块
        for _ in range(1, blocks):
            layers.append(Bottleneck(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, main_input, aux_input):
        """
        前向传播
        Args:
            main_input: RGB输入 [B, 3, 224, 224]
            aux_input: 单波段输入 [B, 1, 112, 112]
        """
        # 编码阶段
        # 主分支
        x1 = self.main_encoder['conv1'](main_input)  # [B, 64, 112, 112]
        x1_pool = self.main_encoder['maxpool'](x1)  # [B, 64, 56, 56]
        x1 = self.main_encoder['layer1'](x1_pool)  # [B, 64/256, 56, 56]

        x2 = self.main_encoder['layer2'](x1)  # [B, 128/512, 28, 28]
        x3 = self.main_encoder['layer3'](x2)  # [B, 256/1024, 14, 14]
        x4 = self.main_encoder['layer4'](x3)  # [B, 512/2048, 7, 7]

        # 辅助分支
        y1 = self.aux_conv1(aux_input)  # [B, 64, 56, 56]
        y1_pool = self.aux_maxpool(y1)  # [B, 64, 28, 28]
        y1 = self.aux_layer1(y1_pool)  # [B, 64/256, 28, 28]

        y2 = self.aux_layer2(y1)  # [B, 128/512, 14, 14]
        y3 = self.aux_layer3(y2)  # [B, 256/1024, 7, 7]
        y4 = self.aux_layer4(y3)  # [B, 512/2048, 4, 4]

        # 特征融合（在每个下采样阶段）
        # 融合1
        y1_resized = F.interpolate(y1, size=x1.shape[-2:], mode='bilinear', align_corners=False)
        fused1 = self.fusion1(x1, y1_resized)

        # 融合2
        y2_resized = F.interpolate(y2, size=x2.shape[-2:], mode='bilinear', align_corners=False)
        fused2 = self.fusion2(x2, y2_resized)

        # 融合3
        y3_resized = F.interpolate(y3, size=x3.shape[-2:], mode='bilinear', align_corners=False)
        fused3 = self.fusion3(x3, y3_resized)

        # 融合4
        y4_resized = F.interpolate(y4, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        fused4 = self.fusion4(x4, y4_resized)

        # Transformer编码
        if self.is_transformer:
            fused4 = self.transformer(fused4)  # [B, 512/2048, 7, 7]

        # 解码阶段
        d1 = self.up1(fused4)  # [B, 256/1024, 14, 14]
        d1 = torch.cat([d1, fused3], dim=1)  # 跳跃连接
        d1 = self.dec1(d1)  # [B, 256/1024, 14, 14]

        d2 = self.up2(d1)  # [B, 128/512, 28, 28]
        d2 = torch.cat([d2, fused2], dim=1)  # 跳跃连接
        d2 = self.dec2(d2)  # [B, 128/512, 28, 28]

        d3 = self.up3(d2)  # [B, 64/256, 56, 56]
        d3 = torch.cat([d3, fused1], dim=1)  # 跳跃连接
        d3 = self.dec3(d3)  # [B, 64/256, 56, 56]

        d4 = self.up4(d3)  # [B, 32/64, 112, 112]
        d4 = self.dec4(d4)  # [B, 32/64, 112, 112]

        # 最终上采样到原始尺寸
        output = F.interpolate(d4, size=main_input.shape[-2:], mode='bilinear', align_corners=False)
        output = self.final_conv(output)  # [B, 1, 224, 224]

        return torch.sigmoid(output)


# 测试模型
def test_model():
    """测试模型是否能正常运行"""
    print("Testing DualInputResNetTransformer model with different backbones...")

    # 测试所有backbone和融合方式
    for backbone in ['resnet18', 'resnet34', 'resnet50']:
        for fusion_type in ['ratio', 'adaptive']:
            print(f"\nTesting {backbone} with {fusion_type} fusion...")

            try:
                # 创建模型实例
                model = DualInputResNetTransformer(num_classes=1, backbone=backbone, fusion_type=fusion_type)

                # 创建模拟输入数据
                batch_size = 2
                main_input = torch.randn(batch_size, 3, 224, 224)  # RGB输入
                aux_input = torch.randn(batch_size, 1, 112, 112)  # 单波段输入

                print(f"Main input shape: {main_input.shape}")
                print(f"Aux input shape: {aux_input.shape}")

                # 前向传播
                model.eval()
                with torch.no_grad():
                    output = model(main_input, aux_input)

                print(f"Output shape: {output.shape}")
                print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")

                # 对于自适应融合，打印学习到的比例
                if fusion_type == 'adaptive':
                    for i, fusion_block in enumerate([model.fusion1, model.fusion2, model.fusion3, model.fusion4], 1):
                        ratio = torch.sigmoid(fusion_block.ratio).item()
                        print(f"Fusion block {i} learned ratio: {ratio:.4f}")

                # 计算参数量
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

                print(f"Total parameters: {total_params:,}")
                print(f"Trainable parameters: {trainable_params:,}")

                print(f"{backbone} with {fusion_type} fusion test passed! ✓")

            except Exception as e:
                print(f"Error testing {backbone} with {fusion_type} fusion: {e}")


if __name__ == "__main__":
    test_model()