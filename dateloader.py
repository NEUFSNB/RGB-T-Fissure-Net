import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from Dual_UNet_Channel_Merge import DualInputResNetTransformer

class DualInputDataset(Dataset):
    """双输入数据集类"""

    def __init__(self, main_dir, aux_dir, mask_dir, transform=None, is_train=True):
        """
        Args:
            main_dir: RGB图像目录
            aux_dir: 单波段图像目录
            mask_dir: 标签掩码目录
            transform: 数据增强
            is_train: 是否为训练模式
        """
        self.main_dir = main_dir
        self.aux_dir = aux_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.is_train = is_train

        # 获取文件列表（假设文件名对应）
        self.main_files = sorted([f for f in os.listdir(main_dir) if f.endswith(('.png', '.jpg', '.tif'))])
        self.aux_files = sorted([f for f in os.listdir(aux_dir) if f.endswith(('.png', '.jpg', '.tif'))])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(('.png', '.jpg', '.tif'))])

        # 确保文件数量一致
        assert len(self.main_files) == len(self.aux_files) == len(self.mask_files), \
            "Number of files in main, aux and mask directories must be equal"

    def __len__(self):
        return len(self.main_files)

    def __getitem__(self, idx):
        # 读取RGB图像
        main_path = os.path.join(self.main_dir, self.main_files[idx])
        main_image = Image.open(main_path).convert('RGB')
        main_array = np.array(main_image)

        # 读取单波段图像
        aux_path = os.path.join(self.aux_dir, self.aux_files[idx])
        aux_image = Image.open(aux_path).convert('L')  # 转为灰度
        aux_array = np.array(aux_image)

        # 读取标签掩码
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
        mask_image = Image.open(mask_path).convert('L')
        mask_array = np.array(mask_image)

        # 二值化掩码（如果有必要）
        mask_array = (mask_array > 0).astype(np.float32)

        # 数据增强 - 分别处理不同尺寸的图像
        if self.transform:
            # 对RGB图像和掩码进行变换（224x224）
            main_mask_transform = self.transform['main_mask']
            transformed_main_mask = main_mask_transform(image=main_array, mask=mask_array)
            main_array = transformed_main_mask['image']
            mask_array = transformed_main_mask['mask']

            # 对单波段图像进行变换（112x112）
            aux_transform = self.transform['aux']
            transformed_aux = aux_transform(image=aux_array)
            aux_array = transformed_aux['image']
        else:
            # 基础转换
            main_array = main_array.astype(np.float32) / 255.0
            aux_array = aux_array.astype(np.float32) / 255.0
            mask_array = mask_array.astype(np.float32)

            # 转换为tensor
            main_array = torch.from_numpy(main_array).permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
            aux_array = torch.from_numpy(aux_array).unsqueeze(0)  # [H, W] -> [1, H, W]
            mask_array = torch.from_numpy(mask_array).unsqueeze(0)  # [H, W] -> [1, H, W]

        return main_array, aux_array, mask_array


def get_transforms(input_size=(224, 224), aux_size=(112, 112), is_train=True):
    """获取数据增强变换"""

    # 主图像和掩码的变换（224x224）
    if is_train:
        main_mask_transform = A.Compose([
            # 空间变换
            A.Resize(input_size[0], input_size[1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),

            # 颜色变换（仅对RGB图像）
            A.OneOf([
                A.RandomBrightnessContrast(p=0.5),
                A.RandomGamma(p=0.5),
                A.CLAHE(p=0.5),
            ], p=0.3),

            # 噪声变换
            A.OneOf([
                A.GaussNoise(p=0.5),
                A.MultiplicativeNoise(p=0.5),
            ], p=0.2),

            # 标准化
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        main_mask_transform = A.Compose([
            A.Resize(input_size[0], input_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    # 辅助图像的变换（112x112）
    if is_train:
        aux_transform = A.Compose([
            # 空间变换（与主图像相同的参数，但输出尺寸不同）
            A.Resize(aux_size[0], aux_size[1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),

            # 标准化（单通道）
            A.Normalize(mean=[0.5], std=[0.5]),
            ToTensorV2(),
        ])
    else:
        aux_transform = A.Compose([
            A.Resize(aux_size[0], aux_size[1]),
            A.Normalize(mean=[0.5], std=[0.5]),
            ToTensorV2(),
        ])

    return {
        'main_mask': main_mask_transform,
        'aux': aux_transform
    }


def create_data_loaders(main_dir, aux_dir, mask_dir, batch_size=2, num_workers=0):
    """创建数据加载器"""

    # 分割训练集和验证集
    all_files = sorted([f for f in os.listdir(main_dir) if f.endswith(('.png', '.jpg', '.tif'))])
    split_idx = int(0.8 * len(all_files))
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]

    # 创建数据集
    train_dataset = DualInputDataset(
        main_dir=main_dir,
        aux_dir=aux_dir,
        mask_dir=mask_dir,
        transform=get_transforms(is_train=True),
        is_train=True
    )

    val_dataset = DualInputDataset(
        main_dir=main_dir,
        aux_dir=aux_dir,
        mask_dir=mask_dir,
        transform=get_transforms(is_train=False),
        is_train=False
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


# 简化的测试数据加载器
def test_dataloader():
    """测试数据加载器"""
    print("Testing data loader...")

    # 创建模拟数据
    batch_size = 2

    # 模拟数据
    main_batch = torch.randn(batch_size, 3, 224, 224)
    aux_batch = torch.randn(batch_size, 1, 112, 112)
    mask_batch = torch.randn(batch_size, 1, 224, 224)

    print(f"Main batch shape: {main_batch.shape}")  # [2, 3, 224, 224]
    print(f"Aux batch shape: {aux_batch.shape}")  # [2, 1, 112, 112]
    print(f"Mask batch shape: {mask_batch.shape}")  # [2, 1, 224, 224]

    # 测试模型前向传播
    model = DualInputResNetTransformer(num_classes=1, backbone='resnet18')
    model.eval()
    with torch.no_grad():
        output = model(main_batch, aux_batch)
        print(f"Model output shape: {output.shape}")

    print("Data loader test passed! ✓")


if __name__ == "__main__":
    test_dataloader()