import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os
import glob
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from Dual_UNet_Channel_Merge_Resnet50_adaptive import DualInputResNetTransformer
from dateloader import create_data_loaders
from tqdm import tqdm
class DualInputPredictor:
    """双输入模型预测器"""

    def __init__(self, model_path, device='cuda', threshold=0.5):
        """
        Args:
            model_path: 训练好的模型路径
            device: 推理设备
            threshold: 二值化阈值
        """
        self.device = device
        self.threshold = threshold

        # 加载模型
        self.model = DualInputResNetTransformer(num_classes=1, backbone='resnet50', fusion_type='adaptive', is_transformer=True)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()

        # 定义预处理变换
        self.main_transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

        self.aux_transform = A.Compose([
            A.Resize(112, 112),
            A.Normalize(mean=[0.5], std=[0.5]),
            ToTensorV2(),
        ])

    def preprocess_images(self, main_image_path, aux_image_path):
        """预处理图像"""
        # 读取RGB图像
        main_image = Image.open(main_image_path).convert('RGB')
        main_array = np.array(main_image)

        # 读取单波段图像
        aux_image = Image.open(aux_image_path).convert('L')
        aux_array = np.array(aux_image)

        # 应用变换
        main_tensor = self.main_transform(image=main_array)['image']
        aux_tensor = self.aux_transform(image=aux_array)['image']

        # 添加batch维度
        main_tensor = main_tensor.unsqueeze(0)
        aux_tensor = aux_tensor.unsqueeze(0)

        return main_tensor, aux_tensor, main_array

    def predict_single(self, main_image_path, aux_image_path, output_path=None):
        """单张图像预测"""
        # 预处理
        main_tensor, aux_tensor, original_main = self.preprocess_images(main_image_path, aux_image_path)

        # 移动到设备
        main_tensor = main_tensor.to(self.device)
        aux_tensor = aux_tensor.to(self.device)

        # 预测
        with torch.no_grad():
            output = self.model(main_tensor, aux_tensor)
            prediction = (output > self.threshold).float()

        # 转换为numpy并调整尺寸
        prediction_np = prediction.squeeze().cpu().numpy()

        # 如果需要，调整到原始尺寸
        if original_main.shape[:2] != prediction_np.shape:
            # 使用PIL进行高质量上采样
            pred_pil = Image.fromarray((prediction_np * 255).astype(np.uint8))
            pred_pil = pred_pil.resize((original_main.shape[1], original_main.shape[0]), Image.BILINEAR)
            prediction_np = np.array(pred_pil) / 255.0

        # 保存结果
        if output_path:
            self.save_prediction(original_main, prediction_np, output_path)

        return prediction_np

    def predict_batch(self, main_dir, aux_dir, output_dir, file_pattern='*.png'):
        """批量预测"""
        os.makedirs(output_dir, exist_ok=True)

        # 获取文件列表
        main_files = sorted(glob.glob(os.path.join(main_dir, file_pattern)))
        aux_files = sorted(glob.glob(os.path.join(aux_dir, file_pattern)))

        assert len(main_files) == len(aux_files), "Number of main and aux files must match"

        results = []

        for main_path, aux_path in zip(tqdm(main_files, desc='Predicting'), aux_files):
            filename = os.path.basename(main_path)
            output_path = os.path.join(output_dir, f'pred_{filename}')

            # 预测
            prediction = self.predict_single(main_path, aux_path, output_path)
            results.append((filename, prediction))

            print(f'Processed: {filename}')

        return results

    def save_prediction(self, original_image, prediction, output_path):
        """保存预测结果"""
        # 创建可视化结果
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 原始图像
        axes[0].imshow(original_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # 预测结果
        axes[1].imshow(prediction, cmap='gray')
        axes[1].set_title('Prediction')
        axes[1].axis('off')

        # 叠加显示
        axes[2].imshow(original_image)
        axes[2].imshow(prediction, cmap='jet', alpha=0.5)
        axes[2].set_title('Overlay')
        axes[2].axis('off')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        # 单独保存二值化掩码
        mask_output_path = output_path.replace('.png', '_mask.png')
        mask_image = (prediction * 255).astype(np.uint8)
        Image.fromarray(mask_image).save(mask_output_path)


# 使用示例
def example_usage():
    """使用示例"""

    # 单张图像预测
    predictor = DualInputPredictor(
        model_path=r"D:\PingshuoAnalysis\pycode\deeplearning\Dual_Unet\checkpoints\Resnet50_adaptive_tranformer8_newdataset\best_model.pth",
        device='cuda' if torch.cuda.is_available() else 'cpu',
        threshold=0.5
    )

    # 批量预测
    results = predictor.predict_batch(
        main_dir=r"C:\Users\PC\Desktop\big_area2\all\folder1",
        aux_dir=r"C:\Users\PC\Desktop\big_area2\all\folder2",
        output_dir=r"C:\Users\PC\Desktop\big_area2\pre"
    )

    print(f"Processed {len(results)} images")


# 实时推理演示
class RealTimePredictor:
    """实时推理演示（适用于视频流或摄像头）"""

    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.model = DualInputResNetTransformer(num_classes=1, backbone='resnet50')
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()

        # 实时推理的简化变换（假设输入已经是正确尺寸）
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def preprocess_frame(self, main_frame, aux_frame):
        """预处理帧"""
        # 调整尺寸
        main_frame = Image.fromarray(main_frame).resize((224, 224))
        aux_frame = Image.fromarray(aux_frame).resize((112, 112))

        # 转换为tensor
        main_tensor = self.transform(main_frame).unsqueeze(0)
        aux_tensor = torch.tensor(np.array(aux_frame), dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0

        return main_tensor, aux_tensor

    def predict_frame(self, main_frame, aux_frame, threshold=0.5):
        """预测单帧"""
        main_tensor, aux_tensor = self.preprocess_frame(main_frame, aux_frame)
        main_tensor = main_tensor.to(self.device)
        aux_tensor = aux_tensor.to(self.device)

        with torch.no_grad():
            output = self.model(main_tensor, aux_tensor)
            prediction = (output > threshold).float()

        return prediction.squeeze().cpu().numpy()


# 评估脚本
def evaluate_model(model_path, test_loader, device='cuda', threshold=0.5):
    """评估模型性能"""

    # 加载模型
    model = DualInputResNetTransformer(num_classes=1, backbone='resnet50')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    criterion = nn.BCELoss()
    test_loss = 0.0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for main_input, aux_input, mask in tqdm(test_loader, desc='Evaluating'):
            main_input = main_input.to(device)
            aux_input = aux_input.to(device)
            mask = mask.to(device)

            # 调整mask形状
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)

            output = model(main_input, aux_input)
            loss = criterion(output, mask)
            test_loss += loss.item()

            # 收集预测结果用于计算指标
            predictions = (output > threshold).float()
            all_predictions.append(predictions.cpu())
            all_targets.append(mask.cpu())

    test_loss /= len(test_loader)

    # 计算评估指标
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)

    # 计算IoU
    intersection = (all_predictions * all_targets).sum()
    union = all_predictions.sum() + all_targets.sum() - intersection
    iou = intersection / union if union > 0 else 0

    # 计算准确率
    accuracy = (all_predictions == all_targets).float().mean()

    print(f"Test Loss: {test_loss:.4f}")
    print(f"IoU: {iou:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

    return {
        'test_loss': test_loss,
        'iou': iou.item(),
        'accuracy': accuracy.item()
    }


if __name__ == "__main__":
    # 使用示例
    example_usage()