import torch
import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm
from Dual_Branch_ARFF_Trans import DualInputResNetTransformer
from dateloader import create_data_loaders
import numpy as np
import time
from nets.deeplabv3_plus import DeepLab



class EarlyStopping:
    """早停类"""

    def __init__(self, patience=10, verbose=False, delta=0, trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.trace_func = trace_func

    def __call__(self, val_loss, model, path):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss


def train_model(model, train_loader, val_loader, num_epochs=100, device='cuda',
                checkpoint_dir='checkpoints', patience=15, is_branch = True):
    """训练模型"""

    # 创建检查点目录
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 优化器和损失函数
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.BCELoss()

    # TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(checkpoint_dir, 'logs'))

    # 早停
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    model.to(device)
    best_val_loss = float('inf')

    # 训练历史记录
    train_history = []
    val_history = []

    # 全局进度条（epoch级别）
    epoch_pbar = tqdm(range(num_epochs), desc='Training Progress', position=0, leave=True)

    for epoch in epoch_pbar:
        start_time = time.time()

        # 训练阶段
        model.train()
        train_loss = 0.0
        train_samples = 0

        # batch级别进度条，设置最小更新间隔
        batch_pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}',
                          position=1, leave=False, mininterval=1.0)

        for main_input, aux_input, mask in batch_pbar:
            main_input = main_input.to(device)
            aux_input = aux_input.to(device)
            mask = mask.to(device)

            # 调整mask形状以匹配模型输出 [B, H, W] -> [B, 1, H, W]
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)

            optimizer.zero_grad()

            # 前向传播
            if is_branch:
                output = model(main_input, aux_input)
            else:
                output = model(main_input)

            loss = criterion(output, mask)

            # 反向传播
            loss.backward()
            optimizer.step()

            batch_size = main_input.size(0)
            train_loss += loss.item() * batch_size
            train_samples += batch_size

            # 更新batch进度条，减少更新频率
            batch_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'LR': f'{scheduler.get_last_lr()[0]:.2e}'
            })

        batch_pbar.close()
        train_loss /= train_samples

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_samples = 0

        val_pbar = tqdm(val_loader, desc='Validating', position=1, leave=False, mininterval=1.0)

        with torch.no_grad():

            for main_input, aux_input, mask in val_pbar:
                main_input = main_input.to(device)
                aux_input = aux_input.to(device)
                mask = mask.to(device)

                # 调整mask形状以匹配模型输出
                if mask.dim() == 3:
                    mask = mask.unsqueeze(1)
                if is_branch:
                    output = model(main_input, aux_input)
                else:
                    output = model(main_input)
                loss = criterion(output, mask)

                batch_size = main_input.size(0)
                val_loss += loss.item() * batch_size
                val_samples += batch_size

                val_pbar.set_postfix({'Val Loss': f'{loss.item():.4f}'})

        val_pbar.close()
        val_loss /= val_samples
        scheduler.step()

        epoch_time = time.time() - start_time

        # 记录历史
        train_history.append(train_loss)
        val_history.append(val_loss)

        # 记录到TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], epoch)

        # 更新epoch进度条
        epoch_pbar.set_postfix({
            'Train Loss': f'{train_loss:.4f}',
            'Val Loss': f'{val_loss:.4f}',
            'LR': f'{scheduler.get_last_lr()[0]:.2e}',
            'Time': f'{epoch_time:.1f}s'
        })

        print(f'\nEpoch {epoch + 1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss:.6f}')
        print(f'  Val Loss: {val_loss:.6f}')
        print(f'  LR: {scheduler.get_last_lr()[0]:.8f}')
        print(f'  Time: {epoch_time:.2f}s')

        # 每5个epoch保存一次检查点
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_history': train_history,
                'val_history': val_history
            }, checkpoint_path)
            print(f'  Saved checkpoint: {checkpoint_path}')

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f'  Saved best model with val_loss: {val_loss:.6f}')

        # 早停检查
        early_stopping(val_loss, model, os.path.join(checkpoint_dir, 'early_stop_best.pth'))
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break

    # 关闭进度条
    epoch_pbar.close()

    # 保存最终模型
    final_model_path = os.path.join(checkpoint_dir, 'final_model.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f'Saved final model: {final_model_path}')

    # 保存训练历史
    history = {
        'train_history': train_history,
        'val_history': val_history
    }
    torch.save(history, os.path.join(checkpoint_dir, 'training_history.pth'))

    writer.close()
    print("Training completed!")

    return train_history, val_history



def load_checkpoint(model, checkpoint_path, optimizer=None, scheduler=None, device='cuda'):
    """加载检查点继续训练"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Train loss: {checkpoint['train_loss']:.6f}")
    print(f"Val loss: {checkpoint['val_loss']:.6f}")

    return checkpoint['epoch'], checkpoint['train_history'], checkpoint['val_history']


# 使用示例
if __name__ == "__main__":
    CUDA_LAUNCH_BLOCKING = 1
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 创建模型
    model = DualInputResNetTransformer(num_classes=1, backbone='resnet50', fusion_type='ratio', enable_dual_branch= False, is_transformer=False)
    #model   = DeepLab(num_classes=1, backbone='mobilenet', downsample_factor=True, pretrained=True)

    # 假设数据目录结构
    # data/
    #   main/    # RGB图像 (224x224x3)
    #   aux/     # 单波段图像 (112x112x1)
    #   mask/    # 标签掩码 (224x224x1)
    # 创建数据加载器
    train_loader, val_loader = create_data_loaders(
        main_dir=r'D:\1024Dataset\train\folder1',
        aux_dir=r'D:\1024Dataset\train\folder2',
        mask_dir=r'D:\1024Dataset\train\folder3',
        batch_size=4
    )

    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    # 开始训练
    train_history, val_history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=200,
        device=device,
        checkpoint_dir='checkpoints',
        patience=15,  # 15个epoch没有改善则早停
        is_branch=False
    )
