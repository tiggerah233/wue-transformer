import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import json
from datetime import datetime

from config import Config
from models.multi_scale_fusion import MultiScaleFusionModel
from data_loader.data_processor import DataProcessor, MultiScaleDataset,Dataset

class PatchDataset(Dataset):
    """分块数据集，用于处理大尺寸图像的分块训练"""

    def __init__(self, data_processor, patch_size=(512, 512), stride=(256, 256)):
        self.data_processor = data_processor
        self.patch_size = patch_size
        self.stride = stride

        # 确保数据已经处理
        if not hasattr(data_processor, 'high_res_data'):
            raise ValueError("请先调用 DataProcessor 的 process_data() 方法")

        # 获取数据维度
        self.time_steps = data_processor.high_res_data.shape[0]
        self.height = data_processor.high_res_data.shape[2]
        self.width = data_processor.high_res_data.shape[3]

        # 计算patch数量
        self.patches_per_time = self._calculate_patches()
        self.total_patches = self.time_steps * self.patches_per_time

        print(
            f"分块数据集: 时间步={self.time_steps}, 每时间步patch数={self.patches_per_time}, 总patch数={self.total_patches}")
        print(f"Patch尺寸: {self.patch_size}, 步长: {self.stride}")

    def _calculate_patches(self):
        """计算每个时间步的patch数量"""
        patches_h = (self.height - self.patch_size[0]) // self.stride[0] + 1
        patches_w = (self.width - self.patch_size[1]) // self.stride[1] + 1
        return patches_h * patches_w

    def _get_patch_coordinates(self, patch_idx):
        """根据patch索引获取坐标"""
        patches_w = (self.width - self.patch_size[1]) // self.stride[1] + 1

        time_idx = patch_idx // self.patches_per_time
        patch_in_time = patch_idx % self.patches_per_time

        patch_h_idx = patch_in_time // patches_w
        patch_w_idx = patch_in_time % patches_w

        h_start = patch_h_idx * self.stride[0]
        w_start = patch_w_idx * self.stride[1]
        h_end = h_start + self.patch_size[0]
        w_end = w_start + self.patch_size[1]

        return time_idx, h_start, h_end, w_start, w_end

    def __len__(self):
        return self.total_patches

    def __getitem__(self, idx):
        if idx >= self.total_patches:
            raise IndexError(f"索引 {idx} 超出范围，数据集大小: {self.total_patches}")

        # 获取patch坐标
        time_idx, h_start, h_end, w_start, w_end = self._get_patch_coordinates(idx)

        # 提取patch数据
        high_res_patch = self.data_processor.high_res_data[time_idx, :, h_start:h_end, w_start:w_end]
        medium_res_patch = self.data_processor.medium_res_data[time_idx, :, h_start:h_end, w_start:w_end]
        low_res_patch = self.data_processor.low_res_data[time_idx, :, h_start:h_end, w_start:w_end]

        # 转换为tensor
        inputs = {
            'high_res': torch.FloatTensor(high_res_patch.astype(np.float32)),
            'medium_res': torch.FloatTensor(medium_res_patch.astype(np.float32)),
            'low_res': torch.FloatTensor(low_res_patch.astype(np.float32))
        }

        # 时间信息
        temporal_info = {
            'year': torch.LongTensor([self.data_processor.temporal_info['year'][time_idx]]),
            'month': torch.LongTensor([self.data_processor.temporal_info['month'][time_idx]])
        }

        # 目标数据（这里需要根据实际WUE数据的空间分布来处理）
        # 假设WUE数据也是空间分布的，与输入数据对齐
        if hasattr(self.data_processor, 'target_data') and len(self.data_processor.target_data.shape) > 1:
            # 如果target_data是空间数据
            target_patch = self.data_processor.target_data[time_idx, h_start:h_end, w_start:w_end]
            target = torch.FloatTensor(target_patch.astype(np.float32))
        else:
            # 如果target_data是点数据，使用对应时间步的第一个值
            target = torch.FloatTensor([self.data_processor.target_data[time_idx]])

        return inputs, temporal_info, target


class ChunkedMultiScaleDataset(Dataset):
    """分块多尺度数据集（如果需要的话）"""

    def __init__(self, data_processor, time_chunk_size=12):
        self.data_processor = data_processor
        self.time_chunk_size = time_chunk_size

        if not hasattr(data_processor, 'high_res_data'):
            raise ValueError("请先调用 DataProcessor 的 process_data() 方法")

        self.total_time_steps = data_processor.high_res_data.shape[0]
        self.num_chunks = (self.total_time_steps + self.time_chunk_size - 1) // self.time_chunk_size

        print(
            f"分块多尺度数据集: 总时间步={self.total_time_steps}, 分块大小={self.time_chunk_size}, 分块数={self.num_chunks}")

    def __len__(self):
        return self.num_chunks

    def __getitem__(self, idx):
        start_idx = idx * self.time_chunk_size
        end_idx = min(start_idx + self.time_chunk_size, self.total_time_steps)

        # 提取时间块数据
        high_res_chunk = self.data_processor.high_res_data[start_idx:end_idx]
        medium_res_chunk = self.data_processor.medium_res_data[start_idx:end_idx]
        low_res_chunk = self.data_processor.low_res_data[start_idx:end_idx]

        inputs = {
            'high_res': torch.FloatTensor(high_res_chunk.astype(np.float32)),
            'medium_res': torch.FloatTensor(medium_res_chunk.astype(np.float32)),
            'low_res': torch.FloatTensor(low_res_chunk.astype(np.float32))
        }

        # 时间信息
        temporal_info = {
            'year': torch.LongTensor(self.data_processor.temporal_info['year'][start_idx:end_idx]),
            'month': torch.LongTensor(self.data_processor.temporal_info['month'][start_idx:end_idx])
        }

        # 目标数据
        if hasattr(self.data_processor, 'target_data'):
            target_chunk = self.data_processor.target_data[start_idx:end_idx]
            target = torch.FloatTensor(target_chunk.astype(np.float32))
        else:
            target = torch.zeros(end_idx - start_idx)

        return inputs, temporal_info, target
from loss_functions.custom_loss import CustomLoss, MaskedMSELoss
from utils.metrics import calculate_metrics
from utils.visualization import plot_results
import gc
from torch.cuda import empty_cache as cuda_empty_cache


def train_model():
    # 初始化配置
    config = Config()

    # 确保输出目录存在
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # 数据处理
    print("开始数据处理...")
    processor = DataProcessor(config)
    processor.process_data()

    # 根据配置选择训练方式
    if config.USE_PATCH_TRAINING:
        print("使用分块训练模式...")
        model, history = train_with_patches(config, processor)
    else:
        print("使用完整数据训练模式...")
        model, history = train_with_full_data(config, processor)

    return model, history


def train_with_patches(config, processor):
    """使用分块训练"""
    # 在函数内部导入，避免循环导入问题
    from data_loader.data_processor import Dataset

    # 创建分块数据集
    dataset = PatchDataset(processor,
                           patch_size=config.PATCH_SIZE,
                           stride=config.STRIDE)
    # ... 其余代码保持不变
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # 初始化模型
    model = MultiScaleFusionModel(config).to(config.DEVICE)

    # 选择损失函数
    if processor.study_area_mask is not None and config.USE_STUDY_AREA_MASK:
        # 将掩膜转换为tensor并移动到设备
        mask_tensor = torch.FloatTensor(processor.study_area_mask.astype(np.float32)).to(config.DEVICE)
        criterion = MaskedMSELoss(mask=mask_tensor, reduction='mean')
        print("使用掩膜感知的损失函数")
    else:
        criterion = CustomLoss()
        print("使用标准损失函数")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS)

    # 训练循环
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'metrics': []}

    for epoch in range(config.NUM_EPOCHS):
        # 训练阶段
        model.train()
        train_loss = 0.0
        batch_count = 0

        for batch_idx, (inputs, temporal_info, targets) in enumerate(train_loader):
            # 移动到设备
            inputs = {k: v.to(config.DEVICE) for k, v in inputs.items()}
            temporal_info = {k: v.to(config.DEVICE) for k, v in temporal_info.items()}
            targets = targets.to(config.DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs, temporal_info)
            loss = criterion(outputs, targets)
            loss.backward()

            # 梯度裁剪
            if config.GRADIENT_CLIPPING:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.MAX_GRAD_NORM)

            optimizer.step()

            train_loss += loss.item()
            batch_count += 1

            # 定期清理内存
            if batch_idx % config.CLEANUP_INTERVAL == 0:
                gc.collect()
                if config.DEVICE.type == 'cuda':
                    cuda_empty_cache()

            if batch_idx % 50 == 0:
                print(
                    f'Epoch {epoch + 1}/{config.NUM_EPOCHS} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}')

        # 更新学习率
        scheduler.step()

        # 验证阶段
        model.eval()
        val_loss = 0.0
        all_outputs = []
        all_targets = []

        with torch.no_grad():
            for inputs, temporal_info, targets in val_loader:
                inputs = {k: v.to(config.DEVICE) for k, v in inputs.items()}
                temporal_info = {k: v.to(config.DEVICE) for k, v in temporal_info.items()}
                targets = targets.to(config.DEVICE)

                outputs = model(inputs, temporal_info)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

                all_outputs.append(outputs.cpu())
                all_targets.append(targets.cpu())

        # 计算指标
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = calculate_metrics(all_outputs, all_targets)

        # 记录历史
        avg_train_loss = train_loss / batch_count
        avg_val_loss = val_loss / len(val_loader)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['metrics'].append(metrics)

        print(
            f'Epoch {epoch + 1}/{config.NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}')
        print(f'Metrics: {metrics}')

        # 保存最佳模型
        if avg_val_loss < best_val_loss and config.SAVE_MODEL:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
                'config': config.__dict__
            }, os.path.join(config.OUTPUT_DIR, 'best_model.pth'))

            print(f"Saved best model with validation loss: {best_val_loss:.4f}")

        # 每个epoch结束后清理内存
        gc.collect()
        if config.DEVICE.type == 'cuda':
            cuda_empty_cache()

    # 保存训练历史
    with open(os.path.join(config.OUTPUT_DIR, 'training_history.json'), 'w') as f:
        # 将numpy数组转换为列表以便JSON序列化
        serializable_history = {
            'train_loss': [float(loss) for loss in history['train_loss']],
            'val_loss': [float(loss) for loss in history['val_loss']],
            'metrics': [
                {k: float(v) if isinstance(v, (np.floating, float)) else v
                 for k, v in metrics.items()}
                for metrics in history['metrics']
            ]
        }
        json.dump(serializable_history, f)

    # 可视化结果
    plot_results(history, config.OUTPUT_DIR)

    return model, history


def train_with_full_data(config, processor):
    """使用完整数据训练"""
    from data_loader.data_processor import MultiScaleDataset

    # 创建数据集
    dataset = MultiScaleDataset(processor)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # 初始化模型
    model = MultiScaleFusionModel(config).to(config.DEVICE)

    # 选择损失函数
    if processor.study_area_mask is not None and config.USE_STUDY_AREA_MASK:
        mask_tensor = torch.FloatTensor(processor.study_area_mask.astype(np.float32)).to(config.DEVICE)
        criterion = MaskedMSELoss(mask=mask_tensor, reduction='mean')
        print("使用掩膜感知的损失函数")
    else:
        criterion = CustomLoss()
        print("使用标准损失函数")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS)

    # 训练循环
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'metrics': []}

    for epoch in range(config.NUM_EPOCHS):
        # 训练阶段
        model.train()
        train_loss = 0.0

        for batch_idx, (inputs, temporal_info, targets) in enumerate(train_loader):
            inputs = {k: v.to(config.DEVICE) for k, v in inputs.items()}
            temporal_info = {k: v.to(config.DEVICE) for k, v in temporal_info.items()}
            targets = targets.to(config.DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs, temporal_info)
            loss = criterion(outputs, targets)
            loss.backward()

            # 梯度裁剪
            if config.GRADIENT_CLIPPING:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.MAX_GRAD_NORM)

            optimizer.step()

            train_loss += loss.item()

            # 定期清理内存
            if batch_idx % config.CLEANUP_INTERVAL == 0:
                gc.collect()
                if config.DEVICE.type == 'cuda':
                    cuda_empty_cache()

            if batch_idx % 50 == 0:
                print(
                    f'Epoch {epoch + 1}/{config.NUM_EPOCHS} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}')

        # 更新学习率
        scheduler.step()

        # 验证阶段
        model.eval()
        val_loss = 0.0
        all_outputs = []
        all_targets = []

        with torch.no_grad():
            for inputs, temporal_info, targets in val_loader:
                inputs = {k: v.to(config.DEVICE) for k, v in inputs.items()}
                temporal_info = {k: v.to(config.DEVICE) for k, v in temporal_info.items()}
                targets = targets.to(config.DEVICE)

                outputs = model(inputs, temporal_info)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

                all_outputs.append(outputs.cpu())
                all_targets.append(targets.cpu())

        # 计算指标
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = calculate_metrics(all_outputs, all_targets)

        # 记录历史
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['metrics'].append(metrics)

        print(
            f'Epoch {epoch + 1}/{config.NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}')
        print(f'Metrics: {metrics}')

        # 保存最佳模型
        if avg_val_loss < best_val_loss and config.SAVE_MODEL:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
                'config': config.__dict__
            }, os.path.join(config.OUTPUT_DIR, 'best_model.pth'))

            print(f"Saved best model with validation loss: {best_val_loss:.4f}")

        # 每个epoch结束后清理内存
        gc.collect()
        if config.DEVICE.type == 'cuda':
            cuda_empty_cache()

    # 保存训练历史
    with open(os.path.join(config.OUTPUT_DIR, 'training_history.json'), 'w') as f:
        # 将numpy数组转换为列表以便JSON序列化
        serializable_history = {
            'train_loss': [float(loss) for loss in history['train_loss']],
            'val_loss': [float(loss) for loss in history['val_loss']],
            'metrics': [
                {k: float(v) if isinstance(v, (np.floating, float)) else v
                 for k, v in metrics.items()}
                for metrics in history['metrics']
            ]
        }
        json.dump(serializable_history, f)

    # 可视化结果
    plot_results(history, config.OUTPUT_DIR)

    return model, history


def predict_with_trained_model(model_path, data_processor):
    """使用训练好的模型进行预测"""
    # 初始化配置
    config = Config()

    # 加载模型
    checkpoint = torch.load(model_path, map_location=config.DEVICE)

    # 初始化模型结构
    model = MultiScaleFusionModel(config).to(config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 根据配置选择数据集类型
    if config.USE_PATCH_TRAINING:
        from data_loader.data_processor import PatchDataset
        dataset = PatchDataset(data_processor,
                               patch_size=config.PATCH_SIZE,
                               stride=config.STRIDE)
    else:
        from data_loader.data_processor import MultiScaleDataset
        dataset = MultiScaleDataset(data_processor)

    data_loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # 进行预测
    predictions = []
    with torch.no_grad():
        for inputs, temporal_info, _ in data_loader:
            inputs = {k: v.to(config.DEVICE) for k, v in inputs.items()}
            temporal_info = {k: v.to(config.DEVICE) for k, v in temporal_info.items()}

            outputs = model(inputs, temporal_info)
            predictions.append(outputs.cpu())

    return torch.cat(predictions, dim=0)


if __name__ == '__main__':
    model, history = train_model()

    # 示例：如何使用训练好的模型进行预测
    # processor = DataProcessor(Config())
    # processor.process_data()
    # predictions = predict_with_trained_model('results/best_model.pth', processor)