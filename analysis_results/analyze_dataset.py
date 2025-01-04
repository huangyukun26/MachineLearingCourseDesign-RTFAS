import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from torchvision.utils import make_grid
import seaborn as sns
import pandas as pd
from datetime import datetime

def save_log(content, filename='analysis_log.txt'):
    """保存分析日志"""
    with open(filename, 'a', encoding='utf-8') as f:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"\n=== {timestamp} ===\n")
        f.write(content)
        f.write("\n")

def load_datasets():
    """加载所有数据集"""
    data_dir = Path("data")
    datasets = {}
    
    # 加载原始训练集
    if (data_dir / 'new_trainingset.npy').exists():
        datasets['original_train'] = np.load(data_dir / 'new_trainingset.npy')
        print(f"Original training set shape: {datasets['original_train'].shape}")
    
    # 加载新的训练集（如果存在）
    if (data_dir / 'new_trainingset.npy').exists():
        datasets['new_train'] = np.load(data_dir / 'new_trainingset.npy')
        print(f"New training set shape: {datasets['new_train'].shape}")
    
    return datasets

def visualize_sample_images(dataset, num_samples=5, title="Sample Images", save_path=None):
    """可视化数据集中的样本图片"""
    # 随机选择样本
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    samples = dataset[indices]
    
    # 创建图像网格
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    for i, ax in enumerate(axes):
        ax.imshow(samples[i])
        ax.axis('off')
    plt.suptitle(title)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def analyze_pixel_distribution(dataset, save_path=None):
    """分析像素值分布"""
    # 计算每个图片的平均像素值和标准差
    means = np.mean(dataset, axis=(1,2,3))
    stds = np.std(dataset, axis=(1,2,3))
    
    # 创建分布图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 平均值分布
    sns.histplot(means, ax=ax1)
    ax1.set_title('Distribution of Mean Pixel Values')
    ax1.set_xlabel('Mean Pixel Value')
    
    # 标准差分布
    sns.histplot(stds, ax=ax2)
    ax2.set_title('Distribution of Pixel Standard Deviations')
    ax2.set_xlabel('Pixel Standard Deviation')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def analyze_dataset_differences(original_dataset, new_dataset, save_path=None):
    """分析原始数据集和新数据集的差异"""
    if len(original_dataset) == len(new_dataset):
        diff = new_dataset - original_dataset
        added_images = 0
    else:
        added_images = len(new_dataset) - len(original_dataset)
    
    analysis_text = [
        f"Dataset Analysis Report:",
        f"Original dataset size: {len(original_dataset)}",
        f"New dataset size: {len(new_dataset)}",
        f"Added images: {added_images}",
        f"Original dataset mean: {np.mean(original_dataset):.3f}",
        f"New dataset mean: {np.mean(new_dataset):.3f}",
        f"Original dataset std: {np.std(original_dataset):.3f}",
        f"New dataset std: {np.std(new_dataset):.3f}"
    ]
    
    # 保存分析结果
    save_log("\n".join(analysis_text))
    print("\n".join(analysis_text))

def main():
    # 创建保存目录
    save_dir = Path("analysis_results")
    save_dir.mkdir(exist_ok=True)
    
    # 加载数据集
    print("Loading datasets...")
    datasets = load_datasets()
    
    # 可视化原始训练集样本
    if 'original_train' in datasets:
        visualize_sample_images(
            datasets['original_train'],
            title="Original Training Set Samples",
            save_path=save_dir / "original_samples.png"
        )
        analyze_pixel_distribution(
            datasets['original_train'],
            save_path=save_dir / "original_distribution.png"
        )
    
    # 可视化新训练集样本
    if 'new_train' in datasets:
        visualize_sample_images(
            datasets['new_train'],
            title="New Training Set Samples",
            save_path=save_dir / "new_samples.png"
        )
        analyze_pixel_distribution(
            datasets['new_train'],
            save_path=save_dir / "new_distribution.png"
        )
    
    # 分析数据集差异
    if 'original_train' in datasets and 'new_train' in datasets:
        analyze_dataset_differences(
            datasets['original_train'],
            datasets['new_train']
        )

if __name__ == "__main__":
    main() 