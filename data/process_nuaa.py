import os
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as transforms

def process_nuaa_dataset(dataset_path, output_size=(64, 64, 3)):
    """处理NUAA数据集
    Args:
        dataset_path: NUAA数据集根目录
        output_size: 输出图像大小
    Returns:
        processed_images: 处理后的图像数组 (N, 64, 64, 3)
        labels: 标签数组
    """
    processed_images = []
    labels = []
    
    transform = transforms.Compose([
        transforms.Resize((output_size[0], output_size[1])),
        transforms.ToTensor(),  # 转换为(C,H,W)格式
    ])
    
    # 处理真实人脸（ClientFace）
    client_dir = Path(dataset_path) / "ClientFace"
    for img_path in client_dir.rglob('*.jpg'):
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img)
            # 转换为(H,W,C)格式并缩放到[0,255]
            img_array = img_tensor.permute(1, 2, 0).numpy() * 255
            processed_images.append(img_array)
            labels.append(1)  # 真实人脸标签为1
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    # 处理攻击样本（ImposterFace）
    imposter_dir = Path(dataset_path) / "ImposterFace"
    for img_path in imposter_dir.rglob('*.jpg'):
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img)
            # 转换为(H,W,C)格式并缩放到[0,255]
            img_array = img_tensor.permute(1, 2, 0).numpy() * 255
            processed_images.append(img_array)
            labels.append(0)  # 攻击样本标签为0
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    return np.array(processed_images), np.array(labels)

def merge_with_original(original_data_path, nuaa_data_path, output_path):
    """将NUAA数据集与原始数据集合并
    Args:
        original_data_path: 原始数据集路径
        nuaa_data_path: NUAA数据集路径
        output_path: 输出路径
    """
    # 加载原始数据集
    print("Loading original dataset...")
    original_data = np.load(original_data_path)
    print(f"Original dataset shape: {original_data.shape}")
    
    # 处理NUAA数据集
    print("Processing NUAA dataset...")
    nuaa_images, nuaa_labels = process_nuaa_dataset(nuaa_data_path)
    print(f"NUAA dataset shape: {nuaa_images.shape}")
    print(f"NUAA labels shape: {nuaa_labels.shape}")
    
    # 确保数据格式一致
    assert original_data.shape[1:] == nuaa_images.shape[1:], \
        f"Shape mismatch: original {original_data.shape} vs NUAA {nuaa_images.shape}"
    
    # 合并数据
    merged_data = np.concatenate([original_data, nuaa_images], axis=0)
    
    # 保存合并后的数据集
    np.save(output_path, merged_data)
    print(f"Merged dataset saved to {output_path}")
    print(f"Final dataset shape: {merged_data.shape}")

if __name__ == "__main__":
    # 设置路径
    data_dir = Path(".")
    nuaa_dir = data_dir / "public_datasets" / "NUAA"
    output_path = data_dir / "enhanced_trainingset.npy"
    
    # 处理并合并数据集
    merge_with_original(
        original_data_path=data_dir / "new_trainingset.npy",
        nuaa_data_path=nuaa_dir,
        output_path=output_path
    ) 