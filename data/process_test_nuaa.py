import os
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as transforms
import shutil

def split_nuaa_test_data(nuaa_path, test_ratio=0.2, seed=42):
    """从NUAA数据集中分离测试数据
    Args:
        nuaa_path: NUAA数据集根目录
        test_ratio: 测试集比例
        seed: 随机种子
    """
    np.random.seed(seed)
    
    # 创建测试数据目录
    test_dir = Path(nuaa_path) / 'test'
    test_dir.mkdir(exist_ok=True)
    (test_dir / 'ClientFace').mkdir(exist_ok=True)
    (test_dir / 'ImposterFace').mkdir(exist_ok=True)
    
    # 处理真实人脸数据
    client_dir = Path(nuaa_path) / 'ClientFace'
    for person_dir in client_dir.iterdir():
        if person_dir.is_dir() and not person_dir.name.startswith('.'):
            files = list(person_dir.glob('*.jpg'))
            n_test = int(len(files) * test_ratio)
            test_files = np.random.choice(files, n_test, replace=False)
            
            # 移动测试文件
            for f in test_files:
                test_path = test_dir / 'ClientFace' / person_dir.name
                test_path.mkdir(exist_ok=True)
                shutil.move(str(f), str(test_path / f.name))
    
    # 处理攻击样本数据
    imposter_dir = Path(nuaa_path) / 'ImposterFace'
    for person_dir in imposter_dir.iterdir():
        if person_dir.is_dir() and not person_dir.name.startswith('.'):
            files = list(person_dir.glob('*.jpg'))
            n_test = int(len(files) * test_ratio)
            test_files = np.random.choice(files, n_test, replace=False)
            
            # 移动测试文件
            for f in test_files:
                test_path = test_dir / 'ImposterFace' / person_dir.name
                test_path.mkdir(exist_ok=True)
                shutil.move(str(f), str(test_path / f.name))

def process_test_images(image_dir, output_size=(64, 64)):
    """处理测试图像
    Args:
        image_dir: 图像目录
        output_size: 输出图像大小
    Returns:
        processed_images: 处理后的图像数组
        labels: 标签数组
    """
    processed_images = []
    labels = []
    
    transform = transforms.Compose([
        transforms.Resize(output_size),
        transforms.ToTensor(),
    ])
    
    for img_path in Path(image_dir).rglob('*.jpg'):
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img)
            img_array = img_tensor.permute(1, 2, 0).numpy() * 255
            processed_images.append(img_array)
            
            # 根据路径确定标签
            label = 1 if 'ClientFace' in str(img_path) else 0
            labels.append(label)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    return np.array(processed_images), np.array(labels)

def create_test_sets(data_dir):
    """创建测试集
    Args:
        data_dir: 数据根目录
    """
    # 处理NUAA测试集
    nuaa_test_dir = data_dir / 'public_datasets/NUAA/test'
    nuaa_images, nuaa_labels = process_test_images(nuaa_test_dir)
    np.save(data_dir / 'nuaa_testset.npy', nuaa_images)
    np.save(data_dir / 'nuaa_testset_labels.npy', nuaa_labels)
    print(f"NUAA test set saved, shape: {nuaa_images.shape}")
    
    # 处理自定义测试集（如果存在）
    custom_test_dir = data_dir / 'custom_test'
    if custom_test_dir.exists():
        custom_images, custom_labels = process_test_images(custom_test_dir)
        np.save(data_dir / 'custom_testset.npy', custom_images)
        np.save(data_dir / 'custom_testset_labels.npy', custom_labels)
        print(f"Custom test set saved, shape: {custom_images.shape}")

if __name__ == "__main__":
    data_dir = Path(".")
    nuaa_dir = data_dir / "public_datasets/NUAA"
    
    # 分离NUAA测试数据
    split_nuaa_test_data(nuaa_dir)
    
    # 创建测试集
    create_test_sets(data_dir) 