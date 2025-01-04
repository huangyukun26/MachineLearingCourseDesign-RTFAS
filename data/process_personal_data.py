import cv2
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from pathlib import Path

def load_and_process_images(personal_data_path):
    """加载并处理个人图片数据"""
    processed_images = []
    image_files = []
    
    #获取所有图片文件
    for img_file in os.listdir(personal_data_path):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_files.append(img_file)
    
    if not image_files:
        raise Exception("No image files found in the personal data directory!")
    
    print(f"Found {len(image_files)} images to process...")
    
    for img_file in image_files:
        img_path = os.path.join(personal_data_path, img_file)
        #使用PIL读取图片
        img = Image.open(img_path)
        #调整大小为64x64
        img = img.resize((64, 64))
        #转换为numpy数组
        img_array = np.array(img)
        
        #确保图片是RGB格式
        if len(img_array.shape) == 2:  #如果是灰度图
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.shape[2] == 4:  #如果是RGBA格式
            img_array = img_array[:, :, :3]
            
        processed_images.append(img_array)
        
    return np.array(processed_images), image_files

def visualize_processed_images(processed_images, image_files, save_path):
    """可视化处理后的图片，确保处理正确"""
    plt.figure(figsize=(15, 5))
    for i in range(min(5, len(processed_images))):
        plt.subplot(1, 5, i + 1)
        plt.imshow(processed_images[i])
        plt.title(f"Processed {image_files[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def merge_with_training_data(processed_images, training_data_path, output_path):
    """将处理后的图片与训练集合并"""
    print("Loading original training data...")
    original_train = np.load(training_data_path)
    print(f"Original training data shape: {original_train.shape}")
    print(f"Processed personal images shape: {processed_images.shape}")
    
    # 合并数据
    new_train = np.concatenate([original_train, processed_images], axis=0)
    print(f"New training data shape: {new_train.shape}")
    
    # 保存新的训练集
    np.save(output_path, new_train)
    print(f"Saved new training data to {output_path}")

def main():
    # 设置路径
    base_dir = Path("")
    personal_data_path = base_dir / "personalData"
    training_data_path = base_dir / "new_trainingset.npy"
    output_path = base_dir / "new_trainingset.npy"
    visualization_path = base_dir / "processed_images_check.png"
    
    try:
        # 1. 处理个人图片
        print("Processing personal images...")
        processed_images, image_files = load_and_process_images(personal_data_path)
        
        # 2. 可视化检查处理后的图片
        print("Generating visualization for verification...")
        visualize_processed_images(processed_images, image_files, visualization_path)
        print(f"Visualization saved to {visualization_path}")
        
        # 3. 合并数据
        print("Merging with training data...")
        merge_with_training_data(processed_images, training_data_path, output_path)
        
        print("Process completed successfully!")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    main() 