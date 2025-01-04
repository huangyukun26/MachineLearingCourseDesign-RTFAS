import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from models.dual_stream_autoencoder import DualStreamAutoencoder
from qqdm import qqdm, format_str
import matplotlib.pyplot as plt

def train_dual_stream_model(
    data_path,
    model_save_dir,
    num_epochs=50,
    batch_size=128,
    learning_rate=1e-4,
    device=None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载数据
    print("Loading enhanced training data...")
    train_data = np.load(data_path)
    train_data = torch.from_numpy(train_data).float()
    train_data = train_data.permute(0, 3, 1, 2)  # NHWC -> NCHW
    train_data = train_data / 255.0  # 归一化到[0,1]
    
    # 创建数据加载器
    train_dataset = torch.utils.data.TensorDataset(train_data)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # 初始化模型
    model = DualStreamAutoencoder().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # 训练循环
    best_loss = float('inf')
    train_losses = []
    
    print(f"Starting training on {device}...")
    for epoch in qqdm(range(num_epochs)):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            imgs = batch[0].to(device)
            
            # 前向传播
            reconstructed = model(imgs)
            loss = criterion(reconstructed, imgs)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # 计算平均损失
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, model_save_dir / 'best_dual_stream_model.pt')
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}')
    
    # 保存训练损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses)
    plt.title('Training Loss Over Time (Dual Stream)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(model_save_dir / 'dual_stream_training_loss.png')
    plt.close()
    
    print(f"Training completed! Best loss: {best_loss:.6f}")
    return model, device

if __name__ == "__main__":
    # 设置路径
    data_dir = Path("data")
    model_save_dir = Path("models")
    model_save_dir.mkdir(exist_ok=True)
    
    # 训练模型
    train_dual_stream_model(
        data_path=data_dir / "enhanced_trainingset.npy",
        model_save_dir=model_save_dir,
        num_epochs=50,
        batch_size=128,
        learning_rate=1e-4
    ) 