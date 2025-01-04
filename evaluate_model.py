import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sns
from models.dual_stream_autoencoder import DualStreamAutoencoder
import gc

def load_model(model_path, device):
    """加载训练好的模型"""
    model = DualStreamAutoencoder().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def calculate_reconstruction_error(model, images, device, batch_size=16):
    """分批计算重建误差"""
    total_errors = []
    n_samples = len(images)
    
    for i in range(0, n_samples, batch_size):
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 处理当前批次
        batch = images[i:min(i + batch_size, n_samples)]
        batch = torch.from_numpy(batch).float().to(device)
        batch = batch.permute(0, 3, 1, 2)  # NHWC -> NCHW
        batch = batch / 255.0  # 归一化
        
        with torch.no_grad():
            reconstructed = model(batch)
            errors = torch.mean((batch - reconstructed) ** 2, dim=(1, 2, 3))
            total_errors.append(errors.cpu().numpy())
        
        # 清理不需要的张量
        del batch, reconstructed
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return np.concatenate(total_errors)

def plot_error_distribution(errors, save_path, threshold=None):
    """绘制重建误差分布"""
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, density=True, alpha=0.7)
    if threshold is not None:
        plt.axvline(x=threshold, color='r', linestyle='--', 
                   label=f'Threshold: {threshold:.4f}')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Density')
    plt.title('Distribution of Reconstruction Errors')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_roc_curve(fpr, tpr, roc_auc, save_path):
    """绘制ROC曲线"""
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(cm, save_path):
    """绘制混淆矩阵"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(save_path)
    plt.close()

def evaluate_labeled_testset(model, test_images, test_labels, name, results_dir, device, batch_size=16):
    """评估有标签的测试集"""
    # 清理内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    # 计算重建误差
    errors = calculate_reconstruction_error(model, test_images, device, batch_size)
    
    # 计算ROC曲线
    fpr, tpr, thresholds = roc_curve(test_labels, errors)
    roc_auc = auc(fpr, tpr)
    
    # 选择最佳阈值（约登指数最大点）
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    # 使用最佳阈值进行预测
    predictions = (errors >= optimal_threshold).astype(int)
    cm = confusion_matrix(test_labels, predictions)
    
    # 保存结果
    results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True)
    
    # 绘制误差分布
    plot_error_distribution(errors, 
                          results_dir / f'error_dist_{name}.png',
                          optimal_threshold)
    
    # 绘制ROC曲线
    plot_roc_curve(fpr, tpr, roc_auc, 
                  results_dir / f'roc_curve_{name}.png')
    
    # 绘制混淆矩阵
    plot_confusion_matrix(cm, 
                        results_dir / f'confusion_matrix_{name}.png')
    
    # 计算指标
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * precision * recall / (precision + recall)
    
    # 保存指标
    metrics = {
        'AUC': roc_auc,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-score': f1_score,
        'Optimal Threshold': optimal_threshold
    }
    
    with open(results_dir / f'metrics_{name}.txt', 'w') as f:
        for metric, value in metrics.items():
            f.write(f'{metric}: {value:.4f}\n')
    
    return metrics, optimal_threshold

def evaluate_unlabeled_testset(model, test_images, name, results_dir, device, threshold, batch_size=16):
    """评估无标签的测试集"""
    # 清理内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    # 计算重建误差
    errors = calculate_reconstruction_error(model, test_images, device, batch_size)
    
    # 使用提供的阈值进行预测
    predictions = (errors >= threshold).astype(int)
    
    # 保存结果
    results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True)
    
    # 绘制误差分布
    plot_error_distribution(errors, 
                          results_dir / f'error_dist_{name}.png',
                          threshold)
    
    # 保存预测结果
    np.save(results_dir / f'predictions_{name}.npy', predictions)
    
    # 统计预测结果
    stats = {
        'Total Samples': len(predictions),
        'Predicted Real': np.sum(predictions == 0),
        'Predicted Fake': np.sum(predictions == 1),
        'Real Ratio': np.mean(predictions == 0),
        'Fake Ratio': np.mean(predictions == 1)
    }
    
    with open(results_dir / f'stats_{name}.txt', 'w') as f:
        for metric, value in stats.items():
            if isinstance(value, float):
                f.write(f'{metric}: {value:.4f}\n')
            else:
                f.write(f'{metric}: {value}\n')
    
    return stats

def main():
    # 设置GPU内存分配策略
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # 设置GPU内存分配比例
        torch.cuda.set_per_process_memory_fraction(0.8)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 设置批处理大小
    batch_size = 16  # 可以根据GPU内存大小调整
    
    # 设置路径
    data_dir = Path("data")
    model_path = Path("models") / "best_dual_stream_model.pt"
    results_dir = Path("evaluation_results")
    
    # 加载模型
    model = load_model(model_path, device)
    
    try:
        # 评估NUAA测试集（有标签）
        print("\nEvaluating NUAA test set...")
        nuaa_test = np.load(data_dir / "nuaa_testset.npy")
        nuaa_labels = np.load(data_dir / "nuaa_testset_labels.npy")
        nuaa_metrics, threshold = evaluate_labeled_testset(
            model, nuaa_test, nuaa_labels, 
            "nuaa", results_dir, device, batch_size
        )
        
        # 使用NUAA测试集得到的阈值评估原始测试集（无标签）
        print("\nEvaluating original test set...")
        original_test = np.load(data_dir / "testingset.npy")
        original_stats = evaluate_unlabeled_testset(
            model, original_test, "original", 
            results_dir, device, threshold, batch_size
        )
        
        # 评估自定义测试集（如果存在）
        custom_test_path = data_dir / "custom_testset.npy"
        if custom_test_path.exists():
            print("\nEvaluating custom test set...")
            custom_test = np.load(custom_test_path)
            custom_labels = np.load(data_dir / "custom_testset_labels.npy")
            custom_metrics, _ = evaluate_labeled_testset(
                model, custom_test, custom_labels, 
                "custom", results_dir, device, batch_size
            )
        
        # 打印总结
        print("\nEvaluation Results Summary:")
        print("\nNUAA Test Set Metrics:")
        for metric, value in nuaa_metrics.items():
            print(f"{metric}: {value:.4f}")
        
        print("\nOriginal Test Set Statistics:")
        for metric, value in original_stats.items():
            if isinstance(value, float):
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: {value}")
        
        if custom_test_path.exists():
            print("\nCustom Test Set Metrics:")
            for metric, value in custom_metrics.items():
                print(f"{metric}: {value:.4f}")
    
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("\nGPU内存不足，尝试减小batch_size或使用CPU进行评估")
            print("建议的batch_size值: 8 或 4")
        raise e
    
    finally:
        # 清理内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main() 