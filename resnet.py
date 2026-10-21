import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# -------------------------- 全局配置（GPU优化版） --------------------------
class Config:
    # 数据路径
    basic_data_path = "data/basic_data"
    pred_data_path = "data/pred_data"
    pred_result_path = "pred_result.csv"
    model_save_path = "best_model.pth"
    
    # 数据参数
    img_size = 150
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15
    batch_size = 64
    num_classes = 6
    class_names = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    # GPU训练参数优化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 30
    lr = 1e-4
    weight_decay = 1e-5
    # 早停机制参数
    patience = 10
    min_delta = 1e-4
    
    # 启用cudnn加速
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

# -------------------------- 数据集类（适配GPU） --------------------------
class SceneDataset(Dataset):
    def __init__(self, img_paths, labels=None, transform=None, is_pred=False):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform
        self.is_pred = is_pred
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            img = self.transform(img)
        
        if self.is_pred:
            img_name = os.path.basename(img_path)
            return img, img_name
        else:
            label = self.labels[idx]
            return img, label

# -------------------------- 提取图片编号 --------------------------
def extract_pic_num(img_name):
    """从文件名中提取数字编号"""
    base_name = os.path.splitext(img_name)[0]
    num_str = ''.join(filter(str.isdigit, base_name))
    return int(num_str) if num_str else -1

# -------------------------- 数据预处理与加载（GPU优化） --------------------------
def load_and_split_data(config):
    img_paths = []
    labels = []
    for cls_name in config.class_names:
        cls_path = os.path.join(config.basic_data_path, cls_name)
        if not os.path.exists(cls_path):
            print(f"警告：类别目录不存在 - {cls_path}")
            continue
        for img_file in os.listdir(cls_path):
            if img_file.lower().endswith((".jpg", ".png", ".jpeg")):
                img_paths.append(os.path.join(cls_path, img_file))
                labels.append(config.class_to_idx[cls_name])
    
    # 数据划分
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        img_paths, labels, test_size=config.test_ratio, random_state=42, stratify=labels
    )
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels, 
        test_size=config.val_ratio/(config.train_ratio+config.val_ratio), 
        random_state=42, stratify=train_val_labels
    )
    
    # 数据增强
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((config.img_size, config.img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2)
    ])
    
    val_test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((config.img_size, config.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 数据加载器
    num_workers = min(os.cpu_count(), 4)
    train_dataset = SceneDataset(train_paths, train_labels, train_transform)
    val_dataset = SceneDataset(val_paths, val_labels, val_test_transform)
    test_dataset = SceneDataset(test_paths, test_labels, val_test_transform)
    
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )
    
    # 预测数据集
    pred_img_paths = []
    if os.path.exists(config.pred_data_path):
        for img_file in os.listdir(config.pred_data_path):
            if img_file.lower().endswith((".jpg", ".png", ".jpeg")):
                pred_img_paths.append(os.path.join(config.pred_data_path, img_file))
    pred_dataset = SceneDataset(pred_img_paths, transform=val_test_transform, is_pred=True)
    pred_loader = DataLoader(
        pred_dataset, batch_size=config.batch_size*2, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    print(f"数据加载完成！")
    print(f"训练集：{len(train_paths)}张，验证集：{len(val_paths)}张，测试集：{len(test_paths)}张，预测集：{len(pred_img_paths)}张")
    return train_loader, val_loader, test_loader, pred_loader

# -------------------------- 模型定义（ResNet18） --------------------------
def build_model(config):
    # 仅保留ResNet18模型
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # 冻结部分层，迁移学习
    for param in list(model.parameters())[:-10]:
        param.requires_grad = False
    # 替换分类头
    model.fc = nn.Linear(model.fc.in_features, config.num_classes)
    model = model.to(config.device)
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(config.device)
    optimizer = optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    return model, criterion, optimizer, scheduler

# -------------------------- 单轮训练/验证函数 --------------------------
def train_one_epoch(model, loader, criterion, optimizer, device, is_train=True):
    model.train() if is_train else model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    with tqdm(loader, desc="训练" if is_train else "验证") as pbar:
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            total_correct += (preds == labels).sum().item()
            total_samples += imgs.size(0)
            
            avg_loss = total_loss / total_samples
            avg_acc = total_correct / total_samples
            pbar.set_postfix({"loss": f"{avg_loss:.4f}", "acc": f"{avg_acc:.4f}"})
    
    return avg_loss, avg_acc

# -------------------------- 训练与评估（带早停机制） --------------------------
def train_evaluate_resnet(config, model, criterion, optimizer, scheduler, train_loader, val_loader, test_loader):
    best_val_acc = 0.0
    best_model_weights = None
    counter = 0  # 早停计数器
    
    print("开始训练ResNet18模型（GPU加速+早停机制）...")
    for epoch in range(config.epochs):
        print(f"\n=== 第 {epoch+1}/{config.epochs} 轮 ===")
        
        # 训练
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, config.device, is_train=True)
        
        # 验证
        with torch.no_grad():
            val_loss, val_acc = train_one_epoch(model, val_loader, criterion, optimizer, config.device, is_train=False)
        
        # 学习率调度
        scheduler.step()
        
        # 早停机制判断
        if val_acc > best_val_acc + config.min_delta:
            best_val_acc = val_acc
            best_model_weights = model.state_dict()
            torch.save(best_model_weights, config.model_save_path)
            print(f"✅ 保存最优模型（验证准确率：{val_acc:.4f}）")
            counter = 0  # 重置计数器
        else:
            counter += 1
            print(f"早停计数器：{counter}/{config.patience}（当前最佳验证准确率：{best_val_acc:.4f}）")
            if counter >= config.patience:
                print(f"早停机制触发！连续{config.patience}个epoch无提升，停止训练")
                break
        
        print(f"训练损失：{train_loss:.4f}，训练准确率：{train_acc:.4f}")
        print(f"验证损失：{val_loss:.4f}，验证准确率：{val_acc:.4f}")
    
    # 测试集评估（使用最佳模型）
    model.load_state_dict(torch.load(config.model_save_path))
    with torch.no_grad():
        test_loss, test_acc = train_one_epoch(model, test_loader, criterion, optimizer, config.device, is_train=False)
    print(f"\n=== 测试集最终评估 ===")
    print(f"测试损失：{test_loss:.4f}，测试准确率：{test_acc:.4f}")
    
    return model

# -------------------------- 预测与结果导出 --------------------------
def predict_and_export(config, model, loader):
    print(f"\n开始预测 {config.pred_data_path} 中的图片...")
    results = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="预测中"):
            imgs, img_names = batch
            imgs = imgs.to(config.device, non_blocking=True)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            preds = preds.cpu().numpy()
            
            for img_name, pred in zip(img_names, preds):
                pic_num = extract_pic_num(img_name)
                pred_cls_name = config.class_names[pred]
                results.append({
                    "pic_num": pic_num,
                    "文件名": img_name,
                    "predict_label": int(pred),
                    "类别名称": pred_cls_name
                })
    
    # 导出为CSV
    df = pd.DataFrame(results)
    df = df.sort_values("pic_num").reset_index(drop=True)
    for col in ["pic_num", "predict_label"]:
        if col not in df.columns:
            df[col] = -1
    df.to_csv(config.pred_result_path, index=False, encoding="utf-8-sig")
    print(f"预测结果已导出到：{config.pred_result_path}")
    print(f"共预测 {len(results)} 张图片，前5条结果：")
    print(df[["pic_num", "文件名", "predict_label", "类别名称"]].head())

# -------------------------- 主函数 --------------------------
def main():
    config = Config()
    print(f"实验配置：设备={config.device}，模型=resnet18，批次大小={config.batch_size}")
    if config.device.type == "cuda":
        print(f"GPU型号：{torch.cuda.get_device_name(0)}，CUDA版本：{torch.version.cuda}")
    
    # 固定随机种子
    seed = 42
    torch.manual_seed(seed)
    if config.device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    
    # 数据加载
    train_loader, val_loader, test_loader, pred_loader = load_and_split_data(config)
    
    # 构建模型
    model, criterion, optimizer, scheduler = build_model(config)
        
    if os.path.exists(config.model_save_path):
        print(f"加载已保存模型：{config.model_save_path}")
        model.load_state_dict(torch.load(config.model_save_path, map_location=config.device))
        model = model.to(config.device)
    else:
        model = train_evaluate_resnet(config, model, criterion, optimizer, scheduler, train_loader, val_loader, test_loader)
        
    predict_and_export(config, model, pred_loader)
    
    print("\n=== 所有流程完成！ ===")
    print(f"1. 模型已保存到：{config.model_save_path}")
    print(f"2. 预测结果已导出到：{config.pred_result_path}")

if __name__ == "__main__":
    main()