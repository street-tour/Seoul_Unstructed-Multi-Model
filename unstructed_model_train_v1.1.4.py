import os
import time
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import multiprocessing
from torch.utils.data import DataLoader, Dataset
from sqlalchemy import create_engine, text
from torchvision import models, transforms
from datetime import datetime
from PIL import Image
import random
import re
import joblib
import traceback
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import urllib.parse

"""
.env 파일 없이 DB 접속 정보를 코드 내에 직접 명시한 버전
- 실행 파일 위치(BASE_DIR) 기준으로 모든 저장/로드 경로를 처리
- DB 접속 정보는 코드 내 상수로 직접 입력
- 외부 리소스(이미지 등)는 절대경로로 접근(기존과 동일)
"""

def get_base_dir():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    else:
        return os.path.dirname(os.path.abspath(__file__))

BASE_DIR = get_base_dir()

# DB 접속 정보 (직접 입력)
DB_USER = "****"
DB_PASSWORD = "****"
DB_HOST = "****"
DB_PORT = "****"
DB_NAME = "****"

# Focal Loss 정의 (확률 뭉침 해결용)
class FocalLoss(nn.Module):
    def __init__(self, pos_weight=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
        self.gamma = gamma
    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

class MultimodalDataset(Dataset):
    def __init__(self, image_paths, tabular_data, labels, transform=None):
        self.image_paths = image_paths
        self.tabular_data = torch.tensor(tabular_data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
        self.transform = transform
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"DB에 기록된 이미지를 다음 경로에서 찾을 수 없습니다: {image_path}")
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image_tensor = self.transform(image)
        else:
            image_tensor = transforms.ToTensor()(image)
        if self.transform is not None and random.random() < 0.10:
            image_tensor = torch.zeros_like(image_tensor)
        return image_tensor, self.tabular_data[idx], self.labels[idx]

class MultimodalFusionModel(nn.Module):
    def __init__(self, num_tabular_features):
        super(MultimodalFusionModel, self).__init__()
        self.vision_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.vision_model.fc = nn.Identity()
        self.tabular_model = nn.Sequential(
            nn.Linear(num_tabular_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
    def forward(self, image, tab_data):
        img_features = self.vision_model(image)
        tab_features = self.tabular_model(tab_data)
        combined = torch.cat((img_features, tab_features), dim=1)
        return self.classifier(combined)

class DefectDetectionTrainer:
    def __init__(self, db_engine):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.engine = db_engine
        self.scaler = StandardScaler()
    def load_data_from_db(self):
        query = "SELECT * FROM AI_PROC_PREVALUE"
        df = pd.read_sql(query, self.engine)
        if df.empty:
            raise ValueError("학습할 데이터가 DB에 없습니다")
        df = df[df['ISERROR'].isin(['정상', '불량'])]
        df['label'] = df['ISERROR'].map({'정상': 0, '불량': 1})
        labels = df['label'].values
        drop_cols = [
            'PPID', 'LOTID', 'DATIME', 'MODITIME', 'FILENAME', 'label', 'REMARK', 'ISERROR',
            'WELD_CURR_VAR', 'WELD_CURR_MAX']
        feature_cols = [col for col in df.columns if col not in drop_cols]
        tabular_data = df[feature_cols].values
        image_paths = []
        for filename in df['FILENAME']:
            clean_name = str(filename).strip()
            if not clean_name.lower().endswith('.jpg'):
                clean_name += '.jpg'
            target_path = os.path.join("D:\\SEOUL\\vision_folder", clean_name).replace("\\", "/")
            image_paths.append(target_path)
        X_tab_train, X_tab_test, y_train, y_test, img_train, img_test = train_test_split(
            tabular_data, labels, image_paths, test_size=0.2, random_state=42, stratify=labels
        )
        return (X_tab_train, X_tab_test, y_train, y_test, img_train, img_test), len(feature_cols)
    def train_and_evaluate(self, data_splits, num_features, model_name, epochs=100):
        X_tab_train, X_tab_test, y_train, y_test, img_train, img_test = data_splits
        X_tab_train_scaled = self.scaler.fit_transform(X_tab_train)
        X_tab_test_scaled = self.scaler.transform(X_tab_test)
        train_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        test_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        train_dataset = MultimodalDataset(img_train, X_tab_train_scaled, y_train, transform=train_transforms)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
        test_dataset = MultimodalDataset(img_test, X_tab_test_scaled, y_test, transform=test_transforms)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
        model = MultimodalFusionModel(num_tabular_features=num_features).to(self.device)
        num_positives = np.sum(y_train == 1)
        num_negatives = np.sum(y_train == 0)
        weight_ratio = num_negatives / max(1, num_positives)
        pos_weight = torch.tensor([weight_ratio], dtype=torch.float32).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        vision_params = model.vision_model.parameters()
        tabular_params = model.tabular_model.parameters()
        classifier_params = model.classifier.parameters()
        best_model_path = os.path.join(BASE_DIR, f"{model_name}.pt")
        best_scaler_path = os.path.join(BASE_DIR, f"{model_name}_scaler.pkl")
        val_recalls = []
        optimizer = optim.Adam([
            {'params': vision_params, 'lr': 1e-5},
            {'params': tabular_params, 'lr': 1e-3},
            {'params': classifier_params, 'lr': 1e-4}
        ], weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
        scaler = GradScaler()
        start_time = time.time()
        print(f"모델 학습을 시작합니다. (Device: {self.device}, Model: {model_name}, Epochs: {epochs}, Class Weight: {weight_ratio:.2f})")
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for step, (images, tabs, targets) in enumerate(train_loader):
                images, tabs, targets = images.to(self.device), tabs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                with torch.amp.autocast('cuda'):
                    outputs = model(images, tabs)
                    loss = criterion(outputs, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item()
                if (step + 1) % 50 == 0:
                    print(f" -> [진행 중] Epoch {epoch+1} - 배치 [{step+1}/{len(train_loader)}] Loss: {loss.item():.4f}")
            avg_train_loss = train_loss / len(train_loader)
            model.eval()
            val_loss = 0.0
            epochs_preds = []
            epoch_actuals = []
            with torch.no_grad():
                for images, tabs, targets in test_loader:
                    images, tabs, targets = images.to(self.device), tabs.to(self.device), targets.to(self.device)
                    outputs = model(images, tabs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    probs = torch.sigmoid(outputs).cpu().numpy()
                    preds = (probs >= 0.5).astype(int).flatten()
                    epochs_preds.extend(preds)
                    epoch_actuals.extend(targets.cpu().numpy().flatten())
            avg_val_loss = val_loss / len(test_loader)
            scheduler.step(avg_val_loss)
            epoch_recall = recall_score(epoch_actuals, epochs_preds, zero_division=0)
            val_recalls.append(epoch_recall)
            print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n=> 검증 손실(Val Loss)이 {patience}회 이상 개선되지 않아 조기 종료(Early Stopping)합니다.")
                    break
        print("\n최적의 가중치(Best Model)를 불러와 최종 평가를 진행합니다...")
        model.load_state_dict(torch.load(best_model_path))
        model.eval()
        predictions = []
        actuals = []
        with torch.no_grad():
            for images, tabs, targets in test_loader:
                images, tabs = images.to(self.device), tabs.to(self.device)
                outputs = model(images, tabs)
                probs = torch.sigmoid(outputs).cpu().numpy()
                preds = (probs >= 0.5).astype(int).flatten()
                predictions.extend(preds)
                actuals.extend(targets.cpu().numpy().flatten())
        end_time = time.time()
        train_duration = round(end_time - start_time, 2)
        recall = recall_score(actuals, predictions, zero_division=0)
        precision = precision_score(actuals, predictions, zero_division=0)
        print(f"\n[검증 결과]")
        print(f"Recall (재현율): {recall:.4f}")
        print(f"Precision (정밀도): {precision:.4f}")
        print(f"학습 소요 시간: {train_duration}초")
        joblib.dump(self.scaler, best_scaler_path)
        self.save_recall_plot(val_recalls, model_name)
        return recall, train_duration
    def save_recall_plot(self, val_recalls, train_id):
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(val_recalls) + 1)
        plt.plot(epochs, val_recalls, label='Validation Recall', color='green', marker='o', linestyle='-')
        plt.title(f'Model Recall Progression - {train_id}')
        plt.xlabel('Epochs')
        plt.ylabel('Recall Score')
        plt.ylim([-0.05, 1.05])
        plt.legend()
        plt.grid(True, alpha=0.3)
        file_name = os.path.join(BASE_DIR, f"recall_{train_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(file_name)
        plt.close()
        print(f"=> Recall 추이 그래프 생성 완료: {file_name}")
    def save_metric_to_db(self, train_id, model_name, datime_str, train_duration, recall):
        try:
            query = text("""
                INSERT INTO AI_UNSTRUCTED_TRAIN_INFO (TRAINID, MODELNAME, DATIME, TRAINTIME, RECALL)
                VALUES (:train_id, :model_name, :datime, :train_duration, :recall)             
            """)
            with self.engine.begin() as conn:
                conn.execute(query, {
                    "train_id": train_id,
                    "model_name": model_name,
                    "datime": datime_str,
                    "train_duration": float(train_duration),
                    "recall": float(recall)
                })
            print(f"=> DB 저장 완료 | TRAINID: {train_id}, RECALL: {recall:.4f}, MODELNAME: {model_name}")
        except Exception as e:
            print(f"DB 저장 중 오류 발생: {e}")

def run_pipeline():
    safe_password = urllib.parse.quote_plus(DB_PASSWORD)
    db_url = f"mysql+pymysql://{DB_USER}:{safe_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(db_url)
    now = datetime.now()
    train_id = now.strftime("%y%m%d-%H%M")
    model_name = f"{train_id}-multi"
    datime_str = now.strftime("%Y-%m-%d %H:%M:%S.000")
    trainer = DefectDetectionTrainer(engine)
    data_splits, num_features = trainer.load_data_from_db()
    recall, train_duration = trainer.train_and_evaluate(data_splits, num_features, model_name)
    trainer.save_metric_to_db(train_id, model_name, datime_str, train_duration, recall)
    print("\n [SUCCESS] 모델 학습 파이프라인이 정상적으로 완료되었습니다.")

if __name__ == "__main__":
    import sys
    try:
        run_pipeline()
        sys.exit(0)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)
