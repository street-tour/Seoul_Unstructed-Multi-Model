import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from torchvision import models, transforms
from datetime import datetime
from PIL import Image
import random
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from torch.cuda.amp import autocast, GradScaler
import urllib.parse


"""
unstructed_model_train_v1.1.py 에서 feature importance가 너무 높았던
'WELD_CURR_VAR', 'WELD_CURR_MAX' 두 개의 컬럼을 제외하고 학습 시킨 코드

+ 이미지에 초점을 둬서 학습을 진행하기 때문에, 이미지 중 30%는 가린채로 학습을 진행
+ 학습률(LR)을 분리시켜, 비전 모델은 미세 조정, 탭 모델은 보폭을 크게 집중학습
"""

class MultimodalDataset(Dataset):
    def __init__(self, image_paths, tabular_data, labels, transform=None):
        self.image_paths = image_paths
        self.tabular_data = torch.tensor(tabular_data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 실제 경로의 이미지를 불러옴
        image_path = self.image_paths[idx]
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"DB에 기록된 이미지를 다음 경로에서 찾을 수 없습니다")
        
        image = Image.open(image_path).convert('RGB')

        if self.transform is not None:
            image_tensor = self.transform(image)
        else:
            image_tensor = transforms.ToTensor()(image)

        if self.transform is not None and random.random() < 0.15:
            image_tensor = torch.zeros_like(image_tensor)

        return image_tensor, self.tabular_data[idx], self.labels[idx]
    
class MultimodalFusionModel(nn.Module):
    def __init__(self, num_tabular_features):
        super(MultimodalFusionModel, self).__init__()
        self.vision_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.vision_model.fc = nn.Identity()

        self.tabular_model = nn.Sequential(
            nn.Linear(num_tabular_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 + 64, 128),
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

    def get_next_sequence(self):
        """DB를 조회하여 UNSTRUCTED_MODEL_N 형태의 다음 번호를 생성"""
        try:
            query = "SELECT TRAINID FROM AI_UNSTRUCTED_TRAIN_INFO ORDER BY DATIME DESC LIMIT 1"
            df = pd.read_sql(query, self.engine)
            if df.empty:
                return "UNSTRUCTED_MODEL_1"
            
            last_id = df.iloc[0]['TRAINID']
            match = re.search(f'\d+', last_id)
            if match:
                next_num = int(match.group()) + 1
                return f"UNSTRUCTED_MODEL_{next_num}"
            else:
                return "UNSTRUCTED_MODEL_1"
        except Exception as e:
            return 

    def load_data_from_db(self):
        """DB에서 데이터를 가져와 학습/검증 데이터로 분리"""
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

        # 경로 병합 로직
        image_paths = []
        for filename in df['FILENAME']:
            clean_name = str(filename).strip()
            if not clean_name.lower().endswith('.jpg'):
                clean_name += '.jpg'
            target_path = os.path.join("D:\\SEOUL\\vision_folder", clean_name).replace("\\", "/")
            image_paths.append(target_path)
        
        # 학습(80%) 및 검증(20%) 데이터 분리
        X_tab_train, X_tab_test, y_train, y_test, img_train, img_test = train_test_split(
            tabular_data, labels, image_paths, test_size=0.2, random_state=42, stratify=labels
        )

        return (X_tab_train, X_tab_test, y_train, y_test, img_train, img_test), len(feature_cols)
    
    def train_and_evaluate(self, data_splits, num_features, epochs=100):
        """모델을 학습하고 검증 데이터로 Recall을 계산합니다"""
        X_tab_train, X_tab_test, y_train, y_test, img_train, img_test = data_splits

        # 학습 데이터 기준으로 스케일링
        X_tab_train_scaled = self.scaler.fit_transform(X_tab_train)
        X_tab_test_scaled = self.scaler.transform(X_tab_test)

        # 데이터 증강 분리
        # 학습용: 이미지를 살짝씩 변형하여 외우기 방지
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
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

        test_dataset = MultimodalDataset(img_test, X_tab_test_scaled, y_test, transform=test_transforms)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

        # 모델 및 데이터로더 초기화
        model = MultimodalFusionModel(num_tabular_features=num_features).to(self.device)

        # 클래스 가중치 동적 계산
        num_positives = np.sum(y_train == 1)
        num_negatives = np.sum(y_train == 0)
        weight_ratio = num_negatives / max(1, num_positives)
        pos_weight = torch.tensor([weight_ratio], dtype=torch.float32).to(self.device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # 모델의 각 부분별 파라미터 묶기
        vision_params = model.vision_model.parameters()
        tabular_params = model.tabular_model.parameters()
        classifier_params = model.classifier.parameters()

        # 각 파라미터 그룹마다 다른 학습률 부여
        optimizer = optim.Adam([
            {'params': vision_params, 'lr': 1e-5},      # 비전 모델: 미세조정
            {'params': tabular_params, 'lr': 1e-3},     # 탭 모델: 보폭을 크게 집중 학습
            {'params': classifier_params, 'lr': 1e-4}   # 분류기: 중간 밸런스 유지
        ], weight_decay=1e-4)

        # AMP Scaler 초기화
        scaler = GradScaler()

        start_time = time.time()
        print(f"모델 학습을 시작합니다. (Device: {self.device}, Epochs: {epochs}, Class Weight: {weight_ratio:.2f})")

        # Early Stopping
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0
        best_model_path = "best_fusion_model_10000_v2.pt"

        # 매 에포크의 Recall을 추적할 리스트
        val_recalls = []

        for epoch in range(epochs):
            # --- 학습(Train) 루프 ---
            model.train()
            train_loss = 0.0
            for images, tabs, targets in train_loader:
                images, tabs, targets = images.to(self.device), tabs.to(self.device), targets.to(self.device)

                optimizer.zero_grad()

                # autocast 블록 안에서 순전파 및 Loss 연산 진행
                with autocast():
                    outputs = model(images, tabs)
                    loss = criterion(outputs, targets)
                
                # Scaler를 통한 역전파 및 가중치 업데이트
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)

            # --- 검증(Validation) 루프 ---
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

            epoch_recall = recall_score(epoch_actuals, epochs_preds, zero_division=0)
            val_recalls.append(epoch_recall)

            print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

            # 조기 종료 및 베스트 모델 저장 로직
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path) # 최고 성능 모델 덮어쓰기
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n=> 검증 손실(Val Loss)이 {patience}회 이상 개선되지 않아 조기 종료(Early Stopping)합니다.")
                    break
        
        # 평가를 위해 가장 성능이 좋았던 베스트 가중치 로드
        print("\n최적의 가중치(Best Model)를 불러와 최종 평가를 진행합니다...")
        model.load_state_dict(torch.load(best_model_path))
        model.eval()

        # 모델 평가 (Recall 계산)
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
        train_duration = f"{end_time - start_time:.2f}s"

        # 실제 불량을 얼마나 잘 찾아냈는지 Recall 계산
        recall = recall_score(actuals, predictions, zero_division=0)
        precision = precision_score(actuals, predictions, zero_division=0)

        print(f"\n[검증 결과]")
        print(f"Recall (재현율): {recall:.4f}")
        print(f"Precision (정밀도): {precision:.4f}") # 0 근처라면 모델이 모든걸 다 불량이라고 찍은 것입니다
        print(f"학습 소요 시간: {train_duration}")

        joblib.dump(self.scaler, 'best_fusion_scaler_10000_v2.pkl')

        seq_name = self.get_next_sequence()
        self.save_recall_plot(val_recalls, seq_name)

        return recall, train_duration
    
    def save_recall_plot(self, val_recalls, train_id):
        """학습 진행에 따른 검증 recall 추이를 그래프로 저장."""
        plt.figure(figsize=(10, 6))

        # x축을 1부터 시작하게 만들기 위해 range 조정
        epochs = range(1, len(val_recalls) + 1)
        plt.plot(epochs, val_recalls, label='Validation Recall', color='green', marker='o', linestyle='-')
        
        plt.title(f'Model Recall Progression - {train_id}')
        plt.xlabel('Epochs')
        plt.ylabel('Recall Score')
        
        # Recall 값은 0.0 ~ 1.0 사이이므로 y축 범위를 고정하면 보기 편합니다
        plt.ylim([-0.05, 1.05])
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 프로젝트 폴더 내 저장
        file_name = f"recall_{train_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(file_name)
        plt.close()  # 메모리 해제
        print(f"=> Recall 추이 그래프 생성 완료: {file_name}")

def run_pipeline():
    load_dotenv()
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = os.getenv("DB_PORT")
    DB_NAME = os.getenv("DB_NAME")

    safe_password = urllib.parse.quote_plus(DB_PASSWORD)

    db_url = f"mysql+pymysql://{DB_USER}:{safe_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(db_url)

    trainer = DefectDetectionTrainer(engine)

    # 데이터 로드 및 분리
    data_splits, num_features = trainer.load_data_from_db()

    # 모델 학습 및 재현율 도출
    recall, train_duration = trainer.train_and_evaluate(data_splits, num_features)

    # 결과를 DB에 저장
    trainer.save_metric_to_db(recall, train_duration)

if __name__ == "__main__":
    run_pipeline()
