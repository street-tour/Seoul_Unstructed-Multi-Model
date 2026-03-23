import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from torch.utils.data import DataLoader, Dataset
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from torchvision import models, transforms
from datetime import datetime
from PIL import Image
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import urllib.parse

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

        if self.transform:
            image = self.transform(image)

        return image, self.tabular_data[idx], self.labels[idx]
    
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
        
        df = df[df['ISERROR'].isin(['normal', 'error'])]
        df['label'] = df['ISERROR'].map({'normal': 0, 'error': 1})

        labels = df['label'].values
        drop_cols = ['PPID', 'LOTID', 'DATIME', 'MODITIME', 'FILENAME', 'label', 'REMARK', 'ISERROR']
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

        return (X_tab_train, X_tab_test, y_train, y_test, img_train, img_test), len(feature_cols), feature_cols
    
    def train_and_evaluate(self, data_splits, num_features, epochs=30):
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
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

        test_dataset = MultimodalDataset(img_test, X_tab_test_scaled, y_test, transform=test_transforms)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

        # 모델 및 데이터로더 초기화
        model = MultimodalFusionModel(num_tabular_features=num_features).to(self.device)

        # 클래스 가중치 동적 계산
        num_positives = np.sum(y_train == 1)
        num_negatives = np.sum(y_train == 0)
        weight_ratio = num_negatives / max(1, num_positives)
        pos_weight = torch.tensor([weight_ratio], dtype=torch.float32).to(self.device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # 정규화 추가
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

        start_time = time.time()
        print(f"모델 학습을 시작합니다. (Device: {self.device}, Epochs: {epochs}, Class Weight: {weight_ratio:.2f})")

        # Early Stopping
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0
        best_model_path = "best_fusion_model.pt"


        for epoch in range(epochs):
            # --- 학습(Train) 루프 ---
            model.train()
            train_loss = 0.0
            for images, tabs, targets in train_loader:
                images, tabs, targets = images.to(self.device), tabs.to(self.device), targets.to(self.device)

                optimizer.zero_grad()
                outputs = model(images, tabs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)

            # --- 검증(Validation) 루프 ---
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, tabs, targets in test_loader:
                    images, tabs, targets = images.to(self.device), tabs.to(self.device), targets.to(self.device)
                    outputs = model(images, tabs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(test_loader)
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

        return recall, train_duration
    
    def save_metric_to_db(self, recall_value, train_duration):
        """AI_UNSTRUCTED_TRAIN_INFO 테이블에 저장합니다."""
        seq_name = self.get_next_sequence()

        result_df = pd.DataFrame([{
            'TRAINID': seq_name,
            'MODELNAME': seq_name,
            'DATIME': datetime.now(),
            'TRAINTIME': train_duration,
            'RECALL': float(recall_value),
            'REMARK': ''
        }])

        try:
            result_df.to_sql(name='AI_UNSTRUCTED_TRAIN_INFO', con=self.engine, if_exists='append', index=False)
            print(f"\n=> DB 저장 완료: {seq_name}")
        except Exception as e:
            print(f"DB 저장 중 오류 발생: {e}")

    def evaluate_saved_model(self, data_splits, num_features, model_path="best_fusion_model.pt"):
        """저장된 모델 가중치를 불러와 혼동행렬을 출력"""
        X_tab_train, X_tab_test, y_train, y_test, img_train, img_test = data_splits

        X_tab_train_scaled = self.scaler.fit_transform(X_tab_train)
        X_tab_test_scaled = self.scaler.transform(X_tab_test)

        test_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        test_dataset = MultimodalDataset(img_test, X_tab_test_scaled, y_test, transform=test_transforms)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

        # 모델 초기화 및 저장된 가중치 로드
        model = MultimodalFusionModel(num_tabular_features=num_features).to(self.device)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"저장된 모델 가중치({model_path})가 없습니다.")
        
        model.load_state_dict(torch.load(model_path))
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

        # Confusion Matrix 계산 및 출력
        cm = confusion_matrix(actuals, predictions)

        print("\n" + "="*40)
        print("[Confusion Matrix 결과]")
        print("="*40)
        # 클래스가 1개로 예측될 경우를 대비한 방어 로직
        if cm.shape == (2, 2):
            print(f"▶ True Negative  (정상 -> 정상 예측) : {cm[0][0]}건")
            print(f"▶ False Positive (정상 -> 불량 오답) : {cm[0][1]}건 (과탐지)")
            print(f"▶ False Negative (불량 -> 정상 오답) : {cm[1][0]}건 (미탐지)")
            print(f"▶ True Positive  (불량 -> 불량 예측) : {cm[1][1]}건")
        else:
            print("검증 데이터의 예측 결과가 단일 클래스로 치우쳐 행렬이 2x2로 생성되지 않았습니다.")
            print("현재 cm 구성:", cm)
        print("="*40)

    def analyze_feature_importance(self, data_splits, feature_cols, model_path="best_fusion_model.pt"):
        """순열 중요도(Permutation Importance)를 통해 수치형 피처의 영향력을 분석"""
        X_tab_train, X_tab_test, y_train, y_test, img_train, img_test = data_splits

        X_tab_train_scaled = self.scaler.fit_transform(X_tab_train)
        X_tab_test_scaled = self.scaler.transform(X_tab_test)

        test_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        model = MultimodalFusionModel(num_tabular_features=len(feature_cols)).to(self.device)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        criterion = nn.BCEWithLogitsLoss()

        # 원본 검증 데이터에 대한 Baseline Loss 계산
        baseline_dataset = MultimodalDataset(img_test, X_tab_test_scaled, y_test, transform=test_transforms)
        baseline_loader = DataLoader(baseline_dataset, batch_size=16, shuffle=False)

        baseline_loss = 0.0
        with torch.no_grad():
            for images, tabs, targets in baseline_loader:
                images, tabs, targets = images.to(self.device), tabs.to(self.device), targets.to(self.device)
                outputs = model(images, tabs)
                baseline_loss += criterion(outputs, targets).item()

        print(f"원본 검증 데이터 Loss (Baseline): {baseline_loss:.4f}")

        # 각 피처별로 순열 적용 후 Loss 변화 측정
        importances = []

        print("피처 중요도 계산")
        for i, col_name in enumerate(feature_cols):
            # 테스트 데이터 복사 후 i번째 컬럼만 무작위로 섞음
            X_tab_shuffled = copy.deepcopy(X_tab_test_scaled)
            np.random.shuffle(X_tab_shuffled[:, i])

            shuffled_dataset = MultimodalDataset(img_test, X_tab_shuffled, y_test, transform=test_transforms)
            shuffled_loader = DataLoader(shuffled_dataset, batch_size=16, shuffle=False)

            shuffled_loss = 0.0
            with torch.no_grad():
                for images, tabs, targets in shuffled_loader:
                    images, tabs, targets = images.to(self.device), tabs.to(self.device), targets.to(self.device)
                    outputs = model(images, tabs)
                    shuffled_loss += criterion(outputs, targets).item()

            # 중요도 = 섞었을 때의 Loss - 원본 Loss
            importance = shuffled_loss - baseline_loss
            importances.append(importance)

        # 결과 정리 및 터미널 출력
        importance_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance_Score': importances
        }).sort_values(by='Importance_Score', ascending=False)

        print("\n" + "="*50)
        print("[수치형 피처 순열 중요도 (Permutation Importance)]")
        print("="*50)
        print(importance_df.to_string(index=False))
        print("="*50)

        # 시각화 및 이미지 저장
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance_Score', y='Feature', data=importance_df, palette='viridis')
        plt.title('Feature Importance (Loss Increase after Permutation)', fontsize=14)
        plt.xlabel('Increase in BCE Loss', fontsize=12)
        plt.ylabel('Tabular Features', fontsize=12)
        plt.tight_layout()

        save_path = 'feature_importance.png'
        plt.savefig(save_path)

        return importance_df
                            

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
    data_splits, num_feature, feature_cols = trainer.load_data_from_db()

    # # 모델 학습 및 재현율 도출
    # recall, train_duration = trainer.train_and_evaluate(data_splits, num_features)

    # # 결과를 DB에 저장
    # trainer.save_metric_to_db(recall, train_duration)

    # # Confusion Matrix 실행
    # trainer.evaluate_saved_model(data_splits, num_features)


    trainer.analyze_feature_importance(data_splits, feature_cols)

if __name__ == "__main__":
    run_pipeline()
