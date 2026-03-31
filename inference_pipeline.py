import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
from torchvision import models, transforms
from PIL import Image
from sqlalchemy import create_engine
from dotenv import load_dotenv
import urllib.parse

"""
best_fusioin_model.pt, best_fusion_scaler.pkl를 불러와 
새로운 데이터에 대한 정확도를 추론하는 파일
"""

# --------------------------------------------------
# 모델 아키텍처
# --------------------------------------------------
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
    
# --------------------------------------------------
# 추론기 (Predictor) 클래스
# --------------------------------------------------
class DefectPredictor:
    def __init__(self, model_path, scaler_path, num_features):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 모델 로드
        self.model = MultimodalFusionModel(num_tabular_features=num_features).to(self.device)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"모델 가중치 파일을 찾을 수 없습니다: {model_path}")
        
        self.model.load_state_dict(torch.load(model_path, weights_only=True, map_location=self.device))
        self.model.eval()   # 추론 모드 전환 (필수)

        # 스케일러 로드
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"스케일러 파일을 찾을 수 없습니다: {scaler_path}")
        self.scaler = joblib.load(scaler_path)

        # 이미지 전처리 (데이터 증강 없이 정규화만 수행)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict_batch(self, new_tabularr_data, new_image_paths):
        # 새로운 수치형 데이터(numpy array)와 이미지 경로 리스트를 받아 예측 결과를 반환합니다
        if len(new_tabularr_data) != len(new_image_paths):
            raise ValueError("수치형 데이터의 개수와 이미지 경로의 개수가 일치하지 않습니다")
        
        # 수치형 데이터 스케일링
        scaled_tab_data = self.scaler.transform(new_tabularr_data)
        tab_tensor = torch.tensor(scaled_tab_data, dtype=torch.float32).to(self.device)

        # 이미지 데이터 로드 및 텐서 변환
        image_tensors = []
        valid_indices = []

        for idx, img_path in enumerate(new_image_paths):
            if not os.path.exists(img_path):
                print(f"예측 건너뜀 - 이미지를 찾을 수 없음: {img_path}")
                continue

            image = Image.open(img_path).convert('RGB')
            img_tensor = self.transform(image)
            image_tensors.append(img_tensor)
            valid_indices.append(idx)

        if not image_tensors:
            print("유효한 이미지 데이터가 없어 예측을 종료합니다.")
            return []
        
        image_tensors = torch.stack(image_tensors).to(self.device)
        valid_tab_tensor = tab_tensor[valid_indices]

        # 모델 추론
        results = []
        with torch.no_grad():   # 메모리 절약 및 기울기 계산 방지
            outputs = self.model(image_tensors, valid_tab_tensor)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            preds = (probs >= 0.5).astype(int)

            for i, idx in enumerate(valid_indices):
                results.append({
                    'image_path': new_image_paths[idx],
                    'probability': float(probs[i]),
                    'prediction': 'error' if preds[i] == 1 else 'normal',
                    'original_index': idx
                })
        
        return results
    
# --------------------------------------------------
# 실행
# --------------------------------------------------
if __name__ == "__main__":
    load_dotenv()
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = os.getenv("DB_PORT")
    DB_NAME = os.getenv("DB_NAME")

    safe_password = urllib.parse.quote_plus(DB_PASSWORD)
    db_url = f"mysql+pymysql://{DB_USER}:{safe_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(db_url)

    # 1. DB에서 추론할 새로운 데이터 가져오기 (예: 최근 5건 테스트)
    query = "SELECT * FROM AI_PROC_PREVALUE"
    df = pd.read_sql(query, engine)

    if df.empty:
        print("추론할 데이터가 DB에 없습니다.")
    else:
        # 2. 학습 시 제외했던 컬럼들 (치트키 2개 포함) 완벽히 동일하게 제외
        drop_cols = [
            'PPID', 'LOTID', 'DATIME', 'MODITIME', 'FILENAME', 'label', 'REMARK', 'ISERROR',
            'WELD_CURR_VAR', 'WELD_CURR_MAX'
        ]
        
        # 실제 데이터프레임에 존재하는 컬럼만 안전하게 필터링
        actual_drop_cols = [col for col in drop_cols if col in df.columns]
        feature_cols = [col for col in df.columns if col not in actual_drop_cols]

        new_tabular_data = df[feature_cols].values
        num_features = len(feature_cols)

        # 3. 이미지 경로 매핑
        new_image_paths = []
        for filename in df['FILENAME']:
            clean_name = str(filename).strip()
            if not clean_name.lower().endswith('.jpg'):
                clean_name += '.jpg'
            target_path = os.path.join("D:\\SEOUL\\vision_folder", clean_name).replace("\\", "/")
            new_image_paths.append(target_path)

        # 4. 예측기 초기화 (저장해둔 모델과 스케일러 파일명 지정)
        # ※ 주의: 앞서 학습 단계에서 스케일러를 'best_fusion_scaler.pkl'로 저장했다고 가정합니다.
        predictor = DefectPredictor(
            model_path=r'C:/Users/wpsol/wb/seoul/best_fusion_model.pt', # 가장 최근 학습된 모델 파일명으로 수정 가능
            scaler_path=r'C:/Users/wpsol/wb/seoul/best_fusion_scaler.pkl',
            num_features=num_features
        )

        # 5. 예측 수행
        predictions = predictor.predict_batch(new_tabular_data, new_image_paths)

        # 6. 결과 출력
        print("\n" + "="*50)
        print("[새로운 데이터 예측 결과]")
        print("="*50)
        for res in predictions:
            idx = res['original_index']
            lot_id = df.iloc[idx]['LOTID']
            print(f"▶ Lot ID: {lot_id}")
            print(f"▶ 이미지: {os.path.basename(res['image_path'])}")
            print(f"▶ 불량 확률: {res['probability'] * 100:.2f}%")
            print(f"▶ 최종 판정: {res['prediction'].upper()}")
            print("-" * 50)
