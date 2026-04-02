import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
from torchvision import models, transforms
from PIL import Image
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from datetime import datetime
from sklearn.metrics import recall_score, precision_score
import urllib.parse

"""
가장 최근의 모델과 스케일러를 불러와 
새로운 데이터에 대한 불량을 추론하는 파일
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
        self.model.eval()   # 추론 모드 전환

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

    def predict_batch(self, new_tabular_data, new_image_paths, lot_ids):
        # 새로운 수치형 데이터(numpy array)와 이미지 경로 리스트를 받아 예측 결과를 반환합니다
        if len(new_tabular_data) != len(new_image_paths):
            raise ValueError("수치형 데이터의 개수와 이미지 경로의 개수가 일치하지 않습니다")
        
        # 수치형 데이터 스케일링
        scaled_tab_data = self.scaler.transform(new_tabular_data)
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
                    'LOTID': lot_ids[idx],
                    'image_path': new_image_paths[idx],
                    'probability': float(probs[i]),
                    'prediction': '불량' if preds[i] == 1 else '정상'
                })
        
        return results
    
# --------------------------------------------------
# DB 컨트롤러 함수
# --------------------------------------------------
def get_latest_model_info(engine):
    query = text("""
        SELECT MODELNAME
        FROM AI_UNSTRUCTED_TRAIN_INFO
        ORDER BY DATIME DESC
        LIMIT 1
    """)
    with engine.connect() as conn:
        result = conn.execute(query).fetchone()

    if result:
        model_name = result[0]
        return f"{model_name}.pt", f"{model_name}_scaler.pkl"
    else:
        raise ValueError("DB에 등록된 모델 학습 이력이 없습니다.")
    
def save_predictions_to_Db(engine, predictions):
    if not predictions:
        return
    
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.000")

    # 중복 검사 쿼리
    check_query = text("SELECT 1 FROM AI_DEFECT_PRED WHERE LOTID = :lot_id")

    # 삽입 쿼리
    insert_query = text("""
        INSERT INTO AI_DEFECT_PRED (PREDID, LOTID, DATIME, ISERROR)
        VALUES (:pred_id, :lot_id, :datime, :is_error)
    """)

    success_count = 0
    skip_count = 0

    with engine.begin() as conn:
        for res in predictions:
            lot_id = res['LOTID']

            # LOTID 중복 체크
            is_exist = conn.execute(check_query, {"lot_id": lot_id}).fetchone()

            if is_exist:
                print(f"중복 스킵: {lot_id}는 이미 DB에 존재합니다.")
                skip_count += 1
                continue

            # PREDID 채번 규칙 적용
            pred_id = f"PREDID_{lot_id}"

            # DB Insert
            try:
                conn.execute(insert_query, {
                    "pred_id": pred_id,
                    "lot_id": lot_id,
                    "datime": current_time,
                    "is_error": res['prediction']
                })
                success_count += 1
            except Exception as e:
                print(f"DB 적재 오류 (LOTID: {lot_id}): {e}")

    print(f"처리 완료: 총 {success_count}건 저장, {skip_count}건 스킵되었습니다.")

    
# --------------------------------------------------
# 실행 파이프라인
# --------------------------------------------------
def run_inference_pipeline():
    load_dotenv()
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = os.getenv("DB_PORT")
    DB_NAME = os.getenv("DB_NAME")

    safe_password = urllib.parse.quote_plus(DB_PASSWORD)
    db_url = f"mysql+pymysql://{DB_USER}:{safe_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(db_url)

    print("1. 최신 모델 정보 조회 중...")
    try:
        model_file, scaler_file = get_latest_model_info(engine)
        print(f"로드 예정 모델: {model_file}")
    except Exception as e:
        print(f"오류: {e}")
        return
    
    print("2. 평가 대상 데이터 로드 중...")
    query = """
        SELECT a.*, b.filepath
        FROM AI_PROC_PREVALUE a
        JOIN AI_VISION_DAVALUE b ON a.LOTID = b.LOTID
        WHERE a.ISERROR IS NULL OR a.ISERROR = ''
        LIMIT 100
    """
    df = pd.read_sql(query, engine)

    if df.empty:
        print("현재 추론을 대기 중인 신규 데이터가 없습니다.")
        return
    
    drop_cols = [
        'PPID', 'LOTID', 'DATIME', 'MODITIME', 'FILENAME', 'REMARK', 'ISERROR', 'FILEPATH',
        'WELD_CURR_VAR', 'WELD_CURR_MAX'
    ]

    actual_drop_cols = [col for col in drop_cols if col in df.columns]
    feature_cols = [col for col in df.columns if col not in actual_drop_cols]

    X_test_df = df[feature_cols].select_dtypes(include=[np.number])
    new_tabular_data = X_test_df.values
    num_features = X_test_df.shape[1]

    new_image_paths = df['filepath'].tolist()
    lot_ids = df['LOTID'].tolist()

    print("3. 모델 및 스케일러 초기화 중...")
    try:
        predictor = DefectPredictor(
            model_path=model_file,
            scaler_path=scaler_file,
            num_features=num_features
        )
    except FileNotFoundError as e:
        print(f"초기화 실패: {e}")
        return
    
    print("4. 추론 수행 중...")
    predictions = predictor.predict_batch(new_tabular_data, new_image_paths, lot_ids)

    print("\n=======================================")
    print(f" [추론 완료] 총 {len(predictions)}")
    print("=======================================")
    for res in predictions:
        print(f"LOTID: {res['LOTID']} | 예측: {res['prediction']}")

    print("\n5. DB 적재 수행 중...")
    save_predictions_to_Db(engine, predictions)
    print("모든 프로세스가 완료되었습니다.")

if __name__ == "__main__":
    run_inference_pipeline()