import os
import re
import pandas as pd
from sqlalchemy import create_engine, text
import numpy as np
import uuid
from dotenv import load_dotenv
from datetime import datetime
import urllib.parse

# .env 파일 로드
load_dotenv()

# 환경 변수에서 DB 정보 가져오기
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

# 필수 정보가 누락되었는지 체크하는 방어 로직
if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME]):
    raise ValueError("DB 연결을 위한 환경 변수가 누락되었습니다. .env 파일을 확인해주세요")

# AI_PROC_DAVALUE 데이터가 전처리 되어 AI_PROC_PREVALUE 테이블에 들어가는 함수
def calculate_and_update_lot_statistics():
    # SQLAlchemy 엔진 생성
    safe_password = urllib.parse.quote_plus(DB_PASSWORD)

    db_url = f"mysql+pymysql://{DB_USER}:{safe_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(db_url)

    query = """
        SELECT
            LOTID, DATIME, WELD_CURR, WELD_VOLT, WELD_TEMP, PAINT_PRESS, PAINT_TEMP, PAINT_HUMID
        FROM AI_PROC_DAVALUE
        ORDER BY LOTID, DATIME ASC
    """

    try:
        df = pd.read_sql(query, engine)
        if df.empty:
            print("처리할 데이터가 없습니다")
            return 
        
        # 사용자 정의 통계 함수 (변화율 계산)
        def calc_rate_of_change(series):
            if len(series) < 2 or series.iloc[0] == 0:
                return 0.0
            return ((series.iloc[-1] - series.iloc[0]) / series.iloc[0]) * 100
        
        # LotID 기준 그룹화 및 통계 연산
        # 연산 대상 컬럼
        sensor_cols = ['WELD_CURR', 'WELD_VOLT', 'WELD_TEMP', 'PAINT_PRESS', 'PAINT_TEMP', 'PAINT_HUMID']

        # Pandas NamedAgg 를 활용한 다중 통계량 계산
        agg_funcs = {
            col: ['mean', 'max', 'min', 'var', calc_rate_of_change] for col in sensor_cols
        }

        summary_df = df.groupby('LOTID').agg(agg_funcs).reset_index()

        # Pandas 집계 함수명을 DB 스키마 접미사로 매핑하기 위한 딕셔너리
        stat_mapping = {
            'mean': 'AVG',
            'max': 'MAX',
            'min': 'MIN',
            'var': 'VAR',
            'calc_rate_of_change': 'RT'
        }

        # 멀티인덱스 컬럼 단일화 
        summary_df.columns = ['LOTID'] + [f"{col}_{stat_mapping.get(stat, stat)}" for col, stat in summary_df.columns[1:]]

        # 결측치 처리
        summary_df = summary_df.fillna(0)

        # 필수 컬럼(일시) 추가
        summary_df['MODITIME'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')


        # DB에 계산 결과 업데이트 (AI_PROC_PREVALUE 테이블)
        try:
            update_cols = [col for col in summary_df.columns if col != 'LOTID']
            set_clause = ", ".join([f"{col} = :{col}" for col in update_cols])

            update_query = text(f"""
                UPDATE AI_PROC_PREVALUE
                SET {set_clause}
                WHERE LOTID = :LOTID
            """)

            data_to_update = summary_df.to_dict(orient='records')

            with engine.begin() as conn:
                conn.execute(update_query, data_to_update)

            print(f"{len(summary_df)}개의 LOTID 통계 데이터가 기존 행에 성공적으로 업데이트 되었습니다.")
                
        except Exception as e:
            print(f"DB 업데이트 중 오류 발생: {e}")
    
    except Exception as e:
        print(f"데이터 조회 중 오류 발생: {e}")


def update_iserror_from_filename():
    """
    AI_PROC_PREVALUE 테이블의 FILENAME을 읽어
    O.jpg로 끝나면 normal, X.jpg로 끝나면 error를 ISERROR 컬럼에 업데이트합니다.
    초기에만 모델학습을 위해 삽입
    """
    safe_password = urllib.parse.quote_plus(DB_PASSWORD)

    db_url = f"mysql+pymysql://{DB_USER}:{safe_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(db_url)

    update_query = text("""
        UPDATE AI_PROC_PREVALUE
        SET ISERROR = CASE
            WHEN FILENAME LIKE '%O.jpg' THEN 'normal'
            WHEN FILENAME LIKE '%X.jpg' THEN 'error'
            ELSE 'unknown'
        END
        WHERE FILENAME IS NOT NULL
        AND (ISERROR IS NULL OR ISERROR = '');            
    """)

    count_query = text("""
        SELECT ISERROR, COUNT(*) as cnt
        FROM AI_PROC_PREVALUE
        WHERE ISERROR IS NOT NULL AND ISERROR != ''
        GROUP BY ISERROR;
    """)

    try:
        # engine.begin()을 사용하여 트랜잭션 자동 커밋 및 에러 시 롤백 보장
        with engine.begin() as conn:
            result = conn.execute(update_query)
            print(f"ISERROR 컬럼 업데이트 완료. 반영된 행(ROW) 수: {result.rowcount}")

            # 전체 카운트 조회
            count_result = conn.execute(count_query).mappings().fetchall()

            print("\n[현재 AI_PROC_PREVALUE 테이블 ISERROR 총 누적 통계]")
            counts = {row['ISERROR']: row['cnt'] for row in count_result}

            print(f"- normal: {counts.get('normal', 0)}건")
            print(f"- error: {counts.get('error', 0)}건")
            print(f"- unknown: {counts.get('unknown', 0)}건")
    
    except Exception as e:
        print(f"ISERROR 컬럼 DB 업데이트 중 오류 발생: {e}")


if __name__ == "__main__":
    calculate_and_update_lot_statistics()
    update_iserror_from_filename()



