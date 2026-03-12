# =============================================================================
# 01_load_data.py
# 데이터 로드 및 기본 확인
#
# [분석 배경]
# 삼성화재 디지털 서비스 기획/운영 직무는 고객의 서비스 이용 데이터를
# 모니터링하고 이를 바탕으로 신규 서비스를 기획하는 업무를 수행한다.
# 실제 보험사 앱 고객 행동 데이터는 공개되어 있지 않으므로,
# 구조적으로 동일한 통신사 구독 고객 데이터(IBM Telco Churn)를 활용한다.
#
# [도메인 매핑]
#   통신사 월정액 구독  →  보험료 월납
#   서비스 해지(Churn)  →  보험 해약
#   고객센터 문의       →  민원/챗봇 이용
#   부가서비스 이용     →  특약/부가 보장 가입
#
# [데이터 출처]
#   IBM Telco Customer Churn Dataset
#   https://www.kaggle.com/datasets/blastchar/telco-customer-churn
#   - 7,043명의 통신사 가입자 정보 + 이탈 여부
#   - 21개 변수 (인구통계, 계약 정보, 서비스 이용, 납입 정보)
# =============================================================================

import os
import pandas as pd
import numpy as np

# 데이터 경로 설정 (이 파일 기준으로 상대경로)
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'Telco-Customer-Churn.csv')


# =============================================================================
# STEP 1. 데이터 로드 및 전처리
# =============================================================================

def load_data() -> pd.DataFrame:
    """
    CSV 파일을 읽고 기본 전처리를 수행한 DataFrame을 반환한다.

    전처리 항목:
    1. TotalCharges: 문자열 → float 변환
       (tenure=0인 신규 가입자는 청구 금액이 없어 공백으로 기록되어 있음)
    2. 공백으로 인한 결측치 행 제거 (11건, 전체의 0.16% — 분석에 영향 미미)
    3. Churn_binary: 타겟 변수 이진화 (Yes=1 이탈/해약, No=0 유지)
    """
    # CSV 직접 로드
    df = pd.read_csv(DATA_PATH)
    print(f"✅ 데이터 로드 완료: {DATA_PATH}")

    # TotalCharges: 공백 문자열 → NaN → float
    # errors='coerce'는 변환 불가 값을 NaN으로 처리하는 옵션
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # 결측치 제거 (신규 가입 직후 해약 케이스 — 이탈 패턴 분석 대상 아님)
    before = len(df)
    df = df.dropna(subset=['TotalCharges']).reset_index(drop=True)
    print(f"  결측치 제거: {before - len(df)}행 제거 → 최종 {len(df):,}행")

    # 타겟 변수 이진화 (보험 도메인: 1=해약, 0=유지)
    df['Churn_binary'] = (df['Churn'] == 'Yes').astype(int)

    return df


# =============================================================================
# STEP 2. 데이터 요약 출력
# =============================================================================

def summarize(df: pd.DataFrame):
    """
    데이터 구조, 결측치, 타겟 분포를 보험 도메인으로 재해석하여 출력한다.
    """
    sep = "=" * 60

    print(f"\n{sep}\n[ 데이터 기본 정보 ]\n{sep}")
    print(f"  크기         : {df.shape[0]:,}행 × {df.shape[1]}열")
    print(f"  수치형 변수  : {df.select_dtypes(include='number').shape[1]}개")
    print(f"  범주형 변수  : {df.select_dtypes(include='object').shape[1]}개")

    print(f"\n{sep}\n[ 전체 변수 목록 및 타입 ]\n{sep}")
    print(df.dtypes.to_string())

    print(f"\n{sep}\n[ 결측치 현황 ]\n{sep}")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    print("  ✅ 결측치 없음" if len(missing) == 0 else missing)

    print(f"\n{sep}\n[ 타겟 변수(Churn) 분포 — 보험 도메인 해석 ]\n{sep}")
    counts = df['Churn'].value_counts()
    pcts   = df['Churn'].value_counts(normalize=True) * 100
    for label, meaning in [('No', '보험 유지'), ('Yes', '보험 해약(이탈)')]:
        print(f"  {label} ({meaning}): {counts[label]:,}명 ({pcts[label]:.1f}%)")
    print(f"\n  → 이탈률 {pcts['Yes']:.1f}%: 클래스 불균형 존재하나 심각하지 않음")
    print(f"     (모델링 시 class_weight='balanced' 적용 예정)")

    print(f"\n{sep}\n[ 주요 수치형 변수 기술통계 ]\n{sep}")
    print(df[['tenure', 'MonthlyCharges', 'TotalCharges']].describe().round(2))

    print(f"\n{sep}\n[ 보험 도메인 변수 매핑 ]\n{sep}")
    mapping = {
        'tenure'          : '계약 유지 기간(월)  →  보험 계약 기간',
        'MonthlyCharges'  : '월 납입액           →  월 보험료',
        'TotalCharges'    : '총 납입액           →  누적 납입 보험료',
        'Contract'        : '계약 유형           →  납입 주기 (월납/연납)',
        'PaperlessBilling': '전자고지 수신 여부  →  디지털 채널 이용 동의',
        'TechSupport'     : '기술 지원 이용      →  고객센터/챗봇 이용',
        'OnlineSecurity'  : '온라인 보안 서비스  →  부가 보장 특약 가입',
        'SeniorCitizen'   : '고령 고객 여부      →  시니어 고객 세그먼트',
    }
    for col, desc in mapping.items():
        print(f"  {col:<22} {desc}")


# =============================================================================
# 실행
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("  보험 디지털 플랫폼 고객 이탈 분석")
    print("  STEP 1: 데이터 로드 및 기본 확인")
    print("=" * 60)

    df = load_data()
    summarize(df)

    print("\n" + "=" * 60)
    print("✅ 01_load_data.py 완료")
    print("   → 다음 단계: python src/02_eda.py")
    print("=" * 60)