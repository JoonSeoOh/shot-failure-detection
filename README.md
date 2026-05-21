# 고전 ML 기반 촬영 실패 사진 자동 선별 시스템

**Automatic Photo Failure Detection using Classical Machine Learning**

> 기계학습 텀 프로젝트 | 인공지능공학과 | 오준서

---

## 프로젝트 개요

스마트폰과 카메라 QA 과정에서 발생하는 수만 장의 사진을 자동으로 선별하는 **3-Stage 파이프라인**입니다.

```
[Stage 1] XGBoost       → 17개 feature 기반 품질 분류 (AUC 0.869)
[Stage 2] Context-Aware → Bokeh / Silhouette 오탐 복구 (37.4% 복구)
[Stage 3] SHAP Coach    → 실패 원인 진단 + 개선 조언 자동 생성
```

---

## Repository 구조

```
shot-failure-detection/
├── README.md
├── requirements.txt
├── src/
│   ├── extract_features.py      # Step 1: 이미지 → 17개 feature 추출
│   ├── train_xgboost.py         # Step 2: XGBoost 학습 (Baseline 포함)
│   ├── tune_hyperparams.py      # Step 3: 하이퍼파라미터 탐색
│   ├── run_experiments_v2.py    # Step 4: 5개 모델 비교 실험
│   ├── stage2_context.py        # Step 5: Context-Aware 재분류
│   ├── stage3_shap.py           # Step 6: SHAP 기반 조언 생성
│   ├── validate_models.py       # 모델 검증 (과적합/누수/CV)
│   └── analyze_my_photos.py     # 개인 사진 3-Stage 분석
└── data/
    ├── koniq10k_scores_and_distributions.csv
    └── koniq10k_distributions_sets.csv
```

---

## 데이터셋

**KonIQ-10k** (Lin et al., 2020)

- 10,073장의 실제 환경 사진
- 레이블: MOS_zscore (0~100, 크라우드소싱 품질 점수)
- 공식 분할: Train 7,058 / Val 1,000 / Test 2,015

**다운로드**: [KonIQ-10k 공식 페이지](http://database.mmsp-kn.de/koniq-10k-database.html)

다운로드 후 아래 구조로 배치하세요:

```
shot-failure-detection/
└── koniq10k/
    ├── images/          ← 이미지 10,073장
    ├── koniq10k_scores_and_distributions.csv
    └── koniq10k_distributions_sets.csv
```

---

## 환경 설정

### 1. Python 버전
```bash
Python 3.10 권장
```

### 2. 패키지 설치
```bash
pip install -r requirements.txt
```

### 3. Mac에서 XGBoost 사용 시 (libomp 필요)
```bash
brew install libomp
```

---

## 재현 방법 (순서대로 실행)

### Step 1. Feature 추출
```bash
python src/extract_features.py \
  --img_dir  ./koniq10k/images \
  --csv      ./koniq10k/koniq10k_distributions_sets.csv \
  --output   ./features.csv
```
> 소요 시간: 약 10~20분 | 출력: `features.csv`

---

### Step 2. 하이퍼파라미터 튜닝 (선택)
```bash
pip install optuna

python src/tune_hyperparams.py \
  --features    ./features.csv \
  --output_dir  ./tuning_results
```
> 소요 시간: 약 10~15분 | 출력: `tuning_results/best_params.json`
>
> 건너뛰어도 됩니다. 기본값이 코드에 내장되어 있습니다.

---

### Step 3. 모델 학습 및 비교 실험
```bash
python src/run_experiments_v2.py \
  --features    ./features.csv \
  --params_json ./tuning_results/best_params.json \
  --output_dir  ./experiment_results
```
> 소요 시간: 약 2~5분 | 출력: 비교표 CSV + 시각화 PNG

---

### Step 4. Stage 2 Context-Aware 분류
```bash
python src/stage2_context.py \
  --failed     ./experiment_results/failed_photos.csv \
  --img_dir    ./koniq10k/images \
  --output_dir ./output
```
> 출력: `stage2_results.csv`, `true_failures.csv`

---

### Step 5. Stage 3 SHAP 조언 생성
```bash
python src/stage3_shap.py \
  --true_failures ./output/true_failures.csv \
  --features      ./features.csv \
  --model         ./experiment_results/xgb_tuned_model.pkl \
  --output_dir    ./output
```
> 출력: `photo_advice_report.csv`, `shap_summary.png`

---

### Step 6. 모델 검증
```bash
python src/validate_models.py \
  --features   ./features.csv \
  --output_dir ./validation_results
```
> 출력: 과적합/누수/CV 검증 결과 + 시각화

---

### 개인 사진 분석 (보너스)
```bash
# my_test_data/ 폴더에 사진을 넣고 실행
python src/analyze_my_photos.py \
  --photo_dir  ./my_test_data \
  --model      ./experiment_results/xgb_tuned_model.pkl \
  --output_dir ./my_results
```

---

## 주요 실험 결과

| 모델 | Accuracy | F1 | AUC-ROC | Precision | Recall |
|---|---|---|---|---|---|
| Rule-based | 0.712 | 0.793 | 0.781 | 0.742 | 0.851 |
| Logistic Regression | 0.699 | 0.744 | 0.781 | 0.830 | 0.674 |
| Random Forest | 0.784 | 0.824 | 0.864 | 0.873 | 0.780 |
| SVM (RBF) | 0.773 | 0.813 | 0.857 | 0.871 | 0.762 |
| **XGBoost (tuned)** | **0.784** | **0.823** | **0.869** | **0.878** | 0.775 |

---

## 개발 환경

| 항목 | 내용 |
|---|---|
| OS | macOS (Apple M3) |
| Python | 3.10 |
| 주요 라이브러리 | XGBoost 1.7.6, scikit-learn 1.3, OpenCV 4.8, SHAP 0.43 |
| 하드웨어 | CPU only (GPU 불필요) |