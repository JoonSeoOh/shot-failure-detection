# 고전 ML 기반 촬영 실패 사진 자동 선별 시스템

**Automatic Photo Failure Detection using Classical Machine Learning**

> 기계학습 텀 프로젝트 | 인공지능공학과 | 오준서

---

## 📌 프로젝트 개요

스마트폰과 카메라 QA 과정에서 발생하는 수만 장의 사진을 자동으로 선별하는 **3-Stage 파이프라인**입니다.

```
[Stage 1] XGBoost       → 17개 feature 기반 품질 분류 (AUC 0.869)
[Stage 2] Context-Aware → Bokeh / Silhouette 오탐 복구 (37.4% 복구)
[Stage 3] SHAP Coach    → 실패 원인 진단 + 개선 조언 자동 생성
```

---

## 📂 Repository 구조

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

## 🗂️ 데이터셋

**KonIQ-10k** (Lin et al., 2020)

- 10,073장의 실제 환경 사진
- 레이블: MOS_zscore (0~100, 크라우드소싱 품질 점수)
- 공식 분할: Train 7,058 / Val 1,000 / Test 2,015

📥 **다운로드**: [KonIQ-10k 공식 페이지](http://database.mmsp-kn.de/koniq-10k-database.html)

다운로드 후 아래 구조로 배치하세요:

```
shot-failure-detection/
└── koniq10k/
    ├── images/          ← 이미지 10,073장
    ├── koniq10k_scores_and_distributions.csv
    └── koniq10k_distributions_sets.csv
```

---

## ⚙️ 환경 설정

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

## 🚀 재현 방법 (순서대로 실행)

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
> 💡 건너뛰어도 됩니다. 기본값이 코드에 내장되어 있습니다.

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

### 🔍 개인 사진 분석 (보너스)
```bash
# my_test_data/ 폴더에 사진을 넣고 실행
python src/analyze_my_photos.py \
  --photo_dir  ./my_test_data \
  --model      ./experiment_results/xgb_tuned_model.pkl \
  --output_dir ./my_results
```

---

## 📊 주요 실험 결과

| 모델 | Accuracy | F1 | AUC-ROC | Precision | Recall |
|---|---|---|---|---|---|
| Rule-based | 0.712 | 0.793 | 0.781 | 0.742 | 0.851 |
| Logistic Regression | 0.699 | 0.744 | 0.781 | 0.830 | 0.674 |
| Random Forest | 0.784 | 0.824 | 0.864 | 0.873 | 0.780 |
| SVM (RBF) | 0.773 | 0.813 | 0.857 | 0.871 | 0.762 |
| **XGBoost (tuned)** | **0.784** | **0.823** | **0.869** | **0.878** | 0.775 |

---

## 🛠️ 개발 환경

| 항목 | 내용 |
|---|---|
| OS | macOS (Apple M3) |
| Python | 3.10 |
| 주요 라이브러리 | XGBoost 1.7.6, scikit-learn 1.3, OpenCV 4.8, SHAP 0.43 |
| 하드웨어 | CPU only (GPU 불필요) |

---

## 📚 References

1. Hosu, V., Lin, H., Sziranyi, T., & Saupe, D. (2020). KonIQ-10k: An ecologically valid database for deep learning of blind image quality assessment. IEEE Transactions on Image Processing, 29, 4041–4056.
2. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD 2016*.
3. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *NeurIPS 2017*.
4. Hasler, D., & Süsstrunk, S. (2003). Measuring colorfulness in natural images. *SPIE 2003*.
5. Pech-Pacheco, J. L., et al. (2000). Diatom autofocusing in brightfield microscopy. *ICPR 2000*.
6. Otsu, N. (1979). A threshold selection method from gray-level histograms. *IEEE TSMC*, 9(1), 62–66.
7. Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019).Optuna: A next-generation hyperparameter optimization framework. Proceedings of KDD 2019, 2623–2631.
8. Breiman, L. (2001). Random forests. Machine Learning, 45, 5–32.
9. Cortes, C., & Vapnik, V. (1995). Support-vector networks. Machine Learning, 20(3), 273–297.
---

## 📝 AI 사용 고지

본 프로젝트는 코드 구현 목적으로 Claude (Anthropic)를 활용하였습니다.
모든 코드와 결과는 직접 검토·실행하였으며, 프로젝트 방향 및 핵심 아이디어는 저자가 직접 수립하였습니다.