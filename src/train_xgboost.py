"""
Stage 1 - XGBoost 품질 예측 학습
features.csv → 모델 학습 → model.pkl + 결과 리포트

실행법:
    python train_xgboost.py --features ./features.csv --output_dir ./output
"""

import argparse
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # 서버 환경 대응

from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, mean_absolute_error, mean_squared_error
)
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings("ignore")


FEATURE_COLS = [
    "sharpness", "colorfulness", "noise", "contrast", "edge_density",
    "dct_hf_ratio", "bright_mean", "bright_std", "bright_low", "bright_high",
    "sat_mean", "sat_std", "under_exp", "over_exp",
    "orig_width", "orig_height", "aspect_ratio"
]

# 이진 분류 threshold (MOS < 45.0 → 실패 사진)
FAIL_THRESHOLD = 45.0


def load_and_split(features_csv):
    df = pd.read_csv(features_csv)
    print(f"총 데이터: {len(df)}장")

    # 공식 분할 사용
    train = df[df["set"] == "training"].copy()
    val   = df[df["set"] == "validation"].copy()
    test  = df[df["set"] == "test"].copy()
    print(f"  train: {len(train)} / val: {len(val)} / test: {len(test)}")

    # 이진 레이블 생성 (0=정상, 1=실패)
    for d in [train, val, test]:
        d["label"] = (d["MOS"] < FAIL_THRESHOLD).astype(int)

    print(f"\n레이블 분포 (train):")
    vc = train["label"].value_counts().sort_index()
    print(f"  정상(0): {vc.get(0, 0)}장 / 실패(1): {vc.get(1, 0)}장")
    ratio = vc.get(0, 1) / max(vc.get(1, 1), 1)
    print(f"  클래스 불균형 비율: {ratio:.1f}:1")

    return train, val, test, ratio


def train_model(train, val, ratio):
    X_tr = train[FEATURE_COLS]
    y_tr = train["label"]
    X_val = val[FEATURE_COLS]
    y_val = val["label"]

    model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=ratio,   # 클래스 불균형 자동 보정
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=30,
    )

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    print(f"\n최적 트리 수: {model.best_iteration}")
    return model


def evaluate(model, test, output_dir):
    X_te = test[FEATURE_COLS]
    y_te = test["label"]

    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]

    print("\n" + "="*50)
    print("📊 테스트셋 평가 결과")
    print("="*50)
    print(classification_report(y_te, y_pred, target_names=["정상", "실패"]))

    auc = roc_auc_score(y_te, y_prob)
    print(f"AUC-ROC: {auc:.4f}")

    # 혼동 행렬 저장
    cm = confusion_matrix(y_te, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["정상(예측)", "실패(예측)"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["정상(실제)", "실패(실제)"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > cm.max()/2 else "black")
    ax.set_title(f"Confusion Matrix (AUC={auc:.3f})")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"혼동 행렬 저장 → {path}")

    return y_pred, y_prob, auc


def plot_feature_importance(model, output_dir):
    fi = pd.Series(model.feature_importances_, index=FEATURE_COLS)
    fi = fi.sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#e74c3c" if v >= fi.quantile(0.75) else "#3498db" for v in fi]
    fi.plot(kind="barh", ax=ax, color=colors)
    ax.set_title("Feature Importance (XGBoost)", fontsize=13)
    ax.set_xlabel("Importance Score")
    ax.axvline(fi.mean(), color="gray", linestyle="--", label="평균")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, "feature_importance.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"특징 중요도 저장 → {path}")


def save_predictions(model, test, output_dir):
    """예측 결과를 CSV로 저장 (Stage 2, 3에서 활용)"""
    X_te = test[FEATURE_COLS]
    test = test.copy()
    test["pred_label"] = model.predict(X_te)
    test["pred_prob"]  = model.predict_proba(X_te)[:, 1]

    # 실패 예측 사진만 따로
    failed = test[test["pred_label"] == 1].copy()
    path_all    = os.path.join(output_dir, "test_predictions.csv")
    path_failed = os.path.join(output_dir, "failed_photos.csv")
    test.to_csv(path_all, index=False)
    failed.to_csv(path_failed, index=False)
    print(f"\n전체 예측 저장   → {path_all}")
    print(f"실패 사진 목록   → {path_failed}  ({len(failed)}장)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features",   default="./features.csv")
    parser.add_argument("--output_dir", default="./output")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 데이터 로드 & 분할
    train, val, test, ratio = load_and_split(args.features)

    # 2. 학습
    print("\n🚀 XGBoost 학습 시작...")
    model = train_model(train, val, ratio)

    # 3. 평가
    y_pred, y_prob, auc = evaluate(model, test, args.output_dir)

    # 4. 시각화
    plot_feature_importance(model, args.output_dir)

    # 5. 예측 결과 저장 (Stage 2 입력용)
    save_predictions(model, test, args.output_dir)

    # 6. 모델 저장
    model_path = os.path.join(args.output_dir, "xgb_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"\n모델 저장 → {model_path}")
    print("\n✅ Stage 1 완료! 다음 단계: python stage2_context.py")


if __name__ == "__main__":
    main()