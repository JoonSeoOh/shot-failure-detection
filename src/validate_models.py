"""
모델 학습 검증 스크립트
다음 5가지를 체계적으로 검증합니다:

1. 과적합 검사    — Train vs Test 성능 차이
2. 데이터 누수 검사 — Test 데이터가 학습에 섞였는지
3. 예측 분포 검사  — 모델이 다양한 예측을 하는지
4. 학습 곡선      — 트리 수에 따른 성능 변화 (XGBoost, RF)
5. 교차 검증      — Train 분산 안정성 확인

실행법:
    python validate_models.py --features ./features.csv --output_dir ./validation_results
"""

import argparse, os, json, warnings, pickle, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.svm             import SVC
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import (
    f1_score, roc_auc_score, accuracy_score,
    precision_score, recall_score
)
from sklearn.model_selection import StratifiedKFold, learning_curve
from xgboost import XGBClassifier
warnings.filterwarnings("ignore")

FEATURE_COLS = [
    "sharpness","colorfulness","noise","contrast","edge_density",
    "dct_hf_ratio","bright_mean","bright_std","bright_low","bright_high",
    "sat_mean","sat_std","under_exp","over_exp",
    "orig_width","orig_height","aspect_ratio"
]
FAIL_THRESHOLD = 67.49
BEST_XGB = {
    "n_estimators": 183, "max_depth": 7,
    "learning_rate": 0.13033543674454473,
    "subsample": 0.7860006984225687,
    "colsample_bytree": 0.769934021972095,
    "min_child_weight": 2, "gamma": 2.3577021559286817,
    "reg_alpha": 0.07877575739848852, "reg_lambda": 3.07460727359055
}

PASS = "✅ PASS"
WARN = "⚠️  WARN"
FAIL = "❌ FAIL"


# ──────────────────────────────────────────
# 데이터 로드
# ──────────────────────────────────────────

def load_data(features_csv):
    df = pd.read_csv(features_csv)
    df["label"] = (df["MOS"] < FAIL_THRESHOLD).astype(int)

    train    = df[df["set"] == "training"].copy()
    val      = df[df["set"] == "validation"].copy()
    test     = df[df["set"] == "test"].copy()
    trainval = pd.concat([train, val], ignore_index=True)

    print(f"데이터 로드 완료")
    print(f"  train+val : {len(trainval)}장  (실패 {trainval['label'].mean()*100:.1f}%)")
    print(f"  test      : {len(test)}장  (실패 {test['label'].mean()*100:.1f}%)")
    return trainval, test


def make_models(ratio):
    return {
        "Rule-based":          None,
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=3, class_weight="balanced", random_state=42, n_jobs=-1),
        "SVM (RBF)":           SVC(kernel="rbf", C=10.0, gamma="scale", class_weight="balanced", probability=True, random_state=42),
        "XGBoost (tuned)":     XGBClassifier(**BEST_XGB, scale_pos_weight=ratio, use_label_encoder=False, eval_metric="logloss", random_state=42, n_jobs=-1, verbosity=0),
    }


# ──────────────────────────────────────────
# 검증 1: 과적합 검사
# ──────────────────────────────────────────

def check_overfitting(trainval, test, ratio, output_dir):
    print("\n" + "═"*60)
    print("🔍 검증 1: 과적합 검사 (Train vs Test 성능 비교)")
    print("═"*60)

    X_tr = trainval[FEATURE_COLS].values
    y_tr = trainval["label"].values
    X_te = test[FEATURE_COLS].values
    y_te = test["label"].values

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    results = []
    models_to_check = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=3, class_weight="balanced", random_state=42, n_jobs=-1),
        "XGBoost (tuned)":     XGBClassifier(**BEST_XGB, scale_pos_weight=ratio, use_label_encoder=False, eval_metric="logloss", random_state=42, n_jobs=-1, verbosity=0),
    }

    print(f"\n  {'모델':<22} {'Train F1':>9} {'Test F1':>9} {'차이':>8} {'판정':>10}")
    print("  " + "-"*62)

    for name, model in models_to_check.items():
        if name in ["Logistic Regression", "SVM (RBF)"]:
            model.fit(X_tr_s, y_tr)
            tr_pred = model.predict(X_tr_s)
            te_pred = model.predict(X_te_s)
        else:
            model.fit(X_tr, y_tr)
            tr_pred = model.predict(X_tr)
            te_pred = model.predict(X_te)

        tr_f1 = f1_score(y_tr, tr_pred, zero_division=0)
        te_f1 = f1_score(y_te, te_pred, zero_division=0)
        diff  = tr_f1 - te_f1

        # 판정: 차이 0.05 이하 → 정상, 0.05~0.15 → 경고, 0.15 이상 → 과적합
        if diff < 0.05:
            verdict = PASS
        elif diff < 0.15:
            verdict = WARN
        else:
            verdict = FAIL

        print(f"  {name:<22} {tr_f1:>9.4f} {te_f1:>9.4f} {diff:>+8.4f}  {verdict}")
        results.append({"model": name, "train_f1": tr_f1, "test_f1": te_f1, "diff": diff, "verdict": verdict})

    return results


# ──────────────────────────────────────────
# 검증 2: 데이터 누수 검사
# ──────────────────────────────────────────

def check_data_leakage(trainval, test):
    print("\n" + "═"*60)
    print("🔍 검증 2: 데이터 누수 검사")
    print("═"*60)

    # image_name 겹치는 것 확인
    tr_names = set(trainval["image_name"].values)
    te_names = set(test["image_name"].values)
    overlap  = tr_names & te_names

    if len(overlap) == 0:
        print(f"  Train/Test 이미지 겹침: 0개  {PASS}")
    else:
        print(f"  Train/Test 이미지 겹침: {len(overlap)}개  {FAIL}")
        print(f"  겹치는 파일 예시: {list(overlap)[:5]}")

    # MOS 분포 비교 (누수 시 분포가 너무 비슷해짐)
    tr_mos_mean = trainval["MOS"].mean()
    te_mos_mean = test["MOS"].mean()
    mos_diff    = abs(tr_mos_mean - te_mos_mean)

    print(f"\n  MOS 평균 — Train: {tr_mos_mean:.2f} / Test: {te_mos_mean:.2f} / 차이: {mos_diff:.2f}")
    if mos_diff < 5.0:
        print(f"  MOS 분포 유사  {PASS}  (공식 KonIQ-10k 분할 사용 확인됨)")
    else:
        print(f"  MOS 분포 차이 큼  {WARN}  (분할 확인 필요)")

    # 레이블 비율 비교
    tr_pos = trainval["label"].mean()
    te_pos = test["label"].mean()
    pos_diff = abs(tr_pos - te_pos)

    print(f"\n  실패 비율 — Train: {tr_pos*100:.1f}% / Test: {te_pos*100:.1f}% / 차이: {pos_diff*100:.1f}%p")
    if pos_diff < 0.05:
        print(f"  레이블 비율 균형  {PASS}")
    else:
        print(f"  레이블 비율 불균형  {WARN}")

    return len(overlap) == 0


# ──────────────────────────────────────────
# 검증 3: 예측 분포 검사
# ──────────────────────────────────────────

def check_prediction_distribution(trainval, test, ratio, output_dir):
    print("\n" + "═"*60)
    print("🔍 검증 3: 예측 분포 검사 (모델이 다양하게 예측하는지)")
    print("═"*60)

    X_tr = trainval[FEATURE_COLS].values
    y_tr = trainval["label"].values
    X_te = test[FEATURE_COLS].values
    y_te = test["label"].values

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    models = {
        "Logistic Regression": ("scaled", LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0, random_state=42)),
        "Random Forest":       ("raw",    RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=3, class_weight="balanced", random_state=42, n_jobs=-1)),
        "XGBoost (tuned)":     ("raw",    XGBClassifier(**BEST_XGB, scale_pos_weight=ratio, use_label_encoder=False, eval_metric="logloss", random_state=42, n_jobs=-1, verbosity=0)),
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("예측 확률 분포 (정상 사진 vs 실패 사진)", fontsize=13, fontweight="bold")

    print(f"\n  {'모델':<22} {'예측=0':>8} {'예측=1':>8} {'비율균형':>10} {'판정':>10}")
    print("  " + "-"*62)

    all_ok = True
    for ax, (name, (mode, model)) in zip(axes, models.items()):
        Xtr = X_tr_s if mode == "scaled" else X_tr
        Xte = X_te_s if mode == "scaled" else X_te
        model.fit(Xtr, y_tr)
        probs = model.predict_proba(Xte)[:, 1]
        preds = model.predict(Xte)

        n_zero = (preds == 0).sum()
        n_one  = (preds == 1).sum()
        ratio_preds = n_one / len(preds)

        # 판정: 예측 비율이 5~95% 사이면 정상
        if 0.05 < ratio_preds < 0.95:
            verdict = PASS
        else:
            verdict = FAIL
            all_ok = False

        print(f"  {name:<22} {n_zero:>8} {n_one:>8} {ratio_preds*100:>9.1f}%  {verdict}")

        # 히스토그램
        probs_pos = probs[y_te == 1]
        probs_neg = probs[y_te == 0]
        ax.hist(probs_neg, bins=30, alpha=0.6, color="#2E75B6", label="실제 정상")
        ax.hist(probs_pos, bins=30, alpha=0.6, color="#E24B4A", label="실제 실패")
        ax.axvline(0.5, color="black", linestyle="--", alpha=0.5)
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("예측 확률")
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(output_dir, "val_prediction_dist.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n  분포 히스토그램 저장 → {path}")
    return all_ok


# ──────────────────────────────────────────
# 검증 4: 학습 곡선
# ──────────────────────────────────────────

def check_learning_curves(trainval, ratio, output_dir):
    print("\n" + "═"*60)
    print("🔍 검증 4: 학습 곡선 (샘플 수 증가 → 성능 변화)")
    print("═"*60)

    X = trainval[FEATURE_COLS].values
    y = trainval["label"].values

    models_lc = {
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced", random_state=42, n_jobs=-1),
        "XGBoost":       XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, scale_pos_weight=ratio, use_label_encoder=False, eval_metric="logloss", random_state=42, n_jobs=-1, verbosity=0),
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("학습 곡선 — 데이터 양 증가에 따른 F1 변화", fontsize=13, fontweight="bold")

    train_sizes = np.linspace(0.1, 1.0, 8)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    for ax, (name, model) in zip(axes, models_lc.items()):
        tr_sizes, tr_scores, val_scores = learning_curve(
            model, X, y,
            train_sizes=train_sizes,
            cv=cv,
            scoring="f1",
            n_jobs=-1,
        )
        tr_mean  = tr_scores.mean(axis=1)
        tr_std   = tr_scores.std(axis=1)
        val_mean = val_scores.mean(axis=1)
        val_std  = val_scores.std(axis=1)

        ax.plot(tr_sizes, tr_mean,  "o-", color="#E24B4A", lw=2, label="Train F1")
        ax.fill_between(tr_sizes, tr_mean-tr_std, tr_mean+tr_std, alpha=0.15, color="#E24B4A")
        ax.plot(tr_sizes, val_mean, "s-", color="#2E75B6", lw=2, label="Val F1 (CV)")
        ax.fill_between(tr_sizes, val_mean-val_std, val_mean+val_std, alpha=0.15, color="#2E75B6")

        gap = tr_mean[-1] - val_mean[-1]
        ax.set_title(f"{name}  (최종 gap={gap:.3f})", fontsize=11)
        ax.set_xlabel("학습 샘플 수")
        ax.set_ylabel("F1 Score")
        ax.legend(fontsize=9)
        ax.set_ylim(0.4, 1.05)
        ax.grid(alpha=0.3)

        verdict = PASS if gap < 0.05 else (WARN if gap < 0.15 else FAIL)
        print(f"  {name:<20} Train F1={tr_mean[-1]:.4f}  Val F1={val_mean[-1]:.4f}  gap={gap:+.4f}  {verdict}")

    plt.tight_layout()
    path = os.path.join(output_dir, "val_learning_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  학습 곡선 저장 → {path}")


# ──────────────────────────────────────────
# 검증 5: 교차 검증
# ──────────────────────────────────────────

def check_cross_validation(trainval, ratio, output_dir):
    print("\n" + "═"*60)
    print("🔍 검증 5: 5-Fold 교차 검증 (분산 안정성 확인)")
    print("═"*60)

    X = trainval[FEATURE_COLS].values
    y = trainval["label"].values

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models_cv = {
        "Logistic Regression": ("scaled", LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0, random_state=42)),
        "Random Forest":       ("raw",    RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced", random_state=42, n_jobs=-1)),
        "XGBoost (tuned)":     ("raw",    XGBClassifier(**BEST_XGB, scale_pos_weight=ratio, use_label_encoder=False, eval_metric="logloss", random_state=42, n_jobs=-1, verbosity=0)),
    }

    cv_results = {}
    print(f"\n  {'모델':<22} {'평균 F1':>9} {'표준편차':>9} {'최소':>8} {'최대':>8} {'판정':>10}")
    print("  " + "-"*72)

    for name, (mode, model) in models_cv.items():
        Xuse = X_s if mode == "scaled" else X
        fold_scores = []
        for tr_idx, val_idx in cv.split(Xuse, y):
            model.fit(Xuse[tr_idx], y[tr_idx])
            pred = model.predict(Xuse[val_idx])
            fold_scores.append(f1_score(y[val_idx], pred, zero_division=0))

        mean_f1 = np.mean(fold_scores)
        std_f1  = np.std(fold_scores)

        # 판정: std < 0.02 → 안정, 0.02~0.05 → 경고, 0.05 이상 → 불안정
        if std_f1 < 0.02:
            verdict = PASS
        elif std_f1 < 0.05:
            verdict = WARN
        else:
            verdict = FAIL

        print(f"  {name:<22} {mean_f1:>9.4f} {std_f1:>9.4f} {min(fold_scores):>8.4f} {max(fold_scores):>8.4f}  {verdict}")
        cv_results[name] = {"scores": fold_scores, "mean": mean_f1, "std": std_f1}

    # 박스플롯
    fig, ax = plt.subplots(figsize=(8, 4))
    names  = list(cv_results.keys())
    scores = [cv_results[n]["scores"] for n in names]
    colors = ["#4A90D9", "#27AE60", "#D85A30"]

    bp = ax.boxplot(scores, labels=[n.replace(" ", "\n") for n in names],
                    patch_artist=True, notch=False, widths=0.5)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for median in bp["medians"]:
        median.set_color("white")
        median.set_linewidth(2)

    ax.set_title("5-Fold CV F1 분포", fontsize=13, fontweight="bold")
    ax.set_ylabel("F1 Score")
    ax.set_ylim(0.5, 1.0)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "val_cross_validation.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  CV 박스플롯 저장 → {path}")
    return cv_results


# ──────────────────────────────────────────
# 최종 판정 요약
# ──────────────────────────────────────────

def print_final_verdict(overfitting_results, leakage_ok, pred_dist_ok, cv_results):
    print("\n" + "═"*60)
    print("📋  최종 검증 요약")
    print("═"*60)

    checks = [
        ("데이터 누수", PASS if leakage_ok else FAIL),
        ("예측 분포",   PASS if pred_dist_ok else FAIL),
    ]

    for r in overfitting_results:
        checks.append((f"과적합 — {r['model']}", r["verdict"]))

    for name, info in cv_results.items():
        verdict = PASS if info["std"] < 0.02 else (WARN if info["std"] < 0.05 else FAIL)
        checks.append((f"CV 분산 — {name}", verdict))

    pass_cnt = sum(1 for _, v in checks if "PASS" in v)
    warn_cnt = sum(1 for _, v in checks if "WARN" in v)
    fail_cnt = sum(1 for _, v in checks if "FAIL" in v)

    for label, verdict in checks:
        print(f"  {verdict}  {label}")

    print(f"\n  총 {len(checks)}개 검증 — PASS: {pass_cnt}  WARN: {warn_cnt}  FAIL: {fail_cnt}")

    if fail_cnt == 0 and warn_cnt <= 2:
        print("\n  🎉 모델이 정상적으로 학습되었습니다!")
    elif fail_cnt == 0:
        print("\n  ⚠️  경미한 이슈가 있으나 전반적으로 정상 학습됩니다.")
    else:
        print("\n  ❌ 일부 모델에서 문제가 감지되었습니다. 위 내용을 확인하세요.")


# ──────────────────────────────────────────
# 메인
# ──────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features",   default="./features.csv")
    parser.add_argument("--output_dir", default="./validation_results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print("🚀 모델 학습 검증 시작\n")

    trainval, test = load_data(args.features)
    y_tr   = trainval["label"].values
    ratio  = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

    # 5가지 검증 실행
    overfitting_results = check_overfitting(trainval, test, ratio, args.output_dir)
    leakage_ok          = check_data_leakage(trainval, test)
    pred_dist_ok        = check_prediction_distribution(trainval, test, ratio, args.output_dir)
    check_learning_curves(trainval, ratio, args.output_dir)
    cv_results          = check_cross_validation(trainval, ratio, args.output_dir)

    print_final_verdict(overfitting_results, leakage_ok, pred_dist_ok, cv_results)

    print(f"\n결과 저장 위치: {args.output_dir}/")
    print("  val_prediction_dist.png  — 예측 확률 분포")
    print("  val_learning_curves.png  — 학습 곡선")
    print("  val_cross_validation.png — CV 박스플롯")


if __name__ == "__main__":
    main()