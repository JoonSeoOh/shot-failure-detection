"""
확장 실험: 5개 모델 비교
Baseline 1 : Rule-based (sharpness 임계값)
Baseline 2 : Logistic Regression
Main 1     : Random Forest
Main 2     : SVM (RBF Kernel)
Main 3     : XGBoost (tuned)

실행법:
    python run_experiments_v2.py --features ./features.csv --output_dir ./experiment_results_v2
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
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix, roc_curve
)
from xgboost import XGBClassifier
warnings.filterwarnings("ignore")

FEATURE_COLS = [
    "sharpness","colorfulness","noise","contrast","edge_density",
    "dct_hf_ratio","bright_mean","bright_std","bright_low","bright_high",
    "sat_mean","sat_std","under_exp","over_exp",
    "orig_width","orig_height","aspect_ratio"
]

BEST_XGB = {
    "n_estimators": 183, "max_depth": 7,
    "learning_rate": 0.13033543674454473,
    "subsample": 0.7860006984225687,
    "colsample_bytree": 0.769934021972095,
    "min_child_weight": 2, "gamma": 2.3577021559286817,
    "reg_alpha": 0.07877575739848852, "reg_lambda": 3.07460727359055
}
FAIL_THRESHOLD = 67.48996559635


# ──────────────────────────────────────────
# 데이터 로드
# ──────────────────────────────────────────

def load_data(features_csv, params_json=None):
    threshold = FAIL_THRESHOLD
    xgb_params = BEST_XGB.copy()

    if params_json and os.path.exists(params_json):
        with open(params_json) as f:
            p = json.load(f)
        threshold  = p.get("FAIL_THRESHOLD", threshold)
        xgb_params = p.get("xgboost", xgb_params)
        print(f"best_params.json 로드 완료 (threshold={threshold:.2f})")

    df = pd.read_csv(features_csv)
    df["label"] = (df["MOS"] < threshold).astype(int)

    train    = df[df["set"] == "training"].copy()
    val      = df[df["set"] == "validation"].copy()
    test     = df[df["set"] == "test"].copy()
    trainval = pd.concat([train, val], ignore_index=True)

    pos = trainval["label"].mean()
    print(f"\n데이터 현황 (threshold={threshold:.1f})")
    print(f"  train+val : {len(trainval)}장  (실패 {pos*100:.1f}%)")
    print(f"  test      : {len(test)}장  (실패 {test['label'].mean()*100:.1f}%)")

    return trainval, test, threshold, xgb_params


# ──────────────────────────────────────────
# 공통 평가 함수
# ──────────────────────────────────────────

def evaluate(name, y_true, y_pred, y_prob, elapsed):
    metrics = {
        "model":     name,
        "accuracy":  accuracy_score(y_true, y_pred),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "auc":       roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "time_sec":  round(elapsed, 2),
        "cm":        confusion_matrix(y_true, y_pred),
        "y_prob":    y_prob,
        "y_true":    y_true,
    }
    cm = metrics["cm"]
    print(f"\n  ┌─ [{name}]  ({elapsed:.1f}s)")
    print(f"  │  Accuracy={metrics['accuracy']:.4f}  F1={metrics['f1']:.4f}  "
          f"AUC={metrics['auc']:.4f}  Prec={metrics['precision']:.4f}  Rec={metrics['recall']:.4f}")
    print(f"  │  TN={cm[0,0]:4d}  FP={cm[0,1]:4d}")
    print(f"  └  FN={cm[1,0]:4d}  TP={cm[1,1]:4d}")
    return metrics


# ──────────────────────────────────────────
# 모델 정의
# ──────────────────────────────────────────

def run_rule_based(trainval, test):
    print("\n" + "="*60)
    print("📏  Baseline 1: Rule-based")
    t0 = time.time()

    X_tr = trainval["sharpness"].values
    y_tr = trainval["label"].values
    best_f1, best_thresh = -1, 0
    for thr in np.percentile(X_tr, np.arange(5, 95, 1)):
        pred = (X_tr < thr).astype(int)
        f1 = f1_score(y_tr, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, thr

    X_te = test["sharpness"].values
    y_te = test["label"].values
    y_pred = (X_te < best_thresh).astype(int)
    y_prob = 1 - (X_te / (X_te.max() + 1e-6))
    print(f"  최적 sharpness 임계값: {best_thresh:.2f}")
    return evaluate("Rule-based", y_te, y_pred, y_prob, time.time()-t0)


def run_logistic(trainval, test):
    print("\n" + "="*60)
    print("📈  Baseline 2: Logistic Regression")
    t0 = time.time()

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(trainval[FEATURE_COLS])
    X_te = scaler.transform(test[FEATURE_COLS])
    y_tr = trainval["label"].values
    y_te = test["label"].values

    model = LogisticRegression(
        max_iter=1000, class_weight="balanced", C=1.0, random_state=42
    )
    model.fit(X_tr, y_tr)
    return evaluate("Logistic Regression", y_te,
                    model.predict(X_te),
                    model.predict_proba(X_te)[:,1],
                    time.time()-t0)


def run_random_forest(trainval, test):
    print("\n" + "="*60)
    print("🌲  Main 1: Random Forest")
    t0 = time.time()

    X_tr = trainval[FEATURE_COLS].values
    y_tr = trainval["label"].values
    X_te = test[FEATURE_COLS].values
    y_te = test["label"].values

    ratio = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_tr, y_tr)

    # 피처 중요도 출력
    fi = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print(f"\n  상위 5개 피처 (RF):")
    for feat, val in fi.head(5).items():
        print(f"    {feat:<22}: {val:.4f}")

    result = evaluate("Random Forest", y_te,
                      model.predict(X_te),
                      model.predict_proba(X_te)[:,1],
                      time.time()-t0)
    result["model_obj"] = model
    return result


def run_svm(trainval, test):
    print("\n" + "="*60)
    print("⚡  Main 2: SVM (RBF Kernel)")
    print("  (학습 시간이 다소 걸릴 수 있습니다...)")
    t0 = time.time()

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(trainval[FEATURE_COLS])
    X_te = scaler.transform(test[FEATURE_COLS])
    y_tr = trainval["label"].values
    y_te = test["label"].values

    model = SVC(
        kernel="rbf",
        C=10.0,
        gamma="scale",
        class_weight="balanced",
        probability=True,   # AUC 계산 위해 필요
        random_state=42
    )
    model.fit(X_tr, y_tr)

    return evaluate("SVM (RBF)", y_te,
                    model.predict(X_te),
                    model.predict_proba(X_te)[:,1],
                    time.time()-t0)


def run_xgboost(trainval, test, xgb_params):
    print("\n" + "="*60)
    print("🌳  Main 3: XGBoost (tuned)")
    t0 = time.time()

    X_tr = trainval[FEATURE_COLS].values
    y_tr = trainval["label"].values
    X_te = test[FEATURE_COLS].values
    y_te = test["label"].values

    ratio = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

    model = XGBClassifier(
        **xgb_params,
        scale_pos_weight=ratio,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    model.fit(X_tr, y_tr)

    fi = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print(f"\n  상위 5개 피처 (XGBoost):")
    for feat, val in fi.head(5).items():
        print(f"    {feat:<22}: {val:.4f}")

    result = evaluate("XGBoost (tuned)", y_te,
                      model.predict(X_te),
                      model.predict_proba(X_te)[:,1],
                      time.time()-t0)
    result["model_obj"] = model
    return result


# ──────────────────────────────────────────
# 시각화
# ──────────────────────────────────────────

MODEL_COLORS = {
    "Rule-based":          "#888780",
    "Logistic Regression": "#4A90D9",
    "Random Forest":       "#27ae60",
    "SVM (RBF)":           "#9B59B6",
    "XGBoost (tuned)":     "#D85A30",
}

def plot_comparison_bar(results, output_dir):
    metrics  = ["accuracy","f1","auc","precision","recall"]
    m_labels = ["Accuracy","F1-Score","AUC-ROC","Precision","Recall"]
    n = len(results)
    x = np.arange(len(metrics))
    width = 0.15

    fig, ax = plt.subplots(figsize=(13, 5))
    for i, r in enumerate(results):
        vals  = [r[m] for m in metrics]
        color = MODEL_COLORS.get(r["model"], "#999")
        offset = (i - n/2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width,
                      label=r["model"], color=color, alpha=0.88, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.004,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7, rotation=45)

    ax.set_xticks(x); ax.set_xticklabels(m_labels, fontsize=11)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("5개 모델 성능 비교 — Test Set", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9, ncol=2)
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.4)
    plt.tight_layout()
    path = os.path.join(output_dir, "model_comparison_v2.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"비교 차트 저장 → {path}")


def plot_roc_all(results, output_dir):
    fig, ax = plt.subplots(figsize=(7, 6))
    for r in results:
        fpr, tpr, _ = roc_curve(r["y_true"], r["y_prob"])
        color = MODEL_COLORS.get(r["model"], "#999")
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{r['model']} (AUC={r['auc']:.3f})")
    ax.plot([0,1],[0,1],"k--",alpha=0.4,label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curve — 5개 모델 비교", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    path = os.path.join(output_dir, "roc_curves_v2.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"ROC 곡선 저장 → {path}")


def plot_confusion_all(results, output_dir):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(n*3.2, 4))
    for ax, r in zip(axes, results):
        cm = r["cm"]
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0,1]); ax.set_xticklabels(["정상","실패"], fontsize=9)
        ax.set_yticks([0,1]); ax.set_yticklabels(["정상","실패"], fontsize=9)
        ax.set_xlabel("예측"); ax.set_ylabel("실제")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                        fontsize=12, fontweight="bold",
                        color="white" if cm[i,j] > cm.max()/2 else "black")
        color = MODEL_COLORS.get(r["model"], "black")
        ax.set_title(f"{r['model']}\nF1={r['f1']:.3f}  AUC={r['auc']:.3f}",
                     fontsize=9, color=color, fontweight="bold")
    plt.suptitle("Confusion Matrix 비교", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrices_v2.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Confusion Matrix 저장 → {path}")


def plot_radar(results, output_dir):
    """레이더 차트 — 5개 지표를 한눈에"""
    metrics  = ["accuracy","f1","auc","precision","recall"]
    m_labels = ["Accuracy","F1","AUC","Precision","Recall"]
    N = len(metrics)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), m_labels, fontsize=10)

    for r in results:
        vals = [r[m] for m in metrics]
        vals += vals[:1]
        color = MODEL_COLORS.get(r["model"], "#999")
        ax.plot(angles, vals, "o-", linewidth=2, color=color, label=r["model"])
        ax.fill(angles, vals, alpha=0.08, color=color)

    ax.set_ylim(0, 1)
    ax.set_title("모델 성능 레이더 차트", fontsize=13,
                 fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)
    plt.tight_layout()
    path = os.path.join(output_dir, "radar_chart.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"레이더 차트 저장 → {path}")


def print_final_table(results, output_dir):
    rows = [{k: r[k] for k in
             ["model","accuracy","f1","auc","precision","recall","time_sec"]}
            for r in results]
    df = pd.DataFrame(rows).set_index("model")

    print("\n" + "="*70)
    print("📊  최종 5개 모델 비교 (Test Set)")
    print("="*70)
    print(df.to_string(float_format=lambda x: f"{x:.4f}"))

    best = df["f1"].idxmax()
    print(f"\n🏆  F1 기준 최고: {best}  (F1={df.loc[best,'f1']:.4f}  AUC={df.loc[best,'auc']:.4f})")

    # XGBoost 기준 상대 개선율
    xgb_f1 = df.loc["XGBoost (tuned)", "f1"]
    print("\n  XGBoost 대비 F1 차이:")
    for model in df.index:
        if model != "XGBoost (tuned)":
            diff = (xgb_f1 - df.loc[model,"f1"]) * 100
            print(f"    vs {model:<25}: {'+' if diff>0 else ''}{diff:.2f}%p")

    df.to_csv(os.path.join(output_dir, "comparison_table_v2.csv"),
              float_format="%.4f")
    print(f"\n비교 테이블 저장 → {output_dir}/comparison_table_v2.csv")


# ──────────────────────────────────────────
# 메인
# ──────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features",    default="./features.csv")
    parser.add_argument("--params_json", default="./tuning_results/best_params.json")
    parser.add_argument("--output_dir",  default="./experiment_results_v2")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print("🚀 5개 모델 비교 실험 시작\n")

    trainval, test, threshold, xgb_params = load_data(
        args.features, args.params_json
    )

    results = []
    results.append(run_rule_based(trainval, test))
    results.append(run_logistic(trainval, test))
    results.append(run_random_forest(trainval, test))
    results.append(run_svm(trainval, test))
    results.append(run_xgboost(trainval, test, xgb_params))

    print("\n\n📊 시각화 생성 중...")
    plot_comparison_bar(results, args.output_dir)
    plot_roc_all(results, args.output_dir)
    plot_confusion_all(results, args.output_dir)
    plot_radar(results, args.output_dir)
    print_final_table(results, args.output_dir)

    # 모델 저장
    for r in results:
        if "model_obj" in r:
            name = r["model"].replace(" ","_").replace("(","").replace(")","")
            path = os.path.join(args.output_dir, f"{name}.pkl")
            with open(path, "wb") as f:
                pickle.dump(r["model_obj"], f)
            print(f"모델 저장 → {path}")

    print("\n✅ 완료!")
    print(f"   결과 폴더: {args.output_dir}/")
    print("   - model_comparison_v2.png")
    print("   - roc_curves_v2.png")
    print("   - confusion_matrices_v2.png")
    print("   - radar_chart.png")
    print("   - comparison_table_v2.csv")


if __name__ == "__main__":
    main()