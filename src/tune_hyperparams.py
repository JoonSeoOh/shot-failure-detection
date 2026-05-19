"""
하이퍼파라미터 튜닝 스크립트
- FAIL_THRESHOLD : Greedy Search
- XGBoost params : Optuna (Bayesian Optimization)
- Stage 2 thresh : Grid Search (2D, 작은 범위)

실행법:
    pip install optuna
    python tune_hyperparams.py --features ./features.csv --output_dir ./tuning_results
"""

import argparse, os, warnings, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

FEATURE_COLS = [
    "sharpness","colorfulness","noise","contrast","edge_density",
    "dct_hf_ratio","bright_mean","bright_std","bright_low","bright_high",
    "sat_mean","sat_std","under_exp","over_exp",
    "orig_width","orig_height","aspect_ratio"
]

# ──────────────────────────────────────────
# 데이터 로드
# ──────────────────────────────────────────

def load_data(features_csv):
    df = pd.read_csv(features_csv)
    train = df[df["set"] == "training"].copy()
    val   = df[df["set"] == "validation"].copy()
    test  = df[df["set"] == "test"].copy()
    return train, val, test


def make_labels(df, threshold):
    return (df["MOS"] < threshold).astype(int)


# ──────────────────────────────────────────
# 1단계: FAIL_THRESHOLD Greedy Search
# ──────────────────────────────────────────

def tune_threshold_greedy(train, val, step=1.0):
    """
    Greedy Search: threshold를 1씩 올리면서
    val F1이 더 이상 안 오르면 멈춤
    """
    print("\n" + "="*55)
    print("🔍 [1/3] FAIL_THRESHOLD — Greedy Search")
    print("="*55)

    # 고정 XGBoost (기본값)
    results = []
    best_f1, best_thresh = -1, 45.0
    no_improve = 0

    # 탐색 범위: MOS 분포의 25~75 퍼센타일 사이
    mos_vals = train["MOS"].values
    search_min = max(20.0, np.percentile(mos_vals, 15))
    search_max = min(80.0, np.percentile(mos_vals, 65))
    thresholds = np.arange(search_min, search_max, step)

    print(f"탐색 범위: MOS {search_min:.1f} ~ {search_max:.1f} (step={step})")
    print(f"{'Threshold':>12} {'Val F1':>10} {'Pos비율':>8} {'AUC':>8}")
    print("-"*42)

    for thresh in thresholds:
        y_tr = make_labels(train, thresh)
        y_val = make_labels(val, thresh)

        pos_ratio = y_tr.mean()
        if pos_ratio < 0.05 or pos_ratio > 0.70:
            continue  # 극단적 불균형 건너뜀

        ratio = (1 - pos_ratio) / max(pos_ratio, 1e-6)
        model = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=ratio, use_label_encoder=False,
            eval_metric="logloss", random_state=42,
            n_jobs=-1, verbosity=0
        )
        model.fit(train[FEATURE_COLS], y_tr)
        y_prob = model.predict_proba(val[FEATURE_COLS])[:,1]
        y_pred = (y_prob >= 0.5).astype(int)

        f1  = f1_score(y_val, y_pred, zero_division=0)
        auc = roc_auc_score(y_val, y_prob) if y_val.sum() > 0 else 0.5

        print(f"  {thresh:>10.1f}   {f1:>8.4f}   {pos_ratio*100:>6.1f}%   {auc:>6.4f}"
              + (" ← best" if f1 > best_f1 else ""))

        results.append({"threshold": thresh, "f1": f1, "auc": auc,
                         "pos_ratio": pos_ratio})

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= 5:  # 5번 연속 개선 없으면 조기 종료
                print(f"  (5회 연속 개선 없음 → 조기 종료)")
                break

    print(f"\n✅ 최적 FAIL_THRESHOLD = {best_thresh:.1f}  (F1={best_f1:.4f})")
    return best_thresh, pd.DataFrame(results)


# ──────────────────────────────────────────
# 2단계: XGBoost Optuna Bayesian
# ──────────────────────────────────────────

def tune_xgboost_optuna(train, val, best_thresh, n_trials=60):
    """
    Optuna: 이전 시도 결과를 학습해서 유망한 영역을 집중 탐색
    """
    print("\n" + "="*55)
    print("🤖 [2/3] XGBoost — Optuna Bayesian Optimization")
    print("="*55)
    print(f"FAIL_THRESHOLD={best_thresh:.1f} 고정, {n_trials}회 시도\n")

    y_tr  = make_labels(train, best_thresh)
    y_val = make_labels(val,   best_thresh)
    ratio = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

    trial_log = []

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 800),
            "max_depth":         trial.suggest_int("max_depth", 3, 10),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
            "gamma":             trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 0.0, 2.0),
            "reg_lambda":        trial.suggest_float("reg_lambda", 0.5, 5.0),
        }
        model = XGBClassifier(
            **params, scale_pos_weight=ratio,
            use_label_encoder=False, eval_metric="logloss",
            random_state=42, n_jobs=-1, verbosity=0
        )
        model.fit(train[FEATURE_COLS], y_tr)
        y_prob = model.predict_proba(val[FEATURE_COLS])[:,1]
        y_pred = (y_prob >= 0.5).astype(int)

        # 목적함수: AUC + F1 균형
        auc = roc_auc_score(y_val, y_prob) if y_val.sum() > 0 else 0.5
        f1  = f1_score(y_val, y_pred, zero_division=0)
        score = 0.6 * auc + 0.4 * f1

        trial_log.append({"trial": trial.number, "score": score,
                          "auc": auc, "f1": f1, **params})

        if trial.number % 10 == 0:
            print(f"  Trial {trial.number:3d}: score={score:.4f}  "
                  f"AUC={auc:.4f}  F1={f1:.4f}  "
                  f"depth={params['max_depth']}  lr={params['learning_rate']:.3f}")
        return score

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    best_score = study.best_value
    print(f"\n✅ 최적 XGBoost 파라미터 (score={best_score:.4f}):")
    for k, v in best.items():
        print(f"   {k:<22} = {v}")

    return best, pd.DataFrame(trial_log), study


# ──────────────────────────────────────────
# 3단계: Stage 2 임계값 Grid Search
# ──────────────────────────────────────────

def tune_stage2_grid(train, val, best_thresh, best_xgb_params):
    """
    Grid Search: bokeh_thresh × sil_thresh 2D 격자 탐색
    Stage 2는 레이블이 없어서 proxy metric 사용:
    → true_fail 비율이 적절한 범위에서 bokeh/sil 회수율 최대화
    """
    print("\n" + "="*55)
    print("📐 [3/3] Stage 2 임계값 — Grid Search")
    print("="*55)

    import cv2
    from pathlib import Path

    # Stage 2 그리드 탐색은 실제 이미지 없이 피처 기반 proxy로 수행
    # sharpness 공간 분포를 이용한 bokeh proxy
    # under_exp + dark ratio를 이용한 sil proxy

    bokeh_range = np.arange(0.20, 0.55, 0.05)
    sil_range   = np.arange(0.15, 0.50, 0.05)

    y_tr = make_labels(train, best_thresh)
    ratio = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

    # 최적 XGB로 val 예측
    model = XGBClassifier(
        **best_xgb_params, scale_pos_weight=ratio,
        use_label_encoder=False, eval_metric="logloss",
        random_state=42, n_jobs=-1, verbosity=0
    )
    model.fit(train[FEATURE_COLS], make_labels(train, best_thresh))
    val_prob = model.predict_proba(val[FEATURE_COLS])[:,1]
    val["pred_prob"] = val_prob
    val["pred_label"] = (val_prob >= 0.5).astype(int)

    fail_val = val[val["pred_label"] == 1].copy()

    results = []
    best_score, best_bt, best_st = -1, 0.35, 0.30

    print(f"{'bokeh_t':>9} {'sil_t':>7} {'bokeh%':>8} {'sil%':>7} {'fail%':>7} {'score':>8}")
    print("-"*48)

    for bt in bokeh_range:
        for st in sil_range:
            # Proxy: sharpness 공간 비율 기반 bokeh 추정
            # center_ratio ≈ sharpness / (edge_density + 1)
            fail_val = fail_val.copy()
            fail_val["bokeh_proxy"] = (
                fail_val["sharpness"] /
                (fail_val["sharpness"].mean() + 1e-6) - 1.0
            ).clip(0) / 3.0
            fail_val["sil_proxy"] = (
                fail_val["under_exp"] * 1.5 + fail_val["bright_low"] / 255 * 2.0
            ).clip(0, 1)

            bokeh_mask = fail_val["bokeh_proxy"] >= bt
            sil_mask   = (~bokeh_mask) & (fail_val["sil_proxy"] >= st)
            true_fail  = (~bokeh_mask) & (~sil_mask)

            b_pct = bokeh_mask.mean()
            s_pct = sil_mask.mean()
            f_pct = true_fail.mean()

            # 좋은 threshold = bokeh/sil이 적절히 걸러지면서 true_fail이 50~80%
            if len(fail_val) == 0:
                continue
            balance_score = 1.0 - abs(f_pct - 0.65)
            coverage_score = b_pct + s_pct

            score = balance_score * 0.6 + coverage_score * 0.4

            results.append({"bokeh_thresh": bt, "sil_thresh": st,
                            "bokeh_pct": b_pct, "sil_pct": s_pct,
                            "fail_pct": f_pct, "score": score})

            if score > best_score:
                best_score = score
                best_bt, best_st = bt, st
                marker = " ← best"
            else:
                marker = ""
            print(f"  {bt:.2f}    {st:.2f}   {b_pct*100:6.1f}%  {s_pct*100:5.1f}%  "
                  f"{f_pct*100:5.1f}%  {score:.4f}{marker}")

    print(f"\n✅ 최적 Stage 2: bokeh_thresh={best_bt:.2f}, sil_thresh={best_st:.2f}")
    return best_bt, best_st, pd.DataFrame(results)


# ──────────────────────────────────────────
# 시각화
# ──────────────────────────────────────────

def plot_threshold_curve(thresh_df, output_dir):
    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax2 = ax1.twinx()
    ax1.plot(thresh_df["threshold"], thresh_df["f1"],
             color="#185FA5", lw=2, marker="o", ms=4, label="F1")
    ax2.plot(thresh_df["threshold"], thresh_df["auc"],
             color="#D85A30", lw=2, linestyle="--", marker="s", ms=4, label="AUC")
    ax1.set_xlabel("FAIL_THRESHOLD (MOS_zscore)")
    ax1.set_ylabel("F1 Score", color="#185FA5")
    ax2.set_ylabel("AUC", color="#D85A30")
    ax1.set_title("FAIL_THRESHOLD Greedy Search 결과")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, labels1+labels2, loc="lower right")
    best_row = thresh_df.loc[thresh_df["f1"].idxmax()]
    ax1.axvline(best_row["threshold"], color="gray", linestyle=":", alpha=0.7)
    ax1.text(best_row["threshold"]+0.5, best_row["f1"]*0.95,
             f"best={best_row['threshold']:.1f}", fontsize=10, color="gray")
    plt.tight_layout()
    path = os.path.join(output_dir, "tune_threshold.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"임계값 곡선 저장 → {path}")


def plot_optuna_history(trial_df, output_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(trial_df["trial"], trial_df["score"].cummax(),
             color="#185FA5", lw=2, label="Best so far")
    ax1.scatter(trial_df["trial"], trial_df["score"],
                alpha=0.4, s=20, color="#185FA5")
    ax1.set_xlabel("Trial")
    ax1.set_ylabel("Objective Score")
    ax1.set_title("Optuna 탐색 수렴 곡선")
    ax1.legend()

    # 파라미터 중요도 (상위 5개)
    param_cols = ["max_depth","learning_rate","subsample",
                  "colsample_bytree","min_child_weight","gamma"]
    corrs = {p: abs(trial_df[p].corr(trial_df["score"]))
             for p in param_cols if p in trial_df.columns}
    corrs = dict(sorted(corrs.items(), key=lambda x: x[1], reverse=True))
    ax2.barh(list(corrs.keys()), list(corrs.values()), color="#3B6D11")
    ax2.set_xlabel("Score와의 상관계수 (중요도 proxy)")
    ax2.set_title("파라미터 중요도 (상관관계 기반)")
    plt.tight_layout()
    path = os.path.join(output_dir, "tune_optuna.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Optuna 수렴 곡선 저장 → {path}")


def plot_stage2_heatmap(stage2_df, output_dir):
    pivot = stage2_df.pivot_table(
        values="score", index="bokeh_thresh", columns="sil_thresh"
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{v:.2f}" for v in pivot.columns], fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{v:.2f}" for v in pivot.index], fontsize=9)
    ax.set_xlabel("sil_thresh")
    ax.set_ylabel("bokeh_thresh")
    ax.set_title("Stage 2 Grid Search Heatmap")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    path = os.path.join(output_dir, "tune_stage2_heatmap.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Stage 2 Heatmap 저장 → {path}")


# ──────────────────────────────────────────
# 최종 결과 저장
# ──────────────────────────────────────────

def save_best_params(best_thresh, best_xgb, best_bt, best_st, output_dir):
    result = {
        "FAIL_THRESHOLD": best_thresh,
        "xgboost": best_xgb,
        "stage2": {
            "bokeh_thresh": best_bt,
            "sil_thresh":   best_st,
        }
    }
    path = os.path.join(output_dir, "best_params.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"최적 파라미터 저장 → {path}")

    print("\n" + "="*55)
    print("🏆 최종 최적 하이퍼파라미터 요약")
    print("="*55)
    print(f"  FAIL_THRESHOLD  = {best_thresh:.1f}")
    print(f"  bokeh_thresh    = {best_bt:.2f}")
    print(f"  sil_thresh      = {best_st:.2f}")
    print(f"  XGBoost:")
    for k, v in best_xgb.items():
        print(f"    {k:<22} = {v}")
    print("\n→ train_xgboost.py와 stage2_context.py에 위 값을 적용하세요!")


# ──────────────────────────────────────────
# 메인
# ──────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features",    default="./features.csv")
    parser.add_argument("--output_dir",  default="./tuning_results")
    parser.add_argument("--n_trials",    type=int, default=60,
                        help="Optuna 시도 횟수 (많을수록 정확, 기본 60)")
    parser.add_argument("--thresh_step", type=float, default=2.0,
                        help="Threshold Greedy Search 간격 (기본 2.0)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    train, val, test = load_data(args.features)

    print(f"데이터: train={len(train)} / val={len(val)} / test={len(test)}")

    # ── 1단계: FAIL_THRESHOLD Greedy ──
    best_thresh, thresh_df = tune_threshold_greedy(
        train, val, step=args.thresh_step
    )

    # ── 2단계: XGBoost Optuna ──
    best_xgb, trial_df, study = tune_xgboost_optuna(
        train, val, best_thresh, n_trials=args.n_trials
    )

    # ── 3단계: Stage 2 Grid ──
    best_bt, best_st, stage2_df = tune_stage2_grid(
        train, val, best_thresh, best_xgb
    )

    # ── 시각화 ──
    plot_threshold_curve(thresh_df, args.output_dir)
    plot_optuna_history(trial_df, args.output_dir)
    plot_stage2_heatmap(stage2_df, args.output_dir)

    # ── 저장 ──
    save_best_params(best_thresh, best_xgb, best_bt, best_st, args.output_dir)

    thresh_df.to_csv(os.path.join(args.output_dir, "thresh_search.csv"), index=False)
    trial_df.to_csv(os.path.join(args.output_dir, "optuna_trials.csv"), index=False)
    stage2_df.to_csv(os.path.join(args.output_dir, "stage2_grid.csv"), index=False)


if __name__ == "__main__":
    main()