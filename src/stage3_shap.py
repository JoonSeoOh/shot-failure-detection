"""
Stage 3 - SHAP 기반 개선 조언 생성
true_failures.csv → 각 사진마다 "왜 실패했는지 + 어떻게 고칠지" 분석

실행법:
    python stage3_shap.py \
        --true_failures ./output/true_failures.csv \
        --features      ./features.csv \
        --model         ./output/xgb_model.pkl \
        --output_dir    ./output
"""

import argparse, os, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
warnings.filterwarnings("ignore")
import json, tempfile



FEATURE_COLS = [
    "sharpness", "colorfulness", "noise", "contrast", "edge_density",
    "dct_hf_ratio", "bright_mean", "bright_std", "bright_low", "bright_high",
    "sat_mean", "sat_std", "under_exp", "over_exp",
    "orig_width", "orig_height", "aspect_ratio"
]

# ──────────────────────────────────────────
# 조언 생성 엔진
# ──────────────────────────────────────────

# 각 특징별 조언 템플릿
# (조건 함수, 조언 텍스트, 심각도)
ADVICE_RULES = {
    "sharpness": [
        (lambda v: v < 30,   "심각한 블러 — 삼각대 사용 또는 셔터 속도를 높이세요 (1/250s 이상 권장)", "high"),
        (lambda v: v < 80,   "약한 블러 — 손떨림 보정(OIS) 활성화 또는 셔터 속도를 높이세요", "mid"),
        (lambda v: v < 150,  "미세한 흔들림 — 셔터 속도를 약간 높이거나 ISO를 낮추세요", "low"),
    ],
    "under_exp": [
        (lambda v: v > 0.40, "심각한 노출 부족 — EV +1.5~2.0 보정 또는 ISO를 높이세요", "high"),
        (lambda v: v > 0.20, "노출 부족 — EV +0.7~1.0 보정을 권장합니다", "mid"),
        (lambda v: v > 0.10, "약간 어두움 — EV +0.3 보정 또는 후보정으로 밝기 조정 가능", "low"),
    ],
    "over_exp": [
        (lambda v: v > 0.30, "심각한 노출 과다 — EV -1.5~2.0 보정 또는 ND필터 사용을 권장합니다", "high"),
        (lambda v: v > 0.15, "노출 과다 — EV -0.7~1.0 보정을 권장합니다", "mid"),
        (lambda v: v > 0.08, "약간 밝음 — EV -0.3 보정 또는 하이라이트 복구 가능", "low"),
    ],
    "noise": [
        (lambda v: v > 8.0,  "심각한 노이즈 — ISO를 낮추고(ISO 800 이하 권장) 조리개를 더 열어보세요", "high"),
        (lambda v: v > 5.0,  "노이즈 과다 — ISO를 낮추거나 노이즈 감소 후보정을 적용하세요", "mid"),
        (lambda v: v > 3.0,  "약간의 노이즈 — 후보정 노이즈 감소로 개선 가능합니다", "low"),
    ],
    "contrast": [
        (lambda v: v < 0.08, "대비가 너무 낮음 — 후보정에서 대비/클래리티를 올려보세요", "mid"),
        (lambda v: v < 0.05, "심각하게 밋밋한 사진 — 히스토그램 스트레칭 또는 커브 조정 필요", "high"),
    ],
    "colorfulness": [
        (lambda v: v < 15,   "색감이 거의 없음 — 채도/생동감을 높이거나 화이트밸런스를 확인하세요", "mid"),
        (lambda v: v < 8,    "완전히 색이 빠진 사진 — 화이트밸런스 오류 또는 안개/흐린 날씨 영향", "high"),
    ],
    "dct_hf_ratio": [
        (lambda v: v < 0.02, "주파수 분석상 블러 확인 — 피사체 가까이에서 재촬영을 권장합니다", "mid"),
    ],
    "edge_density": [
        (lambda v: v < 5,    "엣지 정보가 거의 없음 — 피사체 거리와 초점을 다시 확인하세요", "mid"),
    ],
}

SEVERITY_ORDER = {"high": 0, "mid": 1, "low": 2}
SEVERITY_EMOJI = {"high": "🔴", "mid": "🟡", "low": "🟢"}


def get_advice(feat_name, feat_value):
    """특징값 → 조언 반환"""
    rules = ADVICE_RULES.get(feat_name, [])
    for cond, text, sev in rules:
        try:
            if cond(feat_value):
                return text, sev
        except Exception:
            continue
    return None, None


def generate_advice_for_photo(row, shap_vals, feature_names):
    """
    사진 한 장에 대한 조언 생성
    - SHAP 절댓값 기준 상위 특징만 분석
    - 조언이 있는 것만 반환
    """
    # SHAP 크기 순으로 정렬
    shap_order = np.argsort(np.abs(shap_vals))[::-1]
    advices = []

    for idx in shap_order[:8]:  # 상위 8개 특징만 검토
        fname = feature_names[idx]
        fval  = row.get(fname, None)
        if fval is None:
            continue

        text, sev = get_advice(fname, fval)
        if text:
            advices.append({
                "feature":    fname,
                "value":      round(fval, 4),
                "shap":       round(shap_vals[idx], 4),
                "advice":     text,
                "severity":   sev,
            })

    # 심각도 순 정렬, 최대 3개
    advices.sort(key=lambda x: SEVERITY_ORDER[x["severity"]])
    return advices[:3]


# ──────────────────────────────────────────
# 시각화
# ──────────────────────────────────────────

def plot_shap_summary(explainer, shap_values, X, output_dir):
    """전체 SHAP summary plot"""
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X, show=False, plot_size=None)
    plt.title("SHAP Feature Impact (True Fail 사진 전체)", fontsize=13)
    plt.tight_layout()
    path = os.path.join(output_dir, "shap_summary.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"SHAP summary 저장 → {path}")


def plot_failure_reasons(advice_df, output_dir):
    """실패 원인 빈도 차트"""
    if advice_df.empty:
        return

    feat_counts = advice_df["feature"].value_counts().head(10)
    sev_colors = {
        "high": "#e74c3c",
        "mid":  "#f39c12",
        "low":  "#3498db",
    }

    # 각 특징의 대표 심각도
    feat_sev = (
        advice_df.groupby("feature")["severity"]
        .agg(lambda x: x.value_counts().index[0])
        .to_dict()
    )
    colors = [sev_colors.get(feat_sev.get(f, "low"), "#3498db")
              for f in feat_counts.index]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(feat_counts.index[::-1], feat_counts.values[::-1],
                   color=colors[::-1], edgecolor="white")
    for bar, v in zip(bars, feat_counts.values[::-1]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f"{v}장", va="center", fontsize=10)

    ax.set_xlabel("해당 원인으로 실패한 사진 수")
    ax.set_title("실패 원인 TOP 10", fontsize=13, fontweight="bold")

    from matplotlib.patches import Patch
    legend = [Patch(facecolor=c, label=l)
              for l, c in [("심각(high)","#e74c3c"),("보통(mid)","#f39c12"),("경미(low)","#3498db")]]
    ax.legend(handles=legend, loc="lower right")
    plt.tight_layout()
    path = os.path.join(output_dir, "failure_reasons.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"실패 원인 차트 저장 → {path}")


def plot_severity_pie(advice_df, output_dir):
    """심각도 분포 파이차트"""
    if advice_df.empty:
        return
    counts = advice_df["severity"].value_counts()
    colors = {"high": "#e74c3c", "mid": "#f39c12", "low": "#3498db"}
    labels = [f"{SEVERITY_EMOJI[k]} {k}\n({v}건)"
              for k, v in counts.items() if k in colors]
    cols   = [colors[k] for k in counts.index if k in colors]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.pie(counts.values, labels=labels, colors=cols,
           autopct="%1.1f%%", startangle=90,
           wedgeprops=dict(edgecolor="white", linewidth=2))
    ax.set_title("조언 심각도 분포", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "severity_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"심각도 분포 저장 → {path}")


# ──────────────────────────────────────────
# 메인
# ──────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--true_failures", default="./output/true_failures.csv")
    parser.add_argument("--features",      default="./features.csv")
    parser.add_argument("--model",         default="./output/xgb_model.pkl")
    parser.add_argument("--output_dir",    default="./output")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 모델 로드
    with open(args.model, "rb") as f:
        model = pickle.load(f)
    print("✅ 모델 로드 완료")

    # 실패 사진 + 특징값 합치기
    fail_df = pd.read_csv(args.true_failures)
    feat_df = pd.read_csv(args.features)
    df = fail_df.merge(feat_df[["image_name"] + FEATURE_COLS],
                       on="image_name", how="left")
    df = df.dropna(subset=FEATURE_COLS)
    print(f"분석 대상: {len(df)}장")

    X = df[FEATURE_COLS]

    # SHAP 계산
    print("🔍 SHAP 계산 중...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # XGBoost binary → shap_values shape 처리
    if isinstance(shap_values, list):
        sv = shap_values[1]   # 실패 클래스 SHAP
    else:
        sv = shap_values

    print(f"SHAP 계산 완료 (shape: {sv.shape})")

    # 개별 조언 생성
    print("💬 조언 생성 중...")
    all_advices = []
    photo_reports = []

    for i, (_, row) in enumerate(df.iterrows()):
        row_dict  = row.to_dict()
        advices   = generate_advice_for_photo(row_dict, sv[i], FEATURE_COLS)

        # 사진별 요약
        if advices:
            main_cause = advices[0]["feature"]
            main_sev   = advices[0]["severity"]
            main_advice= advices[0]["advice"]
        else:
            main_cause = "unknown"
            main_sev   = "low"
            main_advice= "특별한 문제가 발견되지 않았습니다"

        photo_reports.append({
            "image_name":   row["image_name"],
            "MOS":          row.get("MOS", None),
            "pred_prob":    row.get("pred_prob", None),
            "main_cause":   main_cause,
            "main_severity":main_sev,
            "advice_1":     advices[0]["advice"] if len(advices) > 0 else "",
            "advice_2":     advices[1]["advice"] if len(advices) > 1 else "",
            "advice_3":     advices[2]["advice"] if len(advices) > 2 else "",
        })

        for adv in advices:
            all_advices.append({"image_name": row["image_name"], **adv})

    report_df = pd.DataFrame(photo_reports)
    advice_df = pd.DataFrame(all_advices)

    # 저장
    report_path = os.path.join(args.output_dir, "photo_advice_report.csv")
    advice_path = os.path.join(args.output_dir, "all_advices.csv")
    report_df.to_csv(report_path, index=False)
    advice_df.to_csv(advice_path, index=False)

    # 결과 출력
    print("\n" + "="*55)
    print("📋 Stage 3 분석 결과")
    print("="*55)
    if not advice_df.empty:
        print("\n🔺 주요 실패 원인 TOP 5:")
        top5 = advice_df["feature"].value_counts().head(5)
        for feat, cnt in top5.items():
            pct = cnt / len(df) * 100
            print(f"  {feat:20s}: {cnt:3d}장 ({pct:.1f}%)")

        print("\n🎯 심각도 분포:")
        for sev, cnt in advice_df["severity"].value_counts().items():
            print(f"  {SEVERITY_EMOJI.get(sev,'?')} {sev}: {cnt}건")

    # 샘플 출력
    print("\n📸 샘플 조언 (상위 5장):")
    for _, row in report_df.head(5).iterrows():
        print(f"\n  [{row['image_name']}]")
        for col in ["advice_1", "advice_2", "advice_3"]:
            if row[col]:
                print(f"    → {row[col]}")

    # 시각화
    plot_shap_summary(explainer, sv, X, args.output_dir)
    plot_failure_reasons(advice_df, args.output_dir)
    plot_severity_pie(advice_df, args.output_dir)

    print(f"\n리포트 저장  → {report_path}")
    print(f"전체 조언    → {advice_path}")
    print("\n🎉 3-Stage 파이프라인 완료!")
    print("   Stage 1: XGBoost 품질 예측  ✅")
    print("   Stage 2: Bokeh/실루엣 구분  ✅")
    print("   Stage 3: SHAP 개선 조언     ✅")


if __name__ == "__main__":
    main()