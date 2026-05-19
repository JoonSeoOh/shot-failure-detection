"""
Stage 2 - Context-Aware 분류기
실패 판정 사진을 3가지로 재분류:
  1. bokeh      : 의도적 아웃포커스 (실패 아님 ✅)
  2. silhouette : 의도적 실루엣/역광 (실패 아님 ✅)
  3. true_fail  : 진짜 촬영 실패 ❌

실행법:
    python stage2_context.py \
        --failed   ./output/failed_photos.csv \
        --img_dir  ./koniq10k/images \
        --output_dir ./output
"""

import argparse, os, warnings
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
warnings.filterwarnings("ignore")


# ──────────────────────────────────────────
# 핵심 분석 함수
# ──────────────────────────────────────────

def spatial_sharpness(gray, zones=3):
    """
    이미지를 zones×zones 격자로 나눠 각 구역 선명도 반환
    Bokeh 판별 핵심: 중앙은 선명 + 주변은 흐릿
    """
    h, w = gray.shape
    zh, zw = h // zones, w // zones
    grid = []
    for r in range(zones):
        row = []
        for c in range(zones):
            tile = gray[r*zh:(r+1)*zh, c*zw:(c+1)*zw]
            row.append(cv2.Laplacian(tile, cv2.CV_64F).var())
        grid.append(row)
    return np.array(grid)


def bokeh_score(gray):
    """
    Bokeh 점수 계산 (0~1)
    원리: 중앙 선명도 / 전체 선명도 비율
    - 중앙이 뚜렷하게 선명하고 주변이 흐리면 → Bokeh
    """
    grid = spatial_sharpness(gray, zones=3)

    center      = grid[1, 1]                          # 중앙 타일
    surround    = np.concatenate([grid[0,:], grid[2,:], grid[:,0], grid[:,2]]).mean()
    total_sharp = grid.mean()

    if total_sharp < 1e-6:
        return 0.0, grid

    # 비율: 중앙이 주변보다 얼마나 선명한가
    ratio = center / (surround + 1e-6)

    # 전체가 너무 흐리면 bokeh 아닌 진짜 블러
    if total_sharp < 50:
        score = 0.0
    else:
        # ratio가 2.0 이상이면 bokeh 가능성 높음
        score = min(1.0, (ratio - 1.0) / 3.0)

    return max(0.0, score), grid


def silhouette_score(gray, bgr):
    """
    실루엣/역광 점수 계산 (0~1)
    원리:
    1. 밝기 히스토그램이 양봉(어두운 주체 + 밝은 배경)이면 실루엣
    2. 엣지가 선명 (실루엣은 경계가 뚜렷)
    3. 어두운 영역이 전체의 10~50% 차지
    """
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist_norm = hist / hist.sum()

    dark_ratio   = hist_norm[:60].sum()    # 어두운 픽셀 비율
    bright_ratio = hist_norm[180:].sum()   # 밝은 픽셀 비율
    mid_ratio    = hist_norm[60:180].sum() # 중간 밝기 비율

    # 실루엣 조건: 어두운 부분 + 밝은 부분이 뚜렷하게 나뉨
    # 중간 밝기가 낮고, 양쪽 극단이 높은 경우
    bimodal = (dark_ratio + bright_ratio) - mid_ratio * 0.5

    # 엣지 선명도 (실루엣은 경계가 뚜렷)
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = edges.mean() / 255.0

    # 구성: 어두운 피사체(10~50%)가 있어야 실루엣
    has_subject = 0.08 < dark_ratio < 0.55

    score = 0.0
    if has_subject and bimodal > 0.1:
        score = min(1.0, bimodal * 1.5 + edge_ratio * 2.0)

    return max(0.0, score), {
        "dark_ratio":   dark_ratio,
        "bright_ratio": bright_ratio,
        "bimodal":      bimodal,
        "edge_ratio":   edge_ratio,
    }


def classify_photo(img_path, bokeh_thresh=0.35, sil_thresh=0.30):
    """
    사진 한 장을 분류
    Returns: (label, bokeh_sc, sil_sc, details)
    """
    bgr = cv2.imread(str(img_path))
    if bgr is None:
        return "error", 0, 0, {}

    # 처리 속도 최적화: 512px 리사이즈
    h, w = bgr.shape[:2]
    scale = 512 / max(h, w)
    if scale < 1.0:
        bgr = cv2.resize(bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    bokeh_sc, grid = bokeh_score(gray)
    sil_sc, sil_details = silhouette_score(gray, bgr)

    # 분류 결정 (bokeh 우선)
    if bokeh_sc >= bokeh_thresh:
        label = "bokeh"
    elif sil_sc >= sil_thresh:
        label = "silhouette"
    else:
        label = "true_fail"

    details = {"bokeh_score": bokeh_sc, "sil_score": sil_sc, **sil_details}
    return label, bokeh_sc, sil_sc, details


# ──────────────────────────────────────────
# 시각화
# ──────────────────────────────────────────

def plot_distribution(result_df, output_dir):
    counts = result_df["context_label"].value_counts()
    labels_order = ["bokeh", "silhouette", "true_fail"]
    colors = {"bokeh": "#2ecc71", "silhouette": "#f39c12", "true_fail": "#e74c3c"}

    vals   = [counts.get(l, 0) for l in labels_order]
    cols   = [colors[l] for l in labels_order]
    names  = [f"Bokeh\n(의도적 블러)", "Silhouette\n(역광/실루엣)", "True Fail\n(진짜 실패)"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 바 차트
    bars = ax1.bar(names, vals, color=cols, edgecolor="white", linewidth=1.5)
    for bar, v in zip(bars, vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                 str(v), ha="center", va="bottom", fontweight="bold", fontsize=12)
    ax1.set_title("Stage 2 분류 결과", fontsize=14, fontweight="bold")
    ax1.set_ylabel("사진 수")
    ax1.set_ylim(0, max(vals) * 1.2)

    # 파이 차트
    non_zero = [(v, n, c) for v, n, c in zip(vals, names, cols) if v > 0]
    ax2.pie([x[0] for x in non_zero],
            labels=[x[1] for x in non_zero],
            colors=[x[2] for x in non_zero],
            autopct="%1.1f%%", startangle=90,
            wedgeprops=dict(edgecolor="white", linewidth=2))
    ax2.set_title("비율", fontsize=14, fontweight="bold")

    plt.suptitle(f"총 {len(result_df)}장 재분류 결과", fontsize=13, y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, "stage2_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"분포 시각화 저장 → {path}")


def plot_score_scatter(result_df, output_dir):
    colors = {"bokeh": "#2ecc71", "silhouette": "#f39c12", "true_fail": "#e74c3c"}
    fig, ax = plt.subplots(figsize=(8, 6))

    for label, grp in result_df.groupby("context_label"):
        ax.scatter(grp["bokeh_score"], grp["sil_score"],
                   c=colors.get(label, "gray"), label=label,
                   alpha=0.6, s=30, edgecolors="none")

    # threshold 선
    ax.axvline(0.35, color="green",  linestyle="--", alpha=0.5, label="bokeh threshold")
    ax.axhline(0.30, color="orange", linestyle="--", alpha=0.5, label="sil threshold")
    ax.set_xlabel("Bokeh Score", fontsize=12)
    ax.set_ylabel("Silhouette Score", fontsize=12)
    ax.set_title("Stage 2 점수 분포", fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, "stage2_scatter.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"점수 산포도 저장 → {path}")


# ──────────────────────────────────────────
# 메인
# ──────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--failed",      default="./output/failed_photos.csv")
    parser.add_argument("--img_dir",     default="./koniq10k/images")
    parser.add_argument("--output_dir",  default="./output")
    parser.add_argument("--bokeh_thresh", type=float, default=0.35)
    parser.add_argument("--sil_thresh",   type=float, default=0.30)
    parser.add_argument("--limit",        type=int,   default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    failed_df = pd.read_csv(args.failed)
    if args.limit:
        failed_df = failed_df.head(args.limit)
        print(f"[테스트 모드] {args.limit}장만 처리")

    print(f"분류 대상: {len(failed_df)}장")
    print(f"Bokeh threshold: {args.bokeh_thresh} / Silhouette threshold: {args.sil_thresh}\n")

    from pathlib import Path
    img_dir = Path(args.img_dir)
    results = []

    for _, row in tqdm(failed_df.iterrows(), total=len(failed_df), desc="Context 분류"):
        img_path = img_dir / row["image_name"]
        label, b_sc, s_sc, details = classify_photo(
            img_path, args.bokeh_thresh, args.sil_thresh
        )
        results.append({
            "image_name":    row["image_name"],
            "MOS":           row["MOS"],
            "pred_prob":     row.get("pred_prob", None),
            "context_label": label,
            "bokeh_score":   round(b_sc, 4),
            "sil_score":     round(s_sc, 4),
            **{k: round(v, 4) if isinstance(v, float) else v
               for k, v in details.items() if k not in ("bokeh_score","sil_score")},
        })

    result_df = pd.DataFrame(results)

    # 결과 출력
    print("\n" + "="*50)
    print("📊 Stage 2 분류 결과")
    print("="*50)
    counts = result_df["context_label"].value_counts()
    total  = len(result_df)
    for label in ["bokeh", "silhouette", "true_fail"]:
        n = counts.get(label, 0)
        icon = {"bokeh":"✅","silhouette":"✅","true_fail":"❌"}[label]
        print(f"  {icon} {label:12s}: {n:4d}장  ({n/total*100:.1f}%)")

    true_fail_df = result_df[result_df["context_label"] == "true_fail"]
    print(f"\n→ Stage 3 (SHAP) 대상: {len(true_fail_df)}장")

    # 저장
    all_path  = os.path.join(args.output_dir, "stage2_results.csv")
    fail_path = os.path.join(args.output_dir, "true_failures.csv")
    result_df.to_csv(all_path,  index=False)
    true_fail_df.to_csv(fail_path, index=False)
    print(f"\n전체 결과 저장  → {all_path}")
    print(f"진짜 실패 저장  → {fail_path}")

    # 시각화
    plot_distribution(result_df, args.output_dir)
    plot_score_scatter(result_df, args.output_dir)

    print("\n✅ Stage 2 완료! 다음 단계: python stage3_shap.py")


if __name__ == "__main__":
    main()