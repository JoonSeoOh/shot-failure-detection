"""
Stage 1 - 특징 추출 스크립트
KonIQ-10k 이미지에서 handcrafted features 추출 → CSV 저장

실행법:
    python extract_features.py --img_dir ./koniq10k/images --csv ./koniq10k/koniq10k_distributions_sets.csv
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse
import warnings
warnings.filterwarnings("ignore")


# ──────────────────────────────────────────
# 특징 추출 함수들
# ──────────────────────────────────────────

def extract_sharpness(gray):
    """선명도: Laplacian 분산 (블러 감지 핵심 지표)"""
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return lap.var()


def extract_brightness(lab):
    """밝기: LAB 색공간의 L채널 통계"""
    L = lab[:, :, 0]
    return {
        "bright_mean": L.mean(),
        "bright_std":  L.std(),
        "bright_low":  np.percentile(L, 5),   # 과노출/암부 체크
        "bright_high": np.percentile(L, 95),
    }


def extract_colorfulness(bgr):
    """
    채도/화려함 지수 (Hasler & Süsstrunk 2003)
    낮으면 색이 빠진 사진 / 회색빛 사진
    """
    (B, G, R) = cv2.split(bgr.astype("float"))
    rg = np.abs(R - G)
    yb = np.abs(0.5 * (R + G) - B)
    return np.sqrt(rg.std()**2 + yb.std()**2) + 0.3 * np.sqrt(rg.mean()**2 + yb.mean()**2)


def extract_noise(gray):
    """
    노이즈 추정: 고주파 잔차 (미디안 필터 전/후 차이)
    높을수록 노이즈가 심함
    """
    blurred = cv2.medianBlur(gray, 5)
    diff = cv2.absdiff(gray, blurred).astype("float")
    return diff.mean()


def extract_contrast(gray):
    """대비: RMS contrast"""
    gray_f = gray.astype("float") / 255.0
    return gray_f.std()


def extract_saturation(hsv):
    """HSV 채도 통계"""
    S = hsv[:, :, 1]
    return {
        "sat_mean": S.mean(),
        "sat_std":  S.std(),
    }


def extract_exposure(gray):
    """
    노출 평가
    - 히스토그램 양 끝단 집중도로 과/부족 노출 감지
    """
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist_norm = hist / hist.sum()
    underexposed = hist_norm[:30].sum()    # 어두운 픽셀 비율
    overexposed  = hist_norm[225:].sum()   # 밝은 픽셀 비율
    return {
        "under_exp": underexposed,
        "over_exp":  overexposed,
    }


def extract_edge_density(gray):
    """
    엣지 밀도: Canny 엣지 비율
    - 너무 낮으면 흐릿한 사진
    - 너무 높으면 노이즈가 심한 사진
    """
    edges = cv2.Canny(gray, 50, 150)
    return edges.mean()


def extract_dct_energy(gray):
    """
    DCT 고주파 에너지 비율
    - 블러 사진 = 고주파 성분 소실
    """
    gray_f = gray.astype("float32")
    # 중앙 512x512 크롭 (속도 최적화)
    h, w = gray_f.shape
    cy, cx = h // 2, w // 2
    crop = gray_f[max(0,cy-256):cy+256, max(0,cx-256):cx+256]
    dct = cv2.dct(crop)
    total   = (dct**2).sum() + 1e-8
    hf_mask = np.zeros_like(dct)
    hf_mask[dct.shape[0]//4:, dct.shape[1]//4:] = 1
    hf_energy = ((dct**2) * hf_mask).sum()
    return hf_energy / total


def extract_all_features(img_path):
    """모든 특징을 하나의 dict로 반환"""
    bgr = cv2.imread(str(img_path))
    if bgr is None:
        return None

    # 처리 속도를 위해 긴 변 기준 512px 리사이즈
    h, w = bgr.shape[:2]
    scale = 512 / max(h, w)
    if scale < 1.0:
        bgr = cv2.resize(bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    lab  = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    hsv  = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    feats = {}
    feats["sharpness"]    = extract_sharpness(gray)
    feats["colorfulness"] = extract_colorfulness(bgr)
    feats["noise"]        = extract_noise(gray)
    feats["contrast"]     = extract_contrast(gray)
    feats["edge_density"] = extract_edge_density(gray)
    feats["dct_hf_ratio"] = extract_dct_energy(gray)
    feats.update(extract_brightness(lab))
    feats.update(extract_saturation(hsv))
    feats.update(extract_exposure(gray))

    # 원본 해상도 정보도 특징으로
    feats["orig_width"]  = w
    feats["orig_height"] = h
    feats["aspect_ratio"] = w / h

    return feats


# ──────────────────────────────────────────
# 메인 실행
# ──────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", default="./koniq10k/images",
                        help="이미지 폴더 경로")
    parser.add_argument("--csv",     default="./koniq10k/koniq10k_distributions_sets.csv",
                        help="레이블 CSV 경로")
    parser.add_argument("--output",  default="./features.csv",
                        help="결과 저장 경로")
    parser.add_argument("--limit",   type=int, default=None,
                        help="테스트용 제한 수 (예: --limit 100)")
    args = parser.parse_args()

    img_dir = Path(args.img_dir)
    label_df = pd.read_csv(args.csv)

    if args.limit:
        label_df = label_df.head(args.limit)
        print(f"[테스트 모드] {args.limit}장만 처리합니다.")

    results = []
    failed  = []

    for _, row in tqdm(label_df.iterrows(), total=len(label_df), desc="특징 추출"):
        img_path = img_dir / row["image_name"]
        feats = extract_all_features(img_path)

        if feats is None:
            failed.append(row["image_name"])
            continue

        feats["image_name"] = row["image_name"]
        feats["MOS"]        = row["MOS"]
        feats["set"]        = row["set"]
        results.append(feats)

    feat_df = pd.DataFrame(results)

    # 컬럼 순서 정렬: 메타 → 특징 → 레이블
    meta_cols  = ["image_name", "set", "MOS"]
    feat_cols  = [c for c in feat_df.columns if c not in meta_cols]
    feat_df    = feat_df[meta_cols + feat_cols]

    feat_df.to_csv(args.output, index=False)

    print(f"\n✅ 완료! {len(feat_df)}장 처리 → {args.output}")
    print(f"   특징 수: {len(feat_cols)}개")
    if failed:
        print(f"   ⚠️  읽기 실패: {len(failed)}장 ({failed[:5]}...)")
    print(f"\n특징 목록: {feat_cols}")


if __name__ == "__main__":
    main()