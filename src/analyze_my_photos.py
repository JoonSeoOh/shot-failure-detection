"""
내 사진 분석 파이프라인 (3-Stage 통합)
my_test_data 폴더의 사진을 Stage1 → Stage2 → Stage3 순서로 분석

실행법:
    python analyze_my_photos.py
    python analyze_my_photos.py --photo_dir ./my_test_data --output_dir ./my_results
"""

import argparse, os, pickle, json, warnings, tempfile
import cv2
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from tqdm import tqdm
warnings.filterwarnings("ignore")

# ── 경로 기본값 ──────────────────────────────
DEFAULT_PHOTO_DIR  = "./my_test_data"
DEFAULT_MODEL_PATH = "./output/xgb_model.pkl"
DEFAULT_OUTPUT_DIR = "./my_results"
FAIL_THRESHOLD     = 45.0   # train_xgboost.py와 동일한 기준
FEATURE_COLS = [
    "sharpness","colorfulness","noise","contrast","edge_density",
    "dct_hf_ratio","bright_mean","bright_std","bright_low","bright_high",
    "sat_mean","sat_std","under_exp","over_exp",
    "orig_width","orig_height","aspect_ratio"
]
IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".webp",".tiff",".tif"}

# ════════════════════════════════════════════
# STAGE 1 — 특징 추출 + XGBoost 품질 예측
# ════════════════════════════════════════════

def extract_features_single(img_path):
    bgr = cv2.imread(str(img_path))
    if bgr is None:
        return None
    h, w = bgr.shape[:2]
    scale = 512 / max(h, w)
    if scale < 1.0:
        bgr = cv2.resize(bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    lab  = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    hsv  = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    lap  = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = lap.var()

    B, G, R = cv2.split(bgr.astype("float"))
    rg = np.abs(R-G); yb = np.abs(0.5*(R+G)-B)
    colorfulness = np.sqrt(rg.std()**2+yb.std()**2)+0.3*np.sqrt(rg.mean()**2+yb.mean()**2)

    blurred = cv2.medianBlur(gray,5)
    noise   = cv2.absdiff(gray,blurred).astype("float").mean()

    gray_f  = gray.astype("float")/255.0
    contrast= gray_f.std()

    edges      = cv2.Canny(gray,50,150)
    edge_density= edges.mean()

    gray_f32 = gray.astype("float32")
    cy,cx    = gray_f32.shape[0]//2, gray_f32.shape[1]//2
    crop     = gray_f32[max(0,cy-256):cy+256, max(0,cx-256):cx+256]

    # --- 추가된 부분: 크기가 홀수일 경우 짝수로 맞춤 ---
    ch, cw = crop.shape
    if ch % 2 != 0: crop = crop[:-1, :]
    if cw % 2 != 0: crop = crop[:, :-1]
    # ---------------------------------------------------

    dct      = cv2.dct(crop)
    total    = (dct**2).sum()+1e-8
    hf_mask  = np.zeros_like(dct)
    hf_mask[dct.shape[0]//4:, dct.shape[1]//4:] = 1
    dct_hf_ratio = ((dct**2)*hf_mask).sum()/total

    L = lab[:,:,0]
    bright_mean = L.mean(); bright_std = L.std()
    bright_low  = np.percentile(L,5); bright_high = np.percentile(L,95)

    S = hsv[:,:,1]
    sat_mean = S.mean(); sat_std = S.std()

    hist = cv2.calcHist([gray],[0],None,[256],[0,256]).flatten()
    hist_norm = hist/hist.sum()
    under_exp = hist_norm[:30].sum()
    over_exp  = hist_norm[225:].sum()

    return {
        "sharpness":sharpness,"colorfulness":colorfulness,"noise":noise,
        "contrast":contrast,"edge_density":edge_density,"dct_hf_ratio":dct_hf_ratio,
        "bright_mean":bright_mean,"bright_std":bright_std,
        "bright_low":bright_low,"bright_high":bright_high,
        "sat_mean":sat_mean,"sat_std":sat_std,
        "under_exp":under_exp,"over_exp":over_exp,
        "orig_width":w,"orig_height":h,"aspect_ratio":w/h,
    }


def run_stage1(photo_dir, model_path):
    print("\n" + "═"*55)
    print("📷  STAGE 1 — XGBoost 품질 예측")
    print("═"*55)

    with open(model_path,"rb") as f:
        model = pickle.load(f)

    photos = sorted([p for p in Path(photo_dir).iterdir()
                     if p.suffix.lower() in IMG_EXTS])
    if not photos:
        raise FileNotFoundError(f"{photo_dir} 에 이미지가 없습니다.")
    print(f"사진 {len(photos)}장 발견")

    rows = []
    for p in tqdm(photos, desc="특징 추출"):
        feats = extract_features_single(p)
        if feats is None:
            print(f"  ⚠️  읽기 실패: {p.name}")
            continue
        feats["image_name"] = p.name
        rows.append(feats)

    feat_df = pd.DataFrame(rows)
    X = feat_df[FEATURE_COLS]
    feat_df["pred_prob"]  = model.predict_proba(X)[:,1]
    feat_df["pred_label"] = model.predict(X)
    feat_df["stage1_result"] = feat_df["pred_label"].map({0:"정상 ✅", 1:"실패 ❌"})

    # MOS_zscore 추정 (모델 출력 확률을 0~100 스케일로 변환)
    feat_df["quality_score"] = ((1 - feat_df["pred_prob"]) * 100).round(1)

    print(f"\n{'파일명':<30} {'품질점수':>8} {'판정':>10} {'확률':>8}")
    print("-"*60)
    for _, r in feat_df.iterrows():
        print(f"  {r['image_name']:<28} {r['quality_score']:>7.1f}점  {r['stage1_result']:>8}  ({r['pred_prob']*100:.1f}%)")

    fail_cnt = feat_df["pred_label"].sum()
    ok_cnt   = len(feat_df) - fail_cnt
    print(f"\n결과: 정상 {ok_cnt}장 / 실패 {fail_cnt}장")
    return feat_df, model


# ════════════════════════════════════════════
# STAGE 2 — Context-Aware 분류
# ════════════════════════════════════════════

def bokeh_score_fn(gray):
    h,w = gray.shape
    zh,zw = h//3, w//3
    grid = []
    for r in range(3):
        row=[]
        for c in range(3):
            tile = gray[r*zh:(r+1)*zh, c*zw:(c+1)*zw]
            row.append(cv2.Laplacian(tile,cv2.CV_64F).var())
        grid.append(row)
    grid = np.array(grid)
    center   = grid[1,1]
    surround = np.concatenate([grid[0,:],grid[2,:],grid[:,0],grid[:,2]]).mean()
    total    = grid.mean()
    if total < 1e-6: return 0.0
    ratio = center/(surround+1e-6)
    score = 0.0 if total < 50 else min(1.0,(ratio-1.0)/3.0)
    return max(0.0, score)

def silhouette_score_fn(gray):
    hist = cv2.calcHist([gray],[0],None,[256],[0,256]).flatten()
    hn = hist/hist.sum()
    dark_ratio   = hn[:60].sum()
    bright_ratio = hn[180:].sum()
    mid_ratio    = hn[60:180].sum()
    bimodal      = (dark_ratio+bright_ratio)-mid_ratio*0.5
    edges        = cv2.Canny(gray,50,150)
    edge_ratio   = edges.mean()/255.0
    has_subject  = 0.08 < dark_ratio < 0.55
    score = 0.0
    if has_subject and bimodal > 0.1:
        score = min(1.0, bimodal*1.5+edge_ratio*2.0)
    return max(0.0, score)

def run_stage2(feat_df, photo_dir, bokeh_thresh=0.35, sil_thresh=0.30):
    print("\n" + "═"*55)
    print("🔍  STAGE 2 — Context-Aware 분류")
    print("═"*55)

    results = []
    for _, row in feat_df.iterrows():
        img_path = Path(photo_dir) / row["image_name"]
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            results.append({"context_label":"error","bokeh_score":0,"sil_score":0})
            continue
        h,w = bgr.shape[:2]
        scale = 512/max(h,w)
        if scale < 1.0:
            bgr = cv2.resize(bgr,(int(w*scale),int(h*scale)),interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)

        b_sc = bokeh_score_fn(gray)
        s_sc = silhouette_score_fn(gray)

        if row["pred_label"] == 0:
            label = "정상"
        elif b_sc >= bokeh_thresh:
            label = "bokeh"
        elif s_sc >= sil_thresh:
            label = "silhouette"
        else:
            label = "true_fail"

        results.append({"context_label":label,
                        "bokeh_score":round(b_sc,4),
                        "sil_score":round(s_sc,4)})

    ctx_df = pd.DataFrame(results)
    feat_df = pd.concat([feat_df.reset_index(drop=True), ctx_df], axis=1)

    LABEL_KO = {"정상":"정상 ✅","bokeh":"Bokeh ✅","silhouette":"실루엣 ✅","true_fail":"진짜 실패 ❌","error":"오류"}
    print(f"\n{'파일명':<30} {'Stage1':>10} {'Stage2':>14}")
    print("-"*58)
    for _, r in feat_df.iterrows():
        s2 = LABEL_KO.get(r["context_label"], r["context_label"])
        print(f"  {r['image_name']:<28} {r['stage1_result']:>10}  {s2:>12}")

    return feat_df


# ════════════════════════════════════════════
# STAGE 3 — SHAP 개선 조언
# ════════════════════════════════════════════

ADVICE_RULES = {
    "sharpness":[
        (lambda v:v<30,  "심각한 블러 — 삼각대 사용 또는 셔터 속도를 높이세요 (1/250s 이상 권장)","high"),
        (lambda v:v<80,  "약한 블러 — 손떨림 보정(OIS) 활성화 또는 셔터 속도를 높이세요","mid"),
        (lambda v:v<150, "미세한 흔들림 — 셔터 속도를 약간 높이거나 ISO를 낮추세요","low"),
    ],
    "under_exp":[
        (lambda v:v>0.40,"심각한 노출 부족 — EV +1.5~2.0 보정 또는 ISO를 높이세요","high"),
        (lambda v:v>0.20,"노출 부족 — EV +0.7~1.0 보정을 권장합니다","mid"),
        (lambda v:v>0.10,"약간 어두움 — EV +0.3 보정 또는 후보정으로 밝기 조정 가능","low"),
    ],
    "over_exp":[
        (lambda v:v>0.30,"심각한 노출 과다 — EV -1.5~2.0 보정 또는 ND필터 사용을 권장합니다","high"),
        (lambda v:v>0.15,"노출 과다 — EV -0.7~1.0 보정을 권장합니다","mid"),
        (lambda v:v>0.08,"약간 밝음 — EV -0.3 보정 또는 하이라이트 복구 가능","low"),
    ],
    "noise":[
        (lambda v:v>8.0, "심각한 노이즈 — ISO를 낮추고(ISO 800 이하 권장) 조리개를 더 열어보세요","high"),
        (lambda v:v>5.0, "노이즈 과다 — ISO를 낮추거나 노이즈 감소 후보정을 적용하세요","mid"),
        (lambda v:v>3.0, "약간의 노이즈 — 후보정 노이즈 감소로 개선 가능합니다","low"),
    ],
    "contrast":[
        (lambda v:v<0.05,"심각하게 밋밋한 사진 — 히스토그램 스트레칭 또는 커브 조정 필요","high"),
        (lambda v:v<0.08,"대비가 너무 낮음 — 후보정에서 대비/클래리티를 올려보세요","mid"),
    ],
    "colorfulness":[
        (lambda v:v<8,   "완전히 색이 빠진 사진 — 화이트밸런스 오류 또는 안개/흐린 날씨 영향","high"),
        (lambda v:v<15,  "색감이 거의 없음 — 채도/생동감을 높이거나 화이트밸런스를 확인하세요","mid"),
    ],
    "dct_hf_ratio":[(lambda v:v<0.02,"주파수 분석상 블러 확인 — 피사체 가까이에서 재촬영을 권장합니다","mid")],
    "edge_density":[(lambda v:v<5,   "엣지 정보가 거의 없음 — 피사체 거리와 초점을 다시 확인하세요","mid")],
}
SEV_ORDER = {"high":0,"mid":1,"low":2}
SEV_ICON  = {"high":"🔴","mid":"🟡","low":"🟢"}

def get_advice(fname, fval):
    for cond, text, sev in ADVICE_RULES.get(fname,[]):
        try:
            if cond(fval): return text, sev
        except: pass
    return None, None

def run_stage3(feat_df, model, output_dir):
    print("\n" + "═"*55)
    print("💡  STAGE 3 — SHAP 개선 조언")
    print("═"*55)

    X = feat_df[FEATURE_COLS]

    # SHAP TreeExplainer (XGBoost 버전 호환 패치)
    try:
        booster = model.get_booster()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp = f.name
        booster.save_model(tmp)
        with open(tmp,"r") as f:
            mj = json.load(f)
        mj["learner"]["learner_model_param"]["base_score"] = "0.5"
        with open(tmp,"w") as f:
            json.dump(mj, f)
        booster.load_model(tmp)
        os.unlink(tmp)
        explainer   = shap.TreeExplainer(booster)
        shap_values = explainer.shap_values(X)
    except Exception:
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values

    final_rows = []
    for i, (_, row) in enumerate(feat_df.iterrows()):
        row_dict = row.to_dict()
        shap_order = np.argsort(np.abs(sv[i]))[::-1]
        advices = []
        for idx in shap_order[:8]:
            fname = FEATURE_COLS[idx]
            fval  = row_dict.get(fname)
            if fval is None: continue
            text, sev = get_advice(fname, fval)
            if text:
                advices.append({"feature":fname,"value":round(fval,4),
                                "shap":round(sv[i][idx],4),"advice":text,"severity":sev})
        advices.sort(key=lambda x: SEV_ORDER[x["severity"]])
        advices = advices[:3]

        final_rows.append({
            "image_name":   row["image_name"],
            "quality_score":row["quality_score"],
            "stage1":       row["stage1_result"],
            "stage2":       row["context_label"],
            "advice_1":     advices[0]["advice"] if len(advices)>0 else "특별한 문제 없음",
            "advice_2":     advices[1]["advice"] if len(advices)>1 else "",
            "advice_3":     advices[2]["advice"] if len(advices)>2 else "",
            "main_cause":   advices[0]["feature"] if advices else "없음",
            "main_severity":advices[0]["severity"] if advices else "low",
        })

    final_df = pd.DataFrame(final_rows)
    return final_df, sv


# ════════════════════════════════════════════
# 최종 리포트 출력 + 저장
# ════════════════════════════════════════════

def print_final_report(final_df):
    print("\n" + "═"*55)
    print("📋  최종 분석 리포트")
    print("═"*55)

    for _, r in final_df.iterrows():
        s2_map = {"정상":"정상 ✅","bokeh":"Bokeh (의도적 블러) ✅",
                  "silhouette":"실루엣/역광 ✅","true_fail":"진짜 실패 ❌","error":"오류"}
        print(f"\n📸  {r['image_name']}")
        print(f"    품질 점수: {r['quality_score']:.1f} / 100")
        print(f"    Stage 1 : {r['stage1']}")
        print(f"    Stage 2 : {s2_map.get(r['stage2'], r['stage2'])}")
        if r["stage2"] == "true_fail":
            print(f"    개선 조언:")
            for col in ["advice_1","advice_2","advice_3"]:
                if r[col]:
                    icon = SEV_ICON.get(r.get("main_severity","mid"),"🟡") if col=="advice_1" else "  →"
                    print(f"      {icon} {r[col]}")
        elif r["stage2"] in ("bokeh","silhouette"):
            print(f"    → 의도된 표현으로 실패가 아닙니다 👍")
        else:
            print(f"    → 정상 사진입니다 👍")


def save_report(final_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "my_photo_analysis.csv")
    final_df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\n✅ 리포트 저장 → {path}")


# ════════════════════════════════════════════
# 메인
# ════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--photo_dir",  default=DEFAULT_PHOTO_DIR)
    parser.add_argument("--model",      default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    print(f"\n🚀 분석 시작: {args.photo_dir}")

    # Stage 1
    feat_df, model = run_stage1(args.photo_dir, args.model)

    # Stage 2
    feat_df = run_stage2(feat_df, args.photo_dir)

    # Stage 3
    final_df, sv = run_stage3(feat_df, model, args.output_dir)

    # 최종 출력
    print_final_report(final_df)
    save_report(final_df, args.output_dir)


if __name__ == "__main__":
    main()