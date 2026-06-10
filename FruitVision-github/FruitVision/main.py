"""
FruitVision backend API (local execution version).
"""

from io import BytesIO

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from scipy import stats
try:
    from skimage.feature import graycomatrix, graycoprops
except Exception:  # noqa: BLE001
    graycomatrix = None
    graycoprops = None

app = FastAPI(title="FruitVision Backend", version="1.0.0")


def _texture_features(gray: np.ndarray) -> tuple[float, float, float, float]:
    if graycomatrix is not None and graycoprops is not None:
        glcm = graycomatrix(gray, [1], [0], 256, symmetric=True, normed=True)
        return (
            float(graycoprops(glcm, "contrast")[0, 0]),
            float(graycoprops(glcm, "homogeneity")[0, 0]),
            float(graycoprops(glcm, "energy")[0, 0]),
            float(graycoprops(glcm, "correlation")[0, 0]),
        )

    # Fallback when scikit-image is unavailable.
    grayf = gray.astype(np.float32)
    contrast = float(np.var(grayf))
    homogeneity = float(1.0 / (1.0 + np.std(grayf)))
    energy = float(np.mean((grayf / 255.0) ** 2))
    correlation = float(np.corrcoef(grayf.flatten(), np.roll(grayf, 1, axis=1).flatten())[0, 1])
    if np.isnan(correlation):
        correlation = 0.0
    return contrast, homogeneity, energy, correlation


def extract_features(img_bgr: np.ndarray) -> dict:
    img = cv2.resize(img_bgr, (128, 128))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    bil = cv2.bilateralFilter(img, 9, 75, 75)
    gray = cv2.cvtColor(bil, cv2.COLOR_BGR2GRAY)

    r = rgb[:, :, 0].astype(float)
    g = rgb[:, :, 1].astype(float)
    b = rgb[:, :, 2].astype(float)

    h_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180]).flatten()
    h_hist /= h_hist.sum() + 1e-9

    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sx**2 + sy**2)
    canny = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    otsu_val, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contrast, homogeneity, energy, correlation = _texture_features(gray)

    return {
        "h_mean": float(np.mean(hsv[:, :, 0])),
        "s_mean": float(np.mean(hsv[:, :, 1])),
        "v_mean": float(np.mean(hsv[:, :, 2])),
        "a_mean": float(np.mean(lab[:, :, 1])),
        "b_mean": float(np.mean(lab[:, :, 2])),
        "redness": float(np.mean(r - g)),
        "greenness": float(np.mean(g - r)),
        "yellowness": float(np.mean((r + g) / 2 - b)),
        "rg_ratio": float(np.mean(r / (g + 1e-9))),
        "rb_ratio": float(np.mean(r / (b + 1e-9))),
        "h_kurt": float(stats.kurtosis(h_hist)),
        "h_skew": float(stats.skew(h_hist)),
        "h_peak": float(np.argmax(h_hist)),
        "s_std": float(np.std(hsv[:, :, 1])),
        "edge_density": float(np.sum(canny > 0) / (128 * 128)),
        "sobel_mean": float(np.mean(sobel)),
        "sobel_std": float(np.std(sobel)),
        "spot_density": float(np.sum(tophat > 25) / (128 * 128)),
        "otsu_thresh": float(otsu_val),
        "contrast": contrast,
        "homogeneity": homogeneity,
        "energy": energy,
        "correlation": correlation,
    }


def classify_fruit_and_stage(feats: dict):
    h = feats["h_mean"]
    s = feats["s_mean"]
    v = feats["v_mean"]
    r = feats["redness"]
    g = feats["greenness"]
    y = feats["yellowness"]
    sp = feats["spot_density"]
    ed = feats["edge_density"]
    hpk = feats["h_peak"]

    fruit_scores = {"Apple": 0.0, "Banana": 0.0, "Mango": 0.0, "Orange": 0.0, "Strawberry": 0.0}

    if (hpk < 20 or hpk > 155) and ed < 0.08:
        fruit_scores["Apple"] += 2.5
    if 35 < hpk < 80 and s > 60:
        fruit_scores["Apple"] += 1.5
    if 20 < hpk < 40 and y > 5:
        fruit_scores["Banana"] += 3.0
    if 22 < h < 38 and s > 80:
        fruit_scores["Banana"] += 2.0
    if 15 < hpk < 35 and s > 100:
        fruit_scores["Mango"] += 2.5
    if y > 10 and r > 5:
        fruit_scores["Mango"] += 1.5
    if 10 < hpk < 22 and s > 120:
        fruit_scores["Orange"] += 3.0
    if 12 < h < 25 and s > 100:
        fruit_scores["Orange"] += 2.0
    if (hpk < 12 or hpk > 165) and r > 20:
        fruit_scores["Strawberry"] += 3.0
    if ed > 0.06 and r > 15:
        fruit_scores["Strawberry"] += 1.5

    fruit = max(fruit_scores, key=fruit_scores.get)
    fruit_total = sum(fruit_scores.values()) + 1e-9
    fruit_probs = {k: round(vv / fruit_total * 100, 1) for k, vv in fruit_scores.items()}

    stage_scores = {"Unripe": 0.0, "Ripe": 0.0, "Overripe": 0.0}
    if g > 15:
        stage_scores["Unripe"] += 3.0
    if hpk > 50:
        stage_scores["Unripe"] += 2.0
    if s < 80:
        stage_scores["Unripe"] += 1.0
    if 100 < s < 220:
        stage_scores["Ripe"] += 2.0
    if 0.9 < feats["rg_ratio"] < 1.4 and fruit in ["Apple", "Strawberry"]:
        stage_scores["Ripe"] += 2.0
    if 22 < h < 40 and fruit == "Banana":
        stage_scores["Ripe"] += 3.0
    if r > 10 and g < 5:
        stage_scores["Ripe"] += 1.5
    if sp > 0.03:
        stage_scores["Overripe"] += 3.0
    if v < 90:
        stage_scores["Overripe"] += 2.0
    if r < -5:
        stage_scores["Overripe"] += 1.5
    if ed > 0.09:
        stage_scores["Overripe"] += 1.5

    stage = max(stage_scores, key=stage_scores.get)
    stage_total = sum(stage_scores.values()) + 1e-9
    stage_probs = {k: round(vv / stage_total * 100, 1) for k, vv in stage_scores.items()}
    return fruit, stage, fruit_probs, stage_probs


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    data = await file.read()
    try:
        pil = Image.open(BytesIO(data)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Invalid image file: {exc}") from exc

    arr = np.array(pil)
    img_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    feats = extract_features(img_bgr)
    fruit, stage, fruit_probs, stage_probs = classify_fruit_and_stage(feats)

    return {
        "fruit": fruit,
        "stage": stage,
        "fruit_confidence": fruit_probs[fruit],
        "stage_confidence": stage_probs[stage],
        "fruit_probabilities": fruit_probs,
        "stage_probabilities": stage_probs,
    }