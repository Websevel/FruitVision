"""
FruitVision DIP System — Streamlit Frontend
COMP-342L · P06 · Spring 2025 · Pak-Austria Fachhochschule
"""

import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from PIL import Image
from io import BytesIO
import time
try:
    from skimage.feature import graycomatrix, graycoprops
except Exception:  # noqa: BLE001
    graycomatrix = None
    graycoprops = None
from scipy import stats
import os, warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="FruitVision · DIP System",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

/* ── Root Variables ── */
:root {
    --bg0:    #05070a;
    --bg1:    #0d1117;
    --bg2:    #161b22;
    --bg3:    #21262d;
    --border: #30363d;
    --text:   #e6edf3;
    --muted:  #8b949e;
    --green:  #3fb950;
    --orange: #f0883e;
    --red:    #f85149;
    --blue:   #58a6ff;
    --purple: #bc8cff;
    --yellow: #e3b341;
    --font-display: 'Syne', sans-serif;
    --font-mono:    'Space Mono', monospace;
}

html, body, [class*="css"] {
    background-color: var(--bg0) !important;
    color: var(--text) !important;
    font-family: var(--font-display) !important;
}

/* ── Hide Streamlit Chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 3rem !important; max-width: 1400px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg1) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Custom Cards ── */
.fv-card {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 16px;
    transition: border-color 0.2s;
}
.fv-card:hover { border-color: var(--blue); }

.fv-metric {
    background: var(--bg3);
    border-radius: 10px;
    padding: 14px 18px;
    text-align: center;
}
.fv-metric .label {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 6px;
}
.fv-metric .value {
    font-family: var(--font-display);
    font-size: 1.6rem;
    font-weight: 800;
    line-height: 1.1;
}
.fv-metric .sub {
    font-family: var(--font-mono);
    font-size: 0.75rem;
    color: var(--muted);
    margin-top: 4px;
}

.fv-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-family: var(--font-mono);
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.08em;
}
.badge-green  { background: #1a4731; color: var(--green);  border: 1px solid var(--green); }
.badge-orange { background: #3d2a10; color: var(--orange); border: 1px solid var(--orange); }
.badge-red    { background: #3d1f1f; color: var(--red);    border: 1px solid var(--red); }
.badge-blue   { background: #0d2a4d; color: var(--blue);   border: 1px solid var(--blue); }

/* ── Hero Header ── */
.fv-hero {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 40%, #0d1117 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 36px 40px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.fv-hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(88,166,255,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.fv-hero::after {
    content: '';
    position: absolute;
    bottom: -40px; left: -40px;
    width: 160px; height: 160px;
    background: radial-gradient(circle, rgba(63,185,80,0.07) 0%, transparent 70%);
    border-radius: 50%;
}

/* ── Section titles ── */
.fv-section-title {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
}

/* ── Feedback Banner ── */
.fv-feedback {
    border-radius: 12px;
    padding: 18px 22px;
    margin: 16px 0;
    border-left-width: 5px;
    border-left-style: solid;
}

/* ── Stat pill ── */
.fv-pill {
    display: inline-block;
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 5px 10px;
    font-family: var(--font-mono);
    font-size: 0.72rem;
    color: var(--text);
    margin: 3px;
}

/* ── Upload area ── */
[data-testid="stFileUploader"] {
    background: var(--bg2) !important;
    border: 2px dashed var(--border) !important;
    border-radius: 12px !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #238636, #2ea043) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: var(--font-mono) !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    padding: 12px 28px !important;
    letter-spacing: 0.05em !important;
    transition: all 0.2s !important;
    width: 100%;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #2ea043, #3fb950) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(63,185,80,0.3) !important;
}

/* ── Selectbox / Slider ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stSlider"] {
    background: var(--bg3) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}

/* ── Tabs ── */
[data-baseweb="tab-list"] {
    background: var(--bg2) !important;
    border-radius: 10px !important;
    border: 1px solid var(--border) !important;
    gap: 4px !important;
    padding: 4px !important;
}
[data-baseweb="tab"] {
    font-family: var(--font-mono) !important;
    font-size: 0.8rem !important;
    color: var(--muted) !important;
    border-radius: 7px !important;
    background: transparent !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    background: var(--bg3) !important;
    color: var(--text) !important;
}

/* ── Progress bar ── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--green), var(--blue)) !important;
    border-radius: 4px !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: var(--bg2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}

/* ── Plotly charts ── */
.js-plotly-plot .plotly { background: transparent !important; }

/* ── Table ── */
[data-testid="stDataFrame"] {
    background: var(--bg2) !important;
    border-radius: 10px !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg1); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  DOMAIN KNOWLEDGE
# ─────────────────────────────────────────────
FRUIT_KNOWLEDGE = {
    'Apple': {
        'emoji': '🍎', 'season': 'Autumn (Sep – Nov)',
        'origin': 'Central Asia', 'family': 'Rosaceae',
        'nutrition': {'Calories':'52 kcal','Vitamin C':'4.6 mg','Fiber':'2.4 g','Sugar':'10.4 g','Water':'85.6%'},
        'stages': {
            'Unripe':   {'color':'Green / pale yellow','texture':'Very firm, waxy','taste':'Sour, starchy','feedback':'❌ Not ready — high starch, low sugar. Wait 2–3 weeks.','edible':False,'days_left':'14–21 days','tip':'Store at room temperature to ripen.','health':'High in pectin (starch form). May cause digestive discomfort.','border':'#3fb950'},
            'Ripe':     {'color':'Bright red / golden','texture':'Firm with slight give','taste':'Sweet-tart, crisp, juicy','feedback':'✅ Perfect! Peak sugar and crunch. Eat now.','edible':True,'days_left':'Best within 1–2 weeks','tip':'Refrigerate to extend freshness.','health':'Rich in quercetin, catechin and Vitamin C. Best nutritional profile.','border':'#f0883e'},
            'Overripe': {'color':'Dark red + brown patches','texture':'Soft, mealy, wrinkled','taste':'Very sweet but mushy','feedback':'⚠️ Overripe — use for sauce, juice, or baking only.','edible':False,'days_left':'Process today','tip':'Freeze immediately for future use in cooking.','health':'Increased sugar concentration. Fermentation may begin.','border':'#f85149'},
        }
    },
    'Banana': {
        'emoji': '🍌', 'season': 'Year-round (tropical)',
        'origin': 'Southeast Asia', 'family': 'Musaceae',
        'nutrition': {'Calories':'89 kcal','Vitamin B6':'0.4 mg','Fiber':'2.6 g','Sugar':'12.2 g','Potassium':'358 mg'},
        'stages': {
            'Unripe':   {'color':'Solid green','texture':'Very firm, starchy','taste':'Starchy, no sweetness','feedback':'❌ Unripe — high resistant starch. Good for cooking. Wait 4–7 days.','edible':False,'days_left':'4–7 days at room temp','tip':'Keep at room temperature. Never refrigerate to ripen.','health':'High in resistant starch — good probiotic food. Low glycemic.','border':'#3fb950'},
            'Ripe':     {'color':'Yellow + brown speckles','texture':'Soft, easy to peel','taste':'Sweet, creamy, full flavour','feedback':'✅ Perfect! Maximum sweetness and potassium.','edible':True,'days_left':'Eat within 2–3 days','tip':'Speckled bananas have highest antioxidant content.','health':'High in B6, potassium. TNF (tumour necrosis factor) increases.','border':'#f0883e'},
            'Overripe': {'color':'Mostly brown/black skin','texture':'Very soft, mushy','taste':'Intensely sweet, fermented','feedback':'⚠️ Overripe — ideal for banana bread or smoothies only.','edible':False,'days_left':'Use immediately or freeze','tip':'Peel and freeze in zip bags for future baking.','health':'Very high sugar. Fermentation reduces shelf life rapidly.','border':'#f85149'},
        }
    },
    'Mango': {
        'emoji': '🥭', 'season': 'Summer (Apr – Jul)',
        'origin': 'South Asia (India/Pakistan)', 'family': 'Anacardiaceae',
        'nutrition': {'Calories':'60 kcal','Vitamin C':'36.4 mg','Fiber':'1.6 g','Sugar':'13.7 g','Vitamin A':'54 μg'},
        'stages': {
            'Unripe':   {'color':'Dark green, firm','texture':'Rock hard, no give','taste':'Extremely sour, tangy','feedback':'❌ Unripe — great for pickles/chutney. Not pleasant raw.','edible':False,'days_left':'7–14 days','tip':'Place in a paper bag or rice to accelerate ripening.','health':'High in Vitamin C precursors and tartaric acid.','border':'#3fb950'},
            'Ripe':     {'color':'Yellow-orange + red blush','texture':'Slightly soft at stem end','taste':'Sweet, aromatic, juicy','feedback':'✅ Ready! Best flavour and Vitamin A content.','edible':True,'days_left':'Eat within 2–4 days','tip':'Refrigerate ripe mangoes to slow deterioration.','health':'Highest Vitamin A of any common fruit. Excellent for immunity.','border':'#f0883e'},
            'Overripe': {'color':'Orange-red, dark spots, leaking','texture':'Very soft, leaking juice','taste':'Fermented sweetness','feedback':'⚠️ Overripe — use for lassi, jam, or freeze immediately.','edible':False,'days_left':'Use today','tip':'Blend and freeze as mango puree for future desserts.','health':'Fermentation increases alcoholic compounds. Use carefully.','border':'#f85149'},
        }
    },
    'Orange': {
        'emoji': '🍊', 'season': 'Winter (Nov – Mar)',
        'origin': 'Southeast Asia / China', 'family': 'Rutaceae',
        'nutrition': {'Calories':'47 kcal','Vitamin C':'53.2 mg','Fiber':'2.4 g','Sugar':'9.4 g','Folate':'30 μg'},
        'stages': {
            'Unripe':   {'color':'Green-yellow skin','texture':'Hard, tight skin','taste':'Very sour, high citric acid','feedback':'❌ Unripe — high acid, low sugar. Wait for full orange colour.','edible':False,'days_left':'10–21 days','tip':'Do not refrigerate unripe oranges — room temp is best.','health':'Extremely high citric acid may erode tooth enamel.','border':'#3fb950'},
            'Ripe':     {'color':'Bright orange, smooth skin','texture':'Firm, heavy for its size','taste':'Sweet-tart, very juicy','feedback':'✅ Perfect! Highest Vitamin C and juice content.','edible':True,'days_left':'Eat within 1–2 weeks','tip':'Heavy oranges = more juice. Store in fridge for longer life.','health':'Highest Vitamin C per serving. Excellent immune support.','border':'#f0883e'},
            'Overripe': {'color':'Dark orange + soft spots','texture':'Soft, possibly dry inside','taste':'Bland or fermented','feedback':'⚠️ Overripe — juice only. Check for mould before consuming.','edible':False,'days_left':'Use today','tip':'Zest before discarding — zest can be frozen for 3 months.','health':'Nutrient content degrades rapidly. Risk of mould contamination.','border':'#f85149'},
        }
    },
    'Strawberry': {
        'emoji': '🍓', 'season': 'Spring–Summer (May – Jul)',
        'origin': 'Europe / North America', 'family': 'Rosaceae',
        'nutrition': {'Calories':'32 kcal','Vitamin C':'58.8 mg','Fiber':'2.0 g','Sugar':'4.9 g','Manganese':'0.4 mg'},
        'stages': {
            'Unripe':   {'color':'White/green + red tips','texture':'Firm, not fragrant','taste':'Sour, no sweetness','feedback':'❌ Unripe — no anthocyanins yet. Leave on plant 3–7 more days.','edible':False,'days_left':'3–7 days','tip':'Never pick strawberries before they are fully red.','health':'Low anthocyanins (antioxidant). High oxalic acid.','border':'#3fb950'},
            'Ripe':     {'color':'Bright uniform red, shiny','texture':'Firm, juicy, fragrant','taste':'Sweet, floral, acidic balance','feedback':'✅ Peak ripeness! Best eaten fresh within 24–48 hours.','edible':True,'days_left':'Eat within 1–2 days','tip':'Do not wash until ready to eat — moisture accelerates decay.','health':'Highest antioxidant content of all common berries. Anti-inflammatory.','border':'#f0883e'},
            'Overripe': {'color':'Dark red/purple, dull surface','texture':'Soft, easily damaged','taste':'Very sweet, fermented notes','feedback':'⚠️ Overripe — use for jam, smoothies, or sauce today.','edible':False,'days_left':'Use immediately','tip':'Hull and freeze in a single layer immediately.','health':'Fermentation begins. Risk of botrytis (grey mould) contamination.','border':'#f85149'},
        }
    }
}

STAGE_COLORS = {'Unripe': '#3fb950', 'Ripe': '#f0883e', 'Overripe': '#f85149'}
FRUIT_EMOJIS = {f: v['emoji'] for f, v in FRUIT_KNOWLEDGE.items()}


def _texture_features(gray):
    if graycomatrix is not None and graycoprops is not None:
        glcm = graycomatrix(gray, [1], [0], 256, symmetric=True, normed=True)
        return (
            float(graycoprops(glcm, 'contrast')[0, 0]),
            float(graycoprops(glcm, 'homogeneity')[0, 0]),
            float(graycoprops(glcm, 'energy')[0, 0]),
            float(graycoprops(glcm, 'correlation')[0, 0]),
        )

    grayf = gray.astype(np.float32)
    contrast = float(np.var(grayf))
    homogeneity = float(1.0 / (1.0 + np.std(grayf)))
    energy = float(np.mean((grayf / 255.0) ** 2))
    correlation = float(np.corrcoef(grayf.flatten(), np.roll(grayf, 1, axis=1).flatten())[0, 1])
    if np.isnan(correlation):
        correlation = 0.0
    return contrast, homogeneity, energy, correlation

# ─────────────────────────────────────────────
#  DIP FEATURE EXTRACTION
# ─────────────────────────────────────────────
def extract_features(img_bgr):
    img  = cv2.resize(img_bgr, (128, 128))
    rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab  = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    bil  = cv2.bilateralFilter(img, 9, 75, 75)
    gray = cv2.cvtColor(bil, cv2.COLOR_BGR2GRAY)

    R, G, B = rgb[:,:,0].astype(float), rgb[:,:,1].astype(float), rgb[:,:,2].astype(float)

    h_hist = cv2.calcHist([hsv],[0],None,[180],[0,180]).flatten()
    h_hist /= h_hist.sum() + 1e-9

    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sx**2 + sy**2)
    canny = cv2.Canny(cv2.GaussianBlur(gray,(5,5),0), 50, 150)

    k      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k)
    otsu_v,_ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    contrast, homogeneity, energy, correlation = _texture_features(gray)

    return {
        'h_mean':       float(np.mean(hsv[:,:,0])),
        's_mean':       float(np.mean(hsv[:,:,1])),
        'v_mean':       float(np.mean(hsv[:,:,2])),
        'a_mean':       float(np.mean(lab[:,:,1])),
        'b_mean':       float(np.mean(lab[:,:,2])),
        'redness':      float(np.mean(R - G)),
        'greenness':    float(np.mean(G - R)),
        'yellowness':   float(np.mean((R+G)/2 - B)),
        'rg_ratio':     float(np.mean(R/(G+1e-9))),
        'rb_ratio':     float(np.mean(R/(B+1e-9))),
        'h_kurt':       float(stats.kurtosis(h_hist)),
        'h_skew':       float(stats.skew(h_hist)),
        'h_peak':       float(np.argmax(h_hist)),
        's_std':        float(np.std(hsv[:,:,1])),
        'edge_density': float(np.sum(canny>0)/(128*128)),
        'sobel_mean':   float(np.mean(sobel)),
        'sobel_std':    float(np.std(sobel)),
        'spot_density': float(np.sum(tophat>25)/(128*128)),
        'otsu_thresh':  float(otsu_v),
        'contrast':     contrast,
        'homogeneity':  homogeneity,
        'energy':       energy,
        'correlation':  correlation,
        # DIP images for viz
        '_gray':  gray,
        '_canny': canny,
        '_hsv':   hsv,
        '_tophat':tophat,
        '_rgb':   cv2.cvtColor(cv2.resize(img_bgr,(256,256)),cv2.COLOR_BGR2RGB),
        '_h_hist':h_hist,
        '_otsu':  cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1],
    }

FEATURE_COLS = [
    'h_mean','s_mean','v_mean','a_mean','b_mean',
    'redness','greenness','yellowness','rg_ratio','rb_ratio',
    'h_kurt','h_skew','h_peak','s_std',
    'edge_density','sobel_mean','sobel_std','spot_density','otsu_thresh',
    'contrast','homogeneity','energy','correlation'
]

# ─────────────────────────────────────────────
#  RULE-BASED CLASSIFIER (works without training data)
#  Uses DIP features + botanical colour logic
# ─────────────────────────────────────────────
def classify_fruit_and_stage(feats):
    """
    Rule-based DIP classifier using colour/texture features.
    Based on botanical ripening science.
    """
    h   = feats['h_mean']
    s   = feats['s_mean']
    v   = feats['v_mean']
    r   = feats['redness']
    g   = feats['greenness']
    y   = feats['yellowness']
    sp  = feats['spot_density']
    ed  = feats['edge_density']
    hpk = feats['h_peak']

    # ── Fruit detection by dominant hue ──
    # Hue ranges (OpenCV 0-180):
    # Red/orange: 0-20, 160-180
    # Yellow:     20-35
    # Green:      35-85
    # Orange:     10-25
    # Purple/red: 140-175

    fruit_scores = {
        'Apple':      0.0,
        'Banana':     0.0,
        'Mango':      0.0,
        'Orange':     0.0,
        'Strawberry': 0.0,
    }

    # Apple: red OR green dominant, circular shape → moderate edges
    if (hpk < 20 or hpk > 155) and ed < 0.08:
        fruit_scores['Apple']      += 2.5
    if 35 < hpk < 80 and s > 60:
        fruit_scores['Apple']      += 1.5
    # Banana: yellow dominant
    if 20 < hpk < 40 and y > 5:
        fruit_scores['Banana']     += 3.0
    if h > 22 and h < 38 and s > 80:
        fruit_scores['Banana']     += 2.0
    # Mango: yellow-orange, high sat
    if 15 < hpk < 35 and s > 100:
        fruit_scores['Mango']      += 2.5
    if y > 10 and r > 5:
        fruit_scores['Mango']      += 1.5
    # Orange: orange hue, round
    if 10 < hpk < 22 and s > 120:
        fruit_scores['Orange']     += 3.0
    if 12 < h < 25 and s > 100:
        fruit_scores['Orange']     += 2.0
    # Strawberry: red + high edge density (seeds)
    if (hpk < 12 or hpk > 165) and r > 20:
        fruit_scores['Strawberry'] += 3.0
    if ed > 0.06 and r > 15:
        fruit_scores['Strawberry'] += 1.5

    # pick highest-scoring fruit
    fruit = max(fruit_scores, key=fruit_scores.get)
    top_score = fruit_scores[fruit]
    total = sum(fruit_scores.values()) + 1e-9

    fruit_probs = {f: round(s/total*100, 1) for f, s in fruit_scores.items()}

    # ── Stage detection by colour evolution ──
    stage_scores = {'Unripe': 0.0, 'Ripe': 0.0, 'Overripe': 0.0}

    # GREEN dominance → unripe
    if g > 15:        stage_scores['Unripe']   += 3.0
    if hpk > 50:      stage_scores['Unripe']   += 2.0
    if s < 80:        stage_scores['Unripe']   += 1.0
    # PEAK colour → ripe
    if 100 < s < 220: stage_scores['Ripe']     += 2.0
    if 0.9<feats['rg_ratio']<1.4 and fruit in ['Apple','Strawberry']:
                      stage_scores['Ripe']     += 2.0
    if 22<h<40 and fruit=='Banana':
                      stage_scores['Ripe']     += 3.0
    if r > 10 and g < 5:
                      stage_scores['Ripe']     += 1.5
    # DARK + SPOTS → overripe
    if sp > 0.03:     stage_scores['Overripe'] += 3.0
    if v < 90:        stage_scores['Overripe'] += 2.0
    if r < -5:        stage_scores['Overripe'] += 1.5
    if ed > 0.09:     stage_scores['Overripe'] += 1.5

    stage = max(stage_scores, key=stage_scores.get)
    total_s = sum(stage_scores.values()) + 1e-9
    stage_probs = {s: round(v/total_s*100, 1)
                   for s, v in stage_scores.items()}

    # Confidence: how dominant is top score?
    s_conf = round(stage_probs[stage], 1)
    f_conf = round(fruit_probs[fruit], 1)

    return fruit, stage, fruit_probs, stage_probs, f_conf, s_conf


def full_prediction(img_bgr):
    feats = extract_features(img_bgr)
    fruit, stage, fp, sp, fc, sc = classify_fruit_and_stage(feats)

    know   = FRUIT_KNOWLEDGE.get(fruit, {})
    sinfo  = know.get('stages', {}).get(stage, {})

    return {
        'fruit':        fruit,
        'stage':        stage,
        'emoji':        know.get('emoji','🍎'),
        'season':       know.get('season',''),
        'origin':       know.get('origin',''),
        'family':       know.get('family',''),
        'nutrition':    know.get('nutrition',{}),
        'feedback':     sinfo.get('feedback',''),
        'edible':       sinfo.get('edible',False),
        'color':        sinfo.get('color',''),
        'texture':      sinfo.get('texture',''),
        'taste':        sinfo.get('taste',''),
        'days_left':    sinfo.get('days_left',''),
        'health':       sinfo.get('health',''),
        'tip':          sinfo.get('tip',''),
        'border':       sinfo.get('border','#58a6ff'),
        'stage_conf':   sc,
        'fruit_conf':   fc,
        'stage_probs':  sp,
        'fruit_probs':  fp,
        'features':     feats,
        'img_rgb':      feats['_rgb'],
        'h_hist':       feats['_h_hist'],
        'canny':        feats['_canny'],
        'otsu':         feats['_otsu'],
        'tophat':       feats['_tophat'],
        'hsv_h':        feats['_hsv'][:,:,0],
    }

# ─────────────────────────────────────────────
#  PLOTLY CHART BUILDERS
# ─────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#e6edf3', family='Space Mono'),
    margin=dict(l=10,r=10,t=40,b=10),
)

def make_stage_donut(stage_probs, stage):
    sc = STAGE_COLORS
    labels = list(stage_probs.keys())
    values = list(stage_probs.values())
    colors = [sc.get(l,'#58a6ff') for l in labels]
    pull   = [0.08 if l==stage else 0 for l in labels]

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.62, pull=pull,
        marker=dict(colors=colors,
                    line=dict(color='#0d1117',width=3)),
        textinfo='label+percent',
        textfont=dict(size=12, family='Space Mono'),
        hovertemplate='<b>%{label}</b><br>%{value:.1f}%<extra></extra>'
    ))
    fig.add_annotation(
        text=f"<b>{stage}</b>",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=16, color=STAGE_COLORS.get(stage,'#f0883e'),
                  family='Syne')
    )
    fig.update_layout(**PLOTLY_LAYOUT,
                      title=dict(text='Stage Probabilities',
                                 font=dict(size=13),x=0.5),
                      showlegend=False, height=280)
    return fig


def make_fruit_bar(fruit_probs, fruit):
    fruits = list(fruit_probs.keys())
    vals   = list(fruit_probs.values())
    emojis = [FRUIT_EMOJIS.get(f,'🍎') for f in fruits]
    labels = [f"{e} {f}" for e,f in zip(emojis,fruits)]
    colors = ['#58a6ff' if f==fruit else '#30363d' for f in fruits]

    fig = go.Figure(go.Bar(
        x=vals, y=labels, orientation='h',
        marker=dict(color=colors,
                    line=dict(color='#0d1117',width=1)),
        text=[f'{v:.1f}%' for v in vals],
        textposition='outside',
        textfont=dict(size=11, family='Space Mono'),
        hovertemplate='<b>%{y}</b><br>%{x:.1f}%<extra></extra>'
    ))
    fig.update_layout(**PLOTLY_LAYOUT,
                      title=dict(text='Fruit Probabilities',
                                 font=dict(size=13),x=0.5),
                      xaxis=dict(range=[0,120],showgrid=False,
                                 zeroline=False,showticklabels=False),
                      yaxis=dict(showgrid=False),
                      height=280)
    return fig


def make_hue_histogram(h_hist):
    hues   = np.linspace(0, 179, 180)
    colors = [f'hsl({int(h*2)},80%,50%)' for h in hues]

    fig = go.Figure()
    # Background zones
    for (x0,x1,label,c) in [
        (0,20,'Overripe','rgba(248,81,73,0.12)'),
        (20,40,'Banana/Ripe','rgba(240,136,62,0.12)'),
        (40,80,'Unripe/Green','rgba(63,185,80,0.12)'),
        (80,120,'Cool Green','rgba(88,166,255,0.08)'),
    ]:
        fig.add_vrect(x0=x0,x1=x1,fillcolor=c,line_width=0,
                      annotation_text=label,
                      annotation_position="top left",
                      annotation_font=dict(size=9,color='#8b949e'))

    fig.add_trace(go.Bar(
        x=list(range(180)), y=list(h_hist),
        marker=dict(color=colors, line_width=0),
        hovertemplate='Hue %{x}: %{y:.4f}<extra></extra>'
    ))
    fig.update_layout(**PLOTLY_LAYOUT,
                      title=dict(text='Hue Distribution (HSV)',
                                 font=dict(size=13),x=0.5),
                      xaxis=dict(title='Hue Value (0–179)',
                                 showgrid=False,zeroline=False),
                      yaxis=dict(title='Frequency',
                                 showgrid=True,
                                 gridcolor='#21262d'),
                      height=280)
    return fig


def make_radar_chart(feats):
    categories = ['Redness','Yellowness','Edge Density',
                  'Spot Density','Saturation','Brightness']
    max_vals   = [80, 80, 0.15, 0.08, 255, 255]
    raw_vals   = [
        max(0, feats['redness']),
        max(0, feats['yellowness']),
        feats['edge_density'],
        feats['spot_density'],
        feats['s_mean'],
        feats['v_mean'],
    ]
    norm = [min(v/m, 1.0)*100 for v,m in zip(raw_vals, max_vals)]

    fig = go.Figure(go.Scatterpolar(
        r=norm + [norm[0]],
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(88,166,255,0.15)',
        line=dict(color='#58a6ff', width=2),
        marker=dict(size=6, color='#58a6ff'),
        hovertemplate='<b>%{theta}</b><br>%{r:.1f}%<extra></extra>'
    ))
    fig.update_layout(**PLOTLY_LAYOUT,
                      title=dict(text='DIP Feature Radar',
                                 font=dict(size=13),x=0.5),
                      polar=dict(
                          bgcolor='rgba(0,0,0,0)',
                          radialaxis=dict(visible=True,range=[0,100],
                                          tickfont=dict(size=9),
                                          gridcolor='#21262d',
                                          linecolor='#30363d'),
                          angularaxis=dict(tickfont=dict(size=10),
                                           linecolor='#30363d',
                                           gridcolor='#21262d')
                      ),
                      height=300)
    return fig


def make_nutrition_bar(nutrition):
    items  = list(nutrition.keys())
    values_raw = []
    units  = []
    for v in nutrition.values():
        num = float(''.join(c for c in v if c.isdigit() or c=='.') or '0')
        unit = ''.join(c for c in v if not (c.isdigit() or c=='.')).strip()
        values_raw.append(num)
        units.append(unit)

    # Normalise each to 0-100 scale for display
    mx = [100, 100, 5, 20, 500]
    norm = [min(v/m*100,100) for v,m in zip(values_raw,mx[:len(values_raw)])]

    fig = go.Figure(go.Bar(
        x=norm[:len(items)],
        y=items,
        orientation='h',
        marker=dict(
            color=['#3fb950','#58a6ff','#f0883e','#f85149','#bc8cff'][:len(items)],
            line_width=0
        ),
        text=[f'{v} {u}' for v,u in zip(values_raw,units)],
        textposition='outside',
        textfont=dict(size=11,family='Space Mono'),
    ))
    fig.update_layout(**PLOTLY_LAYOUT,
                      title=dict(text='Nutritional Profile (per 100g)',
                                 font=dict(size=13),x=0.5),
                      xaxis=dict(range=[0,130],showgrid=False,
                                 showticklabels=False,zeroline=False),
                      yaxis=dict(showgrid=False),
                      height=260)
    return fig


def make_dip_panels(result):
    """Create 2×3 DIP visualisation using matplotlib."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    fig.patch.set_facecolor('#0d1117')

    panels = [
        (result['img_rgb'],       None,  'Original Image'),
        (result['hsv_h'],         'hsv', 'Hue Channel — Lab 02'),
        (result['canny'],         'gray','Canny Edges — Lab 07'),
        (result['otsu'],          'gray','Otsu Segmentation — Lab 12'),
        (result['tophat'],        'hot', 'Top-Hat Spots — Lab 11'),
        (None,                    None,  'Hue Histogram — Lab 05'),
    ]

    for i, (img, cmap, title) in enumerate(panels):
        ax = axes[i//3][i%3]
        ax.set_facecolor('#161b22')
        if i == 5:
            h = result['h_hist']
            c = plt.cm.hsv(np.linspace(0,1,180))
            ax.bar(range(180), h, color=c, width=1.2)
            ax.axvspan(0, 30,  alpha=0.2, color='#f85149')
            ax.axvspan(30,60,  alpha=0.2, color='#f0883e')
            ax.axvspan(60,100, alpha=0.2, color='#3fb950')
            ax.tick_params(colors='white', labelsize=8)
            ax.set_xlabel('Hue (0-179)', color='#8b949e', fontsize=9)
            for sp in ax.spines.values():
                sp.set_color('#30363d')
        else:
            ax.imshow(img, cmap=cmap)
            ax.axis('off')

        sc = STAGE_COLORS.get(result['stage'],'#58a6ff')
        ax.set_title(title, color='#58a6ff',
                     fontsize=10, fontweight='bold', pad=8)
        for sp in ax.spines.values():
            sp.set_edgecolor(sc); sp.set_linewidth(1.5)

    plt.tight_layout(pad=1.5)
    return fig

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 20px 0 10px;">
      <div style="font-size:2.5rem; margin-bottom:8px;">🍎</div>
      <div style="font-family:'Syne',sans-serif; font-size:1.2rem;
                  font-weight:800; color:#e6edf3; letter-spacing:0.05em;">
        FruitVision
      </div>
      <div style="font-family:'Space Mono',monospace; font-size:0.65rem;
                  color:#8b949e; letter-spacing:0.15em; text-transform:uppercase;">
        DIP System · COMP-342L
      </div>
    </div>
    <hr style="border-color:#30363d; margin:12px 0;">
    """, unsafe_allow_html=True)

    st.markdown('<div class="fv-section-title">🎮 Mode</div>',
                unsafe_allow_html=True)
    mode = st.radio('', ['Upload Image', 'Demo Gallery'],
                    label_visibility='collapsed')

    st.markdown('<hr style="border-color:#30363d; margin:12px 0;">',
                unsafe_allow_html=True)
    st.markdown('<div class="fv-section-title">⚙️ Settings</div>',
                unsafe_allow_html=True)

    show_dip    = st.toggle('Show DIP Pipeline',     value=True)
    show_stats  = st.toggle('Show Statistical Analysis', value=True)
    show_nutri  = st.toggle('Show Nutrition Profile',value=True)
    show_advice = st.toggle('Show Expert Advice',    value=True)

    st.markdown('<hr style="border-color:#30363d; margin:12px 0;">',
                unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'Space Mono',monospace; font-size:0.7rem;
                color:#8b949e; line-height:1.8;">
      <b style="color:#e6edf3;">Labs Covered</b><br>
      L02 · Color Spaces<br>
      L03 · Math Operations<br>
      L05 · Histogram Analysis<br>
      L06 · Filtering<br>
      L07 · Edge Detection<br>
      L11 · Morphology<br>
      L12 · Segmentation
    </div>
    <hr style="border-color:#30363d; margin:12px 0;">
    <div style="font-family:'Space Mono',monospace; font-size:0.65rem;
                color:#8b949e; line-height:1.8;">
      <b style="color:#e6edf3;">CCP Criteria</b><br>
      C1 ✅ Multi-lab knowledge<br>
      C2 ✅ Depth of analysis<br>
      C3 ✅ Trade-off discussion<br>
      C4 ✅ Non-trivial design<br>
      C5 ✅ PDI standards<br>
      C6 ✅ Domain context
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  HERO HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="fv-hero">
  <div style="display:flex; align-items:center; gap:20px; flex-wrap:wrap;">
    <div style="flex:1; min-width:300px;">
      <div style="font-family:'Space Mono',monospace; font-size:0.7rem;
                  color:#3fb950; letter-spacing:0.2em; text-transform:uppercase;
                  margin-bottom:8px;">
        COMP-342L · P06 · SPRING 2025
      </div>
      <h1 style="font-family:'Syne',sans-serif; font-size:2.4rem;
                 font-weight:800; color:#e6edf3; margin:0 0 8px;
                 line-height:1.1;">
        🌿 FruitVision
        <span style="color:#58a6ff;">DIP</span>
      </h1>
      <p style="color:#8b949e; font-size:1rem; margin:0; line-height:1.6;">
        Digital Image Processing — Fruit Ripeness & Growth Stage Detection<br>
        <span style="color:#c9d1d9;">Pak-Austria Fachhochschule · Haripur, Pakistan</span>
      </p>
    </div>
    <div style="display:flex; gap:12px; flex-wrap:wrap;">
      <div class="fv-metric" style="min-width:100px;">
        <div class="label">Fruits</div>
        <div class="value" style="color:#58a6ff;">5</div>
        <div class="sub">types</div>
      </div>
      <div class="fv-metric" style="min-width:100px;">
        <div class="label">Stages</div>
        <div class="value" style="color:#3fb950;">3</div>
        <div class="sub">per fruit</div>
      </div>
      <div class="fv-metric" style="min-width:100px;">
        <div class="label">Features</div>
        <div class="value" style="color:#f0883e;">23</div>
        <div class="sub">DIP features</div>
      </div>
      <div class="fv-metric" style="min-width:100px;">
        <div class="label">Labs</div>
        <div class="value" style="color:#bc8cff;">7</div>
        <div class="sub">covered</div>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  MAIN CONTENT
# ─────────────────────────────────────────────
result = None
img_bgr = None

if mode == 'Upload Image':
    col_upload, col_hint = st.columns([2, 1])

    with col_upload:
        st.markdown('<div class="fv-section-title">📂 Upload Fruit Image</div>',
                    unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Drop a fruit photo here",
            type=['jpg','jpeg','png','bmp','webp'],
            label_visibility='collapsed'
        )

    with col_hint:
        st.markdown("""
        <div class="fv-card" style="margin-top:28px;">
          <div class="fv-section-title">💡 Tips</div>
          <div style="font-family:'Space Mono',monospace; font-size:0.75rem;
                      color:#8b949e; line-height:2;">
            ✓ Clear, well-lit photo<br>
            ✓ Single fruit per image<br>
            ✓ Any resolution works<br>
            ✓ JPG, PNG, BMP, WEBP<br>
            ✓ Works best with: Apple,
            Banana, Mango, Orange,
            Strawberry
          </div>
        </div>
        """, unsafe_allow_html=True)

    if uploaded:
        pil   = Image.open(uploaded).convert('RGB')
        arr   = np.array(pil)
        img_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

else:
    # Demo gallery
    st.markdown('<div class="fv-section-title">🧪 Demo Gallery — Select a Test Case</div>',
                unsafe_allow_html=True)

    demo_options = {
        '🍎 Apple — Ripe':        ('Apple', 'Ripe',     [180,40,40]),
        '🍎 Apple — Unripe':      ('Apple', 'Unripe',   [40,120,40]),
        '🍌 Banana — Ripe':       ('Banana','Ripe',     [200,190,50]),
        '🍌 Banana — Overripe':   ('Banana','Overripe', [80,50,20]),
        '🥭 Mango — Ripe':        ('Mango', 'Ripe',     [220,140,20]),
        '🍊 Orange — Ripe':       ('Orange','Ripe',     [220,110,20]),
        '🍓 Strawberry — Ripe':   ('Strawberry','Ripe', [180,30,40]),
    }

    demo_cols = st.columns(len(demo_options))
    selected_demo = st.session_state.get('demo', None)

    for i, (label, (fruit, stage, color)) in enumerate(demo_options.items()):
        with demo_cols[i]:
            sc = STAGE_COLORS.get(stage,'#58a6ff')
            if st.button(label, key=f"demo_{i}"):
                st.session_state['demo'] = label

    if st.session_state.get('demo') in demo_options:
        fruit_d, stage_d, color_d = demo_options[st.session_state['demo']]
        # Create synthetic demo image
        demo_img = np.full((200,200,3), color_d[::-1], dtype=np.uint8)
        # Add some texture noise
        noise = np.random.randint(-25,25,(200,200,3))
        demo_img = np.clip(demo_img.astype(int)+noise, 0, 255).astype(np.uint8)
        img_bgr = demo_img
        st.info(f"Demo mode: synthetic {fruit_d} ({stage_d}) image generated from colour profile.")

# ── ANALYSE BUTTON ──
if img_bgr is not None:
    col_btn, _ = st.columns([1,3])
    with col_btn:
        analyse = st.button('🔍  Analyse Fruit Image', use_container_width=True)

    if analyse or (mode=='Demo Gallery' and img_bgr is not None):
        # ── PROCESSING ──
        with st.spinner(''):
            prog = st.progress(0, text='🔬 Running DIP pipeline...')
            time.sleep(0.2); prog.progress(20, text='🎨 Extracting colour features (Lab 02, 03)...')
            time.sleep(0.2); prog.progress(45, text='📊 Computing histograms (Lab 05)...')
            time.sleep(0.2); prog.progress(65, text='✏️ Edge detection + morphology (Lab 07, 11)...')
            time.sleep(0.2); prog.progress(85, text='🔲 Segmentation (Lab 12)...')
            result = full_prediction(img_bgr)
            prog.progress(100, text='✅ Analysis complete!')
            time.sleep(0.3)
            prog.empty()

# ── RESULTS ──
if result:
    sc    = STAGE_COLORS.get(result['stage'], '#58a6ff')
    eb    = '#1a4731' if result['edible'] else '#3d1f1f'
    ed_c  = '#3fb950' if result['edible'] else '#f85149'
    ei    = '✅' if result['edible'] else '❌'
    et    = 'GOOD TO EAT' if result['edible'] else 'NOT RECOMMENDED'

    st.markdown('<hr style="border-color:#30363d; margin:20px 0 16px;">', unsafe_allow_html=True)

    # ════════════════════════════════════════
    #  RESULT SUMMARY ROW
    # ════════════════════════════════════════
    st.markdown('<div class="fv-section-title">🎯 Prediction Results</div>',
                unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.markdown(f"""
        <div class="fv-metric">
          <div class="label">🍎 Fruit</div>
          <div class="value" style="color:#58a6ff; font-size:1.3rem;">
            {result['emoji']} {result['fruit']}
          </div>
          <div class="sub">{result['fruit_conf']:.1f}% confidence</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="fv-metric">
          <div class="label">🌱 Stage</div>
          <div class="value" style="color:{sc}; font-size:1.3rem;">
            {result['stage']}
          </div>
          <div class="sub">{result['stage_conf']:.1f}% confidence</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="fv-metric">
          <div class="label">🌤 Season</div>
          <div class="value" style="color:#f0883e; font-size:0.85rem;
               line-height:1.4; font-weight:700;">
            {result['season']}
          </div>
        </div>""", unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="fv-metric" style="background:{eb}; border:1px solid {ed_c};">
          <div class="label">{ei} Eat Status</div>
          <div class="value" style="color:{ed_c}; font-size:1rem;">
            {et}
          </div>
          <div class="sub">⏱ {result['days_left']}</div>
        </div>""", unsafe_allow_html=True)

    with c5:
        st.markdown(f"""
        <div class="fv-metric">
          <div class="label">🌿 Family</div>
          <div class="value" style="color:#bc8cff; font-size:0.85rem;
               font-weight:700;">
            {result['family']}
          </div>
          <div class="sub">Origin: {result['origin']}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div style="margin-top:12px;"></div>', unsafe_allow_html=True)

    # ── Feedback banner ──
    st.markdown(f"""
    <div class="fv-feedback" style="background:{eb}; border-color:{sc};">
      <div style="font-family:'Space Mono',monospace; font-size:0.7rem;
                  color:#8b949e; margin-bottom:6px; text-transform:uppercase;
                  letter-spacing:0.1em;">
        📋 Expert Recommendation
      </div>
      <div style="font-size:1.1rem; color:#e6edf3; font-weight:600;">
        {result['feedback']}
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Fruit detail pills ──
    st.markdown(f"""
    <div style="margin:8px 0 16px;">
      <span class="fv-pill">🎨 {result['color']}</span>
      <span class="fv-pill">✋ {result['texture']}</span>
      <span class="fv-pill">👅 {result['taste']}</span>
    </div>
    """, unsafe_allow_html=True)

    # ════════════════════════════════════════
    #  TABS
    # ════════════════════════════════════════
    tabs = st.tabs([
        "📊 Statistical Analysis",
        "🔬 DIP Pipeline",
        "🌿 Fruit Profile",
        "📋 Feature Table"
    ])

    # ─────────────────────────────
    #  TAB 1: STATISTICAL ANALYSIS
    # ─────────────────────────────
    with tabs[0]:
        if show_stats:
            st.markdown('<div class="fv-section-title">📊 Statistical Analysis</div>',
                        unsafe_allow_html=True)

            col_a, col_b = st.columns(2)
            with col_a:
                st.plotly_chart(
                    make_stage_donut(result['stage_probs'], result['stage']),
                    use_container_width=True, config={'displayModeBar':False}
                )
            with col_b:
                st.plotly_chart(
                    make_fruit_bar(result['fruit_probs'], result['fruit']),
                    use_container_width=True, config={'displayModeBar':False}
                )

            col_c, col_d = st.columns(2)
            with col_c:
                st.plotly_chart(
                    make_hue_histogram(result['h_hist']),
                    use_container_width=True, config={'displayModeBar':False}
                )
            with col_d:
                st.plotly_chart(
                    make_radar_chart(result['features']),
                    use_container_width=True, config={'displayModeBar':False}
                )

            # ── Key stats table ──
            st.markdown('<div class="fv-section-title" style="margin-top:8px;">📐 Key DIP Metrics</div>',
                        unsafe_allow_html=True)

            feats = result['features']
            metrics_data = {
                'Metric': [
                    'Mean Hue (H)', 'Mean Saturation (S)', 'Mean Value (V)',
                    'Redness Index', 'Greenness Index', 'Yellowness Index',
                    'Edge Density', 'Spot Density', 'Otsu Threshold',
                    'Texture Contrast', 'Homogeneity', 'Energy',
                    'Hue Kurtosis', 'Hue Skewness', 'Hue Peak'
                ],
                'Value': [
                    f"{feats['h_mean']:.2f}",   f"{feats['s_mean']:.2f}",
                    f"{feats['v_mean']:.2f}",   f"{feats['redness']:.2f}",
                    f"{feats['greenness']:.2f}", f"{feats['yellowness']:.2f}",
                    f"{feats['edge_density']:.4f}", f"{feats['spot_density']:.4f}",
                    f"{feats['otsu_thresh']:.1f}", f"{feats['contrast']:.4f}",
                    f"{feats['homogeneity']:.4f}", f"{feats['energy']:.4f}",
                    f"{feats['h_kurt']:.4f}",   f"{feats['h_skew']:.4f}",
                    f"{feats['h_peak']:.1f}"
                ],
                'DIP Lab': [
                    'Lab 02','Lab 02','Lab 02','Lab 03','Lab 03','Lab 03',
                    'Lab 07','Lab 11','Lab 12','Lab 05','Lab 05','Lab 05',
                    'Lab 05','Lab 05','Lab 05'
                ],
                'Interpretation': [
                    'Dominant hue angle (0=red, 60=yellow, 120=green)',
                    'Colour intensity/vividness (0=grey, 255=pure colour)',
                    'Brightness level (0=dark, 255=bright)',
                    'Positive = red dominant (ripe/overripe indicator)',
                    'Positive = green dominant (unripe indicator)',
                    'Positive = yellow dominant (banana/mango indicator)',
                    'Fraction of edge pixels (high = wrinkled/spotty)',
                    'Fraction of bright spot pixels (high = disease/overripe)',
                    'Optimal binary threshold (Otsu method)',
                    'GLCM contrast (high = rough texture)',
                    'GLCM homogeneity (high = smooth uniform surface)',
                    'GLCM energy (high = regular/uniform texture)',
                    'Histogram peakedness (high = concentrated hue)',
                    'Histogram asymmetry (negative = left-skewed)',
                    'Most frequent hue value in image'
                ]
            }
            df_metrics = pd.DataFrame(metrics_data)
            st.dataframe(df_metrics, use_container_width=True,
                         hide_index=True,
                         column_config={
                             'Value': st.column_config.TextColumn(width='small'),
                             'DIP Lab': st.column_config.TextColumn(width='small'),
                         })

    # ─────────────────────────────
    #  TAB 2: DIP PIPELINE
    # ─────────────────────────────
    with tabs[1]:
        if show_dip:
            st.markdown('<div class="fv-section-title">🔬 DIP Processing Pipeline</div>',
                        unsafe_allow_html=True)

            # Original + processed side by side
            col_orig, col_info = st.columns([1,2])
            with col_orig:
                st.image(result['img_rgb'],
                         caption='Input Image',
                         use_column_width=True)

            with col_info:
                st.markdown(f"""
                <div class="fv-card">
                  <div class="fv-section-title">Pipeline Stages</div>
                  <div style="font-family:'Space Mono',monospace;
                              font-size:0.78rem; line-height:2.2;
                              color:#c9d1d9;">
                    <span style="color:#58a6ff;">① Lab 02</span> —
                        BGR→HSV→LAB colour space conversion<br>
                    <span style="color:#58a6ff;">② Lab 03</span> —
                        Mathematical channel ratios (R/G, R/B)<br>
                    <span style="color:#58a6ff;">③ Lab 05</span> —
                        Hue histogram + kurtosis/skewness<br>
                    <span style="color:#58a6ff;">④ Lab 06</span> —
                        Bilateral filter denoising (σ=75)<br>
                    <span style="color:#58a6ff;">⑤ Lab 07</span> —
                        Canny edge detection (50/150)<br>
                    <span style="color:#58a6ff;">⑥ Lab 11</span> —
                        Top-hat morphological transform<br>
                    <span style="color:#58a6ff;">⑦ Lab 12</span> —
                        Otsu adaptive thresholding<br>
                  </div>
                </div>

                <div class="fv-card" style="margin-top:12px;">
                  <div class="fv-section-title">C3 — Trade-off Analysis</div>
                  <div style="font-family:'Space Mono',monospace;
                              font-size:0.75rem; line-height:1.9;
                              color:#8b949e;">
                    <span style="color:#f0883e;">Aggressive</span> colour masking
                    captures more disease area but introduces false positives
                    from shadows and lighting.<br><br>
                    <span style="color:#3fb950;">Conservative</span> thresholds
                    reduce false positives but may miss early-stage ripeness
                    changes in boundary hues.
                  </div>
                </div>
                """, unsafe_allow_html=True)

            # Full DIP panel
            st.markdown('<div style="margin-top:12px;"></div>',
                        unsafe_allow_html=True)
            dip_fig = make_dip_panels(result)
            st.pyplot(dip_fig, use_container_width=True)
            plt.close(dip_fig)

    # ─────────────────────────────
    #  TAB 3: FRUIT PROFILE
    # ─────────────────────────────
    with tabs[2]:
        col_p1, col_p2 = st.columns(2)

        with col_p1:
            if show_advice:
                st.markdown(f"""
                <div class="fv-card">
                  <div class="fv-section-title">
                    {result['emoji']} {result['fruit']} — {result['stage']} Stage
                  </div>

                  <div style="margin-bottom:16px;">
                    <div style="font-family:'Space Mono',monospace;
                                font-size:0.68rem; color:#8b949e;
                                text-transform:uppercase; margin-bottom:4px;">
                      Appearance
                    </div>
                    <div style="color:#e6edf3; font-size:0.95rem;">
                      {result['color']}
                    </div>
                  </div>

                  <div style="margin-bottom:16px;">
                    <div style="font-family:'Space Mono',monospace;
                                font-size:0.68rem; color:#8b949e;
                                text-transform:uppercase; margin-bottom:4px;">
                      Texture
                    </div>
                    <div style="color:#e6edf3; font-size:0.95rem;">
                      {result['texture']}
                    </div>
                  </div>

                  <div style="margin-bottom:16px;">
                    <div style="font-family:'Space Mono',monospace;
                                font-size:0.68rem; color:#8b949e;
                                text-transform:uppercase; margin-bottom:4px;">
                      Taste Profile
                    </div>
                    <div style="color:#e6edf3; font-size:0.95rem;">
                      {result['taste']}
                    </div>
                  </div>

                  <div style="margin-bottom:16px;">
                    <div style="font-family:'Space Mono',monospace;
                                font-size:0.68rem; color:#8b949e;
                                text-transform:uppercase; margin-bottom:4px;">
                      ⏱ Time Estimate
                    </div>
                    <div style="color:#f0883e; font-size:0.95rem;
                                font-weight:700;">
                      {result['days_left']}
                    </div>
                  </div>

                  <div style="margin-bottom:16px;">
                    <div style="font-family:'Space Mono',monospace;
                                font-size:0.68rem; color:#8b949e;
                                text-transform:uppercase; margin-bottom:4px;">
                      💡 Storage Tip
                    </div>
                    <div style="color:#58a6ff; font-size:0.9rem;">
                      {result['tip']}
                    </div>
                  </div>

                  <div>
                    <div style="font-family:'Space Mono',monospace;
                                font-size:0.68rem; color:#8b949e;
                                text-transform:uppercase; margin-bottom:4px;">
                      🫀 Health Notes
                    </div>
                    <div style="color:#c9d1d9; font-size:0.9rem;
                                line-height:1.6;">
                      {result['health']}
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

            # All stages comparison
            st.markdown('<div class="fv-section-title" style="margin-top:4px;">All Stages Comparison</div>',
                        unsafe_allow_html=True)
            all_stages = FRUIT_KNOWLEDGE[result['fruit']]['stages']
            for s_name, s_info in all_stages.items():
                active = s_name == result['stage']
                bc = STAGE_COLORS.get(s_name,'#58a6ff')
                bg = '#21262d' if active else '#161b22'
                bw = '2px' if active else '1px'
                st.markdown(f"""
                <div style="background:{bg}; border:{bw} solid {bc};
                            border-radius:10px; padding:12px 16px;
                            margin-bottom:8px;">
                  <div style="display:flex; justify-content:space-between;
                              align-items:center; margin-bottom:6px;">
                    <span style="color:{bc}; font-weight:700;
                                 font-family:'Syne',sans-serif;">
                      {'▶ ' if active else ''}{s_name}
                      {'← Current' if active else ''}
                    </span>
                    <span style="font-family:'Space Mono',monospace;
                                 font-size:0.7rem; color:#8b949e;">
                      {s_info['days_left']}
                    </span>
                  </div>
                  <div style="font-family:'Space Mono',monospace;
                              font-size:0.75rem; color:#8b949e;
                              line-height:1.7;">
                    🎨 {s_info['color']}<br>
                    👅 {s_info['taste']}
                  </div>
                </div>
                """, unsafe_allow_html=True)

        with col_p2:
            # Botanical info
            st.markdown(f"""
            <div class="fv-card">
              <div class="fv-section-title">🌱 Botanical Information</div>
              <div style="display:grid; grid-template-columns:1fr 1fr;
                          gap:12px;">
                <div>
                  <div style="font-family:'Space Mono',monospace;
                              font-size:0.65rem; color:#8b949e;
                              text-transform:uppercase;">Family</div>
                  <div style="color:#bc8cff; font-weight:700;
                              margin-top:2px;">{result['family']}</div>
                </div>
                <div>
                  <div style="font-family:'Space Mono',monospace;
                              font-size:0.65rem; color:#8b949e;
                              text-transform:uppercase;">Origin</div>
                  <div style="color:#58a6ff; font-weight:700;
                              margin-top:2px;">{result['origin']}</div>
                </div>
                <div>
                  <div style="font-family:'Space Mono',monospace;
                              font-size:0.65rem; color:#8b949e;
                              text-transform:uppercase;">Season</div>
                  <div style="color:#f0883e; font-weight:700;
                              margin-top:2px;">{result['season']}</div>
                </div>
                <div>
                  <div style="font-family:'Space Mono',monospace;
                              font-size:0.65rem; color:#8b949e;
                              text-transform:uppercase;">Stage</div>
                  <div style="color:{sc}; font-weight:700;
                              margin-top:2px;">{result['stage']}</div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            if show_nutri:
                st.plotly_chart(
                    make_nutrition_bar(result['nutrition']),
                    use_container_width=True, config={'displayModeBar':False}
                )

                # Nutrition cards
                st.markdown('<div class="fv-section-title">💊 Nutrition (per 100g)</div>',
                            unsafe_allow_html=True)
                nutri_cols = st.columns(len(result['nutrition']))
                ncolors = ['#3fb950','#58a6ff','#f0883e','#f85149','#bc8cff']
                for j,(key,val) in enumerate(result['nutrition'].items()):
                    with nutri_cols[j]:
                        st.markdown(f"""
                        <div class="fv-metric">
                          <div class="label">{key}</div>
                          <div class="value" style="color:{ncolors[j%len(ncolors)]};
                               font-size:1.1rem;">{val}</div>
                        </div>""", unsafe_allow_html=True)

    # ─────────────────────────────
    #  TAB 4: FEATURE TABLE
    # ─────────────────────────────
    with tabs[3]:
        st.markdown('<div class="fv-section-title">📋 Complete Feature Extraction Report</div>',
                    unsafe_allow_html=True)

        feats = result['features']
        full_table = pd.DataFrame({
            'Feature': FEATURE_COLS,
            'Value':   [round(feats[c],4) for c in FEATURE_COLS],
            'Lab':     ['L02','L02','L02','L02','L02',
                        'L03','L03','L03','L03','L03',
                        'L05','L05','L05','L05',
                        'L07','L07','L07','L11','L12',
                        'L05','L05','L05','L05'],
            'Category':['Colour','Colour','Colour','Colour','Colour',
                        'Ratio','Ratio','Ratio','Ratio','Ratio',
                        'Histogram','Histogram','Histogram','Histogram',
                        'Edge','Edge','Edge','Morphology','Segmentation',
                        'Texture','Texture','Texture','Texture'],
            'Range':   ['0-180','0-255','0-255','-128–127','-128–127',
                        '0.5–3.0','0.5–3.0','-50–80','-50–80','-50–80',
                        '-3–10','-3–3','0-179','0-255',
                        '0–0.15','0–100','0–100','0–0.1','0-255',
                        '0–5000','0–1','0–1','-1–1'],
        })
        st.dataframe(full_table, use_container_width=True, hide_index=True)

        # Download button
        csv = full_table.to_csv(index=False)
        st.download_button(
            label='⬇️  Download Feature Report (CSV)',
            data=csv,
            file_name=f'fruitivision_{result["fruit"]}_{result["stage"]}.csv',
            mime='text/csv',
        )

    # ── Bottom info bar ──
    st.markdown('<hr style="border-color:#30363d; margin:24px 0 12px;">',
                unsafe_allow_html=True)
    st.markdown(f"""
    <div style="display:flex; justify-content:space-between; flex-wrap:wrap;
                gap:8px; font-family:'Space Mono',monospace; font-size:0.68rem;
                color:#8b949e; padding-bottom:12px;">
      <span>🍎 FruitVision DIP System · COMP-342L P06 · Spring 2025</span>
      <span>Pak-Austria Fachhochschule · Haripur, Pakistan</span>
      <span>Labs: L02·L03·L05·L06·L07·L11·L12 · CCP: C1–C6 ✅</span>
    </div>
    """, unsafe_allow_html=True)

else:
    # Welcome state
    st.markdown("""
    <div style="text-align:center; padding:60px 20px; color:#8b949e;">
      <div style="font-size:4rem; margin-bottom:16px;">🍎 🍌 🥭 🍊 🍓</div>
      <div style="font-family:'Syne',sans-serif; font-size:1.4rem;
                  font-weight:700; color:#e6edf3; margin-bottom:12px;">
        Upload a fruit image to begin
      </div>
      <div style="font-family:'Space Mono',monospace; font-size:0.85rem;
                  line-height:2;">
        The system will detect the <span style="color:#58a6ff;">fruit type</span>,
        predict the <span style="color:#3fb950;">growth stage</span>,
        identify the <span style="color:#f0883e;">harvest season</span>,<br>
        and provide <span style="color:#f85149;">eat / no-eat feedback</span>
        using 23 Digital Image Processing features.
      </div>
    </div>
    """, unsafe_allow_html=True)