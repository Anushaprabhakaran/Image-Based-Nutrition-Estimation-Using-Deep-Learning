import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import time

st.set_page_config(
    page_title="NutriLens · AI Nutrition Estimator",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
.main > div { padding: 0 !important; }
.block-container { padding: 0 !important; max-width: 100% !important; margin: 0 !important; }
[data-testid="stAppViewContainer"] { padding: 0 !important; }
[data-testid="stVerticalBlock"] { gap: 0 !important; padding: 0 !important; }
[data-testid="stVerticalBlockBorderWrapper"] { padding: 0 !important; }
section[data-testid="stSidebar"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }
header[data-testid="stHeader"] { display: none !important; }
.stMarkdown { margin: 0 !important; padding: 0 !important; }
div[data-testid="stMarkdownContainer"] { margin: 0 !important; padding: 0 !important; }
[data-testid="stHorizontalBlock"] { gap: 0 !important; padding: 0 !important; background: #dde6f0; }
[data-testid="stColumn"] { padding: 10px !important; }

/* ── NAV ── */
.topnav {
  background: #0f1e35;
  padding: 14px 36px;
  display: flex; align-items: center; justify-content: space-between;
  border-bottom: 1px solid #1e3050;
}
.logo { display: flex; align-items: center; gap: 12px; }
.logo-text { font-size: 18px; font-weight: 700; color: #fff; letter-spacing: -0.01em; }
.logo-text span { color: #4ade80; }
.nav-tags { display: flex; gap: 8px; }
.nav-tag {
  background: #1a2f4a; color: #7aabcf;
  border: 1px solid #253d5a; border-radius: 20px;
  padding: 5px 14px; font-size: 11px; font-weight: 500;
}

/* ── HERO ── */
.hero {
  background: linear-gradient(135deg, #0f1e35 0%, #162840 60%, #0f1e35 100%);
  padding: 0;
  position: relative;
  overflow: hidden;
  border-bottom: 1px solid #1e3050;
  height: 300px;
  width: 100%;
}

/* Hero text: absolutely placed on the LEFT half */
.hero-content {
  position: absolute;
  top: 0; left: 0; bottom: 0;
  width: 48%;
  z-index: 3;
  display: flex;
  flex-direction: column;
  justify-content: center;
  padding: 38px 0 38px 42px;
}

.hero-eyebrow {
  font-size: 12px; letter-spacing: 0.18em; text-transform: uppercase;
  color: #4ade80; font-weight: 600; margin-bottom: 12px;
}
.hero-title {
  font-size: 42px; font-weight: 700; color: #fff;
  letter-spacing: -0.025em; line-height: 1.1; margin-bottom: 10px;
}
.hero-title span { color: #4ade80; }
.hero-tagline {
  font-size: 15px; color: #7aabcf;
  font-weight: 500; margin-bottom: 10px; letter-spacing: 0.03em;
}
.hero-desc {
  font-size: 13px; color: #5a8ab0;
  line-height: 1.7; margin-bottom: 24px; max-width: 92%;
}
.hero-stats { display: flex; gap: 32px; }
.stat-val { font-size: 22px; font-weight: 700; color: #fff; font-family: 'JetBrains Mono', monospace; }
.stat-label { font-size: 11px; color: #4a7a9a; margin-top: 3px; }

/* food emojis: all absolutely placed on the RIGHT half */
.food-item {
  position: absolute;
  z-index: 2;
  filter: drop-shadow(0 6px 20px rgba(0,0,0,0.55));
}

/* ── PANELS WRAPPER ── */
.panels-outer { background: #dde6f0; padding: 20px 20px; }
.panel-card {
  background: #fff; border-radius: 16px; padding: 24px;
  box-shadow: 0 2px 16px rgba(15,30,53,0.09); border: 1px solid #ccd8e4;
  height: 100%;
}
.panel-head {
  font-size: 12px; font-weight: 700; letter-spacing: 0.13em;
  text-transform: uppercase; color: #3a5a7a; margin-bottom: 5px;
}
.panel-sub { font-size: 13px; color: #6a8aaa; margin-bottom: 18px; }

.upload-zone {
  border: 2px dashed #a8c4dc; border-radius: 12px;
  padding: 32px 20px; text-align: center; background: #f0f6fc; margin-bottom: 4px;
}
.upload-text { font-size: 13px; color: #3a5a7a; font-weight: 500; margin-top: 12px; }
.upload-sub { font-size: 12px; color: #7a9ab8; margin-top: 5px; }

.nut-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-bottom: 18px; }
.nut-card {
  background: #fff; border: 1px solid #ccd8e4; border-radius: 14px;
  padding: 16px 16px 14px; position: relative; overflow: hidden;
  box-shadow: 0 1px 8px rgba(15,30,53,0.06);
}
.nut-top-bar { position: absolute; top: 0; left: 0; right: 0; height: 4px; border-radius: 14px 14px 0 0; }
.nut-label {
  font-size: 11px; font-weight: 700; letter-spacing: 0.12em;
  text-transform: uppercase; color: #4a6a8a;
  margin-bottom: 10px; margin-top: 4px;
  display: flex; justify-content: space-between; align-items: center;
}
.nut-val { font-size: 30px; font-weight: 700; color: #0f1e35; font-family: 'JetBrains Mono', monospace; line-height: 1; }
.nut-unit { font-size: 12px; color: #7a9ab8; margin-top: 5px; }

.section-head {
  font-size: 11px; font-weight: 700; letter-spacing: 0.12em;
  text-transform: uppercase; color: #3a5a7a; margin-bottom: 10px;
}
.macro-bar-wrap { height: 11px; border-radius: 6px; background: #dde6f0; overflow: hidden; display: flex; margin-bottom: 10px; }
.macro-legend { display: flex; gap: 16px; margin-bottom: 16px; }
.macro-dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 6px; vertical-align: middle; }
.macro-leg-item { font-size: 12px; color: #3a5a7a; }

.disclaimer {
  background: #fffbeb; border: 1px solid #d4a72c;
  border-radius: 9px; padding: 10px 14px;
  font-size: 12px; color: #713f12; margin-bottom: 16px;
}

.how-card { background: #f0f6fc; border: 1px solid #ccd8e4; border-radius: 14px; padding: 16px 18px; }
.how-step { display: flex; gap: 12px; margin-bottom: 12px; align-items: flex-start; }
.how-step:last-child { margin-bottom: 0; }
.how-num {
  width: 24px; height: 24px; border-radius: 50%;
  background: #0f1e35; color: #fff;
  font-size: 11px; font-weight: 700;
  display: flex; align-items: center; justify-content: center;
  flex-shrink: 0; margin-top: 1px;
}
.how-text { font-size: 12px; color: #2a4a6a; line-height: 1.6; }
.how-text strong { color: #0f1e35; }

.empty-state {
  display: flex; flex-direction: column; align-items: center;
  justify-content: center; text-align: center; padding: 4rem 2rem;
}
.empty-icon { font-size: 3rem; margin-bottom: 1rem; }
.empty-title { font-size: 1.05rem; font-weight: 600; color: #4a6a8a; margin-bottom: 0.5rem; }
.empty-desc { font-size: 0.9rem; line-height: 1.7; color: #7a9ab8; }

.footer {
  background: #0f1e35; padding: 16px 36px;
  display: flex; justify-content: space-between; align-items: center;
  border-top: 1px solid #1e3050;
}
.footer-l { font-size: 11px; color: #4a7a9a; }
.footer-r { font-size: 11px; color: #4ade80; font-weight: 500; }

[data-testid="stFileUploaderDropzoneInstructions"] { display: none !important; }
[data-testid="stFileUploader"] section { border: none !important; padding: 0 !important; background: transparent !important; }
[data-testid="stFileUploader"] { margin: 0 !important; padding: 0 !important; }
</style>
""", unsafe_allow_html=True)

# ── Model ──────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 4)
    model.load_state_dict(torch.load("nutrition_model_epoch35.pth", map_location="cpu"))
    model.eval()
    return model

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

EB = "https://fonts.gstatic.com/s/e/notoemoji/latest"

# ── NAV + HERO: ONE single st.markdown block so position:absolute works ────────
# The key fix: hero-content AND all food-items are inside ONE HTML block.
# Streamlit wraps each st.markdown in a div, so splitting them breaks abs. positioning.
st.markdown(f"""
<div class="topnav">
  <div class="logo">
    <svg width="40" height="40" viewBox="0 0 38 38">
      <circle cx="16" cy="16" r="13" fill="#0a1828" stroke="#4ade80" stroke-width="2.2"/>
      <circle cx="16" cy="16" r="10.5" fill="#0d2235"/>
      <circle cx="13" cy="14" r="3.5" fill="#16a34a"/>
      <circle cx="16" cy="11.5" r="3" fill="#22c55e"/>
      <circle cx="19" cy="14" r="3.2" fill="#15803d"/>
      <rect x="15" y="17" width="2" height="4" rx="1" fill="#92601a"/>
      <path d="M20,19 Q22,16 23,20 Q22,22 20,21Z" fill="#f97316"/>
      <path d="M21.5,16 L21,14.5 M21.5,16 L23,15 M21.5,16 L20.5,14.8" stroke="#22c55e" stroke-width="0.8" stroke-linecap="round"/>
      <circle cx="10" cy="10" r="2" fill="white" opacity="0.1"/>
      <line x1="26" y1="26" x2="35" y2="35" stroke="#4ade80" stroke-width="3" stroke-linecap="round"/>
      <line x1="25" y1="25" x2="27" y2="27" stroke="#2d9a5a" stroke-width="3.5" stroke-linecap="round"/>
    </svg>
    <div class="logo-text">Nutri<span>Lens</span></div>
  </div>
  <div class="nav-tags">
    <span class="nav-tag">ResNet-50</span>
    <span class="nav-tag">PyTorch</span>
    <span class="nav-tag">Computer Vision</span>
  </div>
</div>

<div class="hero">

  <!-- LEFT: text content -->
  <div class="hero-content">
    <div class="hero-eyebrow">Nutrition Estimation Using AI</div>
    <div class="hero-title">Snap a meal.<br><span>Know your macros.</span></div>
    <div class="hero-tagline">Image Recognition · Deep Learning</div>
    <div class="hero-desc">
      Upload any food photo and get an instant breakdown of calories, fat,
      carbs, and protein — powered by a fine-tuned ResNet-50 model.
    </div>
    <div class="hero-stats">
      <div><div class="stat-val">35</div><div class="stat-label">Training epochs</div></div>
      <div><div class="stat-val">4</div><div class="stat-label">Nutrition outputs</div></div>
      <div><div class="stat-val">224px</div><div class="stat-label">Input resolution</div></div>
      <div><div class="stat-val">&lt;1s</div><div class="stat-label">Inference time</div></div>
    </div>
  </div>

  <!-- RIGHT: scattered food emojis -->
  <!-- ROW 1: top — sandwich, pizza, cake -->
  <div class="food-item" style="top:18px; left:52%; transform:rotate(-8deg);">
    <img src="{EB}/1f96a/emoji.svg" width="82" height="82" alt="sandwich"/>
  </div>
  <div class="food-item" style="top:12px; left:67%; transform:rotate(6deg);">
    <img src="{EB}/1f355/emoji.svg" width="78" height="78" alt="pizza"/>
  </div>
  <div class="food-item" style="top:10px; right:32px; transform:rotate(5deg);">
    <img src="{EB}/1f370/emoji.svg" width="74" height="74" alt="cake"/>
  </div>

  <!-- ROW 2: middle — burger, pasta, carrot -->
  <div class="food-item" style="top:112px; left:51%; transform:rotate(-5deg);">
    <img src="{EB}/1f354/emoji.svg" width="86" height="86" alt="burger"/>
  </div>
  <div class="food-item" style="top:104px; left:65%; transform:rotate(4deg);">
    <img src="{EB}/1f35d/emoji.svg" width="74" height="74" alt="pasta"/>
  </div>
  <div class="food-item" style="top:58px; right:118px; transform:rotate(14deg);">
    <img src="{EB}/1f955/emoji.svg" width="68" height="68" alt="carrot"/>
  </div>

  <!-- ROW 3: bottom — noodles, fries, broccoli -->
  <div class="food-item" style="top:196px; left:52%; transform:rotate(-4deg);">
    <img src="{EB}/1f35c/emoji.svg" width="74" height="74" alt="noodles"/>
  </div>
  <div class="food-item" style="top:188px; left:65%; transform:rotate(7deg);">
    <img src="{EB}/1f35f/emoji.svg" width="68" height="68" alt="fries"/>
  </div>
  <div class="food-item" style="top:182px; right:34px; transform:rotate(-6deg);">
    <img src="{EB}/1f966/emoji.svg" width="74" height="74" alt="broccoli"/>
  </div>

</div>

<div class="panels-outer">
""", unsafe_allow_html=True)

# ── TWO COLUMNS ────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="panel-head">Upload food image</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-sub">JPG or PNG · best with clear, well-lit food photos</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("food", type=["jpg","jpeg","png"], label_visibility="collapsed")

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, use_container_width=True)
    else:
        st.markdown("""
        <div class="upload-zone">
          <svg width="46" height="46" viewBox="0 0 42 42" style="margin:0 auto;display:block;">
            <rect x="3" y="10" width="36" height="26" rx="5" fill="#dde6f0" stroke="#a8c4dc" stroke-width="1.4"/>
            <circle cx="13" cy="20" r="4" fill="#a8c4dc"/>
            <polygon points="3,36 14,22 22,28 30,18 39,36" fill="#b8d4e8" opacity="0.7"/>
            <circle cx="32" cy="6" r="6" fill="#0f1e35"/>
            <line x1="32" y1="2.5" x2="32" y2="9.5" stroke="#fff" stroke-width="1.8" stroke-linecap="round"/>
            <line x1="28.5" y1="6" x2="35.5" y2="6" stroke="#fff" stroke-width="1.8" stroke-linecap="round"/>
          </svg>
          <div class="upload-text">Drag and drop your food photo here</div>
          <div class="upload-sub">or click to browse files</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)

    if uploaded:
        with st.spinner("Running inference…"):
            time.sleep(0.3)
            try:
                mdl = load_model()
            except Exception as e:
                st.error(f"Model load error: {e}")
                st.stop()
            t = transform(image).unsqueeze(0)
            with torch.no_grad():
                p = mdl(t).cpu().numpy()[0]

        cal, fat, carb, prot = p

        st.markdown('<div class="panel-head">Estimated nutrition</div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-sub">Per serving · model estimates, not lab values</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="nut-grid">
          <div class="nut-card">
            <div class="nut-top-bar" style="background:#ef4444;"></div>
            <div class="nut-label">Calories <span>🔥</span></div>
            <div class="nut-val">{cal:.0f}</div>
            <div class="nut-unit">kcal</div>
          </div>
          <div class="nut-card">
            <div class="nut-top-bar" style="background:#f59e0b;"></div>
            <div class="nut-label">Fat <span>🥑</span></div>
            <div class="nut-val">{fat:.1f}</div>
            <div class="nut-unit">grams</div>
          </div>
          <div class="nut-card">
            <div class="nut-top-bar" style="background:#3b82f6;"></div>
            <div class="nut-label">Carbs <span>🌾</span></div>
            <div class="nut-val">{carb:.1f}</div>
            <div class="nut-unit">grams</div>
          </div>
          <div class="nut-card">
            <div class="nut-top-bar" style="background:#22c55e;"></div>
            <div class="nut-label">Protein <span>💪</span></div>
            <div class="nut-val">{prot:.1f}</div>
            <div class="nut-unit">grams</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        total = fat * 9 + carb * 4 + prot * 4
        if total > 0:
            pf = fat * 9 / total * 100
            pc = carb * 4 / total * 100
            pp = prot * 4 / total * 100
            st.markdown(f"""
            <div class="section-head">Macro energy split</div>
            <div class="macro-bar-wrap">
              <div style="width:{pf:.1f}%;background:#f59e0b;"></div>
              <div style="width:{pc:.1f}%;background:#3b82f6;"></div>
              <div style="width:{pp:.1f}%;background:#22c55e;"></div>
            </div>
            <div class="macro-legend">
              <span class="macro-leg-item"><span class="macro-dot" style="background:#f59e0b;"></span>Fat {pf:.0f}%</span>
              <span class="macro-leg-item"><span class="macro-dot" style="background:#3b82f6;"></span>Carbs {pc:.0f}%</span>
              <span class="macro-leg-item"><span class="macro-dot" style="background:#22c55e;"></span>Protein {pp:.0f}%</span>
            </div>
            <div class="disclaimer">⚠️ Estimates may vary with portion size and cooking method.</div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="how-card">
          <div class="section-head" style="margin-bottom:12px;">How it works</div>
          <div class="how-step">
            <div class="how-num">1</div>
            <div class="how-text"><strong>Image preprocessing</strong> — resized to 224×224px, normalised with ImageNet statistics</div>
          </div>
          <div class="how-step">
            <div class="how-num">2</div>
            <div class="how-text"><strong>ResNet-50 inference</strong> — 50-layer CNN extracts visual features from the food image</div>
          </div>
          <div class="how-step">
            <div class="how-num">3</div>
            <div class="how-text"><strong>4-output regression</strong> — custom final layer predicts all 4 nutrients simultaneously</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="panel-head">Estimated nutrition</div>
        <div class="panel-sub">Per serving · model estimates, not lab values</div>
        <div class="empty-state">
          <div class="empty-icon">🍽️</div>
          <div class="empty-title">No image uploaded yet</div>
          <div class="empty-desc">Upload a food photo on the left and the model<br>will estimate its nutritional content in under a second.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# close panels-outer + footer
st.markdown("""
</div>
<div class="footer">
  <div class="footer-l">Built with PyTorch · ResNet-50 · Streamlit · Fine-tuned 35 epochs</div>
  <div class="footer-r">LinkedIn ↗</div>
</div>
""", unsafe_allow_html=True)
