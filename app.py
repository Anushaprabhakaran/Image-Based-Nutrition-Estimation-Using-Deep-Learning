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

html, body, [class*="css"] {
  font-family: 'Inter', sans-serif;
  background-color: #dde6f0;
}
.block-container { padding: 0 !important; max-width: 100% !important; }
[data-testid="collapsedControl"] { display: none; }

/* ── NAV ── */
.topnav {
  background: #0f1e35;
  padding: 13px 32px;
  display: flex; align-items: center; justify-content: space-between;
  border-bottom: 1px solid #1e3050;
}
.logo { display: flex; align-items: center; gap: 11px; }
.logo-text { font-size: 16px; font-weight: 700; color: #fff; letter-spacing: -0.01em; }
.logo-text span { color: #4ade80; }
.nav-tags { display: flex; gap: 7px; }
.nav-tag {
  background: #1a2f4a; color: #7aabcf;
  border: 1px solid #253d5a; border-radius: 20px;
  padding: 4px 12px; font-size: 10.5px; font-weight: 500;
}

/* ── HERO ── */
.hero {
  background: linear-gradient(135deg, #0f1e35 0%, #162840 60%, #0f1e35 100%);
  padding: 38px 36px 32px;
  position: relative; overflow: hidden;
  border-bottom: 1px solid #1e3050;
  min-height: 260px;
}
.hero-content { position: relative; z-index: 3; max-width: 480px; }
.hero-eyebrow {
  font-size: 11px; letter-spacing: 0.18em; text-transform: uppercase;
  color: #4ade80; font-weight: 600; margin-bottom: 10px;
}
.hero-title {
  font-size: 36px; font-weight: 700; color: #fff;
  letter-spacing: -0.025em; line-height: 1.1; margin-bottom: 8px;
}
.hero-title span { color: #4ade80; }
.hero-tagline { font-size: 13px; color: #7aabcf; font-weight: 500; margin-bottom: 8px; letter-spacing: 0.03em; }
.hero-desc { font-size: 12px; color: #5a8ab0; line-height: 1.7; margin-bottom: 20px; max-width: 400px; }
.hero-stats { display: flex; gap: 30px; }
.stat-val { font-size: 19px; font-weight: 700; color: #fff; font-family: 'JetBrains Mono', monospace; }
.stat-label { font-size: 10px; color: #4a7a9a; margin-top: 2px; }

/* food items — right 50% only, clear of text */
.food-item {
  position: absolute; z-index: 2;
  filter: drop-shadow(0 6px 18px rgba(0,0,0,0.5));
}
.food-item img { display: block; }

/* ── PANELS ── */
.panels-bg { background: #dde6f0; padding: 20px; }
.panels {
  display: grid; grid-template-columns: 1fr 1fr;
  gap: 16px; max-width: 1100px; margin: 0 auto;
}
.panel-card {
  background: #fff; border-radius: 14px; padding: 22px;
  box-shadow: 0 2px 12px rgba(15,30,53,0.08); border: 1px solid #ccd8e4;
}
.panel-head {
  font-size: 11px; font-weight: 700; letter-spacing: 0.13em;
  text-transform: uppercase; color: #3a5a7a; margin-bottom: 4px;
}
.panel-sub { font-size: 11.5px; color: #6a8aaa; margin-bottom: 16px; }

/* upload */
.upload-zone {
  border: 2px dashed #a8c4dc; border-radius: 12px;
  padding: 28px 16px; text-align: center; background: #f0f6fc;
}
.upload-text { font-size: 12px; color: #3a5a7a; font-weight: 500; margin-top: 10px; }
.upload-sub { font-size: 11px; color: #7a9ab8; margin-top: 4px; }
.upload-btn {
  display: inline-block; margin-top: 12px;
  background: #0f1e35; color: #fff;
  border-radius: 8px; padding: 8px 18px;
  font-size: 11.5px; font-weight: 600;
}
.preview-area {
  margin-top: 16px; background: #f0f6fc; border-radius: 12px;
  height: 150px; display: flex; align-items: center; justify-content: center;
  border: 1px solid #ccd8e4; color: #8aacca; font-size: 11px;
}

/* nutrition cards */
.nut-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 16px; }
.nut-card {
  background: #fff; border: 1px solid #ccd8e4; border-radius: 12px;
  padding: 14px 14px 12px; position: relative; overflow: hidden;
  box-shadow: 0 1px 6px rgba(15,30,53,0.05);
}
.nut-top-bar { position: absolute; top: 0; left: 0; right: 0; height: 4px; border-radius: 12px 12px 0 0; }
.nut-label {
  font-size: 10px; font-weight: 700; letter-spacing: 0.12em;
  text-transform: uppercase; color: #4a6a8a;
  margin-bottom: 8px; margin-top: 2px;
  display: flex; justify-content: space-between; align-items: center;
}
.nut-val { font-size: 26px; font-weight: 700; color: #0f1e35; font-family: 'JetBrains Mono', monospace; line-height: 1; }
.nut-unit { font-size: 11px; color: #7a9ab8; margin-top: 4px; }

/* macro */
.section-head {
  font-size: 10.5px; font-weight: 700; letter-spacing: 0.12em;
  text-transform: uppercase; color: #3a5a7a; margin-bottom: 9px;
}
.macro-bar-wrap { height: 10px; border-radius: 6px; background: #dde6f0; overflow: hidden; display: flex; margin-bottom: 9px; }
.macro-legend { display: flex; gap: 14px; margin-bottom: 14px; }
.macro-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 5px; vertical-align: middle; }
.macro-leg-item { font-size: 11px; color: #3a5a7a; }

.disclaimer {
  background: #fffbeb; border: 1px solid #d4a72c;
  border-radius: 8px; padding: 8px 12px;
  font-size: 11px; color: #713f12; margin-bottom: 14px;
}

/* how it works */
.how-card { background: #f0f6fc; border: 1px solid #ccd8e4; border-radius: 12px; padding: 14px 16px; }
.how-step { display: flex; gap: 10px; margin-bottom: 10px; align-items: flex-start; }
.how-step:last-child { margin-bottom: 0; }
.how-num {
  width: 22px; height: 22px; border-radius: 50%;
  background: #0f1e35; color: #fff;
  font-size: 10px; font-weight: 700;
  display: flex; align-items: center; justify-content: center;
  flex-shrink: 0; margin-top: 1px;
}
.how-text { font-size: 11px; color: #2a4a6a; line-height: 1.6; }
.how-text strong { color: #0f1e35; }

/* empty state */
.empty-state {
  display: flex; flex-direction: column; align-items: center;
  justify-content: center; text-align: center;
  padding: 4rem 2rem; color: #8aacca;
}
.empty-icon { font-size: 2.8rem; margin-bottom: 1rem; }
.empty-title { font-size: 0.95rem; font-weight: 600; color: #4a6a8a; margin-bottom: 0.4rem; }
.empty-desc { font-size: 0.83rem; line-height: 1.6; }

/* footer */
.footer {
  background: #0f1e35; padding: 14px 32px;
  display: flex; justify-content: space-between; align-items: center;
  border-top: 1px solid #1e3050;
}
.footer-l { font-size: 10.5px; color: #4a7a9a; }
.footer-r { font-size: 10.5px; color: #4ade80; font-weight: 500; }

/* hide streamlit file uploader chrome */
[data-testid="stFileUploaderDropzoneInstructions"] { display: none; }
[data-testid="stFileUploader"] section { border: none; padding: 0; background: transparent; }
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

# ── NAV ────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="topnav">
  <div class="logo">
    <svg width="38" height="38" viewBox="0 0 38 38">
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
""", unsafe_allow_html=True)

# ── HERO ───────────────────────────────────────────────────────────────────────
# Using static SVG versions of Noto Emoji (no animation)
EMOJI_BASE = "https://fonts.gstatic.com/s/e/notoemoji/latest"

st.markdown(f"""
<div class="hero">

  <!-- ROW 1: top — sandwich, pizza, cake -->
  <div class="food-item" style="top:16px;left:52%;transform:rotate(-8deg);">
    <img src="{EMOJI_BASE}/1f96a/emoji.svg" width="78" height="78" alt="sandwich"/>
  </div>
  <div class="food-item" style="top:10px;left:66%;transform:rotate(6deg);">
    <img src="{EMOJI_BASE}/1f355/emoji.svg" width="74" height="74" alt="pizza"/>
  </div>
  <div class="food-item" style="top:8px;right:28px;transform:rotate(5deg);">
    <img src="{EMOJI_BASE}/1f370/emoji.svg" width="70" height="70" alt="cake"/>
  </div>

  <!-- ROW 2: middle — burger, pasta, carrot -->
  <div class="food-item" style="top:108px;left:50%;transform:rotate(-5deg);">
    <img src="{EMOJI_BASE}/1f354/emoji.svg" width="82" height="82" alt="burger"/>
  </div>
  <div class="food-item" style="top:100px;left:64%;transform:rotate(4deg);">
    <img src="{EMOJI_BASE}/1f35d/emoji.svg" width="70" height="70" alt="pasta"/>
  </div>
  <div class="food-item" style="top:55px;right:110px;transform:rotate(14deg);">
    <img src="{EMOJI_BASE}/1f955/emoji.svg" width="64" height="64" alt="carrot"/>
  </div>

  <!-- ROW 3: bottom — noodles, fries, broccoli -->
  <div class="food-item" style="top:168px;left:50%;transform:rotate(-4deg);">
    <img src="{EMOJI_BASE}/1f35c/emoji.svg" width="70" height="70" alt="noodles"/>
  </div>
  <div class="food-item" style="top:162px;left:63%;transform:rotate(7deg);">
    <img src="{EMOJI_BASE}/1f35f/emoji.svg" width="64" height="64" alt="fries"/>
  </div>
  <div class="food-item" style="top:158px;right:30px;transform:rotate(-6deg);">
    <img src="{EMOJI_BASE}/1f966/emoji.svg" width="70" height="70" alt="broccoli"/>
  </div>

  <!-- Text content — left 48% only, never overlaps food -->
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
</div>
""", unsafe_allow_html=True)

# ── TWO PANELS ─────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="small")

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
          <svg width="42" height="42" viewBox="0 0 42 42" style="margin:0 auto;display:block;">
            <rect x="3" y="10" width="36" height="26" rx="5" fill="#dde6f0" stroke="#a8c4dc" stroke-width="1.4"/>
            <circle cx="13" cy="20" r="4" fill="#a8c4dc"/>
            <polygon points="3,36 14,22 22,28 30,18 39,36" fill="#b8d4e8" opacity="0.7"/>
            <circle cx="32" cy="6" r="6" fill="#0f1e35"/>
            <line x1="32" y1="2.5" x2="32" y2="9.5" stroke="#fff" stroke-width="1.8" stroke-linecap="round"/>
            <line x1="28.5" y1="6" x2="35.5" y2="6" stroke="#fff" stroke-width="1.8" stroke-linecap="round"/>
          </svg>
          <div class="upload-text">Drag and drop your food photo here</div>
          <div class="upload-sub">or use the button above to browse files</div>
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
          <div class="section-head" style="margin-bottom:10px;">How it works</div>
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

# ── FOOTER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  <div class="footer-l">Built with PyTorch · ResNet-50 · Streamlit · Fine-tuned 35 epochs</div>
  <div class="footer-r">LinkedIn ↗</div>
</div>
""", unsafe_allow_html=True)