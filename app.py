import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import time

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NutriLens · AI Nutrition Estimator",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ---------- global ---------- */
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: #1a1a2e;
  }

  /* Remove default Streamlit top padding */
  .block-container { padding-top: 2rem; max-width: 960px; }

  /* ---------- hero ---------- */
  .hero {
    background: linear-gradient(135deg, #0f3460 0%, #16213e 60%, #1a1a2e 100%);
    border-radius: 20px;
    padding: 3rem 2.5rem 2.5rem;
    margin-bottom: 2rem;
    color: white;
    position: relative;
    overflow: hidden;
  }
  .hero::before {
    content: "";
    position: absolute;
    top: -60px; right: -60px;
    width: 280px; height: 280px;
    border-radius: 50%;
    background: rgba(0,188,212,0.12);
    pointer-events: none;
  }
  .hero-eyebrow {
    font-size: 0.75rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #00bcd4;
    font-weight: 600;
    margin-bottom: 0.6rem;
  }
  .hero-title {
    font-size: 2.6rem;
    font-weight: 600;
    line-height: 1.15;
    margin: 0 0 0.8rem;
  }
  .hero-sub {
    font-size: 1.05rem;
    color: rgba(255,255,255,0.72);
    max-width: 540px;
    line-height: 1.6;
    margin: 0 0 1.6rem;
  }
  .badge-row { display: flex; gap: 0.6rem; flex-wrap: wrap; }
  .badge {
    background: rgba(0,188,212,0.18);
    border: 1px solid rgba(0,188,212,0.4);
    color: #00e5ff;
    border-radius: 40px;
    padding: 0.25rem 0.85rem;
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.04em;
  }

  /* ---------- upload zone ---------- */
  .upload-card {
    border: 2px dashed #c9d6df;
    border-radius: 16px;
    padding: 2.5rem 2rem;
    text-align: center;
    background: #f8fafc;
    transition: border-color 0.2s;
  }
  .upload-card:hover { border-color: #0f3460; }

  /* ---------- metric cards ---------- */
  .metric-grid { display: flex; gap: 1rem; flex-wrap: wrap; margin-top: 1.2rem; }
  .metric-card {
    flex: 1 1 120px;
    background: white;
    border-radius: 14px;
    padding: 1.2rem 1rem 1rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    border-top: 3px solid var(--accent);
    text-align: center;
  }
  .metric-label {
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #6b7a8d;
    font-weight: 600;
    margin-bottom: 0.4rem;
  }
  .metric-value {
    font-family: 'DM Mono', monospace;
    font-size: 1.75rem;
    font-weight: 500;
    color: #1a1a2e;
    line-height: 1;
  }
  .metric-unit { font-size: 0.8rem; color: #6b7a8d; margin-top: 0.2rem; }

  /* accent colours per nutrient */
  .cal  { --accent: #ef4444; }
  .fat  { --accent: #f59e0b; }
  .carb { --accent: #3b82f6; }
  .prot { --accent: #10b981; }

  /* ---------- result section ---------- */
  .result-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #1a1a2e;
    margin: 1.8rem 0 0.3rem;
  }
  .result-sub {
    font-size: 0.88rem;
    color: #6b7a8d;
    margin-bottom: 0.8rem;
  }

  /* ---------- info panel ---------- */
  .info-pill {
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: #eef2ff; color: #4f46e5;
    border-radius: 8px; padding: 0.4rem 0.9rem;
    font-size: 0.82rem; font-weight: 500;
  }

  /* ---------- footer ---------- */
  .footer {
    margin-top: 3rem;
    padding-top: 1.5rem;
    border-top: 1px solid #e2e8f0;
    display: flex; justify-content: space-between; align-items: center;
    flex-wrap: wrap; gap: 0.5rem;
    font-size: 0.8rem; color: #94a3b8;
  }
  .footer a { color: #0f3460; text-decoration: none; font-weight: 500; }

  /* hide default streamlit file uploader label clutter */
  [data-testid="stFileUploaderDropzoneInstructions"] { display: none; }
  [data-testid="stFileUploader"] section { border: none; padding: 0; background: transparent; }
</style>
""", unsafe_allow_html=True)

# ── Model ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 4)
    model.load_state_dict(
        torch.load("nutrition_model_epoch35.pth", map_location="cpu")
    )
    model.eval()
    return model

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-eyebrow">AI · Computer Vision · ResNet-50</div>
  <div class="hero-title">NutriLens</div>
  <div class="hero-sub">
    Upload any food photo and get an instant estimate of calories, fat,
    carbohydrates, and protein — powered by a fine-tuned deep learning model.
  </div>
  <div class="badge-row">
    <span class="badge">ResNet-50 backbone</span>
    <span class="badge">4-output regression</span>
    <span class="badge">Trained 35 epochs</span>
    <span class="badge">PyTorch · Streamlit</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Layout: two columns ───────────────────────────────────────────────────────
col_upload, col_result = st.columns([1, 1], gap="large")

with col_upload:
    st.markdown("#### Upload a food image")
    st.markdown(
        '<p style="font-size:0.88rem;color:#6b7a8d;margin-top:-0.6rem;margin-bottom:1rem;">'
        "JPG or PNG · best results with a clear, top-down food photo</p>",
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        label="food image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_container_width=True, caption="")

with col_result:
    if uploaded_file:
        with st.spinner("Analysing nutritional content…"):
            time.sleep(0.4)          # subtle UX pause so the spinner is visible
            try:
                model = load_model()
            except Exception:
                st.error("⚠️ Model file not found. Make sure `nutrition_model_epoch35.pth` is in the same directory.")
                st.stop()

            img_tensor = transform(image).unsqueeze(0)
            with torch.no_grad():
                pred = model(img_tensor).cpu().numpy()[0]

        calories, fat, carbs, protein = pred

        st.markdown("#### Estimated nutrition")
        st.markdown(
            '<p class="result-sub">Per serving · values are model estimates, not lab measurements</p>',
            unsafe_allow_html=True,
        )

        st.markdown(f"""
        <div class="metric-grid">
          <div class="metric-card cal">
            <div class="metric-label">Calories</div>
            <div class="metric-value">{calories:.0f}</div>
            <div class="metric-unit">kcal</div>
          </div>
          <div class="metric-card fat">
            <div class="metric-label">Fat</div>
            <div class="metric-value">{fat:.1f}</div>
            <div class="metric-unit">grams</div>
          </div>
          <div class="metric-card carb">
            <div class="metric-label">Carbs</div>
            <div class="metric-value">{carbs:.1f}</div>
            <div class="metric-unit">grams</div>
          </div>
          <div class="metric-card prot">
            <div class="metric-label">Protein</div>
            <div class="metric-value">{protein:.1f}</div>
            <div class="metric-unit">grams</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Macro bar
        total_macro = fat * 9 + carbs * 4 + protein * 4
        if total_macro > 0:
            pct_fat  = fat  * 9 / total_macro * 100
            pct_carb = carbs * 4 / total_macro * 100
            pct_prot = protein * 4 / total_macro * 100

            st.markdown('<div class="result-header">Macro breakdown</div>', unsafe_allow_html=True)
            st.markdown(
                f'<p class="result-sub">Fat {pct_fat:.0f}% · Carbs {pct_carb:.0f}% · Protein {pct_prot:.0f}%</p>',
                unsafe_allow_html=True,
            )
            bar_html = f"""
            <div style="display:flex;height:12px;border-radius:6px;overflow:hidden;background:#f1f5f9;margin-bottom:1.6rem;">
              <div style="width:{pct_fat:.1f}%;background:#f59e0b;"></div>
              <div style="width:{pct_carb:.1f}%;background:#3b82f6;"></div>
              <div style="width:{pct_prot:.1f}%;background:#10b981;"></div>
            </div>
            """
            st.markdown(bar_html, unsafe_allow_html=True)

        st.markdown(
            '<div class="info-pill">ℹ️ Estimates may vary with portion size and food preparation method</div>',
            unsafe_allow_html=True,
        )

    else:
        # Empty state
        st.markdown("""
        <div style="height:100%;display:flex;flex-direction:column;justify-content:center;
                    align-items:center;padding:3rem 1rem;text-align:center;color:#94a3b8;">
          <div style="font-size:3rem;margin-bottom:1rem;">🥗</div>
          <div style="font-size:1rem;font-weight:600;color:#64748b;margin-bottom:0.4rem;">
            No image uploaded yet
          </div>
          <div style="font-size:0.85rem;line-height:1.6;">
            Upload a food photo on the left and the model<br>will estimate its nutritional content instantly.
          </div>
        </div>
        """, unsafe_allow_html=True)

# ── How it works (collapsible) ────────────────────────────────────────────────
with st.expander("How it works"):
    st.markdown("""
    **Model architecture:** ResNet-50 pre-trained on ImageNet, with the final fully-connected
    layer replaced by a 4-output linear layer for multi-target regression (calories, fat, carbs, protein).

    **Training:** Fine-tuned for 35 epochs on a labelled food image dataset. Input images are resized
    to 224 × 224 and normalised using ImageNet statistics before inference.

    **Limitations:** The model estimates nutrition from visual features alone — it cannot account for
    hidden ingredients, cooking oils, or exact portion weights. Treat outputs as approximate starting points.
    """)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  <span>Built with PyTorch · Streamlit · ResNet-50</span>
  <span>AI/ML Course Project · <a href="https://linkedin.com" target="_blank">LinkedIn</a></span>
</div>
""", unsafe_allow_html=True)