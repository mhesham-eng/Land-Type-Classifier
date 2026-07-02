import gradio as gr
import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import io
import traceback
import random

# ── 1. Class definitions ──────────────────────────────────────────────────────

CLASSES = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
    'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake'
]

CLASS_INFO = {
    'AnnualCrop':           {'code': 'AC', 'color': '#E8A838', 'desc': 'Seasonal crops harvested within one cycle — wheat, corn, sunflower fields.'},
    'Forest':               {'code': 'FR', 'color': '#3EB489', 'desc': 'Dense tree cover spanning coniferous and broadleaf canopy.'},
    'HerbaceousVegetation': {'code': 'HV', 'color': '#7FD68C', 'desc': 'Grassland and meadow cover without woody structure.'},
    'Highway':              {'code': 'HW', 'color': '#9AA7B5', 'desc': 'Motorway and major road infrastructure corridors.'},
    'Industrial':           {'code': 'IN', 'color': '#B07CC6', 'desc': 'Factory, warehouse, and industrial zone development.'},
    'Pasture':              {'code': 'PA', 'color': '#F2A341', 'desc': 'Livestock grazing land and permanent grassland.'},
    'PermanentCrop':        {'code': 'PC', 'color': '#5FBF7F', 'desc': 'Multi-year cultivation — vineyards, orchards, olive groves.'},
    'Residential':          {'code': 'RS', 'color': '#E8716B', 'desc': 'Urban housing, suburbs, and built-up settlement areas.'},
    'River':                {'code': 'RV', 'color': '#5EB3D6', 'desc': 'Flowing freshwater — rivers, streams, and canal networks.'},
    'SeaLake':              {'code': 'SL', 'color': '#2E7FA8', 'desc': 'Open standing water — seas, lakes, and major reservoirs.'},
}

# ── 2. Model loading ───────────────────────────────────────────────────────────

try:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(
        torch.load("resnet18_eurosat.pth", map_location=torch.device('cpu'), weights_only=True)
    )
    model.eval()
    MODEL_LOADED = True
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    MODEL_LOADED = False

img_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ── 3. Grad-CAM implementation ────────────────────────────────────────────────
# Hooks the last convolutional block of ResNet18 (layer4) to extract activations
# and gradients, then computes a class-discriminative localization map.

_activations = {}
_gradients = {}

def _save_activation(module, inp, out):
    _activations['value'] = out.detach()

def _save_gradient(module, grad_in, grad_out):
    _gradients['value'] = grad_out[0].detach()

if MODEL_LOADED:
    target_layer = model.layer4[-1]
    target_layer.register_forward_hook(_save_activation)
    target_layer.register_full_backward_hook(_save_gradient)


def generate_gradcam(tensor_input, class_idx):
    """Returns a normalized 0-1 heatmap (H,W) for the given class index."""
    model.zero_grad()
    output = model(tensor_input)
    score = output[0, class_idx]
    score.backward()

    grads = _gradients['value'][0]        # (C, H, W)
    acts  = _activations['value'][0]      # (C, H, W)

    weights = grads.mean(dim=(1, 2))      # Global Average Pool of gradients -> (C,)
    cam = torch.zeros(acts.shape[1:], dtype=torch.float32)
    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = F.relu(cam)
    cam -= cam.min()
    if cam.max() > 0:
        cam /= cam.max()
    return cam.numpy()


def overlay_heatmap(pil_img, cam, alpha=0.45):
    """Resize cam to image size, apply colormap, blend with original image."""
    img_resized = pil_img.resize((256, 256)).convert('RGB')
    img_np = np.array(img_resized).astype(np.float32) / 255.0

    cam_resized = Image.fromarray((cam * 255).astype('uint8')).resize((256, 256), Image.BICUBIC)
    cam_np = np.array(cam_resized).astype(np.float32) / 255.0

    colormap = matplotlib.colormaps['inferno']
    heat_rgba = colormap(cam_np)
    heat_rgb = heat_rgba[:, :, :3]

    blended = (1 - alpha) * img_np + alpha * heat_rgb
    blended = np.clip(blended, 0, 1)
    return Image.fromarray((blended * 255).astype('uint8'))


def build_chart(probs, sorted_idx):
    plt.rcParams['font.family'] = 'monospace'
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    fig.patch.set_facecolor('#0a0f17')
    ax.set_facecolor('#0a0f17')

    colors = [CLASS_INFO[CLASSES[i]]['color'] for i in sorted_idx]
    labels = [f"{CLASS_INFO[CLASSES[i]]['code']}  {CLASSES[i]}" for i in sorted_idx]
    values = [probs[i] * 100 for i in sorted_idx]

    bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1], edgecolor='none', height=0.58)
    for bar, val in zip(bars, values[::-1]):
        ax.text(min(val + 1.5, 94), bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}', va='center', ha='left', fontsize=9, color='#EAEFF3', fontweight='bold')

    ax.set_xlim(0, 108)
    ax.tick_params(colors='#5A7088', labelsize=8.5)
    for s in ax.spines.values():
        s.set_edgecolor('#152031')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('CONFIDENCE  %', color='#5A7088', fontsize=8, labelpad=10, fontweight='bold')
    ax.yaxis.set_tick_params(labelcolor='#C8D6E5')
    ax.grid(axis='x', color='#152031', linewidth=0.7, linestyle=(0, (3, 3)))
    ax.set_axisbelow(True)
    plt.tight_layout(pad=1.4)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=140, bbox_inches='tight', facecolor='#0a0f17')
    buf.seek(0)
    chart_img = Image.open(buf).copy()
    plt.close(fig)
    buf.close()
    return chart_img


# ── 4. Prediction function ────────────────────────────────────────────────────

def scan_id():
    return f"EUS-{random.randint(100000,999999)}"

PLACEHOLDER_HTML = """
<div style="display:flex; flex-direction:column; align-items:center; justify-content:center;
            min-height:240px; color:#3A4A5C; font-family:'JetBrains Mono', monospace; text-align:center;">
  <div style="font-size:0.78rem; letter-spacing:0.15em; text-transform:uppercase; color:#3A4A5C; margin-bottom:10px;">
    Awaiting Input
  </div>
  <div style="width:46px; height:46px; border:1.5px solid #1A2738; border-radius:50%;
              display:flex; align-items:center; justify-content:center; margin-bottom:14px;">
    <div style="width:8px; height:8px; background:#2C4356; border-radius:50%;"></div>
  </div>
  <div style="font-size:0.74rem; color:#2C4356; max-width:240px; line-height:1.6;">
    Upload satellite imagery to begin terrain classification
  </div>
</div>
"""

def predict(img):
    if img is None:
        return PLACEHOLDER_HTML, None, None

    if not MODEL_LOADED:
        return (
            "<div style='color:#E8716B; padding:24px; font-family:monospace; font-size:0.85rem;'>"
            "⚠ MODEL_LOAD_ERROR — resnet18_eurosat.pth not found</div>",
            None, None
        )

    try:
        pil_img = Image.fromarray(img.astype('uint8'), 'RGB')
        tensor  = img_transforms(pil_img).unsqueeze(0)
        tensor.requires_grad_(False)

        # Forward pass for prediction (no grad needed here)
        with torch.no_grad():
            logits = model(tensor)
            probs  = torch.nn.functional.softmax(logits[0], dim=0).numpy()

        sorted_idx = np.argsort(probs)[::-1]
        top_class  = CLASSES[sorted_idx[0]]
        top_idx    = int(sorted_idx[0])
        top_prob   = float(probs[sorted_idx[0]])
        info       = CLASS_INFO[top_class]

        # Grad-CAM needs its own forward+backward pass with grad enabled
        cam = generate_gradcam(tensor, top_idx)
        heatmap_img = overlay_heatmap(pil_img, cam)

        conf_color = '#3EB489' if top_prob >= 0.75 else '#F2A341' if top_prob >= 0.45 else '#E8716B'
        conf_label = 'HIGH CERTAINTY' if top_prob >= 0.75 else 'MODERATE CERTAINTY' if top_prob >= 0.45 else 'LOW CERTAINTY'

        entropy = float(-(probs * np.log(probs + 1e-12)).sum())
        max_entropy = np.log(len(CLASSES))
        clarity = max(0.0, 1 - entropy / max_entropy) * 100

        # Focus spread: how much of the image the model is "looking at"
        focus_ratio = float((cam > 0.5).sum()) / cam.size * 100
        if focus_ratio < 15:
            focus_note = "Tight, localized focus — the model is keying on a small, distinctive region."
        elif focus_ratio < 40:
            focus_note = "Moderate focus spread — attention concentrated on a clear sub-area of the image."
        else:
            focus_note = "Broad focus spread — the model is drawing on texture cues across most of the frame."

        rank_rows = "".join([
            f"""
            <div style="display:flex; align-items:center; gap:12px; padding:9px 0;
                        border-bottom:1px solid #131D2A;">
              <span style="font-family:'JetBrains Mono',monospace; font-size:0.68rem; color:#3A4A5C; width:18px;">
                {str(rank+1).zfill(2)}
              </span>
              <span style="width:7px; height:7px; border-radius:2px; background:{CLASS_INFO[CLASSES[i]]['color']}; flex-shrink:0;"></span>
              <span style="font-size:0.82rem; color:#C8D6E5; flex:1;">{CLASSES[i]}</span>
              <span style="font-family:'JetBrains Mono',monospace; font-size:0.76rem; color:#7A8FA6; font-weight:600;">
                {probs[i]*100:.1f}%
              </span>
            </div>"""
            for rank, i in enumerate(sorted_idx[:5])
        ])

        result_html = f"""
        <div style="font-family:'Inter',sans-serif; color:#EAEFF3;">

          <div style="display:flex; justify-content:space-between; align-items:flex-start;
                      margin-bottom:18px; padding-bottom:16px; border-bottom:1px solid #131D2A;">
            <div>
              <div style="font-family:'JetBrains Mono',monospace; font-size:0.68rem; color:#3A4A5C;
                          letter-spacing:0.1em; margin-bottom:6px;">SCAN_ID · {scan_id()}</div>
              <div style="display:flex; align-items:baseline; gap:10px;">
                <span style="font-size:1.9rem; font-weight:700; color:#fff; letter-spacing:-0.01em;">{top_class}</span>
                <span style="font-family:'JetBrains Mono',monospace; font-size:0.78rem; color:{info['color']};
                            border:1px solid {info['color']}55; border-radius:4px; padding:1px 7px;">{info['code']}</span>
              </div>
            </div>
            <div style="text-align:right;">
              <div style="font-family:'JetBrains Mono',monospace; font-size:1.5rem; font-weight:700; color:{conf_color};">
                {top_prob*100:.1f}<span style="font-size:0.8rem;">%</span>
              </div>
              <div style="font-family:'JetBrains Mono',monospace; font-size:0.62rem; color:{conf_color}; opacity:0.85;
                          letter-spacing:0.08em;">{conf_label}</div>
            </div>
          </div>

          <div style="font-size:0.86rem; color:#8FA3B8; line-height:1.65; margin-bottom:20px;">
            {info['desc']}
          </div>

          <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-bottom:16px;">
            <div style="background:#0d1520; border:1px solid #131D2A; border-radius:8px; padding:12px 14px;">
              <div style="font-family:'JetBrains Mono',monospace; font-size:0.62rem; color:#3A4A5C;
                          letter-spacing:0.08em; margin-bottom:6px;">CONFIDENCE</div>
              <div style="background:#131D2A; border-radius:999px; height:5px; overflow:hidden;">
                <div style="width:{top_prob*100:.1f}%; height:100%; background:{conf_color}; border-radius:999px;"></div>
              </div>
            </div>
            <div style="background:#0d1520; border:1px solid #131D2A; border-radius:8px; padding:12px 14px;">
              <div style="font-family:'JetBrains Mono',monospace; font-size:0.62rem; color:#3A4A5C;
                          letter-spacing:0.08em; margin-bottom:6px;">SIGNAL CLARITY</div>
              <div style="background:#131D2A; border-radius:999px; height:5px; overflow:hidden;">
                <div style="width:{clarity:.1f}%; height:100%; background:#7FC8E8; border-radius:999px;"></div>
              </div>
            </div>
          </div>

          <div style="background:#0d1520; border:1px solid #131D2A; border-left:2px solid #F2A341;
                      border-radius:8px; padding:11px 14px; margin-bottom:22px;">
            <div style="font-family:'JetBrains Mono',monospace; font-size:0.6rem; color:#F2A341;
                        letter-spacing:0.08em; margin-bottom:4px;">⌖ MODEL ATTENTION</div>
            <div style="font-size:0.78rem; color:#8FA3B8; line-height:1.55;">{focus_note}</div>
          </div>

          <div style="font-family:'JetBrains Mono',monospace; font-size:0.66rem; color:#3A4A5C;
                      letter-spacing:0.1em; margin-bottom:4px;">CLASSIFICATION RANKING</div>
          <div>{rank_rows}</div>

        </div>
        """

        chart_img = build_chart(probs, sorted_idx)
        return result_html, heatmap_img, chart_img

    except Exception as e:
        print(f"[ERROR] predict(): {traceback.format_exc()}")
        return (
            f"<div style='color:#E8716B; padding:20px; font-family:monospace; font-size:0.8rem;'>"
            f"⚠ INFERENCE_ERROR — {str(e)}</div>",
            None, None
        )

# ── 5. CSS ─────────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

* { box-sizing: border-box; }

:root {
    --bg-deep: #06090F;
    --bg-panel: #0a0f17;
    --bg-card: #0d1520;
    --border: #131D2A;
    --border-soft: #1A2738;
    --terrain: #3EB489;
    --signal: #F2A341;
    --ice: #7FC8E8;
    --text-primary: #EAEFF3;
    --text-secondary: #8FA3B8;
    --text-dim: #3A4A5C;
}

body, .gradio-container { background: var(--bg-deep) !important; font-family: 'Inter', sans-serif !important; }
.gradio-container { max-width: 1180px !important; margin: 0 auto !important; padding: 0 20px 40px !important; }

.gradio-container::before {
    content: ""; position: fixed; inset: 0; pointer-events: none;
    background-image:
        linear-gradient(rgba(62,180,137,0.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(62,180,137,0.025) 1px, transparent 1px);
    background-size: 48px 48px; z-index: 0;
}

.upload-zone { background: var(--bg-panel) !important; border: 1px solid var(--border-soft) !important; border-radius: 16px !important; }
.upload-zone:hover { border-color: var(--terrain) !important; }

#classify-btn {
    background: linear-gradient(135deg, #2E9670, var(--terrain)) !important;
    border: none !important; border-radius: 10px !important; color: #04130D !important;
    font-weight: 700 !important; font-size: 0.86rem !important; letter-spacing: 0.04em;
    font-family: 'Inter', sans-serif !important; text-transform: uppercase;
}
#classify-btn:hover { filter: brightness(1.1) !important; }

#clear-btn {
    background: transparent !important; border: 1px solid var(--border-soft) !important;
    border-radius: 10px !important; color: var(--text-secondary) !important; font-size: 0.82rem !important;
}
#clear-btn:hover { border-color: var(--text-dim) !important; color: var(--text-primary) !important; }

#chart-output img, #heatmap-output img {
    border-radius: 12px; border: 1px solid var(--border);
}

.gr-box, .gr-panel, .gr-form, .gr-block, .gr-padded { background: transparent !important; }
label.block { color: var(--text-dim) !important; font-size: 0.74rem !important;
              text-transform: uppercase; letter-spacing: 0.08em; font-family:'JetBrains Mono',monospace !important; }

#heatmap-tabs .tab-nav button {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
"""

# ── 6. HTML blocks ────────────────────────────────────────────────────────────

HEADER_HTML = """
<div style="position:relative; z-index:1; padding:36px 4px 28px; border-bottom:1px solid #131D2A; margin-bottom:28px;">
  <div style="display:flex; justify-content:space-between; align-items:flex-end; flex-wrap:wrap; gap:16px;">
    <div>
      <div style="font-family:'JetBrains Mono',monospace; font-size:0.7rem; color:#3EB489;
                  letter-spacing:0.2em; margin-bottom:10px; display:flex; align-items:center; gap:8px;">
        <span style="width:6px; height:6px; background:#3EB489; border-radius:50%; display:inline-block;
                     box-shadow:0 0 8px #3EB489;"></span>
        SYSTEM ONLINE
      </div>
      <h1 style="font-family:'Space Grotesk',sans-serif; font-size:2.1rem; font-weight:700;
                 color:#fff; margin:0; letter-spacing:-0.02em; line-height:1.1;">
        Terrain<span style="color:#3EB489;">Scope</span>
      </h1>
      <div style="font-size:0.86rem; color:#5A7088; margin-top:6px; max-width:460px; line-height:1.5;">
        Land cover classification from Sentinel-2 satellite imagery using a fine-tuned ResNet-18 network
        — with Grad-CAM visual explainability.
      </div>
    </div>
    <div style="display:flex; gap:8px; flex-wrap:wrap;">
      <div style="background:#0d1520; border:1px solid #131D2A; border-radius:8px; padding:8px 14px; text-align:center;">
        <div style="font-family:'JetBrains Mono',monospace; font-size:0.95rem; font-weight:700; color:#7FC8E8;">ResNet-18</div>
        <div style="font-size:0.62rem; color:#3A4A5C; letter-spacing:0.06em;">ARCHITECTURE</div>
      </div>
      <div style="background:#0d1520; border:1px solid #131D2A; border-radius:8px; padding:8px 14px; text-align:center;">
        <div style="font-family:'JetBrains Mono',monospace; font-size:0.95rem; font-weight:700; color:#F2A341;">Grad-CAM</div>
        <div style="font-size:0.62rem; color:#3A4A5C; letter-spacing:0.06em;">EXPLAINABILITY</div>
      </div>
      <div style="background:#0d1520; border:1px solid #131D2A; border-radius:8px; padding:8px 14px; text-align:center;">
        <div style="font-family:'JetBrains Mono',monospace; font-size:0.95rem; font-weight:700; color:#7FC8E8;">EuroSAT</div>
        <div style="font-size:0.62rem; color:#3A4A5C; letter-spacing:0.06em;">DATASET</div>
      </div>
    </div>
  </div>
</div>
"""

TIPS_HTML = """
<div style="margin-top:14px; padding:16px 18px; background:#0d1520; border:1px solid #131D2A;
            border-radius:12px; font-family:'JetBrains Mono',monospace;">
  <div style="color:#3EB489; font-weight:700; font-size:0.68rem; letter-spacing:0.1em; margin-bottom:10px;">
    ▸ INPUT GUIDELINES
  </div>
  <div style="font-size:0.74rem; color:#5A7088; line-height:2; letter-spacing:0.01em;">
    FORMAT &nbsp;&nbsp;PNG · JPG · WEBP<br>
    RESIZE &nbsp;&nbsp;auto → 64×64px<br>
    SOURCE &nbsp;&nbsp;top-down aerial / satellite<br>
    BANDS &nbsp;&nbsp;&nbsp;RGB only
  </div>
</div>
"""

CLASS_LEGEND_HTML = """
<div style="margin-top:14px; padding:16px 18px; background:#0d1520; border:1px solid #131D2A; border-radius:12px;">
  <div style="font-family:'JetBrains Mono',monospace; color:#3EB489; font-weight:700; font-size:0.68rem;
              letter-spacing:0.1em; margin-bottom:12px;">▸ CLASS LEGEND</div>
  <div style="display:grid; grid-template-columns:1fr 1fr; gap:7px;">
""" + "".join([
    f"""<div style="display:flex; align-items:center; gap:7px;">
          <span style="width:7px;height:7px;border-radius:2px;background:{v['color']};flex-shrink:0;"></span>
          <span style="font-size:0.72rem; color:#5A7088;">{k}</span>
        </div>"""
    for k, v in CLASS_INFO.items()
]) + """
  </div>
</div>
"""

GRADCAM_EXPLAINER_HTML = """
<div style="margin-top:10px; padding:13px 16px; background:#0d1520; border:1px solid #131D2A;
            border-left:2px solid #7FC8E8; border-radius:8px; font-size:0.76rem; color:#5A7088; line-height:1.6;">
  <span style="color:#7FC8E8; font-family:'JetBrains Mono',monospace; font-size:0.66rem; letter-spacing:0.06em;">
    ⌖ HOW TO READ THIS &nbsp;</span>
  Bright (yellow/white) regions are where the network's last convolutional layer
  contributed most to its decision. Dark regions were effectively ignored.
</div>
"""

FOOTER_HTML = """
<div style="margin-top:32px; padding-top:20px; border-top:1px solid #131D2A;
            display:flex; justify-content:space-between; flex-wrap:wrap; gap:12px;
            font-family:'JetBrains Mono',monospace; font-size:0.68rem; color:#2C4356;">
  <div>RESNET-18 · FINE-TUNED ON EUROSAT (SENTINEL-2 RGB) · 64×64 INPUT · GRAD-CAM ON LAYER4</div>
  <div>BUILT WITH PYTORCH + GRADIO</div>
</div>
"""

# ── 7. Gradio Blocks interface ────────────────────────────────────────────────

with gr.Blocks(title="TerrainScope — Satellite Land Classifier") as demo:

    gr.HTML(HEADER_HTML)

    with gr.Row(equal_height=False):
        with gr.Column(scale=5):
            image_input = gr.Image(
                label="UPLOAD IMAGERY",
                type="numpy",
                elem_classes=["upload-zone"],
                height=300,
            )
            with gr.Row():
                classify_btn = gr.Button("Run Classification", elem_id="classify-btn", variant="primary", size="lg")
                clear_btn = gr.ClearButton([image_input], value="Clear", elem_id="clear-btn", size="lg")

            gr.HTML(TIPS_HTML)
            gr.HTML(CLASS_LEGEND_HTML)

        with gr.Column(scale=7):
            with gr.Group():
                result_html = gr.HTML(value=PLACEHOLDER_HTML)

            with gr.Tabs(elem_id="heatmap-tabs"):
                with gr.Tab("Grad-CAM Heatmap"):
                    heatmap_output = gr.Image(
                        label="MODEL ATTENTION MAP",
                        elem_id="heatmap-output",
                        height=300,
                    )
                    gr.HTML(GRADCAM_EXPLAINER_HTML)
                with gr.Tab("Probability Chart"):
                    chart_output = gr.Image(
                        label="PROBABILITY DISTRIBUTION",
                        elem_id="chart-output",
                        height=300,
                    )

    gr.HTML(FOOTER_HTML)

    classify_btn.click(fn=predict, inputs=[image_input], outputs=[result_html, heatmap_output, chart_output])
    image_input.upload(fn=predict, inputs=[image_input], outputs=[result_html, heatmap_output, chart_output])
    clear_btn.add([result_html, heatmap_output, chart_output])

demo.launch(css=CUSTOM_CSS)
