# app.py  — EVA-Tiny HAM10000 demo (auto-matches checkpoint head, metadata-safe)
# Requirements: torch, timm, torchvision, pillow, numpy, streamlit
# Optional: opencv-python (for nicer overlays); if missing we fall back to PIL.

from __future__ import annotations
import os, io, json, math, warnings
from typing import Tuple, Dict, Any, List, Optional

import numpy as np
from PIL import Image, ImageOps

import torch
device = torch.device("cpu")
weights = torch.load("best.pt", map_location=device)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

import streamlit as st

try:
    import timm
except Exception as e:
    st.error("timm is required. Install with:  pip install timm")
    raise

# ---------------------------
# Configuration
# ---------------------------
CLASSES = ["AKIEC", "BCC", "BKL", "DF", "MEL", "NV", "VASC"]  # training order
BACKBONE = "eva02_tiny_patch14_224"                           # your model
IMG_SIZE = 224                                                # 224 for EVA-tiny
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Make sure we stay in float32
torch.set_float32_matmul_precision("high")
torch.set_default_dtype(torch.float32)

# ---------------------------
# Small utilities
# ---------------------------
def try_extract_state_dict(obj: Any) -> Dict[str, torch.Tensor]:
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        if "model" in obj and isinstance(obj["model"], dict):
            # common export style
            return obj["model"]
        # sometimes the dict already *is* the state_dict
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj
    raise RuntimeError("Could not find state_dict in checkpoint file.")

def strip_prefix_if_needed(sd: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    out = {}
    n = len(prefix)
    for k, v in sd.items():
        if k.startswith(prefix):
            out[k[n:]] = v
        else:
            out[k] = v
    return out

def softmax_np(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / (e.sum() + 1e-12)

# ---------------------------
# Model with optional metadata
# ---------------------------
class MetaMLP(nn.Module):
    """age(1) + sex(2) + site(10) -> 64-d embedding."""
    def __init__(self, in_dim: int = 13, hidden: int = 64, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)

class EVAMeta(nn.Module):
    def __init__(self, backbone_name: str, num_classes: int, meta_dims: int = 0):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        feat_dim = self.backbone.num_features  # 192 for eva02_tiny_patch14_224
        self.meta_mlps: Optional[MetaMLP] = None
        self.meta_out = 0
        if meta_dims > 0:
            self.meta_mlps = MetaMLP(in_dim=meta_dims, hidden=64, out_dim=64)
            self.meta_out = 64
        self.head = nn.Linear(feat_dim + self.meta_out, num_classes)

    def forward_features(self, x):
        return self.backbone.forward_features(x)

    def forward(self, x, meta: Optional[torch.Tensor] = None):
        feats = self.forward_features(x)
        if feats.ndim == 3:  # tokens (B,N,C) → pool
            feats = feats.mean(1)
        if feats.ndim == 4:  # (B,C,H,W)
            feats = F.adaptive_avg_pool2d(feats, 1).flatten(1)
        if self.meta_mlps is not None and meta is not None:
            meta_emb = self.meta_mlps(meta)
            feats = torch.cat([feats, meta_emb], dim=1)
        return self.head(feats)

# ---------------------------
# Preprocess
# ---------------------------
def build_transform(img_size: int = IMG_SIZE) -> T.Compose:
    return T.Compose([
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

def preprocess_image(pil: Image.Image) -> torch.Tensor:
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    tfm = build_transform(IMG_SIZE)
    t = tfm(pil)  # [3,H,W], float32
    return t.unsqueeze(0)  # [1,3,H,W]

def build_meta_vector(age: int, sex: str, site: str) -> np.ndarray:
    # sex → {male, female}
    sex_map = {"male": 0, "female": 1}
    sex_vec = np.zeros(2, dtype=np.float32)
    sex_vec[sex_map.get(sex.lower(), 0)] = 1.0

    # site → simple 10 slot one-hot (same as training)
    sites = ["trunk","lower extremity","upper extremity","head/neck","back",
             "abdomen","face","chest","hand","foot"]
    site_vec = np.zeros(10, dtype=np.float32)
    # light mapping (prefix match)
    picked = 0
    s = site.lower().strip()
    for i, name in enumerate(sites):
        if name in s:
            site_vec[i] = 1.0
            picked = 1
            break
    if not picked:
        site_vec[0] = 1.0  # fallback to trunk

    return np.concatenate([np.array([float(age)], np.float32), sex_vec, site_vec], axis=0)  # (13,)

# ---------------------------
# Loader that AUTO-MATCHES the checkpoint head
# ---------------------------
@torch.no_grad()
def load_model(ckpt_path: str, ui_wants_meta: bool, device: torch.device) -> Tuple[nn.Module, Dict[str, Any]]:
    msg: Dict[str, Any] = {"missing_keys": [], "unexpected_keys": [], "note": ""}

    # build temp backbone to know feat_dim
    tmp = timm.create_model(BACKBONE, pretrained=False, num_classes=0)
    feat_dim = int(tmp.num_features)

    # if no file → build a fresh no-meta model
    if not os.path.isfile(ckpt_path):
        msg["note"] = f"Checkpoint not found: {ckpt_path}"
        model = EVAMeta(BACKBONE, num_classes=len(CLASSES), meta_dims=0).to(device).eval()
        return model, msg

    raw = torch.load(ckpt_path, map_location="cpu")
    sd = try_extract_state_dict(raw)
    if any(k.startswith("model.") for k in sd): sd = strip_prefix_if_needed(sd, "model.")
    if any(k.startswith("module.") for k in sd): sd = strip_prefix_if_needed(sd, "module.")

    head_w = sd.get("head.weight", None)
    if isinstance(head_w, torch.Tensor) and head_w.ndim == 2:
        ckpt_head_in = int(head_w.shape[1])
        msg["note"] = f"Detected checkpoint head_in={ckpt_head_in} (backbone feat_dim={feat_dim})."
    else:
        ckpt_head_in = feat_dim
        msg["note"] = "head.weight not found; assuming no metadata (head_in=feat_dim)."

    # decide meta from head_in
    if ckpt_head_in == feat_dim:
        use_meta = False
        meta_dims = 0
        head_in = feat_dim
        msg["note"] += " → using NO metadata."
    elif ckpt_head_in == feat_dim + 64:
        use_meta = True
        meta_dims = 13  # age(1)+sex(2)+site(10)
        head_in = feat_dim + 64
        msg["note"] += " → using metadata (64-d meta MLP)."
    else:
        # unusual: match head size exactly; disable meta so shapes line up
        use_meta = False
        meta_dims = 0
        head_in = ckpt_head_in
        msg["note"] += f" → unusual head; building head_in={head_in}, metadata disabled."

    model = EVAMeta(BACKBONE, num_classes=len(CLASSES), meta_dims=meta_dims).to(device).eval()
    if model.head.in_features != head_in:
        model.head = nn.Linear(head_in, len(CLASSES)).to(device)

    load_res = model.load_state_dict(sd, strict=False)
    msg["missing_keys"] = list(load_res.missing_keys)
    msg["unexpected_keys"] = list(load_res.unexpected_keys)

    if ui_wants_meta and not use_meta:
        msg["note"] += "  (UI requested metadata, but checkpoint didn’t use it — continuing without metadata.)"

    return model, msg

# ---------------------------
# Prediction
# ---------------------------
@torch.no_grad()
def predict(model: nn.Module, pil_img: Image.Image,
            age: Optional[int], sex: Optional[str], site: Optional[str]) -> Tuple[np.ndarray, int]:
    x = preprocess_image(pil_img).to(DEVICE)                 # [1,3,H,W] float32
    meta_t = None
    if age is not None and sex and site and hasattr(model, "meta_mlps") and model.meta_mlps is not None:
        meta = build_meta_vector(int(age), sex, site)        # (13,)
        meta_t = torch.from_numpy(meta).unsqueeze(0).to(DEVICE)  # [1,13]
    logits = model(x, meta_t)                                # [1,7]
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]    # (7,)
    idx = int(probs.argmax())
    return probs, idx

# ---------------------------
# Simple saliency fallback (if Grad-CAM isn’t available)
# ---------------------------
def input_gradient_saliency(model: nn.Module, pil_img: Image.Image, target_idx: int) -> Optional[Image.Image]:
    try:
        x = preprocess_image(pil_img).to(DEVICE).requires_grad_(True)
        logits = model(x, None)
        score = logits[0, target_idx]
        model.zero_grad(set_to_none=True)
        score.backward()
        g = x.grad.detach().abs().max(dim=1)[0]  # [1,H,W] → [H,W]
        g = (g - g.min()) / (g.max() - g.min() + 1e-8)
        g = (g * 255).byte().cpu().numpy()
        g = np.repeat(g[:, :, None], 3, axis=2)
        img = Image.fromarray(g)
        img = img.resize(pil_img.size, resample=Image.BILINEAR)
        # overlay (purple-ish)
        overlay = Image.new("RGBA", pil_img.size, (0,0,0,0))
        heat = Image.fromarray(g).convert("RGBA")
        heat_np = np.array(heat, dtype=np.uint8)
        heat_np[..., 3] = (heat_np[...,0] * 0.6).astype(np.uint8)  # alpha from intensity
        heat = Image.fromarray(heat_np, mode="RGBA")
        base = pil_img.convert("RGBA")
        out = Image.alpha_composite(base, heat)
        return out.convert("RGB")
    except Exception:
        return None

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Dermoscopic Image Classifier — EVA-Tiny", layout="wide")

st.title("Dermoscopic Image Classifier — EVA-Tiny")

with st.sidebar:
    st.header("Model & Settings")
    ckpt_choice = st.selectbox("Checkpoint file", options=[f for f in os.listdir(".") if f.endswith(".pt")],
                               index=0 if any(p.endswith(".pt") for p in os.listdir(".")) else None)
    ui_use_meta = st.checkbox("Use metadata (age/sex/site)", value=True,
                              help="If the checkpoint wasn’t trained with metadata, the app will auto-disable it.")

    st.markdown("---")
    st.subheader("Metadata (optional)")
    age = st.number_input("Age", min_value=0, max_value=110, value=45, step=1)
    sex = st.selectbox("Sex", options=["Male","Female"])
    site = st.text_input("Anatomical site (e.g., 'trunk', 'acral', 'face')", value="trunk")

st.info("**This tool is not a diagnosis.** High-risk predictions (especially **Melanoma**) should prompt consultation with a qualified dermatologist.", icon="⚠️")

uploaded = st.file_uploader("Upload a dermoscopic image", type=["jpg","jpeg","png"])
if uploaded:
    pil = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
    st.image(pil, caption="Uploaded image", use_column_width=True)

    # Load/prepare model
    if ckpt_choice:
        model, load_msg = load_model(ckpt_choice, ui_use_meta, DEVICE)
    else:
        model, load_msg = load_model("", ui_use_meta, DEVICE)

    with st.expander("Checkpoint load report", expanded=False):
        st.write(load_msg.get("note",""))
        if load_msg["missing_keys"]:
            st.write("Missing keys:", load_msg["missing_keys"][:10], "…")
        if load_msg["unexpected_keys"]:
            st.write("Unexpected keys:", load_msg["unexpected_keys"][:10], "…")

    # Decide metadata actually used this run
    effective_meta = (model.meta_mlps is not None) and ui_use_meta
    age_val = int(age) if effective_meta else None
    sex_val = sex if effective_meta else None
    site_val = site if effective_meta else None

    # Predict
    probs, idx = predict(model, pil, age_val, sex_val, site_val)

    # Display predictions
    st.subheader("Predictions")
    grid = st.columns(3)
    for i, name in enumerate(CLASSES):
        col = grid[i % 3]
        with col:
            st.metric(label=name, value=f"{probs[i]*100:0.1f}%")

    st.caption(f"Top class: **{CLASSES[idx]}**. If this lesion is changing, symptomatic, or concerning, seek medical advice.")

    st.markdown("---")
    st.subheader("Explanation")
    explain = st.toggle("Show explanation (saliency fallback if Grad-CAM is unavailable)", value=False)
    if explain:
        overlay = input_gradient_saliency(model, pil, idx)
        if overlay is None:
            st.warning("Explanation could not be generated for this model/layer combo. You can still use the predictions above.")
        else:
            st.image(overlay, caption="Saliency overlay (stronger color = regions influencing the decision)", use_column_width=True)

st.markdown("---")
st.caption("EVA-Tiny via timm; demo app for educational use. This interface does not provide medical advice.")


