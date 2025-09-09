# Dermoscopy EVA-Tiny demo (CPU-friendly, metadata aware)
# If torch isn't installed, show a helpful message instead of crashing.

from __future__ import annotations
import os, io
from typing import Dict, Any, Optional, Tuple, List

import streamlit as st
st.set_page_config(page_title="Dermoscopic Image Classifier — EVA-Tiny", layout="wide")

# --- Safe imports (give guidance if deps missing) ---
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision.transforms as T
except ModuleNotFoundError as e:
    st.error(
        "PyTorch is not installed in the deployment environment.\n\n"
        "➡️ Make sure your repository has a **requirements.txt** at the root with:\n"
        "```\n"
        "--extra-index-url https://download.pytorch.org/whl/cpu\n"
        "torch==2.2.2\n"
        "torchvision==0.17.2\n"
        "torchaudio==2.2.2\n"
        "streamlit==1.37.1\n"
        "timm==0.9.16\n"
        "pillow==10.3.0\n"
        "numpy==1.26.4\n"
        "opencv-python-headless==4.10.0.84\n"
        "scikit-learn==1.4.2\n"
        "```\n"
        "Then click **Manage app → Restart** in Streamlit Cloud."
    )
    st.stop()

try:
    import timm
except ModuleNotFoundError:
    st.error("Missing `timm`. Add `timm==0.9.16` to requirements.txt and restart.")
    st.stop()

from PIL import Image
import numpy as np

# --- Config ---
CLASSES: List[str] = ["AKIEC","BCC","BKL","DF","MEL","NV","VASC"]
BACKBONE = "eva02_tiny_patch14_224"
IMG_SIZE = 224
DEVICE = torch.device("cpu")  # force CPU on Streamlit Cloud

# Keep math stable on CPU
torch.set_default_dtype(torch.float32)

# --- Model with optional metadata ---
class MetaMLP(nn.Module):
    def __init__(self, in_dim: int = 13, hidden: int = 64, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class EVAMeta(nn.Module):
    def __init__(self, backbone_name: str, num_classes: int, meta_dims: int = 0):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        feat_dim = self.backbone.num_features
        self.meta_mlps: Optional[MetaMLP] = None
        self.meta_out = 0
        if meta_dims > 0:
            self.meta_mlps = MetaMLP(in_dim=meta_dims, hidden=64, out_dim=64)
            self.meta_out = 64
        self.head = nn.Linear(feat_dim + self.meta_out, num_classes)

    def forward_features(self, x): return self.backbone.forward_features(x)

    def forward(self, x, meta: Optional[torch.Tensor] = None):
        feats = self.forward_features(x)
        if feats.ndim == 3:  # transformer tokens
            feats = feats.mean(1)
        if feats.ndim == 4:  # CNN map
            feats = F.adaptive_avg_pool2d(feats, 1).flatten(1)
        if self.meta_mlps is not None and meta is not None:
            feats = torch.cat([feats, self.meta_mlps(meta)], dim=1)
        return self.head(feats)

# --- Utilities ---
def try_extract_state_dict(obj: Any) -> Dict[str, torch.Tensor]:
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict): return obj["state_dict"]
        if "model" in obj and isinstance(obj["model"], dict): return obj["model"]
        if all(isinstance(v, torch.Tensor) for v in obj.values()): return obj
    raise RuntimeError("Could not find state_dict in checkpoint.")

def strip(sd: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    n = len(prefix); return { (k[n:] if k.startswith(prefix) else k): v for k,v in sd.items() }

def transform(img_size: int = IMG_SIZE) -> T.Compose:
    return T.Compose([
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
    ])

def preprocess(pil: Image.Image) -> torch.Tensor:
    if pil.mode != "RGB": pil = pil.convert("RGB")
    return transform(IMG_SIZE)(pil).unsqueeze(0)

def build_meta(age: int, sex: str, site: str) -> np.ndarray:
    sex_map = {"male":0, "female":1}
    sex_vec = np.zeros(2, np.float32); sex_vec[sex_map.get(sex.lower(),0)] = 1.0
    sites = ["trunk","lower extremity","upper extremity","head/neck","back","abdomen","face","chest","hand","foot"]
    site_vec = np.zeros(10, np.float32)
    s = site.lower().strip()
    for i,name in enumerate(sites):
        if name in s: site_vec[i] = 1.0; break
    if site_vec.sum() == 0: site_vec[0] = 1.0
    return np.concatenate([np.array([float(age)],np.float32), sex_vec, site_vec], axis=0)

@torch.no_grad()
def load_model(ckpt_path: str, ui_wants_meta: bool) -> Tuple[nn.Module, Dict[str,Any]]:
    report: Dict[str,Any] = {"missing_keys":[], "unexpected_keys":[], "note":""}
    tmp = timm.create_model(BACKBONE, pretrained=False, num_classes=0)
    feat_dim = int(tmp.num_features)

    if not os.path.isfile(ckpt_path):
        report["note"] = f"Checkpoint not found: {ckpt_path}. Using random weights."
        return EVAMeta(BACKBONE, len(CLASSES), meta_dims=0).to(DEVICE).eval(), report

    raw = torch.load(ckpt_path, map_location="cpu")
    sd = try_extract_state_dict(raw)
    if any(k.startswith("model.") for k in sd): sd = strip(sd, "model.")
    if any(k.startswith("module.") for k in sd): sd = strip(sd, "module.")
    head_w = sd.get("head.weight", None)
    if isinstance(head_w, torch.Tensor) and head_w.ndim == 2:
        head_in = int(head_w.shape[1])
    else:
        head_in = feat_dim

    if head_in == feat_dim:
        meta_dims, note = 0, "no metadata"
    elif head_in == feat_dim + 64:
        meta_dims, note = 13, "with metadata (64-d meta MLP)"
    else:
        meta_dims, note = 0, f"unusual head_in={head_in}; disabling metadata"

    model = EVAMeta(BACKBONE, len(CLASSES), meta_dims=meta_dims).to(DEVICE).eval()
    if model.head.in_features != head_in:
        model.head = nn.Linear(head_in, len(CLASSES)).to(DEVICE)

    res = model.load_state_dict(sd, strict=False)
    report["missing_keys"] = list(res.missing_keys)
    report["unexpected_keys"] = list(res.unexpected_keys)
    if ui_wants_meta and meta_dims == 0:
        note += " (UI asked for metadata, but checkpoint was trained without it)."
    report["note"] = note
    return model, report

@torch.no_grad()
def predict(model: nn.Module, pil: Image.Image,
            age: Optional[int], sex: Optional[str], site: Optional[str]) -> Tuple[np.ndarray, int]:
    x = preprocess(pil).to(DEVICE)
    meta_t = None
    if age is not None and sex and site and getattr(model, "meta_mlps", None) is not None:
        meta_t = torch.from_numpy(build_meta(int(age), sex, site)).unsqueeze(0).to(DEVICE)
    logits = model(x, meta_t)
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    idx = int(probs.argmax())
    return probs, idx

# --- UI ---
st.title("Dermoscopic Image Classifier — EVA-Tiny")
with st.sidebar:
    st.header("Model & Settings")
    ckpt = st.selectbox("Checkpoint (.pt) in repo root", options=[f for f in os.listdir(".") if f.endswith(".pt")] or ["(none)"])
    use_meta = st.checkbox("Use metadata (age/sex/site)", value=True)
    st.markdown("---")
    st.subheader("Metadata")
    age = st.number_input("Age", 0, 110, 45)
    sex = st.selectbox("Sex", ["Male","Female"])
    site = st.text_input("Site (e.g., trunk, face, hand)", "trunk")

st.info("Educational demo — not medical advice.", icon="⚠️")

up = st.file_uploader("Upload a dermoscopic image", type=["jpg","jpeg","png"])
if up:
    img = Image.open(io.BytesIO(up.read())).convert("RGB")
    st.image(img, caption="Uploaded image", use_column_width=True)

    model, rep = load_model(ckpt if ckpt.endswith(".pt") else "", use_meta)
    with st.expander("Checkpoint load report"):
        st.write(rep.get("note",""))
        if rep["missing_keys"]: st.write("Missing keys:", rep["missing_keys"][:10], "…")
        if rep["unexpected_keys"]: st.write("Unexpected keys:", rep["unexpected_keys"][:10], "…")

    eff_meta = (model.meta_mlps is not None) and use_meta
    probs, idx = predict(model, img, int(age) if eff_meta else None,
                         sex if eff_meta else None, site if eff_meta else None)

    st.subheader("Predictions")
    cols = st.columns(3)
    for i, name in enumerate(CLASSES):
        with cols[i % 3]:
            st.metric(name, f"{probs[i]*100:0.1f}%")
    st.caption(f"Top class: **{CLASSES[idx]}**")

    st.markdown("---")
    st.subheader("Quick saliency (input-grad)")
    if st.toggle("Show explanation", value=False):
        x = preprocess(img).to(DEVICE).requires_grad_(True)
        y = model(x, None)[0, idx]
        model.zero_grad(set_to_none=True)
        y.backward()
        g = x.grad.detach().abs().max(dim=1)[0]
        g = (g - g.min()) / (g.max() - g.min() + 1e-8)
        heat = (g[0].cpu().numpy() * 255).astype(np.uint8)
        heat = np.stack([heat]*3, axis=-1)
        heat_img = Image.fromarray(heat).resize(img.size)
        st.image(heat_img, caption="Saliency overlay (brighter = more influence)", use_column_width=True)



