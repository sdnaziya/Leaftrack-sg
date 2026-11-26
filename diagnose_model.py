"""
Run actual model inference and print full softmax distribution.
Writes results to diagnose_model.log
Run: venv\Scripts\python.exe diagnose_model.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import Image, ImageOps
import config

LOG = os.path.join(os.path.dirname(__file__), "diagnose_model.log")
lines = []
def p(s=""): lines.append(s); print(s)

p("=" * 65)
p("LEAFTRACK-SG MODEL DIAGNOSIS")
p("=" * 65)

# ── Load class names ──────────────────────────────────────────────
with open(config.CLASS_NAMES_PATH) as f:
    class_names = json.load(f)
p(f"\nClass names ({len(class_names)} total):")
for i, name in enumerate(class_names):
    p(f"  [{i}] {name}")

# ── Load model ────────────────────────────────────────────────────
p(f"\nLoading model from: {config.CLASSIFIER_MODEL_PATH}")
import keras
model = keras.models.load_model(config.CLASSIFIER_MODEL_PATH, compile=False)
p(f"Model input shape: {model.input_shape}")
p(f"Model output shape: {model.output_shape}")
p(f"Model layers: {len(model.layers)}")

# ── Check model output range with a blank image ───────────────────
p("\n--- Blank image test (should be near-uniform distribution) ---")
blank = np.zeros((1, 224, 224, 3), dtype=np.float32)
blank_pred = model.predict(blank, verbose=0)[0]
p(f"Blank image predictions: {[round(float(x)*100, 2) for x in blank_pred]}")
p(f"Top class on blank: [{np.argmax(blank_pred)}] {class_names[np.argmax(blank_pred)]}")

# ── Check model output range with a uniform gray image ────────────
p("\n--- Gray image test (0.5 normalized) ---")
gray = np.full((1, 224, 224, 3), 0.5, dtype=np.float32)
gray_pred = model.predict(gray, verbose=0)[0]
p(f"Gray image predictions: {[round(float(x)*100, 2) for x in gray_pred]}")
p(f"Top class on gray: [{np.argmax(gray_pred)}] {class_names[np.argmax(gray_pred)]}")

# ── Check model output range with random noise ────────────────────
p("\n--- Random noise test ---")
np.random.seed(42)
noise = np.random.rand(1, 224, 224, 3).astype(np.float32)
noise_pred = model.predict(noise, verbose=0)[0]
p(f"Noise predictions: {[round(float(x)*100, 2) for x in noise_pred]}")
p(f"Top class on noise: [{np.argmax(noise_pred)}] {class_names[np.argmax(noise_pred)]}")

# ── Inspect the final Dense layer weights ─────────────────────────
p("\n--- Final dense layer weight statistics ---")
for layer in reversed(model.layers):
    if hasattr(layer, 'kernel') and layer.kernel is not None:
        w = layer.kernel.numpy()
        p(f"Layer '{layer.name}': weight shape {w.shape}, "
          f"mean={w.mean():.4f}, std={w.std():.4f}, "
          f"min={w.min():.4f}, max={w.max():.4f}")
        # Check per-output-unit norms
        norms = np.linalg.norm(w, axis=0)
        p(f"  Per-class weight norms: {[round(float(n),3) for n in norms]}")
        bias_layer = layer
        if layer.use_bias:
            b = layer.bias.numpy()
            p(f"  Biases: {[round(float(x),4) for x in b]}")
        break

# ── Test with real images from uploads/ ───────────────────────────
upload_dir = config.UPLOADS_DIR
imgs = [f for f in os.listdir(upload_dir)
        if f.lower().endswith(('.jpg','.jpeg','.png'))][:5]

if imgs:
    p(f"\n--- Testing {len(imgs)} uploaded image(s) ---")
    H, W = config.IMAGE_SIZE
    for fname in imgs:
        path = os.path.join(upload_dir, fname)
        try:
            img = Image.open(path).convert("RGB")
            img = ImageOps.exif_transpose(img)
            img = img.resize((H, W), Image.LANCZOS)
            arr = np.array(img).astype(np.float32) / 255.0
            inp = np.expand_dims(arr, 0)
            pred = model.predict(inp, verbose=0)[0]
            top3 = np.argsort(pred)[::-1][:3]
            p(f"\n  File: {fname}")
            p(f"  Image stats: min={arr.min():.3f} max={arr.max():.3f} "
              f"mean={arr.mean():.3f}")
            for idx in top3:
                p(f"    [{idx}] {class_names[idx]:<45} {pred[idx]*100:.2f}%")
        except Exception as e:
            p(f"  Error on {fname}: {e}")
else:
    p("\nNo uploaded images found in uploads/. Run a prediction first.")

p("\n" + "=" * 65)

with open(LOG, "w") as f:
    f.write("\n".join(lines))
p(f"\nLog written to {LOG}")
