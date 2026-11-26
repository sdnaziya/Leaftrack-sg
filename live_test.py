"""Live model test — writes live_test.log"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = []
def p(s=""): OUT.append(str(s)); print(s, flush=True)

import numpy as np
import keras
from PIL import Image, ImageOps
import config

clf      = config.CLASSIFIER_MODEL_PATH
cls_json = config.CLASS_NAMES_PATH
H, W     = config.IMAGE_SIZE

model = keras.models.load_model(clf, compile=False)
with open(cls_json, encoding="utf-8") as f:
    class_names = json.load(f)

p(f"Model input  : {model.input_shape}")
p(f"Model output : {model.output_shape}")
p(f"Classes      : {len(class_names)}")
p(f"Model size   : {os.path.getsize(clf)/1024/1024:.1f} MB\n")

# Test 1 — Blank black image
p("--- TEST: Blank (black) image ---")
blank = np.zeros((1, H, W, 3), dtype=np.float32)
blank_norm = (blank / 127.5) - 1.0
pred = model.predict(blank_norm, verbose=0)[0]
top3 = np.argsort(pred)[::-1][:3]
for i in top3: p(f"  {pred[i]*100:5.1f}%  {class_names[i]}")

# Test 2 — Uniform gray image
p("\n--- TEST: Gray image (128,128,128) ---")
gray = np.full((1, H, W, 3), 128.0, dtype=np.float32)
gray_norm = (gray / 127.5) - 1.0
pred = model.predict(gray_norm, verbose=0)[0]
top3 = np.argsort(pred)[::-1][:3]
for i in top3: p(f"  {pred[i]*100:5.1f}%  {class_names[i]}")

# Test 3 — Green leaf-like image
p("\n--- TEST: Green image (leaf-like) ---")
green = np.zeros((1, H, W, 3), dtype=np.float32)
green[:,:,:,1] = 180  # green channel dominant
green_norm = (green / 127.5) - 1.0
pred = model.predict(green_norm, verbose=0)[0]
top3 = np.argsort(pred)[::-1][:3]
for i in top3: p(f"  {pred[i]*100:5.1f}%  {class_names[i]}")

# Test 4 — Real uploaded images (deduplicated)
p("\n--- TEST: Real uploaded images ---")
upload_dir = os.path.join(ROOT, "uploads")
seen = set()
tested = 0
for fname in sorted(os.listdir(upload_dir)):
    if not fname.lower().endswith(('.jpg','.jpeg','.png','.webp')): continue
    path = os.path.join(upload_dir, fname)
    try:
        img = Image.open(path).convert("RGB")
        arr = np.array(img)
        h = hash(arr.flatten()[:30].tobytes())
        if h in seen: continue
        seen.add(h)
        img_r = img.resize((H, W), Image.LANCZOS)
        arr_r = np.array(img_r).astype(np.float32)
        arr_n = (arr_r / 127.5) - 1.0
        inp   = np.expand_dims(arr_n, 0)
        pred  = model.predict(inp, verbose=0)[0]
        top3  = np.argsort(pred)[::-1][:3]
        entropy = float(-np.sum((pred+1e-10)*np.log(pred+1e-10)))
        max_e   = float(np.log(len(pred)))
        p(f"\n  {fname[:40]}")
        p(f"  size={img.size} entropy={entropy/max_e*100:.0f}%")
        for i in top3: p(f"  {pred[i]*100:5.1f}%  {class_names[i]}")
        tested += 1
        if tested >= 3: break
    except: pass

# Test 5 — One real training image per class
p("\n--- TEST: One training image per class ---")
pv = config.PLANTVILLAGE_DIR
correct = 0; total = 0
for idx, cls in enumerate(class_names):
    cls_dir = os.path.join(pv, cls)
    imgs = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    if not imgs: continue
    path = os.path.join(cls_dir, imgs[5] if len(imgs) > 5 else imgs[0])
    try:
        img = Image.open(path).convert("RGB").resize((H,W), Image.LANCZOS)
        arr = np.array(img).astype(np.float32)
        arr_n = (arr / 127.5) - 1.0
        inp   = np.expand_dims(arr_n, 0)
        pred  = model.predict(inp, verbose=0)[0]
        pred_idx  = int(np.argmax(pred))
        pred_conf = float(np.max(pred)) * 100
        ok = "✓" if pred_idx == idx else "✗"
        if pred_idx == idx: correct += 1
        total += 1
        p(f"  {ok} TRUE:[{idx}]{cls[:35]:<35} → PRED:[{pred_idx}]{class_names[pred_idx][:35]:<35} {pred_conf:.1f}%")
    except Exception as e:
        p(f"  ? ERROR on {cls}: {e}")

p(f"\n  Per-class accuracy: {correct}/{total} = {100*correct/max(total,1):.0f}%")
if correct/max(total,1) < 0.5:
    p("  ⚠ BELOW 50% — model is undertrained or preprocessing mismatch")
else:
    p("  ✓ Above 50% — model is differentiating classes")

LOG = os.path.join(ROOT, "live_test.log")
with open(LOG, "w", encoding="utf-8") as f:
    f.write("\n".join(OUT))
p(f"\nLog → {LOG}")
