"""
Deep model diagnosis — checks preprocessing and per-image uniqueness.
Run: venv\Scripts\python.exe diagnose_model2.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from PIL import Image, ImageOps
import config

LOG = "diagnose_model2.log"
lines = []
def p(s=""): lines.append(s); print(s)

p("=" * 65)
p("PREPROCESSING & IMAGE UNIQUENESS CHECK")
p("=" * 65)

# Check all uploaded images
upload_dir = config.UPLOADS_DIR
imgs = sorted([f for f in os.listdir(upload_dir)
               if f.lower().endswith(('.jpg','.jpeg','.png','.webp'))])

p(f"\nImages in uploads/: {len(imgs)}")
H, W = config.IMAGE_SIZE

# Check if images are genuinely different
p("\n--- Per-image pixel statistics ---")
hashes = []
for fname in imgs[:8]:
    path = os.path.join(upload_dir, fname)
    img  = Image.open(path).convert("RGB")
    img  = ImageOps.exif_transpose(img)
    arr  = np.array(img)
    # Hash first 100 pixels to check uniqueness
    flat = arr.flatten()[:100]
    h    = hash(flat.tobytes())
    p(f"  {fname[:30]:<32} shape={arr.shape} mean={arr.mean():.1f} "
      f"hash={h % 99999:05d}")
    hashes.append(h)

unique = len(set(hashes))
p(f"\nUnique images (by first-100-pixel hash): {unique} / {len(hashes)}")
if unique < len(hashes):
    p("  !! DUPLICATE IMAGES DETECTED — this explains identical predictions !!")

# Check data/plantvillage structure
p("\n--- PlantVillage dataset check ---")
pv_dir = config.PLANTVILLAGE_DIR
if os.path.isdir(pv_dir):
    classes = [d for d in os.listdir(pv_dir) if os.path.isdir(os.path.join(pv_dir, d))]
    p(f"  Dataset classes: {len(classes)}")
    for cls in sorted(classes):
        cls_path = os.path.join(pv_dir, cls)
        imgs_in_cls = [f for f in os.listdir(cls_path)
                       if f.lower().endswith(('.jpg','.jpeg','.png'))]
        p(f"  [{len(imgs_in_cls):>5}] {cls}")
else:
    p("  PLANTVILLAGE_DIR not found")

# Check whether training used correct normalization
p("\n--- Normalization check ---")
p("  Training uses: arr / 255.0  (EfficientNetB0 without built-in rescaling)")
p("  EfficientNetB0 built-in preprocessing expects: 0-255 range if using")
p("  tf.keras.applications.efficientnet.preprocess_input()")
p("  Current code: divides by 255 manually -> range [0,1]")
p("  This is INCONSISTENT if the model was built with preprocess_input.")
p("  Check training/train_classifier.py for how EfficientNetB0 was called.")

# Load and check the model's first few layer outputs
p("\n--- First layer output for a real image ---")
import keras
model = keras.models.load_model(config.CLASSIFIER_MODEL_PATH, compile=False)

# Get the EfficientNetB0 base model (layer index 1 in the wrapper)
p(f"\nModel layers:")
for i, layer in enumerate(model.layers):
    p(f"  [{i}] {layer.name} — {type(layer).__name__}")

# Test two different images to see if outputs differ
test_imgs = imgs[:2]
if len(test_imgs) >= 2:
    preds_list = []
    p(f"\nComparing first 2 images after resize:")
    for fname in test_imgs:
        path = os.path.join(upload_dir, fname)
        img = Image.open(path).convert("RGB")
        img = ImageOps.exif_transpose(img)
        orig_size = img.size
        img_r = img.resize((H, W), Image.LANCZOS)
        arr = np.array(img_r).astype(np.float32) / 255.0
        inp = np.expand_dims(arr, 0)
        pred = model.predict(inp, verbose=0)[0]
        preds_list.append(pred)
        p(f"  {fname[:30]}: orig={orig_size} -> resized=({H},{W}) "
          f"mean_after_resize={arr.mean():.4f} "
          f"top1=[{np.argmax(pred)}] {pred[np.argmax(pred)]*100:.2f}%")
    
    diff = np.abs(preds_list[0] - preds_list[1]).max()
    p(f"\n  Max prediction difference between img1 and img2: {diff:.6f}")
    if diff < 1e-5:
        p("  !! IDENTICAL MODEL OUTPUTS — model is not differentiating images !!")
        p("  This strongly suggests the model is undertrained / collapsed.")

with open(LOG, "w") as f:
    f.write("\n".join(lines))
p(f"\nLog saved to {LOG}")
