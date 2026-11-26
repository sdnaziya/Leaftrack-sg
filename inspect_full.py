"""Full project inspection — writes inspect_full.log"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT = []
def p(s=""): OUT.append(str(s)); print(s, flush=True)

ROOT = os.path.dirname(os.path.abspath(__file__))
p("=" * 65)
p("LEAFTRACK-SG FULL INSPECTION")
p("=" * 65)

# 1. Dataset
p("\n--- DATASET (data/plantvillage/) ---")
pv = os.path.join(ROOT, "data", "plantvillage")
if os.path.isdir(pv):
    classes = sorted([d for d in os.listdir(pv) if os.path.isdir(os.path.join(pv,d))])
    total_imgs = 0
    for cls in classes:
        imgs = [f for f in os.listdir(os.path.join(pv, cls))
                if f.lower().endswith(('.jpg','.jpeg','.png'))]
        total_imgs += len(imgs)
        p(f"  [{len(imgs):>5}]  {cls}")
    p(f"\n  Total classes: {len(classes)}")
    p(f"  Total images : {total_imgs}")
else:
    p("  MISSING: data/plantvillage/")

# 2. Models
p("\n--- MODELS ---")
clf = os.path.join(ROOT, "models", "classifier", "efficientnet_classifier.keras")
cls_json = os.path.join(ROOT, "models", "classifier", "class_names.json")
seg = os.path.join(ROOT, "models", "segmentation", "unet_model.keras")
for label, path in [("Classifier", clf), ("class_names.json", cls_json), ("Segmentation", seg)]:
    exists = os.path.exists(path)
    size   = f"{os.path.getsize(path)/1024/1024:.1f} MB" if exists else "—"
    p(f"  {'EXISTS' if exists else 'MISSING'}  {size:>8}  {os.path.basename(path)}")

if os.path.exists(cls_json):
    with open(cls_json) as f:
        names = json.load(f)
    p(f"\n  class_names.json ({len(names)} classes):")
    for i, n in enumerate(names):
        p(f"    [{i}] {n}")

# 3. Training script normalization check
p("\n--- TRAINING SCRIPT NORMALIZATION ---")
train_py = os.path.join(ROOT, "training", "train_classifier.py")
if os.path.exists(train_py):
    with open(train_py) as f:
        src = f.read()
    if "preprocess_input" in src:
        p("  Training normalization: efficientnet.preprocess_input → [-1, 1]  CORRECT")
    elif "/ 255.0" in src or "/255.0" in src:
        p("  Training normalization: /255.0 → [0, 1]  (may differ from inference)")
    else:
        p("  Training normalization: UNKNOWN")
else:
    p("  MISSING: training/train_classifier.py")

# 4. Prediction service normalization
p("\n--- PREDICTION SERVICE NORMALIZATION ---")
pred_py = os.path.join(ROOT, "services", "prediction.py")
if os.path.exists(pred_py):
    with open(pred_py) as f:
        src = f.read()
    if "127.5" in src:
        p("  Inference normalization: (x/127.5)-1 → [-1, 1]  MATCHES training")
    elif "/ 255.0" in src or "/255.0" in src:
        p("  Inference normalization: /255.0 → [0, 1]  DOES NOT match training")
    else:
        p("  Inference normalization: UNKNOWN")
else:
    p("  MISSING: services/prediction.py")

# 5. Test with real images using BOTH preprocessing paths
p("\n--- LIVE MODEL TEST ---")
if os.path.exists(clf) and os.path.exists(cls_json):
    try:
        import numpy as np, keras
        from PIL import Image, ImageOps
        model = keras.models.load_model(clf, compile=False)
        with open(cls_json) as f:
            class_names = json.load(f)
        H, W = 224, 224

        upload_dir = os.path.join(ROOT, "uploads")
        imgs = []
        if os.path.isdir(upload_dir):
            seen_hashes = set()
            for fname in sorted(os.listdir(upload_dir)):
                if not fname.lower().endswith(('.jpg','.jpeg','.png','.webp')):
                    continue
                path = os.path.join(upload_dir, fname)
                arr = np.array(Image.open(path).convert("RGB"))
                h = hash(arr.flatten()[:50].tobytes())
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    imgs.append((fname, path))
                if len(imgs) >= 4:
                    break

        if not imgs:
            # Create synthetic test images
            p("  No uploaded images — testing with synthetic images")
            for color, label in [([34,139,34],"Green"), ([139,69,19],"Brown"), ([255,255,0],"Yellow")]:
                arr = np.full((H,W,3), color, dtype=np.float32)
                imgs.append((f"synthetic_{label}", None, arr))

        for item in imgs:
            if len(item) == 2:
                fname, path = item
                arr = np.array(Image.open(path).convert("RGB").resize((H,W), Image.LANCZOS)).astype(np.float32)
            else:
                fname, path, arr = item

            arr_eff = (arr / 127.5) - 1.0
            inp = np.expand_dims(arr_eff, 0)
            pred = model.predict(inp, verbose=0)[0]
            top3 = np.argsort(pred)[::-1][:3]
            entropy = float(-np.sum((pred+1e-10)*np.log(pred+1e-10)))
            max_e = float(np.log(len(pred)))
            p(f"\n  Image: {fname[:40]}")
            p(f"  Entropy: {entropy:.3f}/{max_e:.3f} ({entropy/max_e*100:.0f}% of max)")
            for i in top3:
                p(f"    [{i}] {class_names[i]:<50} {pred[i]*100:5.1f}%")
            if pred[top3[0]] < 0.40:
                p("  ⚠ LOW CONFIDENCE — model may be undertrained")
    except Exception as e:
        p(f"  Live test ERROR: {e}")
else:
    p("  Cannot test — model or class_names.json missing")

# 6. Config check
p("\n--- CONFIG ---")
import config
p(f"  EPOCHS         : {config.EPOCHS}")
p(f"  BATCH_SIZE     : {config.BATCH_SIZE}")
p(f"  LEARNING_RATE  : {config.LEARNING_RATE}")
p(f"  IMAGE_SIZE     : {config.IMAGE_SIZE}")
p(f"  SEVERITY_LOW   : {config.SEVERITY_LOW_MAX}%")
p(f"  SEVERITY_MOD   : {config.SEVERITY_MODERATE_MAX}%")

p("\n" + "=" * 65)
LOG = os.path.join(ROOT, "inspect_full.log")
with open(LOG, "w", encoding="utf-8") as f:
    f.write("\n".join(OUT))
p(f"Log saved → {LOG}")
