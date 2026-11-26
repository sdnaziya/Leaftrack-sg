"""Inspect current project state and write to inspect_state.log"""
import os, sys, json

BASE = os.path.dirname(os.path.abspath(__file__))
lines = []

def p(s=""):
    lines.append(s)
    print(s)

p("=" * 60)
p("LEAFTRACK-SG STATE INSPECTION")
p("=" * 60)

# Models
p("\n--- MODELS ---")
clf  = os.path.join(BASE, "models", "classifier", "efficientnet_classifier.keras")
cls  = os.path.join(BASE, "models", "classifier", "class_names.json")
seg  = os.path.join(BASE, "models", "segmentation", "unet_model.keras")
for label, path in [("Classifier model", clf), ("Class names JSON", cls), ("Segmentation model", seg)]:
    exists = os.path.exists(path)
    size   = f"{os.path.getsize(path)/1024/1024:.1f} MB" if exists else "—"
    p(f"  {label}: {'EXISTS ' + size if exists else 'MISSING'}")
    p(f"    Path: {path}")

# Class names content
if os.path.exists(cls):
    with open(cls) as f:
        names = json.load(f)
    p(f"\n  class_names.json has {len(names)} classes")
    p(f"  First 5: {names[:5]}")

# Dataset
p("\n--- DATASET ---")
pv = os.path.join(BASE, "data", "plantvillage")
if os.path.isdir(pv):
    class_dirs = [d for d in os.listdir(pv) if os.path.isdir(os.path.join(pv, d))]
    p(f"  Class folders: {len(class_dirs)}")
    if class_dirs:
        total = 0
        for cd in class_dirs:
            cd_path = os.path.join(pv, cd)
            imgs = [f for f in os.listdir(cd_path) if f.lower().endswith((".jpg",".jpeg",".png"))]
            total += len(imgs)
        p(f"  Total images: {total}")
        p(f"  Sample classes: {sorted(class_dirs)[:5]}")
else:
    p("  MISSING: data/plantvillage/")

# TF / Keras
p("\n--- TENSORFLOW / KERAS ---")
try:
    import tensorflow as tf
    p(f"  TensorFlow: {tf.__version__}")
except Exception as e:
    p(f"  TensorFlow: ERROR — {e}")
try:
    import keras
    p(f"  Keras: {keras.__version__}")
    # check EfficientNetB0
    from keras.applications import EfficientNetB0
    p("  EfficientNetB0: available")
except Exception as e:
    p(f"  Keras/EfficientNetB0: ERROR — {e}")

# Services
p("\n--- SERVICE FILES ---")
for svc in ["prediction.py", "segmentation.py", "severity.py",
            "explainability.py", "recommendations.py", "report_generator.py"]:
    path = os.path.join(BASE, "services", svc)
    p(f"  {'EXISTS' if os.path.exists(path) else 'MISSING'}: services/{svc}")

# Routes
p("\n--- ROUTE FILES ---")
for rt in ["prediction_routes.py", "history_routes.py"]:
    path = os.path.join(BASE, "routes", rt)
    p(f"  {'EXISTS' if os.path.exists(path) else 'MISSING'}: routes/{rt}")

p("\n" + "=" * 60)

with open(os.path.join(BASE, "inspect_state.log"), "w") as f:
    f.write("\n".join(lines))
