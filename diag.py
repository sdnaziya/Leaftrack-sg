"""Full diagnostic. Run: venv\Scripts\python.exe diag.py"""
import json, os, sys
sys.path.insert(0, '.')
import config

OUT = []
def p(s=""): OUT.append(str(s)); print(s, flush=True)

p("=== CLASS NAMES ===")
if os.path.exists(config.CLASS_NAMES_PATH):
    with open(config.CLASS_NAMES_PATH) as f:
        names = json.load(f)
    for i, n in enumerate(names):
        p(f"  [{i}] {n}")
else:
    p("  MISSING")
    names = []

p("")
p("=== MODEL STATUS ===")
clf = config.CLASSIFIER_MODEL_PATH
if os.path.exists(clf):
    mb = round(os.path.getsize(clf)/1024/1024, 1)
    p(f"  Classifier: EXISTS {mb} MB")
else:
    p("  Classifier: MISSING")

p("")
p("=== PREPROCESSING (services/prediction.py) ===")
src = open("services/prediction.py", encoding="utf-8").read()
if "/ 127.5" in src or "/127.5" in src:
    p("  Normalization: (x/127.5)-1 → [-1, 1]  (EfficientNet style)")
elif "/ 255.0" in src or "/255.0" in src:
    p("  Normalization: x/255 → [0, 1]  (custom CNN style)")
else:
    p("  Normalization: NOT FOUND")

p("")
p("=== PARSE CLASS LABEL TEST ===")
# import and test the parser
try:
    from services.prediction import _parse_class_label
    test_labels = [
        "Pepper__bell___Bacterial_spot",
        "Pepper__bell___healthy",
        "Potato___Early_blight",
        "Tomato_Late_blight",
        "Tomato_Septoria_leaf_spot",
    ]
    if names:
        test_labels = names
    for lbl in test_labels:
        crop, disease, is_healthy, leaf_name = _parse_class_label(lbl)
        ok = "BAD" if crop == "Unknown" else "OK "
        p(f"  {ok}  {lbl}")
        p(f"       → crop={crop!r}  disease={disease!r}  leaf={leaf_name!r}  healthy={is_healthy}")
except Exception as e:
    p(f"  Error importing parser: {e}")

p("")
p("=== LIVE MODEL TEST ===")
if os.path.exists(clf) and names:
    try:
        import numpy as np
        import keras
        from PIL import Image, ImageOps

        model = keras.models.load_model(clf, compile=False)
        H, W  = config.IMAGE_SIZE

        p(f"  Model input  : {model.input_shape}")
        p(f"  Model output : {model.output_shape}")

        # Test blank black image (should be near-uniform if model is good)
        blank = np.zeros((1, H, W, 3), dtype=np.float32)
        pred  = model.predict(blank, verbose=0)[0]
        p(f"  Blank image top class: [{np.argmax(pred)}] {names[np.argmax(pred)]} ({np.max(pred)*100:.1f}%)")
        p(f"  Entropy ratio (1.0=random): {float(-np.sum((pred+1e-10)*np.log(pred+1e-10)) / np.log(len(pred))):.3f}")

        # Test training images
        pv = config.PLANTVILLAGE_DIR
        p("")
        p("  Per-class test (one training image each):")
        correct = 0
        for idx, cls in enumerate(names):
            cls_dir = os.path.join(pv, cls)
            if not os.path.isdir(cls_dir): continue
            imgs = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
            if not imgs: continue
            img_path = os.path.join(cls_dir, imgs[min(5, len(imgs)-1)])
            img = Image.open(img_path).convert("RGB")
            img = ImageOps.exif_transpose(img).resize((H, W), Image.LANCZOS)
            arr = np.array(img).astype(np.float32) / 255.0
            inp = np.expand_dims(arr, 0)
            pred = model.predict(inp, verbose=0)[0]
            pred_i = int(np.argmax(pred))
            conf   = float(np.max(pred)) * 100
            ok = "PASS" if pred_i == idx else "FAIL"
            p(f"  {ok} [{idx}]{cls[:40]:<41} → [{pred_i}]{names[pred_i][:25]:<26} {conf:.1f}%")
            if pred_i == idx: correct += 1
        p(f"\n  Per-class accuracy: {correct}/{len(names)} = {100*correct//max(len(names),1)}%")
    except Exception as e:
        import traceback
        p(f"  Live test error: {e}")
        p(traceback.format_exc())
else:
    p("  Cannot run — model or class names missing")

p("")
p("=== QUALITY CHECK CODE ===")
p("  _check_image_quality in prediction.py:")
if "_check_image_quality" in src:
    # Find the organic_pct threshold
    import re
    m = re.search(r"organic_pct\s*<\s*([\d.]+)", src)
    if m:
        p(f"  organic_pct threshold = {m.group(1)}% (below this → 'not a leaf')")
    else:
        p("  organic_pct threshold not found")
else:
    p("  _check_image_quality function not found")

LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diag.log")
with open(LOG, "w", encoding="utf-8") as f:
    f.write("\n".join(OUT))
p(f"\nLog saved → {LOG}")
