"""
Inspect existing dataset, model, and class names.
Run: venv\Scripts\python.exe inspect_dataset.py
Writes: inspect_dataset.log
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

OUT = []
def p(s=""): OUT.append(str(s)); print(s, flush=True)

ROOT = os.path.dirname(os.path.abspath(__file__))
p("=" * 65)
p("LEAFTRACK-SG DATASET + MODEL INSPECTION")
p("=" * 65)

# ── Current class_names.json ───────────────────────────────────────
p("\n--- CURRENT class_names.json ---")
if os.path.exists(config.CLASS_NAMES_PATH):
    with open(config.CLASS_NAMES_PATH, encoding="utf-8") as f:
        names = json.load(f)
    p(f"  {len(names)} classes currently trained:")
    for i, n in enumerate(names):
        p(f"  [{i:>2}] {n}")
else:
    p("  MISSING — no trained model yet")
    names = []

# ── Current model ─────────────────────────────────────────────────
p("\n--- CURRENT MODEL ---")
clf = config.CLASSIFIER_MODEL_PATH
if os.path.exists(clf):
    mb = os.path.getsize(clf) / 1024 / 1024
    p(f"  EXISTS  {mb:.1f} MB  →  {clf}")
else:
    p("  MISSING")

# ── PlantVillage dataset ───────────────────────────────────────────
p("\n--- PLANTVILLAGE DATASET ---")
pv = config.PLANTVILLAGE_DIR
if os.path.isdir(pv):
    classes = sorted([d for d in os.listdir(pv) if os.path.isdir(os.path.join(pv, d))])
    total = 0
    # group by crop
    crops = {}
    for cls in classes:
        crop = cls.split("___")[0].split("_")[0] if "_" in cls else cls
        # better: take everything before first ___
        if "___" in cls:
            crop = cls.split("___")[0]
        elif "_" in cls:
            # Tomato_Early_blight → Tomato
            for known in ["Tomato","Potato","Pepper","Corn","Apple","Grape",
                           "Strawberry","Peach","Cherry","Blueberry","Raspberry",
                           "Soybean","Squash","Orange"]:
                if cls.lower().startswith(known.lower()):
                    crop = known; break
            else:
                crop = cls
        n = len([f for f in os.listdir(os.path.join(pv, cls))
                 if f.lower().endswith((".jpg",".jpeg",".png"))])
        total += n
        crops.setdefault(crop, []).append((cls, n))
    p(f"  {len(classes)} class folders  |  {total} total images")
    p(f"  {len(crops)} unique crops:\n")
    for crop, entries in sorted(crops.items()):
        crop_total = sum(n for _, n in entries)
        p(f"  {crop}  ({crop_total} images, {len(entries)} classes):")
        for cls, n in sorted(entries):
            already = "✓ TRAINED" if cls in names else "  NOT YET"
            p(f"      {already}  [{n:>5}]  {cls}")
else:
    p("  PLANTVILLAGE_DIR not found — dataset not downloaded")

# ── Missing classes (in PV but not yet trained) ────────────────────
if os.path.isdir(pv) and names:
    pv_classes = set(sorted([d for d in os.listdir(pv) if os.path.isdir(os.path.join(pv, d))]))
    trained    = set(names)
    not_trained = sorted(pv_classes - trained)
    p(f"\n--- CLASSES IN DATASET BUT NOT TRAINED ({len(not_trained)}) ---")
    for c in not_trained:
        n = len([f for f in os.listdir(os.path.join(pv, c)) if f.lower().endswith((".jpg",".jpeg",".png"))])
        p(f"  [{n:>5}]  {c}")

p("\n" + "=" * 65)
LOG = os.path.join(ROOT, "inspect_dataset.log")
with open(LOG, "w", encoding="utf-8") as f:
    f.write("\n".join(OUT))
p(f"Log → {LOG}")
