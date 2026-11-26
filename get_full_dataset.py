"""
LeafTrack-SG – Full PlantVillage Dataset Downloader & Organiser
================================================================
Downloads the complete PlantVillage dataset (38 classes, ~54 k images)
covering all 14 crop species requested.

After running this script you will have:
  data/plantvillage/
    Apple___Apple_scab/
    Apple___Black_rot/
    Apple___Cedar_apple_rust/
    Apple___healthy/
    Blueberry___healthy/
    Cherry_(including_sour)___Powdery_mildew/
    Cherry_(including_sour)___healthy/
    Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot/
    Corn_(maize)___Common_rust_/
    Corn_(maize)___Northern_Leaf_Blight/
    Corn_(maize)___healthy/
    Grape___Black_rot/
    Grape___Esca_(Black_Measles)/
    Grape___Leaf_blight_(Isariopsis_Leaf_Spot)/
    Grape___healthy/
    Orange___Haunglongbing_(Citrus_greening)/
    Peach___Bacterial_spot/
    Peach___healthy/
    Pepper__bell___Bacterial_spot/      ← already present
    Pepper__bell___healthy/             ← already present
    Potato___Early_blight/              ← already present
    Potato___Late_blight/               ← already present
    Potato___healthy/                   ← already present
    Raspberry___healthy/
    Soybean___healthy/
    Squash___Powdery_mildew/
    Strawberry___Leaf_scorch/
    Strawberry___healthy/
    Tomato___Bacterial_spot/
    Tomato___Early_blight/
    Tomato___Late_blight/
    Tomato___Leaf_Mold/
    Tomato___Septoria_leaf_spot/
    Tomato___Spider_mites Two-spotted_spider_mite/
    Tomato___Target_Spot/
    Tomato___Tomato_Yellow_Leaf_Curl_Virus/
    Tomato___Tomato_mosaic_virus/
    Tomato___healthy/

HOW TO RUN:
    venv\\Scripts\\python.exe get_full_dataset.py

The script tries three methods in order:
  1. tensorflow_datasets (auto-download, ~1.3 GB)
  2. Kaggle CLI (needs kaggle.json in ~/.kaggle/)
  3. Prints manual download instructions

After completion, run:
    venv\\Scripts\\python.exe retrain.py
"""
import os, sys, json, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

PV  = config.PLANTVILLAGE_DIR
LOG = os.path.join(os.path.dirname(__file__), "get_full_dataset.log")
OUT = []


def p(s=""): OUT.append(str(s)); print(s, flush=True)


# ── Count how many images we already have ─────────────────────────
def count_existing():
    if not os.path.isdir(PV):
        return 0
    n = 0
    for cls in os.listdir(PV):
        d = os.path.join(PV, cls)
        if os.path.isdir(d):
            n += len([f for f in os.listdir(d) if f.lower().endswith(('.jpg','.jpeg','.png'))])
    return n


# ── Method 1: tensorflow_datasets ─────────────────────────────────
def download_via_tfds() -> bool:
    p("[Method 1] Trying tensorflow_datasets...")
    try:
        import tensorflow_datasets as tfds
    except ImportError:
        p("  tensorflow_datasets not installed. Installing...")
        import subprocess
        r = subprocess.run(
            [sys.executable, "-m", "pip", "install", "tensorflow_datasets", "--quiet"],
            capture_output=True
        )
        if r.returncode != 0:
            p(f"  Install failed: {r.stderr.decode()[:200]}")
            return False
        import tensorflow_datasets as tfds

    cache = os.path.join(config.DATA_DIR, "_tfds_cache")
    os.makedirs(cache, exist_ok=True)

    try:
        p("  Downloading plant_village (~1.3 GB). This may take 20–40 minutes...")
        ds, info = tfds.load(
            "plant_village",
            split=["train"],
            with_info=True,
            as_supervised=True,
            data_dir=cache,
        )
        label_names = info.features["label"].names
        p(f"  {len(label_names)} classes found.")

        # Create output folders
        for label in label_names:
            os.makedirs(os.path.join(PV, label), exist_ok=True)

        saved = 0
        for img_tensor, label_tensor in ds[0]:
            label_name = label_names[int(label_tensor.numpy())]
            out_path   = os.path.join(PV, label_name, f"{saved:08d}.jpg")
            if not os.path.exists(out_path):
                import numpy as np
                from PIL import Image
                Image.fromarray(img_tensor.numpy().astype("uint8")).save(
                    out_path, quality=95, optimize=True
                )
            saved += 1
            if saved % 5000 == 0:
                p(f"  Saved {saved} images...")

        p(f"  TFDS download complete — {saved} images saved.")
        return True
    except Exception as e:
        p(f"  TFDS failed: {e}")
        return False


# ── Method 2: Kaggle CLI ───────────────────────────────────────────
def download_via_kaggle() -> bool:
    p("[Method 2] Trying Kaggle CLI...")
    kaggle_json = os.path.join(os.path.expanduser("~"), ".kaggle", "kaggle.json")
    if not os.path.exists(kaggle_json):
        p("  kaggle.json not found. Skipping Kaggle.")
        return False

    import subprocess, zipfile
    p("  Downloading via Kaggle (~1.3 GB)...")
    tmp = os.path.join(config.DATA_DIR, "_kaggle_tmp")
    os.makedirs(tmp, exist_ok=True)

    r = subprocess.run(
        [sys.executable, "-m", "kaggle", "datasets", "download",
         "-d", "abdallahalidev/plantvillage-dataset", "-p", tmp],
        capture_output=True, text=True
    )
    if r.returncode != 0:
        p(f"  Kaggle download failed: {r.stderr[:300]}")
        return False

    # Unzip
    zips = [f for f in os.listdir(tmp) if f.endswith(".zip")]
    for z in zips:
        p(f"  Extracting {z}...")
        with zipfile.ZipFile(os.path.join(tmp, z), "r") as zf:
            zf.extractall(tmp)

    # Find the class folders inside the extracted content
    _move_extracted_folders(tmp)
    shutil.rmtree(tmp, ignore_errors=True)
    p("  Kaggle download and extraction complete.")
    return True


def _move_extracted_folders(tmp_dir: str):
    """Move extracted class folders into PV, handling nested directories."""
    for dirpath, dirs, files in os.walk(tmp_dir):
        imgs = [f for f in files if f.lower().endswith(('.jpg','.jpeg','.png'))]
        if len(imgs) >= 10:
            # This looks like a class folder
            cls_name = os.path.basename(dirpath)
            dest = os.path.join(PV, cls_name)
            os.makedirs(dest, exist_ok=True)
            for img in imgs:
                src = os.path.join(dirpath, img)
                dst = os.path.join(dest, img)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)


# ── Method 3: Manual instructions ─────────────────────────────────
def print_manual_instructions():
    p("""
╔══════════════════════════════════════════════════════════════════╗
║          MANUAL DOWNLOAD REQUIRED                                ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  Option A — Kaggle (fastest, ~1.3 GB zip):                      ║
║                                                                  ║
║  1. Create account at: kaggle.com                                ║
║  2. Settings → API → Create New Token → kaggle.json             ║
║  3. Place kaggle.json in: C:\\Users\\AQSA\\.kaggle\\             ║
║  4. Run in terminal:                                             ║
║                                                                  ║
║  venv\\Scripts\\python.exe -m kaggle datasets download \\        ║
║    -d abdallahalidev/plantvillage-dataset                        ║
║                                                                  ║
║  5. Unzip the downloaded file into:                              ║
║       data\\plantvillage\\                                       ║
║                                                                  ║
║  Option B — Mendeley (direct browser download, free):           ║
║                                                                  ║
║  1. Open in browser:                                             ║
║     https://data.mendeley.com/datasets/tywbtsjrjv/1              ║
║  2. Click "Download All" (~1.3 GB zip)                           ║
║  3. Extract all class folders to:                                ║
║       data\\plantvillage\\                                       ║
║                                                                  ║
║  EXPECTED RESULT:                                                ║
║  data\\plantvillage\\                                            ║
║      Apple___Apple_scab\\      ← 630+ images                    ║
║      Apple___healthy\\         ← 1640+ images                   ║
║      Corn_(maize)___Common_rust_\\                               ║
║      Grape___Black_rot\\                                         ║
║      Tomato___Early_blight\\                                     ║
║      ... (38 class folders total)                                ║
║                                                                  ║
║  After downloading, run:                                         ║
║    venv\\Scripts\\python.exe get_full_dataset.py   ← verify      ║
║    venv\\Scripts\\python.exe retrain.py            ← retrain     ║
║    venv\\Scripts\\python.exe app.py                ← website     ║
╚══════════════════════════════════════════════════════════════════╝
""")


# ── Validate and clean what's on disk ─────────────────────────────
def validate_dataset():
    p("\n--- VALIDATING DATASET ---")
    if not os.path.isdir(PV):
        p("  Dataset directory not found.")
        return

    from PIL import Image
    import hashlib

    classes = sorted([d for d in os.listdir(PV) if os.path.isdir(os.path.join(PV, d))])
    total_kept = total_removed = 0

    for cls in classes:
        cls_dir = os.path.join(PV, cls)
        imgs    = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
        seen    = set()
        kept    = 0

        for fname in imgs:
            path = os.path.join(cls_dir, fname)
            try:
                # Verify image is readable
                with Image.open(path) as img:
                    img.verify()
                # Duplicate detection by MD5
                with open(path, "rb") as f:
                    h = hashlib.md5(f.read()).hexdigest()
                if h in seen:
                    os.remove(path)
                    total_removed += 1
                    continue
                seen.add(h)
                kept += 1
            except Exception:
                os.remove(path)
                total_removed += 1

        total_kept += kept

    p(f"  Valid images kept    : {total_kept}")
    p(f"  Corrupted/duplicates : {total_removed}")


# ── Print final summary ────────────────────────────────────────────
def print_summary():
    p("\n--- FINAL DATASET SUMMARY ---")
    if not os.path.isdir(PV):
        return

    classes = sorted([d for d in os.listdir(PV) if os.path.isdir(os.path.join(PV, d))])
    crops   = {}
    grand   = 0

    for cls in classes:
        n = len([f for f in os.listdir(os.path.join(PV, cls))
                 if f.lower().endswith(('.jpg','.jpeg','.png'))])
        grand += n
        crop  = cls.split("___")[0] if "___" in cls else cls.split("_")[0]
        crops.setdefault(crop, []).append((cls, n))

    p(f"  {len(classes)} disease classes  |  {grand} images  |  {len(crops)} crops")
    p()
    for crop, entries in sorted(crops.items()):
        ct = sum(n for _, n in entries)
        p(f"  {crop}  ({ct} images, {len(entries)} classes):")
        for cls, n in sorted(entries):
            p(f"    [{n:>5}]  {cls}")


# ── MAIN ───────────────────────────────────────────────────────────
def main():
    p("=" * 65)
    p("LeafTrack-SG – Full PlantVillage Dataset Downloader")
    p("=" * 65)

    existing = count_existing()
    p(f"\nExisting images on disk: {existing}")

    if existing >= 50000:
        p("Dataset already looks complete (50k+ images).")
        validate_dataset()
        print_summary()
    elif existing > 0:
        p(f"Partial dataset found ({existing} images).")
        p("Attempting to download the missing classes...\n")
        ok = download_via_tfds()
        if not ok:
            ok = download_via_kaggle()
        if not ok:
            print_manual_instructions()
        else:
            validate_dataset()
            print_summary()
    else:
        p("No images found. Attempting download...\n")
        ok = download_via_tfds()
        if not ok:
            ok = download_via_kaggle()
        if not ok:
            print_manual_instructions()
        else:
            validate_dataset()
            print_summary()

    p("\n")
    p("After the dataset is ready, run:")
    p("  venv\\Scripts\\python.exe retrain.py")

    with open(LOG, "w", encoding="utf-8") as f:
        f.write("\n".join(OUT))
    p(f"Log → {LOG}")


if __name__ == "__main__":
    main()
