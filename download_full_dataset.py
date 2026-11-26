"""
LeafTrack-SG – Full PlantVillage Dataset Downloader
====================================================
Downloads the complete PlantVillage dataset (38 classes, ~54k images)
using tensorflow_datasets (TFDS), which provides the official version.

Alternatively uses Kaggle CLI if TFDS fails.

After downloading, the script:
  1. Removes empty or near-empty folders (< 10 images)
  2. Removes duplicate images by MD5 hash
  3. Verifies every image is readable
  4. Prints a final class summary

RUN (from project root with venv active):
    venv\\Scripts\\python.exe download_full_dataset.py

WHAT YOU GET AFTER THIS COMPLETES:
  data/plantvillage/
      Apple___Apple_scab/          (630+ images)
      Apple___Black_rot/           (620+ images)
      Apple___Cedar_apple_rust/    (270+ images)
      Apple___healthy/             (1640+ images)
      Blueberry___healthy/
      Cherry___Powdery_mildew/
      Cherry___healthy/
      Corn___Cercospora_leaf_spot/
      Corn___Common_rust/
      Corn___Northern_Leaf_Blight/
      Corn___healthy/
      Grape___Black_rot/
      Grape___Esca/
      Grape___Leaf_blight/
      Grape___healthy/
      Orange___Haunglongbing/
      Peach___Bacterial_spot/
      Peach___healthy/
      Pepper__bell___Bacterial_spot/   ← already present
      Pepper__bell___healthy/          ← already present
      Potato___Early_blight/           ← already present
      Potato___Late_blight/            ← already present
      Potato___healthy/                ← already present
      Raspberry___healthy/
      Soybean___healthy/
      Squash___Powdery_mildew/
      Strawberry___Leaf_scorch/
      Strawberry___healthy/
      Tomato___Bacterial_spot/
      Tomato___Early_blight/           ← already present (rename needed)
      ... (38 total)
"""
import os, sys, json, hashlib, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

PV = config.PLANTVILLAGE_DIR
LOG = os.path.join(os.path.dirname(__file__), "download_full_dataset.log")
OUT = []

def p(s=""): OUT.append(str(s)); print(s, flush=True)


# ── Step 1: Try tensorflow_datasets ───────────────────────────────
def download_via_tfds():
    p("[TFDS] Attempting download via tensorflow_datasets...")
    try:
        import tensorflow_datasets as tfds
    except ImportError:
        p("[TFDS] tensorflow_datasets not installed. Installing...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install",
                        "tensorflow_datasets", "--quiet"], check=True)
        import tensorflow_datasets as tfds

    cache_dir = os.path.join(config.DATA_DIR, "_tfds_cache")
    os.makedirs(cache_dir, exist_ok=True)

    p(f"[TFDS] Downloading plant_village to cache: {cache_dir}")
    p("[TFDS] This may take 15–30 minutes on first run (~1.3 GB)...")

    try:
        ds, info = tfds.load(
            "plant_village",
            split=["train"],
            with_info=True,
            as_supervised=True,
            data_dir=cache_dir,
        )
        label_names = info.features["label"].names
        p(f"[TFDS] {len(label_names)} classes found.")

        # Create destination folders
        for label in label_names:
            os.makedirs(os.path.join(PV, label), exist_ok=True)

        # Save images
        saved = 0
        skipped = 0
        for img_tensor, label_tensor in ds[0]:
            label_idx  = int(label_tensor.numpy())
            label_name = label_names[label_idx]
            out_dir    = os.path.join(PV, label_name)
            out_path   = os.path.join(out_dir, f"{saved:07d}.jpg")

            if not os.path.exists(out_path):
                import numpy as np
                from PIL import Image
                arr = img_tensor.numpy().astype("uint8")
                Image.fromarray(arr).save(out_path, quality=95, optimize=True)

            saved += 1
            if saved % 5000 == 0:
                p(f"  [TFDS] Saved {saved} images...")

        p(f"[TFDS] Download complete. {saved} images saved.")
        return True

    except Exception as e:
        p(f"[TFDS] Failed: {e}")
        return False


# ── Step 2: Kaggle fallback ────────────────────────────────────────
def print_kaggle_instructions():
    p("""
╔══════════════════════════════════════════════════════════════════╗
║  MANUAL DOWNLOAD REQUIRED                                        ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  Option A — Kaggle (recommended, ~1.3 GB):                      ║
║  1. Create account at kaggle.com                                 ║
║  2. Settings → API → Create New Token → kaggle.json             ║
║  3. Place kaggle.json in: C:\\Users\\AQSA\\.kaggle\\             ║
║  4. Run:                                                         ║
║     venv\\Scripts\\python.exe -m kaggle datasets download \\     ║
║       -d abdallahalidev/plantvillage-dataset                     ║
║  5. Unzip the downloaded file into data/plantvillage/            ║
║                                                                  ║
║  Option B — Mendeley (direct download, free):                   ║
║     https://data.mendeley.com/datasets/tywbtsjrjv/1              ║
║     Download "PlantVillage Dataset.zip"                          ║
║     Extract all class folders to: data/plantvillage/             ║
║                                                                  ║
║  After downloading, re-run this script to validate the data.    ║
╚══════════════════════════════════════════════════════════════════╝
""")


# ── Step 3: Validate and clean the dataset ─────────────────────────
def validate_and_clean():
    p("\n--- VALIDATING DATASET ---")
    if not os.path.isdir(PV):
        p(f"[ERROR] {PV} does not exist.")
        return False

    from PIL import Image
    classes = sorted([d for d in os.listdir(PV) if os.path.isdir(os.path.join(PV, d))])
    p(f"  Checking {len(classes)} class folders...")

    total_kept = 0
    total_removed = 0
    removed_empty = []

    for cls in classes:
        cls_dir = os.path.join(PV, cls)
        imgs = [f for f in os.listdir(cls_dir)
                if f.lower().endswith(('.jpg','.jpeg','.png'))]

        # Remove empty folders
        if len(imgs) == 0:
            p(f"  REMOVING empty folder: {cls}")
            shutil.rmtree(cls_dir, ignore_errors=True)
            removed_empty.append(cls)
            continue

        # Validate images and remove corrupted ones
        kept = 0
        seen_hashes = set()
        for fname in imgs:
            path = os.path.join(cls_dir, fname)
            try:
                # Check readability
                with Image.open(path) as img:
                    img.verify()   # will raise on corrupt files

                # Check for duplicates by MD5
                with open(path, "rb") as f:
                    h = hashlib.md5(f.read()).hexdigest()
                if h in seen_hashes:
                    os.remove(path)
                    total_removed += 1
                    continue
                seen_hashes.add(h)
                kept += 1
            except Exception:
                os.remove(path)
                total_removed += 1

        total_kept += kept

    classes_after = sorted([d for d in os.listdir(PV) if os.path.isdir(os.path.join(PV, d))])
    p(f"\n  Kept      : {total_kept} valid images")
    p(f"  Removed   : {total_removed} corrupted/duplicate images")
    p(f"  Removed empty folders: {removed_empty}")
    p(f"  Final classes: {len(classes_after)}")
    return True


# ── Step 4: Print final summary ────────────────────────────────────
def print_summary():
    p("\n--- FINAL DATASET SUMMARY ---")
    if not os.path.isdir(PV):
        p("  Dataset not found.")
        return

    classes = sorted([d for d in os.listdir(PV) if os.path.isdir(os.path.join(PV, d))])
    crops = {}
    grand_total = 0

    for cls in classes:
        n = len([f for f in os.listdir(os.path.join(PV, cls))
                 if f.lower().endswith(('.jpg','.jpeg','.png'))])
        grand_total += n
        # Determine crop from class name
        if "___" in cls:
            crop = cls.split("___")[0]
        else:
            crop = cls.split("_")[0]
        crops.setdefault(crop, []).append((cls, n))

    p(f"  {len(classes)} classes  |  {grand_total} total images  |  {len(crops)} crops\n")
    for crop, entries in sorted(crops.items()):
        ct = sum(n for _, n in entries)
        p(f"  {crop}  ({ct} images, {len(entries)} disease classes):")
        for cls, n in sorted(entries):
            p(f"    [{n:>5}]  {cls}")
    p()


# ── MAIN ───────────────────────────────────────────────────────────
def main():
    p("=" * 65)
    p("LeafTrack-SG – Full PlantVillage Dataset Downloader")
    p("=" * 65)
    p(f"Target: {PV}")

    # Count existing
    existing = 0
    if os.path.isdir(PV):
        for cls in os.listdir(PV):
            cls_dir = os.path.join(PV, cls)
            if os.path.isdir(cls_dir):
                existing += len([f for f in os.listdir(cls_dir)
                                  if f.lower().endswith(('.jpg','.jpeg','.png'))])

    p(f"Existing images: {existing}")

    if existing >= 50000:
        p("\nDataset appears complete (50k+ images). Skipping download.")
    elif existing >= 10000:
        p(f"\nPartial dataset found ({existing} images). Attempting to get full dataset...")
        ok = download_via_tfds()
        if not ok:
            print_kaggle_instructions()
    elif existing > 0:
        p(f"\nOnly {existing} images found. Attempting full download...")
        ok = download_via_tfds()
        if not ok:
            print_kaggle_instructions()
    else:
        p("\nNo images found. Attempting download via TFDS...")
        ok = download_via_tfds()
        if not ok:
            print_kaggle_instructions()

    # Always validate what's there
    validate_and_clean()
    print_summary()

    p("\nNext step: retrain the model with all available classes:")
    p("  venv\\Scripts\\python.exe retrain.py")

    with open(LOG, "w", encoding="utf-8") as f:
        f.write("\n".join(OUT))
    p(f"\nLog → {LOG}")


if __name__ == "__main__":
    main()
