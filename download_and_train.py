"""
LeafTrack-SG – Dataset Download + Classifier Training
======================================================
This script does everything in order:
  1. Checks if PlantVillage dataset already exists
  2. If not, downloads it (tries tf-datasets first, then direct URL)
  3. Verifies the dataset structure
  4. Trains EfficientNetB0 classifier
  5. Saves models/classifier/efficientnet_classifier.keras
  6. Saves models/classifier/class_names.json

Run from the project root:
    venv\Scripts\python.exe download_and_train.py

For faster training on first run (fewer epochs, can be resumed):
    venv\Scripts\python.exe download_and_train.py --fast
"""
import os
import sys
import json
import argparse

# ── Make sure we run from project root ───────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

import config

VENV_PY = os.path.join(PROJECT_ROOT, "venv", "Scripts", "python.exe")
LOG_FILE = os.path.join(PROJECT_ROOT, "training_log.txt")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 – CHECK / DOWNLOAD DATASET
# ══════════════════════════════════════════════════════════════════════════════

def count_classes():
    """Return number of class subdirectories found in PLANTVILLAGE_DIR."""
    if not os.path.isdir(config.PLANTVILLAGE_DIR):
        return 0
    return len([
        d for d in os.listdir(config.PLANTVILLAGE_DIR)
        if os.path.isdir(os.path.join(config.PLANTVILLAGE_DIR, d))
    ])


def dataset_ready():
    n = count_classes()
    if n >= 10:
        total_imgs = 0
        for cls in os.listdir(config.PLANTVILLAGE_DIR):
            cls_path = os.path.join(config.PLANTVILLAGE_DIR, cls)
            if os.path.isdir(cls_path):
                total_imgs += sum(
                    1 for f in os.listdir(cls_path)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                )
        print(f"[Dataset] {n} class folders found, {total_imgs} total images.")
        return total_imgs > 0
    return False


def download_via_tfds():
    """
    Download PlantVillage via tensorflow_datasets and unpack it
    into data/plantvillage/<ClassName>/<image>.jpg
    """
    print("[Download] Trying tensorflow_datasets...")
    try:
        import tensorflow_datasets as tfds
    except ImportError:
        print("[Download] tensorflow_datasets not installed. Installing...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install",
                        "tensorflow_datasets", "--quiet"], check=True)
        import tensorflow_datasets as tfds

    try:
        print("[Download] Loading plant_village from TFDS (first run downloads ~1.3 GB)...")
        ds, info = tfds.load(
            "plant_village",
            split=["train"],
            with_info=True,
            as_supervised=True,
            data_dir=os.path.join(config.DATA_DIR, "_tfds_cache"),
        )
        label_names = info.features["label"].names
        print(f"[Download] {len(label_names)} classes in TFDS dataset.")

        from PIL import Image as PILImage
        import numpy as np

        for label_idx, label_name in enumerate(label_names):
            os.makedirs(os.path.join(config.PLANTVILLAGE_DIR, label_name), exist_ok=True)

        saved = 0
        for img_tensor, label_tensor in ds[0]:
            label_idx = int(label_tensor.numpy())
            label_name = label_names[label_idx]
            out_dir = os.path.join(config.PLANTVILLAGE_DIR, label_name)
            out_path = os.path.join(out_dir, f"{saved:07d}.jpg")
            if not os.path.exists(out_path):
                img_arr = img_tensor.numpy().astype("uint8")
                PILImage.fromarray(img_arr).save(out_path, quality=95)
            saved += 1
            if saved % 2000 == 0:
                print(f"  [TFDS] Saved {saved} images...")

        print(f"[Download] TFDS download complete. {saved} images saved.")
        return True

    except Exception as e:
        print(f"[Download] TFDS failed: {e}")
        return False


def show_manual_instructions():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║          PLANTVILLAGE DATASET – MANUAL DOWNLOAD REQUIRED            ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  Automatic download failed. Please download manually:               ║
║                                                                      ║
║  OPTION A – Kaggle (fastest, ~1.3 GB zip):                          ║
║  1. Create account at https://www.kaggle.com                        ║
║  2. Settings → API → "Create New Token" → saves kaggle.json         ║
║  3. Place kaggle.json in: C:\\Users\\AQSA\\.kaggle\\kaggle.json       ║
║  4. Open a terminal IN the project folder and run:                   ║
║                                                                      ║
║     venv\\Scripts\\python.exe -m kaggle datasets download \\          ║
║       -d abdallahalidev/plantvillage-dataset                         ║
║                                                                      ║
║  5. Unzip the downloaded file, then copy the folder that contains    ║
║     subfolders like "Apple___Apple_scab", "Tomato___Early_blight"   ║
║     into:  data\\plantvillage\\                                       ║
║                                                                      ║
║  OPTION B – Direct from Mendeley (~1.3 GB):                         ║
║  https://data.mendeley.com/datasets/tywbtsjrjv/1                    ║
║  Download "PlantVillage Dataset.zip", extract, and copy the         ║
║  class folders into:  data\\plantvillage\\                            ║
║                                                                      ║
║  WHAT THE FOLDER MUST LOOK LIKE:                                     ║
║  data\\plantvillage\\                                                 ║
║      Apple___Apple_scab\\      ← folder containing .jpg files        ║
║      Apple___Black_rot\\                                              ║
║      Tomato___Early_blight\\                                          ║
║      ... (38 total class folders)                                    ║
║                                                                      ║
║  After placing the data, re-run:                                     ║
║      venv\\Scripts\\python.exe download_and_train.py                  ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 – TRAIN
# ══════════════════════════════════════════════════════════════════════════════

def run_training(fast_mode: bool = False):
    """
    Import and run train() from the training script.
    If fast_mode=True, override epochs for a quicker run
    (useful for verifying the pipeline works before a full overnight train).
    """
    import config as cfg

    if fast_mode:
        # Override config values in-process for fast testing
        cfg.EPOCHS      = 5
        cfg.BATCH_SIZE  = 16
        print("[Training] FAST MODE: 5 epochs, batch=16 (for pipeline testing only)")
        print("           For full accuracy, run WITHOUT --fast flag.")
    else:
        print(f"[Training] Full training: {cfg.EPOCHS} epochs, batch={cfg.BATCH_SIZE}")
        print("[Training] Estimated time: 1-3 hours on CPU, 20-40 min on GPU")
        print("[Training] The script saves checkpoints — it is safe to interrupt and resume.")

    # Import and run the training script
    # We do this via importlib so we stay in the same process (venv stays active)
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "train_classifier",
        os.path.join(PROJECT_ROOT, "training", "train_classifier.py"),
    )
    train_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_mod)
    train_mod.train()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true",
                        help="Use 5 epochs for quick pipeline test (not for final model)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download attempt and go straight to training")
    args = parser.parse_args()

    print("\n" + "="*65)
    print("  LeafTrack-SG – Dataset Download & Classifier Training")
    print("="*65 + "\n")

    # ── Check if dataset is already in place ──────────────────────────────────
    if dataset_ready():
        print("[OK] PlantVillage dataset already in place. Skipping download.")
    elif args.skip_download:
        n = count_classes()
        if n == 0:
            print(f"[ERROR] No dataset found and --skip-download was set.")
            show_manual_instructions()
            sys.exit(1)
        print(f"[Warning] Only {n} class folders found — training will proceed anyway.")
    else:
        print("[Download] PlantVillage dataset not found. Attempting download...")
        success = download_via_tfds()
        if not success or not dataset_ready():
            show_manual_instructions()
            sys.exit(1)

    # ── Verify before training ────────────────────────────────────────────────
    n = count_classes()
    if n < 5:
        print(f"\n[ERROR] Only {n} class folders in data/plantvillage/.")
        print("        At least 5 are needed. Please check your dataset structure.")
        show_manual_instructions()
        sys.exit(1)

    print(f"\n[OK] Dataset ready: {n} classes found.")
    print(f"[OK] Model will be saved to: {config.CLASSIFIER_MODEL_PATH}")
    print(f"[OK] Class names will be saved to: {config.CLASS_NAMES_PATH}\n")

    # ── Run training ──────────────────────────────────────────────────────────
    run_training(fast_mode=args.fast)

    # ── Verify outputs ────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  Verifying output files...")
    print("="*65)

    model_ok     = os.path.exists(config.CLASSIFIER_MODEL_PATH)
    classes_ok   = os.path.exists(config.CLASS_NAMES_PATH)

    print(f"  efficientnet_classifier.keras : {'✓ EXISTS' if model_ok   else '✗ MISSING'}")
    print(f"  class_names.json             : {'✓ EXISTS' if classes_ok else '✗ MISSING'}")

    if model_ok and classes_ok:
        with open(config.CLASS_NAMES_PATH) as f:
            names = json.load(f)
        size_mb = os.path.getsize(config.CLASSIFIER_MODEL_PATH) / (1024*1024)
        print(f"\n  Classes: {len(names)}")
        print(f"  Model size: {size_mb:.1f} MB")
        print(f"\n{'='*65}")
        print("  ✓ Training complete! The website is ready to use.")
        print(f"{'='*65}")
        print("\n  Start the website:")
        print("    venv\\Scripts\\python.exe app.py")
        print("  Then open: http://localhost:5000\n")
    else:
        print("\n[ERROR] Training completed but output files are missing.")
        print("Check training_log.txt for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
