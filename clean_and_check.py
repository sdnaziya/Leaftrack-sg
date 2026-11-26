"""
Clean empty junk folders and check syntax.
Run: venv\Scripts\python.exe clean_and_check.py
"""
import os, sys, shutil, ast
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

ROOT = os.path.dirname(os.path.abspath(__file__))
pv = config.PLANTVILLAGE_DIR

print("=" * 60)
print("STEP 1 – Clean empty/junk dataset folders")
print("=" * 60)

JUNK = {"color", "grayscale", "segmented"}
if os.path.isdir(pv):
    removed = []
    for d in os.listdir(pv):
        dp = os.path.join(pv, d)
        if not os.path.isdir(dp): continue
        imgs = [f for f in os.listdir(dp) if f.lower().endswith(('.jpg','.jpeg','.png'))]
        if len(imgs) == 0 or d in JUNK:
            shutil.rmtree(dp)
            removed.append(d)
            print(f"  Removed empty folder: {d}")
    if not removed:
        print("  No empty folders to remove.")

    # Print what's left
    classes = sorted([d for d in os.listdir(pv) if os.path.isdir(os.path.join(pv, d))])
    total = 0
    print(f"\n  Remaining: {len(classes)} class folders")
    for c in classes:
        n = len([f for f in os.listdir(os.path.join(pv, c))
                 if f.lower().endswith(('.jpg','.jpeg','.png'))])
        total += n
        print(f"    [{n:>5}]  {c}")
    print(f"\n  Total images: {total}")
else:
    print(f"  Dataset directory not found: {pv}")

print()
print("=" * 60)
print("STEP 2 – Python syntax check")
print("=" * 60)

errors = []
ok = 0
for dirpath, dirs, files in os.walk(ROOT):
    dirs[:] = [d for d in dirs if d not in ('venv', '__pycache__', '.git')]
    for f in files:
        if not f.endswith('.py'): continue
        path = os.path.join(dirpath, f)
        try:
            with open(path, encoding='utf-8') as fh:
                ast.parse(fh.read())
            ok += 1
        except SyntaxError as e:
            errors.append(f"  FAIL {path}  line {e.lineno}: {e.msg}")

print(f"  Files checked: {ok + len(errors)}")
if errors:
    for e in errors: print(e)
    print(f"\n  {len(errors)} SYNTAX ERROR(S) FOUND")
else:
    print("  All files valid — 0 syntax errors")

print()
print("=" * 60)
print("WHAT TO DO NEXT")
print("=" * 60)
print()
print("Your dataset currently has ONLY 3 crops (Pepper, Potato, Tomato).")
print("Apple, Grape, Corn, Cherry, Peach, Strawberry, etc. are NOT present.")
print()
print("To add them, you must download the full PlantVillage dataset.")
print("Run ONE of these options:")
print()
print("Option A — Automatic (tries tensorflow_datasets, ~1.3 GB):")
print("  venv\\Scripts\\python.exe download_full_dataset.py")
print()
print("Option B — Manual Kaggle download:")
print("  1. Create a Kaggle account at kaggle.com")
print("  2. Settings → API → Create New Token → saves kaggle.json")
print("  3. Place kaggle.json in: C:\\Users\\AQSA\\.kaggle\\")
print("  4. Run:")
print("     venv\\Scripts\\python.exe -m kaggle datasets download -d abdallahalidev/plantvillage-dataset")
print("  5. Unzip the downloaded file into data\\plantvillage\\")
print()
print("Option C — Mendeley (browser download, no account needed):")
print("  1. Go to: https://data.mendeley.com/datasets/tywbtsjrjv/1")
print("  2. Click 'Download All' (~1.3 GB zip file)")
print("  3. Extract all class folders into: data\\plantvillage\\")
print()
print("After downloading:")
print("  venv\\Scripts\\python.exe clean_and_check.py    ← verify everything looks right")
print("  venv\\Scripts\\python.exe retrain.py            ← retrain with all classes")
print("  venv\\Scripts\\python.exe app.py                ← start website")
