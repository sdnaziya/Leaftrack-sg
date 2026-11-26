"""
LeafTrack-SG – Environment Check & Auto-Installer
Run this with the venv Python:
    venv\Scripts\python.exe check_env.py
Or from PowerShell (bypasses execution policy):
    powershell -Command "& 'venv\Scripts\python.exe' check_env.py"
"""
import sys
import subprocess

PY = sys.executable
print(f"Python: {sys.version}")
print(f"Executable: {PY}\n")

PACKAGES = [
    # (pip_name, import_name, version_attr)
    ("flask",          "flask",      "__version__"),
    ("flask-cors",     "flask_cors", "__version__"),
    ("numpy",          "numpy",      "__version__"),
    ("Pillow",         "PIL",        "__version__"),
    ("opencv-python",  "cv2",        "__version__"),
    ("scikit-learn",   "sklearn",    "__version__"),
    ("matplotlib",     "matplotlib", "__version__"),
    ("seaborn",        "seaborn",    "__version__"),
    ("tqdm",           "tqdm",       "__version__"),
    ("reportlab",      "reportlab",  "Version"),
    ("requests",       "requests",   "__version__"),
    ("pandas",         "pandas",     "__version__"),
    ("pytest",         "pytest",     "__version__"),
]

TF_PACKAGES = [
    ("tensorflow==2.17.0", "tensorflow", "__version__"),
]

def check_or_install(pip_name, import_name, version_attr):
    try:
        mod = __import__(import_name)
        ver = getattr(mod, version_attr, "?")
        print(f"  ✓  {import_name:<20} {ver}")
        return True
    except ImportError:
        print(f"  ✗  {import_name:<20} NOT FOUND — installing {pip_name}...")
        result = subprocess.run(
            [PY, "-m", "pip", "install", pip_name, "--quiet"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            try:
                # Re-import after install
                import importlib
                mod = importlib.import_module(import_name)
                ver = getattr(mod, version_attr, "?")
                print(f"       → Installed: {ver}")
                return True
            except Exception as e:
                print(f"       → Installed but import still failed: {e}")
                return False
        else:
            print(f"       → pip install failed: {result.stderr[:200]}")
            return False

print("=" * 55)
print("  Checking / Installing Core Packages")
print("=" * 55)
all_ok = True
for pkg in PACKAGES:
    ok = check_or_install(*pkg)
    if not ok:
        all_ok = False

print()
print("=" * 55)
print("  Checking / Installing TensorFlow")
print("=" * 55)
print("  (This may take several minutes — TF is ~500 MB)")
tf_ok = False
for pip_name, import_name, version_attr in TF_PACKAGES:
    tf_ok = check_or_install(pip_name, import_name, version_attr)

if not tf_ok:
    print("  Trying tensorflow-cpu as fallback...")
    tf_ok = check_or_install("tensorflow-cpu", "tensorflow", "__version__")

if not tf_ok:
    print("""
  ⚠  TensorFlow could not be installed on Python 3.14.
  
  Options:
  1. Install Python 3.12 from https://www.python.org/downloads/
     Then re-create the venv with Python 3.12 and re-run this script.
  
  2. The Flask website WILL still run — but AI inference endpoints 
     (/api/predict) will return an error until TF is available.
""")

print()
print("=" * 55)
print("  Summary")
print("=" * 55)
print(f"  Core packages : {'✓ All OK' if all_ok else '✗ Some failed'}")
print(f"  TensorFlow    : {'✓ OK' if tf_ok else '✗ Not available'}")
print()

if all_ok:
    print("  Flask website is ready to run!")
    print("  Start it with:")
    print("    venv\\Scripts\\python.exe app.py")
    print("  Then open: http://localhost:5000")
