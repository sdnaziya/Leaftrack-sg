"""
LeafTrack-SG GitHub Push Script
Inspects, stages, commits and pushes the project to GitHub.
Run: venv\Scripts\python.exe git_push.py
"""
import subprocess, os, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)

def run(cmd, capture=True):
    r = subprocess.run(cmd, shell=True, capture_output=capture,
                       text=True, cwd=ROOT)
    out = (r.stdout or "") + (r.stderr or "")
    return out.strip(), r.returncode

def p(s=""): print(s, flush=True)

p("=" * 60)
p("LeafTrack-SG GitHub Push Script")
p("=" * 60)

# ── 1. Check git status ───────────────────────────────────────────
p("\n[1] Current git state:")
out, _ = run("git status")
p(out)

out, _ = run("git remote -v")
p("\nRemotes:")
p(out)

out, _ = run("git log --oneline -5")
p("\nLast 5 commits:")
p(out)

# ── 2. Find files larger than 90 MB ──────────────────────────────
p("\n[2] Checking for large files (>90 MB)...")
large = []
for dirpath, dirs, files in os.walk(ROOT):
    # skip git internals and venv
    dirs[:] = [d for d in dirs if d not in ('.git', 'venv', '__pycache__')]
    for f in files:
        fp = os.path.join(dirpath, f)
        try:
            sz = os.path.getsize(fp)
            if sz > 90 * 1024 * 1024:
                rel = os.path.relpath(fp, ROOT)
                large.append((rel, round(sz / 1024 / 1024, 1)))
        except: pass

if large:
    p("  LARGE FILES FOUND:")
    for name, mb in large:
        p(f"  !! {mb} MB  {name}")
else:
    p("  No files > 90 MB found. Safe to push.")

# ── 3. Update .gitignore to exclude large/unnecessary files ───────
p("\n[3] Updating .gitignore...")
gitignore_content = """# Virtual environment
venv/
env/

# Python cache
__pycache__/
*.pyc
*.pyo

# Large dataset and zip files
*.zip
*.tar.gz
data/plantvillage/
data/plantdoc/
data/_tfds_cache/
data/segmentation/images/
data/segmentation/masks/

# Trained model weights (large binary — retrain on server)
models/classifier/*.keras
models/segmentation/*.keras

# User uploads and reports
uploads/
reports/

# Database
database/leaftrack.db
*.db

# Log files
*.log
training_history.csv

# OS / IDE
.DS_Store
Thumbs.db
.vscode/
.idea/
*.swp
*.bak
*.tmp
"""
with open(".gitignore", "w", encoding="utf-8") as f:
    f.write(gitignore_content)
p("  .gitignore written.")

# ── 4. Remove any large files from git index ──────────────────────
p("\n[4] Removing large/excluded files from git index (not from disk)...")
for name, mb in large:
    out, code = run(f'git rm --cached "{name}" --ignore-unmatch')
    p(f"  Removed from index: {name} ({mb} MB)")

# Also ensure the zip specifically is removed
run('git rm --cached plantvillage-dataset.zip --ignore-unmatch')

# ── 5. Stage all project files ────────────────────────────────────
p("\n[5] Staging all project files...")
out, code = run("git add -A")
p(f"  git add result: {out if out else 'OK'} (code {code})")

# Show what will be committed
out, _ = run("git status --short")
lines = [l for l in out.splitlines() if l.strip()]
p(f"\n  Files staged: {len(lines)}")
for l in lines[:30]:
    p(f"  {l}")
if len(lines) > 30:
    p(f"  ... and {len(lines)-30} more")

# ── 6. Commit ─────────────────────────────────────────────────────
p("\n[6] Committing...")
out, code = run('git commit -m "Initial LeafTrack-SG Project Upload"')
p(out if out else "Nothing new to commit.")

# ── 7. Push ───────────────────────────────────────────────────────
p("\n[7] Pushing to GitHub (origin main)...")
p("  This may take 1-2 minutes depending on your connection.")
p("  If it asks for credentials:")
p("  - Username: sdnaziya")
p("  - Password: your Personal Access Token from github.com/settings/tokens")
p()

out, code = run("git push origin main", capture=False)
if code == 0:
    p("\n✓ PUSH SUCCESSFUL")
    p("  Your project is live at: https://github.com/sdnaziya/Leaftrack-sg")
else:
    p(f"\n✗ Push failed (exit code {code})")
    p("  Run this manually in Command Prompt:")
    p('  git push origin main')
    p()
    p("  If still failing, try:")
    p('  git push origin main --force')

p("\nDone.")
