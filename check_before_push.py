"""Check what will be pushed. Run: venv\Scripts\python.exe check_before_push.py"""
import subprocess, os, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)

def run(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=ROOT)
    return ((r.stdout or "") + (r.stderr or "")).strip()

LOG = os.path.join(ROOT, "push_check.log")
OUT = []
def p(s=""): OUT.append(str(s)); print(s, flush=True)

p("=" * 60)
p("Pre-push check")
p("=" * 60)

p("\nGit status:")
p(run("git status"))

p("\nRemote:")
p(run("git remote -v"))

p("\nLast 5 commits:")
p(run("git log --oneline -5"))

p("\nFiles tracked by git that exist on disk:")
tracked = run("git ls-files").splitlines()
total_mb = 0
large = []
for f in tracked:
    fp = os.path.join(ROOT, f)
    if os.path.exists(fp):
        mb = os.path.getsize(fp) / 1024 / 1024
        total_mb += mb
        if mb > 50:
            large.append((f, mb))

p(f"  Total tracked files: {len(tracked)}")
p(f"  Total size: {total_mb:.1f} MB")

if large:
    p(f"\n  !! LARGE FILES (>50 MB) — will cause push failure:")
    for f, mb in large:
        p(f"     {mb:.1f} MB  {f}")
else:
    p(f"\n  No files >50 MB. Safe to push.")

p("\nSample tracked files:")
for f in sorted(tracked)[:40]:
    p(f"  {f}")

p("\nFiles in .gitignore (excluded):")
ignored_check = run("git check-ignore -v plantvillage-dataset.zip 2>&1")
p(f"  plantvillage-dataset.zip: {ignored_check if ignored_check else 'NOT ignored - fix needed'}")

with open(LOG, "w", encoding="utf-8") as f:
    f.write("\n".join(OUT))
p(f"\nLog saved to: {LOG}")
p("\nIf no large files found, run: push_to_github.bat")
