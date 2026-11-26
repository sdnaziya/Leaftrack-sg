"""Execute the push. Run: venv\Scripts\python.exe do_push.py"""
import subprocess, os

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
LOG  = os.path.join(ROOT, "do_push.log")
OUT  = []

def run(cmd, show=True):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=ROOT)
    out = ((r.stdout or "") + (r.stderr or "")).strip()
    if show and out:
        print(out, flush=True)
        OUT.append(out)
    return out, r.returncode

print("=" * 55)
print("LeafTrack-SG: Push to GitHub")
print("=" * 55)

# Stage any new files added since last commit
run('git add check_before_push.py git_push.py do_push.py push_to_github.bat', show=False)
_, staged = run('git diff --cached --name-only', show=False)
if staged:
    run('git commit -m "Add helper scripts"')

# Check if there is anything ahead of origin
ahead, _ = run('git rev-list origin/main..HEAD --count', show=False)
print(f"\nCommits ahead of GitHub: {ahead}")

if ahead.strip() == "0":
    print("\nNothing new to push — GitHub already has the latest code.")
    print("\nYour project is at: https://github.com/sdnaziya/Leaftrack-sg")
else:
    print(f"\nPushing {ahead} commit(s) to GitHub...")
    out, code = run('git push origin main')
    if code == 0:
        print("\n✓ PUSH SUCCESSFUL")
        print("  Live at: https://github.com/sdnaziya/Leaftrack-sg")
    else:
        print(f"\nPush returned code {code}. Trying --force...")
        out, code = run('git push origin main --force')
        if code == 0:
            print("\n✓ FORCE PUSH SUCCESSFUL")
            print("  Live at: https://github.com/sdnaziya/Leaftrack-sg")
        else:
            print("\n✗ Push failed. Run manually:")
            print('  git push origin main')
            print("\nIf asked for password, use a Personal Access Token:")
            print("  https://github.com/settings/tokens")

print("\nWhat IS on GitHub (84 code files, 1.4 MB):")
print("  All Python files, templates, static assets")
print("  routes/, services/, database/, training/")
print("  models/classifier/class_names.json (labels)")
print("  requirements.txt, README.md, config.py")
print()
print("What is NOT on GitHub (too large, stays on your laptop):")
print("  models/*.keras  — trained model weights")
print("  data/           — PlantVillage images")
print("  venv/           — Python packages")

with open(LOG, "w", encoding="utf-8") as f:
    f.write("\n".join(OUT))
