"""Run this to check all Python files for syntax errors:
   venv\Scripts\python.exe check_syntax.py
"""
import ast, os, sys

root = os.path.dirname(os.path.abspath(__file__))
errors = []
ok = []

for dirpath, dirs, files in os.walk(root):
    dirs[:] = [d for d in dirs if d not in ('venv', '__pycache__', '.git',
                                              'node_modules', '.idea')]
    for fname in files:
        if not fname.endswith('.py'):
            continue
        path = os.path.join(dirpath, fname)
        rel  = os.path.relpath(path, root)
        try:
            with open(path, encoding='utf-8') as f:
                src = f.read()
            ast.parse(src, filename=rel)
            ok.append(rel)
        except SyntaxError as e:
            errors.append(f"  SYNTAX ERROR  {rel}  line {e.lineno}: {e.msg}")
        except Exception as e:
            errors.append(f"  READ ERROR    {rel}: {e}")

print(f"\nLeafTrack-SG Python Syntax Check")
print(f"{'='*50}")
print(f"Files checked : {len(ok) + len(errors)}")
print(f"Clean         : {len(ok)}")
print(f"Errors        : {len(errors)}")
if errors:
    print("\nFailed files:")
    for e in errors:
        print(e)
    sys.exit(1)
else:
    print("\nAll files are syntactically valid.")
