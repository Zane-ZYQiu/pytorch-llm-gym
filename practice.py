#!/usr/bin/env python3
"""PyTorch LLM Gym · interactive practice runner

Usage:
  python practice.py                # show progress + the next problem
  python practice.py list           # list all problems and their pass status
  python practice.py next           # show the next unsolved problem statement
  python practice.py show  <id>     # show a problem statement (e.g. show 03)
  python practice.py check [id]     # run tests; no id = run everything
  python practice.py hint  <id>     # reveal hints one by one
  python practice.py solution <id>  # show the reference solution
  python practice.py status         # progress bar

<id> can be a number (03) or a directory name (03_indexing).
"""
import sys
import os
import re
import ast
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EX_DIR = ROOT / "exercises"
SOL_DIR = ROOT / "solutions"
PROGRESS = ROOT / ".progress.json"
MARK = "# ===== reference solution ====="

USE_COLOR = sys.stdout.isatty()


def _c(s, code):
    return f"\033[{code}m{s}\033[0m" if USE_COLOR else str(s)


def GREEN(s): return _c(s, "32")
def RED(s):   return _c(s, "31")
def YEL(s):   return _c(s, "33")
def CYAN(s):  return _c(s, "36")
def MAG(s):   return _c(s, "35")
def BOLD(s):  return _c(s, "1")
def DIM(s):   return _c(s, "2")


# ---------------------------------------------------------------- discovery
def exercises():
    out = []
    if EX_DIR.exists():
        for d in sorted(EX_DIR.iterdir()):
            if d.is_dir() and re.match(r"\d+_", d.name):
                out.append(d)
    return out


def ex_by_id(eid):
    eid = str(eid).strip()
    cands = exercises()
    for d in cands:
        if d.name == eid:
            return d
    num = eid.zfill(2)
    for d in cands:
        if d.name.split("_")[0] == num:
            return d
    return None


def _docstring(d):
    try:
        return ast.get_docstring(ast.parse((d / "task.py").read_text())) or ""
    except Exception:
        return ""


def title_of(d):
    for line in _docstring(d).splitlines():
        line = line.strip()
        if line:
            return line.lstrip("# ").strip()
    return d.name


def level_of(d):
    m = re.search(r"Level:\s*(\d+)", _docstring(d))
    return m.group(1) if m else "?"


# ----- data accessors (shared by the CLI and the web backend) -----
def hints_of(d):
    sol = SOL_DIR / f"{d.name}.py"
    if not sol.exists():
        return []
    try:
        tree = ast.parse(sol.read_text())
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id == "HINTS":
                        return ast.literal_eval(node.value)
    except Exception:
        pass
    return []


def solution_code(d):
    sol = SOL_DIR / f"{d.name}.py"
    if not sol.exists():
        return ""
    text = sol.read_text()
    return text.split(MARK, 1)[1].strip() if MARK in text else text


def starter_code(d):
    return (d / "task.py").read_text()


def description(d):
    return _docstring(d)


# ---------------------------------------------------------------- progress
def load_progress():
    if PROGRESS.exists():
        try:
            return json.loads(PROGRESS.read_text())
        except Exception:
            return {}
    return {}


def save_progress(p):
    PROGRESS.write_text(json.dumps(p, indent=2, ensure_ascii=False))


def mark(d, passed):
    p = load_progress()
    rec = p.get(d.name, {})
    rec["passed"] = passed
    p[d.name] = rec
    save_progress(p)


# ---------------------------------------------------------------- tests
def run_tests(d, use_solution=False):
    env = os.environ.copy()
    if use_solution:
        env["PRACTICE_SOLUTION"] = "1"
    cmd = [sys.executable, "-m", "pytest", str(d / "test.py"),
           "-q", "--no-header", "--tb=short"]
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env,
                          capture_output=True, text=True)
    return proc.returncode == 0, (proc.stdout + proc.stderr)


def cmd_check(eid=None):
    if eid is None:
        targets = exercises()
    else:
        d = ex_by_id(eid)
        if not d:
            print(RED(f"Problem {eid} not found"))
            return
        targets = [d]

    for d in targets:
        passed, out = run_tests(d)
        mark(d, passed)
        tag = GREEN("✓ PASS") if passed else RED("✗ FAIL")
        print(f"{tag}  {BOLD(d.name)}  {DIM(title_of(d))}")
        if not passed and len(targets) == 1:
            print(DIM("-" * 60))
            # show the tail of pytest output (the useful part)
            lines = out.strip().splitlines()
            print("\n".join(lines[-40:]))
            print(DIM("-" * 60))
            print(f"Hint: {CYAN(f'python practice.py hint {d.name.split(chr(95))[0]}')}"
                  f"  Solution: {CYAN(f'python practice.py solution {d.name.split(chr(95))[0]}')}")
    if len(targets) > 1:
        cmd_status()


# ---------------------------------------------------------------- views
def cmd_list():
    p = load_progress()
    last_level = None
    for d in exercises():
        lv = level_of(d)
        if lv != last_level:
            print(BOLD(MAG(f"\n── Level {lv} " + "─" * 40)))
            last_level = lv
        st = p.get(d.name, {}).get("passed")
        if st is True:
            badge = GREEN("✓")
        elif st is False:
            badge = RED("✗")
        else:
            badge = DIM("·")
        num = d.name.split("_")[0]
        print(f"  {badge}  {BOLD(num)}  {title_of(d)}")
    print()
    cmd_status()


def cmd_status():
    exs = exercises()
    p = load_progress()
    done = sum(1 for d in exs if p.get(d.name, {}).get("passed"))
    total = len(exs)
    width = 30
    filled = int(width * done / total) if total else 0
    bar = GREEN("█" * filled) + DIM("░" * (width - filled))
    print(f"Progress {bar} {BOLD(f'{done}/{total}')}")


def cmd_show(eid):
    d = ex_by_id(eid)
    if not d:
        print(RED(f"Problem {eid} not found"))
        return
    print(BOLD(CYAN(f"\n{'='*64}")))
    print((d / "task.py").read_text().split('"""')[1].strip()
          if '"""' in (d / "task.py").read_text() else _docstring(d))
    print(BOLD(CYAN(f"{'='*64}")))
    print(f"\nEdit file: {YEL(str((d / 'task.py').relative_to(ROOT)))}")
    print(f"Run tests: {CYAN(f'python practice.py check {eid}')}\n")


def cmd_next():
    p = load_progress()
    for d in exercises():
        if not p.get(d.name, {}).get("passed"):
            cmd_show(d.name.split("_")[0])
            return
    print(GREEN("🎉 All passed! You've implemented every problem from scratch."))


def cmd_hint(eid):
    d = ex_by_id(eid)
    if not d:
        print(RED(f"Problem {eid} not found"))
        return
    hints = hints_of(d)
    if not hints:
        print(DIM("No hints for this problem. Try `solution` for the reference answer."))
        return
    for i, h in enumerate(hints, 1):
        print(f"{YEL(f'Hint {i}')}: {h}")


def cmd_solution(eid):
    d = ex_by_id(eid)
    if not d:
        print(RED(f"Problem {eid} not found"))
        return
    body = solution_code(d)
    if not body:
        print(RED("No reference solution available."))
        return
    print(BOLD(MAG(f"Reference solution · {d.name}")))
    print(DIM("(try writing it yourself first, then compare)\n"))
    print(body)


def cmd_default():
    print(BOLD(CYAN("PyTorch LLM Gym")))
    cmd_status()
    print(DIM("\nCommands: list / next / show <id> / check [id] / hint <id> / solution <id>"))
    print()
    cmd_next()


def main():
    args = sys.argv[1:]
    if not args:
        return cmd_default()
    cmd, rest = args[0], args[1:]
    if cmd in ("list", "ls"):
        cmd_list()
    elif cmd == "status":
        cmd_status()
    elif cmd == "next":
        cmd_next()
    elif cmd == "show":
        cmd_show(rest[0]) if rest else print(RED("Usage: show <id>"))
    elif cmd == "check":
        cmd_check(rest[0] if rest else None)
    elif cmd == "hint":
        cmd_hint(rest[0]) if rest else print(RED("Usage: hint <id>"))
    elif cmd in ("solution", "sol"):
        cmd_solution(rest[0]) if rest else print(RED("Usage: solution <id>"))
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
