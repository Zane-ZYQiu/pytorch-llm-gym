#!/usr/bin/env python3
"""PyTorch LLM Gym · local web server (pure standard library, zero extra dependencies)

Start:  python web/app.py
Then open http://127.0.0.1:8000 in your browser.

It reuses the repo's exercises/ and their tests/, running the code you write with real pytest
on your own machine.
"""
import json
import os
import re
import ast
import sys
import subprocess
import tempfile
import threading
import webbrowser
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STATIC = HERE / "static"
EXPLAINERS = HERE / "explainers"
APIS_FILE = HERE / "problem_apis.json"

# make `import practice` resolve against the repo root
sys.path.insert(0, str(ROOT))
import practice  # noqa: E402

HOST, PORT = "127.0.0.1", 8000
RUN_TIMEOUT = 60


# ----------------------------------------------------------------- data
def list_problems():
    out = []
    for d in practice.exercises():
        out.append({
            "id": d.name.split("_")[0],
            "dir": d.name,
            "title": practice.title_of(d),
            "level": practice.level_of(d),
        })
    return out


def apis_for(dirname):
    try:
        return json.loads(APIS_FILE.read_text(encoding="utf-8")).get(dirname, [])
    except Exception:
        return []


def get_problem(pid):
    d = practice.ex_by_id(pid)
    if not d:
        return None
    expl = EXPLAINERS / f"{d.name}.html"
    return {
        "id": d.name.split("_")[0],
        "dir": d.name,
        "title": practice.title_of(d),
        "level": practice.level_of(d),
        "description": practice.description(d),
        "starter": practice.starter_code(d),
        "hints": practice.hints_of(d),
        "explainer": expl.read_text(encoding="utf-8") if expl.exists() else None,
        "apis": apis_for(d.name),
        "has_solution": bool(practice.solution_code(d)),
    }


def get_solution(pid):
    d = practice.ex_by_id(pid)
    return practice.solution_code(d) if d else ""


def _parse_pytest(out, code=""):
    """Turn raw pytest output into a clean structured result the UI can render."""
    # top-level function/class names the student defined (to map a test -> their function)
    funcs = []
    try:
        tree = ast.parse(code)
        funcs = [n.name for n in tree.body
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
    except Exception:  # noqa: BLE001
        funcs = []

    def _count(word):
        m = re.search(r"(\d+) " + word, out)
        return int(m.group(1)) if m else 0

    summary = {"passed": _count("passed"), "failed": _count("failed"), "errors": _count("error")}

    failures = []
    start = re.search(r"=+ (FAILURES|ERRORS) =+", out)
    if start:
        endm = re.search(r"=+ short test summary", out)
        seg = out[start.end():endm.start() if endm else len(out)]
        parts = re.split(r"\n_{3,} (.+?) _{3,}\n", seg)  # [pre, name, body, name, body, ...]
        for i in range(1, len(parts), 2):
            name = parts[i].strip()
            body = parts[i + 1] if i + 1 < len(parts) else ""
            loc = None
            mloc = re.search(r"(exercises/\S+\.py):(\d+):", body)
            if mloc:
                loc = f"{mloc.group(1)}:{mloc.group(2)}"
            errs = [ln[1:].strip() for ln in body.splitlines() if ln.startswith("E ")]
            message = errs[-1] if errs else ""
            fn = next((f for f in funcs if f and f in name), None)
            failures.append({"test": name, "func": fn, "location": loc, "message": message})
    return summary, failures


def run_user_code(pid, code):
    d = practice.ex_by_id(pid)
    if not d:
        return {"ok": False, "error": "unknown problem"}
    tmp = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False,
                                         encoding="utf-8") as f:
            f.write(code)
            tmp = f.name
        env = os.environ.copy()
        env["PRACTICE_USER_CODE"] = tmp
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", str(d / "test.py"),
             "-q", "--no-header", "--tb=short", "--assert=plain", "-p", "no:cacheprovider"],
            cwd=str(ROOT), env=env, capture_output=True, text=True,
            timeout=RUN_TIMEOUT,
        )
        out = (proc.stdout + proc.stderr).strip()
        summary, failures = _parse_pytest(out, code)
        return {"ok": True, "passed": proc.returncode == 0,
                "output": out, "summary": summary, "failures": failures}
    except subprocess.TimeoutExpired:
        return {"ok": True, "passed": False, "output": f"Timed out (>{RUN_TIMEOUT}s)"}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": str(e)}
    finally:
        if tmp:
            try:
                os.unlink(tmp)
            except OSError:
                pass


def lint_code(code):
    """Syntax-check the code with real CPython. Returns annotations for the editor.
    Uses stdlib compile() for hard syntax errors; pyflakes (if installed) for warnings."""
    anns = []
    try:
        compile(code, "<editor>", "exec")
    except SyntaxError as e:
        anns.append({"line": e.lineno or 1, "col": e.offset or 1,
                     "message": f"SyntaxError: {e.msg}", "severity": "error"})
        return {"ok": True, "annotations": anns}
    except Exception as e:  # e.g. ValueError: source code contains null bytes
        anns.append({"line": 1, "col": 1,
                     "message": f"{type(e).__name__}: {e}", "severity": "error"})
        return {"ok": True, "annotations": anns}
    # optional richer checks (undefined names, unused imports, ...) if pyflakes exists
    try:
        import pyflakes.api  # noqa: WPS433

        class _Collector:
            def __init__(self):
                self.items = []

            def unexpectedError(self, *a):
                pass

            def syntaxError(self, *a):
                pass

            def flake(self, m):
                self.items.append({
                    "line": getattr(m, "lineno", 1),
                    "col": getattr(m, "col", 0) + 1,
                    "message": (m.message % m.message_args),
                    "severity": "warning",
                })

        c = _Collector()
        pyflakes.api.check(code, "<editor>", c)
        anns.extend(c.items)
    except Exception:  # noqa: BLE001 — pyflakes optional
        pass
    return {"ok": True, "annotations": anns}


# ----------------------------------------------------------------- http
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # keep it quiet
        pass

    def _send_json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path):
        if not path.exists() or not path.is_file():
            self.send_error(404)
            return
        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype + ("; charset=utf-8" if "text" in ctype or "javascript" in ctype else ""))
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/" or path == "/index.html":
            return self._send_file(STATIC / "index.html")
        if path.startswith("/static/"):
            return self._send_file(STATIC / path[len("/static/"):])
        if path == "/api/problems":
            return self._send_json({"problems": list_problems()})
        if path.startswith("/api/problems/") and path.endswith("/solution"):
            pid = path[len("/api/problems/"):-len("/solution")]
            return self._send_json({"code": get_solution(pid)})
        if path.startswith("/api/problems/"):
            pid = path[len("/api/problems/"):]
            prob = get_problem(pid)
            return self._send_json(prob) if prob else self._send_json({"error": "not found"}, 404)
        self.send_error(404)

    def _read_json(self):
        length = int(self.headers.get("Content-Length", 0))
        try:
            return json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return None

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/api/run":
            payload = self._read_json()
            if payload is None:
                return self._send_json({"ok": False, "error": "bad json"}, 400)
            return self._send_json(run_user_code(str(payload.get("id", "")), payload.get("code", "")))
        if path == "/api/lint":
            payload = self._read_json()
            if payload is None:
                return self._send_json({"ok": False, "error": "bad json"}, 400)
            return self._send_json(lint_code(payload.get("code", "")))
        self.send_error(404)


def main():
    n = len(practice.exercises())
    url = f"http://{HOST}:{PORT}"
    print(f"\n  PyTorch LLM Gym  ·  {n} problems ready")
    print(f"  Open in your browser:  {url}\n")
    try:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    except Exception:
        pass
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
