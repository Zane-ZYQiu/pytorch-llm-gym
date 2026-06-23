"use strict";

const $ = (sel) => document.querySelector(sel);
const api = (p, opts) => fetch(p, opts).then((r) => r.json());

let problems = [];
let current = null;      // current problem object
let hintsShown = 0;
let editor = null;

// ---- local persistence (per-problem code + solved flag) ----
const codeKey = (dir) => `gym:code:${dir}`;
const solvedKey = (dir) => `gym:solved:${dir}`;
const isSolved = (dir) => localStorage.getItem(solvedKey(dir)) === "1";

// ---- editor: autocomplete vocabulary ----
const PY_KEYWORDS = [
  "def", "return", "if", "elif", "else", "for", "while", "in", "not", "and", "or",
  "import", "from", "as", "class", "with", "lambda", "None", "True", "False", "pass",
  "raise", "try", "except", "finally", "yield", "assert", "break", "continue",
  "global", "nonlocal", "is", "del", "print", "range", "len", "enumerate", "zip",
  "isinstance", "int", "float", "str", "list", "dict", "tuple", "set", "bool", "super",
];
const TORCH_API = [
  "torch", "nn", "Tensor", "tensor", "reshape", "view", "contiguous", "transpose",
  "permute", "matmul", "mm", "bmm", "einsum", "softmax", "log_softmax", "logsumexp",
  "gather", "scatter", "scatter_", "masked_fill", "triu", "tril", "arange", "zeros",
  "zeros_like", "ones", "ones_like", "full", "randn", "rand", "randint", "cat", "stack",
  "split", "chunk", "unbind", "unsqueeze", "squeeze", "expand", "repeat",
  "repeat_interleave", "mean", "var", "std", "sum", "max", "min", "maximum", "minimum",
  "sqrt", "rsqrt", "exp", "log", "pow", "sigmoid", "tanh", "clamp", "clamp_min", "norm",
  "normalize", "topk", "multinomial", "where", "sort", "cumsum", "argmax", "argmin",
  "flatten", "dim", "keepdim", "shape", "dtype", "device", "requires_grad_", "backward",
  "detach", "clone", "item", "numel", "size", "long", "values", "indices",
  "Linear", "Embedding", "RMSNorm", "LayerNorm", "Dropout", "Parameter", "Module",
  "ModuleList", "Sequential", "scaled_dot_product_attention", "cross_entropy", "silu",
  "gelu", "relu", "one_hot", "logsigmoid", "no_grad", "manual_seed", "erf",
];
const HINT_WORDS = Array.from(new Set([...PY_KEYWORDS, ...TORCH_API]));

function pythonHint(cm) {
  const cur = cm.getCursor();
  const line = cm.getLine(cur.line);
  let start = cur.ch;
  while (start > 0 && /[A-Za-z0-9_]/.test(line.charAt(start - 1))) start--;
  const word = line.slice(start, cur.ch);
  // words already present in the file (so your own functions/vars autocomplete too)
  const docWords = new Set();
  let m; const re = /[A-Za-z_][A-Za-z0-9_]*/g;
  while ((m = re.exec(cm.getValue()))) docWords.add(m[0]);
  const pool = Array.from(new Set([...HINT_WORDS, ...docWords]));
  const lw = word.toLowerCase();
  const list = pool
    .filter((w) => w !== word && (lw === "" || w.toLowerCase().startsWith(lw)))
    .sort();
  return { list, from: CodeMirror.Pos(cur.line, start), to: CodeMirror.Pos(cur.line, cur.ch) };
}

// async linter: real CPython syntax check via the backend
function pyLint(content, updateLinting, options, cm) {
  api("/api/lint", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ code: content }),
  })
    .then((res) => {
      const anns = (res.annotations || []).map((a) => {
        const ln = Math.max(0, (a.line || 1) - 1);
        const txt = cm.getLine(ln) || "";
        const col = Math.max(0, (a.col || 1) - 1);
        return {
          message: a.message,
          severity: a.severity === "warning" ? "warning" : "error",
          from: CodeMirror.Pos(ln, Math.min(col, txt.length)),
          to: CodeMirror.Pos(ln, Math.max(col + 1, txt.length)),
        };
      });
      updateLinting(anns);
    })
    .catch(() => updateLinting([]));
}

// ---- editor ----
function initEditor() {
  const ta = $("#code");
  if (!window.CodeMirror) return;
  editor = CodeMirror.fromTextArea(ta, {
    mode: "python",
    theme: "material-darker",
    lineNumbers: true,
    indentUnit: 4,
    tabSize: 4,
    indentWithTabs: false,
    scrollPastEnd: true,
    gutters: ["CodeMirror-lint-markers"],
    lint: { getAnnotations: pyLint, async: true, delay: 500 },
    extraKeys: {
      "Cmd-Enter": runTests,
      "Ctrl-Enter": runTests,
      "Ctrl-Space": (cm) => cm.showHint({ hint: pythonHint, completeSingle: false }),
      Tab: (cm) => cm.replaceSelection("    "),
    },
  });
  // pop the autocomplete dropdown as you type an identifier (or after a dot)
  editor.on("inputRead", (cm, change) => {
    if (cm.state.completionActive) return;
    const ch = (change.text && change.text[0]) || "";
    if (ch === ".") {
      cm.showHint({ hint: pythonHint, completeSingle: false });
      return;
    }
    if (/[A-Za-z_]/.test(ch)) {
      const cur = cm.getCursor(), line = cm.getLine(cur.line);
      let s = cur.ch;
      while (s > 0 && /[A-Za-z0-9_]/.test(line.charAt(s - 1))) s--;
      if (cur.ch - s >= 2) cm.showHint({ hint: pythonHint, completeSingle: false });
    }
  });
}
const getCode = () => (editor ? editor.getValue() : $("#code").value);
const setCode = (v) => (editor ? editor.setValue(v) : ($("#code").value = v));

// ---- sidebar ----
function renderSidebar() {
  const box = $("#problist");
  box.innerHTML = "";
  let lastLevel = null;
  let group = null;
  for (const p of problems) {
    if (p.level !== lastLevel) {
      group = document.createElement("div");
      group.className = "level-group";
      const lab = document.createElement("div");
      lab.className = "level-label";
      lab.textContent = `Level ${p.level}`;
      group.appendChild(lab);
      box.appendChild(group);
      lastLevel = p.level;
    }
    const el = document.createElement("div");
    el.className = "prob";
    el.dataset.id = p.id;
    el.dataset.dir = p.dir;
    el.innerHTML =
      `<span class="mark ${isSolved(p.dir) ? "solved" : "todo"}">${isSolved(p.dir) ? "✓" : "•"}</span>` +
      `<span class="num">${p.id}</span><span class="ttl"></span>`;
    el.querySelector(".ttl").textContent = stripNum(p.title);
    el.onclick = () => loadProblem(p.id);
    group.appendChild(el);
  }
  updateProgress();
}

const stripNum = (t) => t.replace(/^\d+\s*[·.\-]?\s*/, "");

function updateProgress() {
  const solved = problems.filter((p) => isSolved(p.dir)).length;
  $("#progress").innerHTML = `Solved <b>${solved}</b> / ${problems.length}`;
}

function markActive(id) {
  document.querySelectorAll(".prob").forEach((e) =>
    e.classList.toggle("active", e.dataset.id === id)
  );
}

// ---- inject explainer HTML and execute its <script> tags ----
function setHTMLWithScripts(el, html) {
  el.innerHTML = html || "";
  el.querySelectorAll("script").forEach((old) => {
    const s = document.createElement("script");
    if (old.src) s.src = old.src;
    else s.textContent = old.textContent;
    old.replaceWith(s);
  });
}

// ---- load a problem ----
async function loadProblem(id) {
  const p = await api(`/api/problems/${id}`);
  if (p.error) return;
  current = p;
  hintsShown = 0;
  markActive(id);
  $("#desc-level").textContent = `Level ${p.level}`;
  $("#desc-title").textContent = stripNum(p.title);
  setHTMLWithScripts($("#explainer"), p.explainer);
  $("#desc-body").innerHTML = window.marked
    ? marked.parse(p.description || "")
    : `<pre>${escapeHtml(p.description || "")}</pre>`;
  const saved = localStorage.getItem(codeKey(p.dir));
  setCode(saved != null ? saved : p.starter);
  $("#status").className = "status";
  $("#status").textContent = isSolved(p.dir) ? "✓ solved" : "";
  setConsole('Implement the TODOs, then click “Run tests”.', "dim");
}

function escapeHtml(s) {
  return s.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
}

function setConsole(text, cls) {
  const c = $("#console");
  c.className = "";
  c.innerHTML = cls ? `<span class="${cls}">${escapeHtml(text)}</span>` : escapeHtml(text);
}

// ---- run ----
async function runTests() {
  if (!current) return;
  const code = getCode();
  localStorage.setItem(codeKey(current.dir), code);
  $("#run").disabled = true;
  $("#status").className = "status run";
  $("#status").textContent = "running…";
  setConsole("Running pytest on your machine…", "dim");
  try {
    const res = await api("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id: current.id, code }),
    });
    if (!res.ok) {
      $("#status").className = "status fail";
      $("#status").textContent = "error";
      setConsole(res.error || "unknown error", "err");
      return;
    }
    const c = $("#console");
    c.className = "";
    c.innerHTML =
      `<span class="${res.passed ? "ok" : "err"}">${res.passed ? "✓ All tests passed" : "✗ Tests failed"}</span>\n\n` +
      escapeHtml(res.output || "");
    if (res.passed) {
      $("#status").className = "status pass";
      $("#status").textContent = "✓ passed";
      localStorage.setItem(solvedKey(current.dir), "1");
      const mark = document.querySelector(`.prob[data-id="${current.id}"] .mark`);
      if (mark) { mark.textContent = "✓"; mark.className = "mark solved"; }
      updateProgress();
    } else {
      $("#status").className = "status fail";
      $("#status").textContent = "✗ failed";
    }
  } catch (e) {
    $("#status").className = "status fail";
    $("#status").textContent = "error";
    setConsole(String(e), "err");
  } finally {
    $("#run").disabled = false;
  }
}

// ---- hint / reset / solution ----
function showHint() {
  if (!current || !current.hints || !current.hints.length) {
    setConsole("No hints for this problem.", "dim");
    return;
  }
  if (hintsShown >= current.hints.length) hintsShown = 0;
  hintsShown += 1;
  const lines = current.hints
    .slice(0, hintsShown)
    .map((h, i) => `Hint ${i + 1}: ${h}`)
    .join("\n\n");
  const more = hintsShown < current.hints.length ? `\n\n(${current.hints.length - hintsShown} more — click again)` : "";
  setConsole(lines + more, "dim");
}

function resetCode() {
  if (!current) return;
  setCode(current.starter);
  localStorage.removeItem(codeKey(current.dir));
  setConsole("Reset to starter code.", "dim");
}

async function showSolution() {
  if (!current) return;
  const res = await api(`/api/problems/${current.id}/solution`);
  const c = $("#console");
  c.className = "";
  c.innerHTML =
    `<span class="dim">Reference solution — ${escapeHtml(stripNum(current.title))}:</span>\n\n` +
    escapeHtml(res.code || "No solution available.");
}

// ---- navigation ----
function gotoOffset(delta) {
  if (!current) return;
  const idx = problems.findIndex((p) => p.id === current.id);
  if (idx < 0) return;
  const nxt = problems[idx + delta];
  if (nxt) loadProblem(nxt.id);
}

// ---- theme ----
function applyTheme(light) {
  document.body.classList.toggle("light", light);
  const btn = $("#theme");
  if (btn) btn.textContent = light ? "☀️" : "🌙";
  if (editor) editor.setOption("theme", light ? "default" : "material-darker");
  localStorage.setItem("gym:theme", light ? "light" : "dark");
}

// ---- resizable splitters (sidebar | description | editor / console) ----
function initSplitters() {
  const layout = $("#layout"), sidebar = $("#sidebar"), desc = $("#desc"),
        work = $("#work"), consoleEl = $("#console");
  const refresh = () => { if (editor) editor.refresh(); };

  // restore saved sizes
  const sw = localStorage.getItem("gym:sidebarWidth");
  if (sw && sidebar) sidebar.style.width = sw;
  const db = localStorage.getItem("gym:descBasis");
  if (db && desc) { desc.style.flexBasis = db; desc.style.flexGrow = "0"; if (work) work.style.flexGrow = "1"; }
  const ch = localStorage.getItem("gym:consoleHeight");
  if (ch && consoleEl) consoleEl.style.height = ch;

  function makeDrag(handle, onMove, onEnd) {
    if (!handle) return;
    let dragging = false;
    handle.addEventListener("mousedown", (e) => {
      dragging = true; handle.classList.add("dragging");
      document.body.style.userSelect = "none"; e.preventDefault();
    });
    window.addEventListener("mousemove", (e) => { if (dragging) onMove(e); });
    window.addEventListener("mouseup", () => {
      if (!dragging) return;
      dragging = false; handle.classList.remove("dragging");
      document.body.style.userSelect = ""; if (onEnd) onEnd();
    });
  }

  // sidebar | description
  makeDrag($("#splitter-left"), (e) => {
    const r = layout.getBoundingClientRect();
    const w = Math.max(150, Math.min(440, e.clientX - r.left));
    sidebar.style.width = w + "px"; refresh();
  }, () => localStorage.setItem("gym:sidebarWidth", sidebar.style.width));

  // description | editor
  makeDrag($("#splitter"), (e) => {
    const r = layout.getBoundingClientRect();
    const left = r.left + sidebar.offsetWidth;
    const max = r.width - sidebar.offsetWidth - 360;
    const w = Math.max(280, Math.min(max, e.clientX - left));
    desc.style.flexBasis = w + "px"; desc.style.flexGrow = "0"; work.style.flexGrow = "1"; refresh();
  }, () => { if (desc.style.flexBasis) localStorage.setItem("gym:descBasis", desc.style.flexBasis); });

  // editor | console (vertical)
  makeDrag($("#splitter-h"), (e) => {
    const r = work.getBoundingClientRect();
    const h = Math.max(80, Math.min(r.height - 160, r.bottom - e.clientY));
    consoleEl.style.height = h + "px"; refresh();
  }, () => localStorage.setItem("gym:consoleHeight", consoleEl.style.height));
}

// ---- boot ----
async function boot() {
  initEditor();
  applyTheme(localStorage.getItem("gym:theme") === "light");
  initSplitters();
  const data = await api("/api/problems");
  problems = data.problems || [];
  renderSidebar();
  $("#run").onclick = runTests;
  $("#reset").onclick = resetCode;
  $("#hint").onclick = showHint;
  $("#solution").onclick = showSolution;
  $("#prev").onclick = () => gotoOffset(-1);
  $("#next-prob").onclick = () => gotoOffset(1);
  $("#theme").onclick = () => applyTheme(!document.body.classList.contains("light"));
  if (problems.length) loadProblem(problems[0].id);
}

boot();
