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

// real CPython lint via the backend, surfaced through Monaco's marker API
let _lintTimer = null;
function scheduleLint() {
  clearTimeout(_lintTimer);
  _lintTimer = setTimeout(runLint, 450);
}
function runLint() {
  if (!editor || !window.monaco) return;
  const model = editor.getModel();
  api("/api/lint", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ code: model.getValue() }),
  })
    .then((res) => {
      const markers = (res.annotations || []).map((a) => ({
        startLineNumber: a.line || 1,
        startColumn: a.col || 1,
        endLineNumber: a.line || 1,
        endColumn: (a.col || 1) + 1,
        message: a.message,
        severity: a.severity === "warning"
          ? monaco.MarkerSeverity.Warning
          : monaco.MarkerSeverity.Error,
      }));
      monaco.editor.setModelMarkers(model, "pylint", markers);
    })
    .catch(() => {});
}

// register theme + Python autocomplete (PyTorch API + keywords + in-file identifiers) once
let _monacoRegistered = false;
function registerMonaco() {
  if (_monacoRegistered) return;
  _monacoRegistered = true;
  monaco.editor.defineTheme("gym-dark", {
    base: "vs-dark",
    inherit: true,
    rules: [],
    colors: { "editor.background": "#0d1117" },
  });
  const KW = new Set(PY_KEYWORDS);
  monaco.languages.registerCompletionItemProvider("python", {
    provideCompletionItems(model, position) {
      const w = model.getWordUntilPosition(position);
      const range = {
        startLineNumber: position.lineNumber,
        endLineNumber: position.lineNumber,
        startColumn: w.startColumn,
        endColumn: w.endColumn,
      };
      const docWords = new Set(model.getValue().match(/[A-Za-z_][A-Za-z0-9_]*/g) || []);
      const pool = Array.from(new Set([...HINT_WORDS, ...docWords]));
      const suggestions = pool.map((label) => ({
        label,
        kind: KW.has(label)
          ? monaco.languages.CompletionItemKind.Keyword
          : monaco.languages.CompletionItemKind.Function,
        insertText: label,
        range,
      }));
      return { suggestions };
    },
  });
}

// ---- editor (Monaco — the VS Code engine) ----
function initEditor() {
  registerMonaco();
  editor = monaco.editor.create($("#code"), {
    value: "",
    language: "python",
    theme: "gym-dark",
    automaticLayout: true,
    fontSize: 13.5,
    fontFamily: 'ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace',
    tabSize: 4,
    insertSpaces: true,
    minimap: { enabled: true },
    scrollBeyondLastLine: true,
    smoothScrolling: true,
    cursorBlinking: "smooth",
    padding: { top: 10 },
    fixedOverflowWidgets: true,
    bracketPairColorization: { enabled: true },
  });
  editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.Enter, () => runTests());
  editor.onDidChangeModelContent(scheduleLint);
}

function whenMonacoReady() {
  return new Promise((resolve) => {
    if (window.monaco && window.__MONACO_READY) return resolve();
    document.addEventListener("monaco-ready", () => resolve(), { once: true });
  });
}
const getCode = () => (editor ? editor.getValue() : "");
const setCode = (v) => { if (editor) editor.setValue(v != null ? v : ""); };

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

// ---- "new functions in this task" panel ----
function renderApis(list) {
  const box = $("#apis");
  if (!box) return;
  if (!list || !list.length) { box.innerHTML = ""; return; }
  let html = '<div class="apis-card"><h4>🔧 New functions in this task</h4>';
  for (const a of list) {
    let item =
      '<div class="api">' +
      `<code class="api-sig">${escapeHtml(a.sig || a.name)}</code>` +
      `<div class="api-desc">${escapeHtml(a.desc || "")}</div>`;
    if (a.long) {
      item +=
        '<details class="api-more"><summary>More detail</summary>' +
        `<div class="api-long">${escapeHtml(a.long)}</div></details>`;
    }
    item += "</div>";
    html += item;
  }
  html += "</div>";
  box.innerHTML = html;
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
  renderApis(p.apis);
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

// render a structured test result: a clean summary + one card per failing test
function renderRunResult(res) {
  const c = $("#console");
  c.className = "";
  const s = res.summary || {};
  if (res.passed) {
    const n = s.passed || 0;
    c.innerHTML = `<span class="ok">✓ All ${n} test${n === 1 ? "" : "s"} passed</span>`;
    return;
  }
  let html = '<div class="run-summary">' +
    `<span class="err">✗ ${s.failed || 0} failed</span>` +
    (s.passed ? ` <span class="ok">· ${s.passed} passed</span>` : "") + "</div>";
  const fails = res.failures || [];
  if (fails.length) {
    for (const f of fails) {
      html +=
        '<div class="fail">' +
        `<span class="fail-test">✗ ${escapeHtml(f.test)}</span>` +
        (f.func ? ` <span class="fail-fn">— checks your <b>${escapeHtml(f.func)}()</b></span>` : "") +
        (f.location ? `<span class="fail-loc">${escapeHtml(f.location)}</span>` : "") +
        (f.message ? `<span class="fail-msg">${escapeHtml(f.message)}</span>` : "") +
        "</div>";
    }
  } else {
    html += `<pre class="raw-pre">${escapeHtml(res.output || "")}</pre>`;
  }
  html += `<details class="raw"><summary>Show raw pytest output</summary>` +
          `<pre>${escapeHtml(res.output || "")}</pre></details>`;
  c.innerHTML = html;
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
    renderRunResult(res);
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
  if (window.monaco) monaco.editor.setTheme(light ? "vs" : "gym-dark");
  localStorage.setItem("gym:theme", light ? "light" : "dark");
}

// ---- resizable splitters (sidebar | description | editor / console) ----
function initSplitters() {
  const layout = $("#layout"), sidebar = $("#sidebar"), desc = $("#desc"),
        work = $("#work"), consoleEl = $("#console");
  const refresh = () => { if (editor) editor.layout(); };

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
  await whenMonacoReady();
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
