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

// ---- editor ----
function initEditor() {
  const ta = $("#code");
  if (window.CodeMirror) {
    editor = CodeMirror.fromTextArea(ta, {
      mode: "python",
      theme: "material-darker",
      lineNumbers: true,
      indentUnit: 4,
      tabSize: 4,
      indentWithTabs: false,
      extraKeys: {
        "Cmd-Enter": runTests,
        "Ctrl-Enter": runTests,
        Tab: (cm) => cm.replaceSelection("    "),
      },
    });
  }
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

// ---- resizable splitter (description | editor) ----
function initSplitter() {
  const sp = $("#splitter"), desc = $("#desc"), work = $("#work"),
        layout = $("#layout"), sidebar = $("#sidebar");
  if (!sp || !desc || !work) return;
  const saved = localStorage.getItem("gym:descBasis");
  if (saved) { desc.style.flexBasis = saved; desc.style.flexGrow = "0"; work.style.flexGrow = "1"; }
  let dragging = false;
  sp.addEventListener("mousedown", (e) => {
    dragging = true; sp.classList.add("dragging");
    document.body.style.userSelect = "none"; e.preventDefault();
  });
  window.addEventListener("mousemove", (e) => {
    if (!dragging) return;
    const rect = layout.getBoundingClientRect();
    const left = rect.left + (sidebar ? sidebar.offsetWidth : 0);
    const max = rect.width - (sidebar ? sidebar.offsetWidth : 0) - 360;
    const w = Math.max(280, Math.min(max, e.clientX - left));
    desc.style.flexBasis = w + "px"; desc.style.flexGrow = "0";
    work.style.flexGrow = "1";
  });
  window.addEventListener("mouseup", () => {
    if (!dragging) return;
    dragging = false; sp.classList.remove("dragging"); document.body.style.userSelect = "";
    if (desc.style.flexBasis) localStorage.setItem("gym:descBasis", desc.style.flexBasis);
  });
}

// ---- boot ----
async function boot() {
  initEditor();
  applyTheme(localStorage.getItem("gym:theme") === "light");
  initSplitter();
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
