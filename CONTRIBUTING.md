# Contributing

Thanks for helping make PyTorch LLM Gym better! The most valuable contributions are
**new problems** and **interactive explainers**.

## Add a new problem

A problem is three files (the `<id>` is a zero-padded number, `<name>` is snake_case):

```
exercises/<id>_<name>/task.py     # docstring (statement) + function stubs with `raise NotImplementedError`
exercises/<id>_<name>/test.py     # pytest tests
solutions/<id>_<name>.py          # HINTS = [...] + a `# ===== reference solution =====` marker + the reference impl
```

Conventions:
- **`task.py` docstring**: the first line is the title `NN · Title`; include a `Level: N` line; end with `## Your task` and `Run: python practice.py check NN`.
- **`test.py`**: start with
  ```python
  import sys, pathlib
  sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
  from practharness import load_task
  task = load_task(__file__)
  ```
  Then write tests against `task.<your_function>`. Prefer asserting **real properties** (compare to a
  reference computed with PyTorch, or check an invariant) rather than hard-coding outputs.
- **`solutions/<id>_<name>.py`**: a `HINTS` list (progressive, code-snippet hints) above the
  `# ===== reference solution =====` marker, then the working implementation.

The new problem then shows up automatically in both the CLI and the web app.

### Verify

```bash
# your reference solution must pass its own tests:
PRACTICE_SOLUTION=1 python -m pytest exercises/<id>_<name>/test.py -q
# the starter (task.py) must fail cleanly:
python practice.py check <id>
```

All content must be in **English**.

## Add an interactive explainer (optional but loved)

Drop a self-contained widget at `web/explainers/<id>_<name>.html`. It's injected into the problem
panel. Follow the existing ones (`07_softmax.html`, `13_attention.html`, `19_transformer_block.html`):

- One outer `<div class="viz" id="vizNN">…</div>` + one `<script>(function(){ … })();</script>` IIFE.
- Scope every element id with a `vNN-` prefix; query only within your widget.
- No external libraries or CDNs — plain JS/SVG/canvas/DOM. Must be idempotent (it may be re-run).
- Reuse the shared `.viz` / `.row` / `.val` / `.btn` classes; English only.

## Style

Match the surrounding code. Keep solutions readable and idiomatic — they double as teaching material.
