"""Shared test harness helpers.

Each exercise's test file uses `load_task(__file__)` to import the module under
test. By default it loads the student's `task.py`. When the env var
`PRACTICE_SOLUTION=1` is set, it loads the reference solution instead — this is
how we self-verify that every reference solution passes its own tests.
"""
import os
import importlib.util
import pathlib


def load_task(test_file):
    here = pathlib.Path(test_file).resolve().parent
    name = here.name  # e.g. "01_tensor_basics"
    user_code = os.environ.get("PRACTICE_USER_CODE")
    if user_code:
        # The web backend writes the submitted code to a temp file; load it to run the tests.
        path = pathlib.Path(user_code)
        modname = f"user_{name}"
    elif os.environ.get("PRACTICE_SOLUTION"):
        path = here.parents[1] / "solutions" / f"{name}.py"
        modname = f"sol_{name}"
    else:
        path = here / "task.py"
        modname = f"task_{name}"
    spec = importlib.util.spec_from_file_location(modname, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
