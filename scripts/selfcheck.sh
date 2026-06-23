#!/usr/bin/env bash
# Verify every reference solution passes its own test suite.
set -u
cd "$(dirname "$0")/.."
pass=0; fail=0; failed=""
for d in exercises/*/; do
  name=$(basename "$d")
  out=$(PRACTICE_SOLUTION=1 python -m pytest "$d/test.py" -q --no-header -p no:cacheprovider 2>&1 | tail -1)
  if echo "$out" | grep -qiE "failed|error"; then
    fail=$((fail+1)); failed="$failed $name"
  else
    pass=$((pass+1))
  fi
done
echo "reference solutions passing: $pass / $((pass+fail))"
if [ -n "$failed" ]; then echo "FAILED:$failed"; exit 1; fi
echo "OK"
