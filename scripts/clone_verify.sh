#!/usr/bin/env bash
# Clone-and-run proof: a clueless developer clones with NOTHING and gets a working
# diagnosis + live web app, step by step, in a clean git worktree. Every step is
# the literal command a cloner would run. Exits non-zero on the first failure.
#
#   bash scripts/clone_verify.sh
#
# Needs: git, uv, ffmpeg, yt-dlp on PATH. The ~2GB CLAP weights are reused from the
# shared HuggingFace cache (no re-download). No scrape/train-from-scratch is run.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && git rev-parse --show-toplevel)"
WT="${TMPDIR:-/tmp}/cardiag_clone_verify"
PASS=0; FAIL=0
step(){ printf "\n\033[1m▶ %s\033[0m\n" "$1"; }
ok(){   printf "  \033[32m✓ %s\033[0m\n" "$1"; PASS=$((PASS+1)); }
bad(){  printf "  \033[31m✗ %s\033[0m\n" "$1"; FAIL=$((FAIL+1)); }
run(){  if eval "$1" >/tmp/cv.out 2>&1; then ok "$2"; else bad "$2 — see below"; tail -8 /tmp/cv.out|sed 's/^/    /'; fi; }

step "0. fresh worktree at HEAD (a clone with nothing)"
cd "$ROOT"
git worktree remove --force "$WT" 2>/dev/null || true
git worktree add --detach "$WT" HEAD >/tmp/cv.out 2>&1 && ok "worktree at $WT" || { bad "worktree add"; exit 1; }
cd "$WT"
# a fresh clone has no scraped data and no user-trained model
rm -rf data/training 2>/dev/null || true

step "1. install (uv venv + editable install with extras)"
uv venv --python 3.11 .venv >/tmp/cv.out 2>&1 && ok "venv" || bad "venv"
source .venv/bin/activate
run "uv pip install -e '.[scrape,web,dev,viz]'" "pip install -e .[scrape,web,dev,viz]"

step "2. quality gates (what a contributor runs)"
run "ruff check src/cardiag tests scripts --output-format=concise" "ruff"
run "mypy" "mypy"
run "python -m pytest -p no:cacheprovider -q" "pytest (offline suite)"

step "3. wheel builds with the bundled model-less data + demo clip"
run "uv build --wheel" "wheel build"
run "python - <<'PY'
import zipfile,glob
z=zipfile.ZipFile(sorted(glob.glob('dist/*.whl'))[-1])
n=z.namelist()
assert any(p.endswith('_fixtures/demo.wav') for p in n), 'demo.wav missing from wheel'
assert any(p.endswith('web/static/index.html') for p in n), 'web UI missing from wheel'
print('wheel contains demo.wav + web UI')
PY" "wheel packages demo clip + web UI"

step "4. onboarding: cardiag doctor + start (offline)"
run "cardiag doctor" "cardiag doctor"
run "cardiag train --fixtures" "cardiag train --fixtures (offline model in ~2s)"

step "5. diagnose the bundled demo clip (NO scrape, NO model needed beyond fixtures/shipped)"
run "cardiag diagnose \"\$(python -c 'from cardiag import paths;print(paths.DEMO_CLIP)')\"" "cardiag diagnose <demo clip>"
run "cardiag clean \"\$(python -c 'from cardiag import paths;print(paths.DEMO_CLIP)')\" --no-music-gate" "cardiag clean <demo clip>"

step "6. the shipped pre-trained model works via the fallback"
run "rm -rf data/training; python -c 'from cardiag import Classifier; Classifier.load(); print(\"shipped model loads\")'" "shipped model auto-fallback (no data/training)"

step "7. live web app: serve + every endpoint"
PORT=8791
cardiag serve --model "$ROOT/models" --port $PORT >/tmp/cv_srv.log 2>&1 &
SRV=$!
for i in $(seq 1 40); do curl -sf http://127.0.0.1:$PORT/health >/dev/null && break; sleep 0.5; done
run "curl -sf http://127.0.0.1:$PORT/health | grep -q '\"model_loaded\":true'" "GET /health (model loaded)"
run "curl -sf http://127.0.0.1:$PORT/favicon.svg | grep -q svg" "GET /favicon.svg"
DEMO="$(python -c 'from cardiag import paths;print(paths.DEMO_CLIP)')"
run "curl -sf -X POST http://127.0.0.1:$PORT/api/diagnose/stream -F file=@$DEMO | grep -q 'event: diagnosis'" "POST /api/diagnose/stream (upload -> diagnosis)"
run "test \$(curl -s -o /dev/null -w '%{http_code}' -X POST http://127.0.0.1:$PORT/api/diagnose/stream --data 'url=http://169.254.169.254/') = 200 && curl -s -X POST http://127.0.0.1:$PORT/api/diagnose/stream --data 'url=http://169.254.169.254/' | grep -q 'only YouTube'" "SSRF guard rejects internal URL"
run "curl -sf -X POST http://127.0.0.1:$PORT/api/explain -F file=@$DEMO | grep -q '\"available\"'" "POST /api/explain (occlusion saliency)"
run "test \$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/api/audio/zzzz) = 400" "GET /api/audio bad-id rejected"
kill $SRV 2>/dev/null

step "summary"
printf "\n  \033[32m%d passed\033[0m, \033[31m%d failed\033[0m\n" "$PASS" "$FAIL"
cd "$ROOT"; git worktree remove --force "$WT" >/dev/null 2>&1 || true
[ "$FAIL" -eq 0 ] && { printf "\n\033[1;32mCLONE-AND-RUN VERIFIED — a stupid developer can do this.\033[0m\n"; exit 0; } || exit 1
