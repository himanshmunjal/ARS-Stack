#!/usr/bin/env bash
# =========================================================
# build.sh — regenerate the paper assets and compile both drafts.
#
#   bash paper/build.sh
#
# 1. paper/src/make_assets.py builds every results table and figure from
#    reports/ablation_results (so numbers are never copied by hand).
# 2. The shared sources in paper/src are copied into two self-contained
#    folders (each can be uploaded to Overleaf on its own):
#       paper/single_column/  article class, single column
#       paper/ieee/           IEEEtran conference class, two columns
# 3. Each is compiled with tectonic (or latexmk if tectonic is missing).
# =========================================================
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-venv/bin/python}"
"$PY" paper/src/make_assets.py

build() {  # $1 = output folder, $2 = main file in src
  local out="paper/$1"
  rm -rf "$out"
  mkdir -p "$out"
  cp -R paper/src/sections paper/src/figures paper/src/tables paper/src/references.bib "$out"/
  cp "paper/src/$2" "$out/main.tex"
  (
    cd "$out"
    if command -v tectonic >/dev/null 2>&1; then
      tectonic --keep-logs main.tex
    else
      latexmk -pdf -interaction=nonstopmode main.tex
    fi
  )
  echo "Built $out/main.pdf"
}

build single_column main_single.tex
build ieee main_ieee.tex
