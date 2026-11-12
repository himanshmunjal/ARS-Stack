#!/usr/bin/env bash
# Rebuilds every figure, table and the three paper formats from the saved results.
#   bash final_paper/build.sh
set -euo pipefail
cd "$(dirname "$0")"
python3 -I scripts/make_figures.py > /dev/null
python3 -I scripts/make_tables.py > /dev/null
tectonic -X compile ars_stack_paper.tex --keep-intermediates
python3 -I scripts/build_docx.py
rm -f ars_stack_paper.aux ars_stack_paper.bbl ars_stack_paper.blg ars_stack_paper.out ars_stack_paper.log
rm -f ars_stack_paper_overleaf.zip
zip -qr ars_stack_paper_overleaf.zip ars_stack_paper.tex references.bib figures/*.png tables/*.tex
echo "built: ars_stack_paper.pdf  ars_stack_paper.tex  ars_stack_paper.docx  ars_stack_paper_overleaf.zip"
