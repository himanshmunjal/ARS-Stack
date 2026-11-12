"""
final_paper/scripts/build_docx.py

Builds ars_stack_paper.docx from ars_stack_paper.tex with pandoc, keeping the
PDF's figure/table/equation/section numbering (read from the LaTeX .aux file)
and IEEE-style numbered references.

    tectonic -X compile ars_stack_paper.tex --keep-intermediates   # writes the .aux
    python3 scripts/build_docx.py
"""

import json
import re
import subprocess
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, RGBColor

PAPER = Path(__file__).resolve().parents[1]
BUILD = PAPER / "build"
BUILD.mkdir(exist_ok=True)
TEX = PAPER / "ars_stack_paper.tex"
AUX = PAPER / "ars_stack_paper.aux"

ALGORITHM_TEXT = r"""
\begin{verbatim}
Algorithm 1  ARS-Stack training on one (inner) training split
Require: training split (X, y), validation split (Xv, yv), balance ratio beta
 1  fit scaler on X; transform X and Xv
 2  D~ <- majority class c*
 3  for each class c != c*:
 4      B_c <- max(0, round(beta * N_c*) - N_c)                 # Eq. 1
 5      {C_c,j} <- Cluster(X_c)                    # KMeans, k by silhouette
 6      for each cluster C_c,j:
 7          o <- overlap(C_c,j); rho <- density ratio(C_c,j)   # Eqs. 3-5
 8          r <- s(o, rho); b_c,j <- allocation                # Eqs. 6-7
 9          M <- k_m majority points nearest to centroid(C_c,j)
10          (S, M') <- r.FitResample(C_c,j U M) to |C_c,j| + b_c,j minority
                       points (SMOTE fallback)
11          add retained minority and synthetic points S to D~;
            mark M \ M' as removed
12  remove marked majority points from D~
13  train RF, XGBoost, LightGBM on D~
14  w_l <- normalised validation AUC-PR                        # Eq. 8
15  tau* <- F1-optimal threshold of sum_l w_l p_l on validation  # Eq. 9
16  return (scaler, {h_l}, {w_l}, tau*)
\end{verbatim}
"""


def read_labels():
    labels = {}
    for m in re.finditer(r"\\newlabel\{([^}]+)\}\{\{([^}]*)\}", AUX.read_text()):
        labels[m.group(1)] = m.group(2)
    return labels


def inline_inputs(src: str) -> str:
    return re.sub(r"\\input\{([^}]+)\}", lambda m: (PAPER / m.group(1)).read_text(), src)


def number_floats(src: str, labels: dict) -> str:
    def fix(block, kind):
        lab = re.search(r"\\label\{([^}]+)\}", block)
        if not lab or lab.group(1) not in labels:
            return block
        return block.replace("\\caption{", f"\\caption{{{kind} {labels[lab.group(1)]}: ", 1)

    for env, kind in [("figure", "Figure"), ("table", "Table"), ("longtable", "Table")]:
        pattern = re.compile(rf"\\begin\{{{env}\}}.*?\\end\{{{env}\}}", re.S)
        src = pattern.sub(lambda m: fix(m.group(0), kind), src)
    return src


def main():
    labels = read_labels()
    src = inline_inputs(TEX.read_text())
    src = re.sub(r"%ALG-BEGIN.*?%ALG-END", lambda m: ALGORITHM_TEXT, src, flags=re.S)
    src = number_floats(src, labels)
    # custom list labels (G1, C1, H1) are dropped by pandoc: make them inline bold text
    src = re.sub(r"\\item\[(\\textbf\{[^}]+\})\]", lambda m: "\\item " + m.group(1) + " ", src)
    # plain-text math in the abstract (it becomes document metadata)
    a0, a1 = src.index("\\begin{abstract}"), src.index("\\end{abstract}")
    abstract = src[a0:a1]
    for old, new in [("$t$", "t"), ("$p=0.88$", "p = 0.88"), ("$p<0.05$", "p < 0.05"), ("1.2$\\times$", "1.2×")]:
        abstract = abstract.replace(old, new)
    assert "$" not in abstract, "math left in abstract"
    src = src[:a0] + abstract + src[a1:]
    src = re.sub(r"\\paragraph\{([^}]+)\}", lambda m: "\\textbf{" + m.group(1) + "}", src)
    # small caps: \text{} inside equations, plain capitals elsewhere
    src = re.sub(r"\\begin\{equation\}.*?\\end\{equation\}",
                 lambda m: re.sub(r"\\textsc\{([^}]+)\}", lambda n: "\\text{" + n.group(1).upper() + "}", m.group(0)),
                 src, flags=re.S)
    src = re.sub(r"\\textsc\{([^}]+)\}", lambda n: n.group(1).upper(), src)
    # equation labels -> visible tags; references -> resolved numbers
    src = re.sub(r"\\label\{(eq:[^}]+)\}", lambda m: f"\\tag{{{labels.get(m.group(1), '?')}}}", src)
    src = re.sub(r"Eqs?\.~\\ref\{([^}]+)\}", lambda m: f"Eq. ({labels.get(m.group(1), '?')})", src)
    src = re.sub(r"\\ref\{([^}]+)\}", lambda m: labels.get(m.group(1), "?"), src)
    # appendix: lettered headings outside the automatic numbering
    src = src.replace("\\appendix\n\\section{Supplementary Design Diagrams}",
                      "\\section*{Appendix A: Supplementary Design Diagrams}")
    src = src.replace("\\section{Full Per-Dataset Results}", "\\section*{Appendix B: Full Per-Dataset Results}")
    src = src.replace("\\section*{Data and Code Availability}", "\\section*{Data and Code Availability}")
    flat = BUILD / "paper_flat.tex"
    flat.write_text(src)

    # tex -> pandoc AST; put the reference list before the appendix
    ast_path = BUILD / "paper.json"
    subprocess.run(["pandoc", str(flat), "-f", "latex", "-t", "json", "-o", str(ast_path)], check=True, cwd=PAPER)
    ast = json.loads(ast_path.read_text())
    blocks = ast["blocks"]
    idx = next(i for i, b in enumerate(blocks) if b["t"] == "Header" and "Supplementary" in json.dumps(b["c"][2]))
    refs_header = {"t": "Header", "c": [1, ["references", ["unnumbered"], []], [{"t": "Str", "c": "References"}]]}
    refs_div = {"t": "Div", "c": [["refs", [], []], []]}
    ast["blocks"] = blocks[:idx] + [refs_header, refs_div] + blocks[idx:]
    ast_path.write_text(json.dumps(ast))

    ref_doc = BUILD / "reference.docx"
    make_reference_doc(ref_doc)
    out = PAPER / "ars_stack_paper.docx"
    subprocess.run(["pandoc", str(ast_path), "-f", "json", "-o", str(out), "--citeproc",
                    "--bibliography", str(PAPER / "references.bib"), "--csl", str(BUILD / "ieee.csl"),
                    "--number-sections", "--reference-doc", str(ref_doc),
                    "--resource-path", f"{PAPER}:{PAPER / 'figures'}"], check=True, cwd=PAPER)
    polish(out)
    print("wrote", out)


def make_reference_doc(path: Path):
    """pandoc's default reference.docx with a serif paper style."""
    subprocess.run(["pandoc", "-o", str(path), "--print-default-data-file", "reference.docx"], check=True)
    doc = Document(str(path))
    for name in ["Normal", "Body Text", "First Paragraph", "Compact", "Abstract", "Bibliography",
                 "Title", "Author", "Image Caption", "Table Caption", "Caption"]:
        if name in [s.name for s in doc.styles]:
            st = doc.styles[name]
            st.font.name = "Times New Roman"
            st.font.size = Pt(11)
    for lvl, size in [(1, 14), (2, 12), (3, 11)]:
        name = f"Heading {lvl}"
        if name in [s.name for s in doc.styles]:
            st = doc.styles[name]
            st.font.name = "Times New Roman"
            st.font.size = Pt(size)
            st.font.bold = True
            st.font.color.rgb = RGBColor(0, 0, 0)
    doc.styles["Title"].font.size = Pt(17)
    doc.styles["Title"].font.color.rgb = RGBColor(0, 0, 0)
    for name in ["Image Caption", "Table Caption"]:
        if name in [s.name for s in doc.styles]:
            doc.styles[name].font.size = Pt(9.5)
            doc.styles[name].font.italic = False
    doc.save(str(path))


def polish(path: Path):
    """Body paragraphs justified; table cells compact."""
    doc = Document(str(path))
    for p in doc.paragraphs:
        if p.style.name in ("Body Text", "First Paragraph"):
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    for table in doc.tables:
        table.style = doc.styles["Table"] if "Table" in [s.name for s in doc.styles] else table.style
        for row in table.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    for r in p.runs:
                        r.font.size = Pt(8.5)
    doc.save(str(path))


if __name__ == "__main__":
    main()
