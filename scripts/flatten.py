"""Inline all \\input{...} files of Paper/main.tex into Paper/main_flat.tex.

The Springer Nature template requires the submitted manuscript to be one
.tex document. Run this before uploading to the submission system:

    python scripts/flatten.py
"""
import re
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1] / "Paper"


def flatten(text):
    def repl(m):
        sub = (PAPER / (m.group(1) + ".tex"))
        if not sub.exists():
            sub = PAPER / m.group(1)
        return flatten(sub.read_text(encoding="utf-8"))

    out_lines = []
    for line in text.splitlines(keepends=True):
        if line.lstrip().startswith("%"):  # keep comments verbatim
            out_lines.append(line)
        else:
            out_lines.append(re.sub(r"\\input\{([^}]+)\}", repl, line))
    return "".join(out_lines)


src = (PAPER / "main.tex").read_text(encoding="utf-8")
out = flatten(src)
(PAPER / "main_flat.tex").write_text(out, encoding="utf-8")
print("written:", PAPER / "main_flat.tex", f"({len(out.splitlines())} lines)")
