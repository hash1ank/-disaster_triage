"""Convert a `# %%` cell-style .py file into a Kaggle-ready .ipynb (stdlib only).

Usage: python src/make_notebook.py notebooks/01_eda_baseline.py notebooks/kaggle_01/humaid-01-eda-baseline.ipynb
"""
import json
import os
import sys


def convert(src_path, dst_path):
    cells, kind, buf = [], "code", []

    def flush():
        text = "\n".join(buf).strip("\n")
        if not text:
            return
        if kind == "markdown":
            text = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in text.split("\n"))
            cells.append({"cell_type": "markdown", "metadata": {}, "source": text})
        else:
            cells.append({"cell_type": "code", "metadata": {}, "source": text,
                          "execution_count": None, "outputs": []})

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for line in open(src_path, encoding="utf-8").read().split("\n"):
        if line.startswith("# %embed "):
            # "# %embed NAME path" becomes NAME = "<file contents>", so the notebook is self-contained
            _, _, name, path = line.split()
            buf.append(f"{name} = {open(os.path.join(root, path), encoding='utf-8-sig').read()!r}")
        elif line.startswith("# %%"):
            flush()
            kind = "markdown" if "[markdown]" in line else "code"
            buf = []
        else:
            buf.append(line)
    flush()

    nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}}}
    with open(dst_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"wrote {dst_path} ({len(cells)} cells)")


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])
