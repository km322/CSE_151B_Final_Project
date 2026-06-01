"""Execute notebooks/improvements_cse151b_comp.ipynb end-to-end.

Skips shell-magic cells (the install cell). Use inside tmux for durable runs.
"""
import argparse
import os
import sys
from pathlib import Path

import nbformat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--notebook", default="notebooks/improvements_cse151b_comp.ipynb")
    ap.add_argument("--start", type=int, default=0,
                    help="Index of first code cell to run (0-based among code cells)")
    ap.add_argument("--stop", type=int, default=None,
                    help="Index AFTER the last code cell to run (exclusive)")
    ap.add_argument("--skip", default="",
                    help="Comma-separated code cell indices to skip (in addition to magics)")
    ap.add_argument("--list", action="store_true",
                    help="List code cells with their indices and don't execute")
    args = ap.parse_args()
    skip_set = {int(x) for x in args.skip.split(",") if x.strip()}

    nb = nbformat.read(args.notebook, as_version=4)
    code_cells = []
    for nb_idx, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        is_magic = any(l.lstrip().startswith("!") for l in cell.source.splitlines())
        code_cells.append((nb_idx, cell.source, is_magic))

    if args.list:
        for ci, (nb_idx, src, is_magic) in enumerate(code_cells):
            preview = next((l for l in src.splitlines() if l.strip()), "").strip()[:80]
            tag = "[skip-magic]" if is_magic else "          "
            print(f"  code#{ci:02d}  nb#{nb_idx:02d}  {tag}  {preview}")
        return

    stop = args.stop if args.stop is not None else len(code_cells)
    ns = {"__name__": "__main__"}
    # Make notebooks/ the cwd so relative paths like ../data work.
    os.chdir(Path(args.notebook).parent.resolve())

    for ci, (nb_idx, src, is_magic) in enumerate(code_cells):
        if ci < args.start or ci >= stop:
            continue
        if is_magic:
            print(f"\n>>> SKIP code cell #{ci:02d} (shell magic)\n")
            continue
        if ci in skip_set:
            print(f"\n>>> SKIP code cell #{ci:02d} (--skip)\n")
            continue
        print(f"\n>>> RUN code cell #{ci:02d} (nb cell #{nb_idx})\n")
        try:
            exec(compile(src, f"<cell {ci}>", "exec"), ns)
        except SystemExit:
            raise
        except Exception as e:
            print(f"!!! Cell #{ci} raised: {type(e).__name__}: {e}")
            raise


if __name__ == "__main__":
    main()
