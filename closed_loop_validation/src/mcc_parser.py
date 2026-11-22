# -*- coding: utf-8 -*-
"""Lenient parser for IBA .mcc measurement files (skeleton; calibrate against a real export first).

An .mcc file is the ASCII interchange format written by IBA OmniPro / myQA. It
consists of keyword lines (comments, units, device/measurement info) plus
numeric lines grouped into scan or matrix blocks. Layouts differ between myQA
versions, so this module stays permissive and ships an inspector:

- inspect_mcc(path) : print the file structure (keyword lines, shape and sample
                      values of each numeric block) to calibrate the parser on
                      a real sample;
- parse_mcc(path)   : parse metadata and numeric blocks, auto-detect an N x N
                      square matrix (default 32x32) or a 3-column x-y-value
                      triplet grid, and return the matrix with its metadata.

First use:
  1) python mcc_parser.py <sample.mcc> --inspect > inspect_out.txt
  2) identify from the output which block holds the dose matrix;
  3) confirm the dose unit (cGy? counts?) and the absolute-dose calibration
     before trusting the numbers.

Dependencies: numpy only (plus the standard library).
"""

from __future__ import annotations

import argparse
import re
import sys
import numpy as np

_FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _is_numeric_line(line):
    """True if the line is (almost) entirely numeric tokens (>= 1)."""
    tokens = line.split()
    if not tokens:
        return False
    n_num = 0
    for tok in tokens:
        if _FLOAT_RE.fullmatch(tok):
            n_num += 1
        else:
            return False
    return n_num >= 1


def _split_blocks(lines):
    """Split lines into (keyword lines, numeric blocks). A block = maximal run of numeric lines."""
    keywords = []
    blocks = []
    cur = []
    cur_start = None
    for idx, line in enumerate(lines, start=1):
        s = line.strip()
        if not s:
            continue
        if _is_numeric_line(s):
            if not cur:
                cur_start = idx
            cur.append(s)
        else:
            if cur:
                blocks.append({"start_line": cur_start, "lines": cur})
                cur = []
            keywords.append({"line_no": idx, "text": s})
    if cur:
        blocks.append({"start_line": cur_start, "lines": cur})
    return keywords, blocks


def _block_to_array(block):
    """Numeric block -> (M, K) float array; ragged rows are truncated to the shortest."""
    rows = [[float(t) for t in ln.split()] for ln in block["lines"]]
    width = min(len(r) for r in rows)
    uniform = all(len(r) == width for r in rows)
    arr = np.asarray([r[:width] for r in rows], dtype=float)
    return arr, uniform


def inspect_mcc(path, max_show=6):
    """Print a structure summary: keyword lines plus size/sample of each numeric block."""
    with open(path, "r", errors="replace") as f:
        lines = f.read().splitlines()
    keywords, blocks = _split_blocks(lines)

    print(f"file: {path}")
    print(f"total lines: {len(lines)} | keyword lines: {len(keywords)} | numeric blocks: {len(blocks)}")
    print("\n-- keyword lines (first 60) --")
    for kw in keywords[:60]:
        print(f"  L{kw['line_no']:>5}: {kw['text'][:120]}")
    print("\n-- numeric blocks --")
    for bi, b in enumerate(blocks):
        arr, uniform = _block_to_array(b)
        print(
            f"  block#{bi}: starts at L{b['start_line']}, shape {arr.shape}, "
            f"uniform row width={uniform}, value range [{np.nanmin(arr):.4g}, {np.nanmax(arr):.4g}]"
        )
        show = min(max_show, arr.shape[0])
        for r in range(show):
            print("    ", " ".join(f"{v:.5g}" for v in arr[r, : min(10, arr.shape[1])]))
        if arr.shape[0] > show:
            print(f"    ... ({arr.shape[0]} rows in total)")
    print("\nhint: use this output to pin down which block parse_mcc should return.")


def _find_square_block(arr, n):
    """Look for an n x n square in a block: the whole block first, then row/column splits."""
    if arr.shape[0] == n and arr.shape[1] == n:
        return arr
    if arr.shape[0] == 1 and arr.shape[1] >= n * n:
        flat = arr.ravel()
        for s in range(0, arr.shape[1] - n * n + 1):
            cand = flat[s : s + n * n].reshape(n, n)
            if np.isfinite(cand).all():
                return cand
    if arr.shape[1] == n * n and arr.shape[0] >= n:
        for s in range(0, arr.shape[0] - n + 1):
            cand = arr[s : s + n, :].reshape(n * n)
            cand = cand.reshape(n, n)
            if np.isfinite(cand).all():
                return cand
    return None


def _triplet_block_to_grid(arr, n):
    """3-column (x, y, value) block -> n x n grid (indexed by sorted unique x/y)."""
    if arr.shape[1] < 3:
        return None
    xyv = arr[:, -3:]
    xs = np.unique(xyv[:, 0])
    ys = np.unique(xyv[:, 1])
    if xs.size != n or ys.size != n:
        return None
    grid = np.full((n, n), np.nan)
    ix = {v: i for i, v in enumerate(xs)}
    iy = {v: i for i, v in enumerate(ys)}
    for x, y, v in xyv:
        grid[iy[y], ix[x]] = v
    if np.isfinite(grid).all():
        return grid
    return None


def parse_mcc(path, expected_shape=(32, 32), block_index=None, return_all=False):
    """Parse an .mcc file and return candidate matrices.

    Parameters
    ----------
    expected_shape : expected detector matrix shape (default (32, 32); MatriXX
                     Evolution/ONE read out 32x32, some models leave the four
                     corners as dead channels but still return 32x32).
    block_index : index of the numeric block to use (decided manually after --inspect).
    return_all : when True, return the list of all matrix-like candidates (first calibration).

    Returns
    -------
    dict(matrix=2D array, source_block=..., candidates=[...], meta={...})
    """
    with open(path, "r", errors="replace") as f:
        lines = f.read().splitlines()
    keywords, blocks = _split_blocks(lines)

    candidates = []
    for bi, b in enumerate(blocks):
        arr, _ = _block_to_array(b)
        if arr.size < 16:
            continue
        g_sq = _find_square_block(arr, expected_shape[0])
        if g_sq is not None:
            candidates.append({"block": bi, "how": "square", "matrix": g_sq})
            continue
        g_tri = _triplet_block_to_grid(arr, expected_shape[0])
        if g_tri is not None:
            candidates.append({"block": bi, "how": "triplet", "matrix": g_tri})

    meta = {
        "n_keywords": len(keywords),
        "keyword_texts": [k["text"] for k in keywords[:40]],
        "n_blocks": len(blocks),
        "block_shapes": [
            (b["start_line"], _block_to_array(b)[0].shape) for b in blocks
        ],
    }

    if return_all:
        return {"meta": meta, "candidates": candidates}

    chosen = None
    if block_index is not None:
        for c in candidates:
            if c["block"] == block_index:
                chosen = c
                break
        if chosen is None:
            raise ValueError(f"no matrix-like block found for block_index={block_index} in this file")
    else:
        if not candidates:
            raise ValueError(
                "no matrix-like block detected. Run "
                f"`python mcc_parser.py {path} --inspect` first and check which block holds the dose matrix."
            )
        if len(candidates) > 1:
            # default: take the block with the widest value range (usually the main
            # dose matrix) -- confirm manually before trusting it
            chosen = max(candidates, key=lambda c: np.nanmax(c["matrix"]))
        else:
            chosen = candidates[0]
    return {
        "matrix": chosen["matrix"],
        "source_block": chosen["block"],
        "how": chosen["how"],
        "n_candidates": len(candidates),
        "meta": meta,
    }


def write_synthetic_mcc(path, matrix32, comments=None):
    """Write a synthetic .mcc in the simplest layout this parser understands (for selftests; not a real export)."""
    with open(path, "w") as f:
        f.write("#COMMENT synthetic mcc for pipeline testing\n")
        f.write("#UNITS cGy\n")
        for c in comments or []:
            f.write(f"#COMMENT {c}\n")
        f.write("MEASINFO absolute\n")
        for row in np.asarray(matrix32):
            f.write(" ".join(f"{v:.6f}" for v in row) + "\n")
        f.write("ENDFILE\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="IBA .mcc structure inspection / parsing (skeleton)")
    ap.add_argument("mcc_file")
    ap.add_argument("--inspect", action="store_true", help="print a structure summary")
    ap.add_argument("--shape", default="32", help="expected matrix side length (default 32)")
    args = ap.parse_args()
    if args.inspect:
        inspect_mcc(args.mcc_file)
    else:
        res = parse_mcc(args.mcc_file, expected_shape=(int(args.shape), int(args.shape)))
        print("chosen matrix from block", res["source_block"], "detected as", res["how"])
        print("shape:", res["matrix"].shape, "max value:", np.nanmax(res["matrix"]))
