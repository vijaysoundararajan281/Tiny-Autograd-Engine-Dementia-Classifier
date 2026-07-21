"""
data.py — the dementia dataset.

Uses the OASIS clinical feature structure (age, education, MMSE, nWBV, eTIV, ASF).
By default it GENERATES a small dataset with the same realistic clinical
relationships, so the whole project runs on any laptop with no download.

TO USE THE REAL OASIS DATA INSTEAD:
  1. Get oasis_cross-sectional.csv (or oasis_longitudinal.csv) from
     https://www.oasis-brains.org/  (free, requires a quick registration).
  2. Set USE_REAL = True and point CSV_PATH at the file.
  3. The loader below keeps the clinically-relevant columns and labels
     CDR == 0 as non-demented (0) and CDR > 0 as demented (1).
"""

import random

USE_REAL = True
CSV_PATH = r"D:\Downloads\Autograd Engine\oasis_cross-sectional-5708aa0a98d82080.xlsx"

FEATURE_NAMES = ["age", "educ", "mmse", "nwbv", "etiv", "asf"]


def _load_real():
    import openpyxl
    wb = openpyxl.load_workbook(CSV_PATH, read_only=True, data_only=True)
    ws = wb.active
    rows = []
    headers = None
    for row in ws.iter_rows(values_only=True):
        if headers is None:
            headers = [str(c).strip() if c is not None else "" for c in row]
            continue
        r = dict(zip(headers, row))
        try:
            cdr = float(r.get("CDR") or "nan")
            age = float(r["Age"]); educ = float(r["Educ"])
            mmse = float(r["MMSE"]); nwbv = float(r["nWBV"])
            etiv = float(r["eTIV"]); asf = float(r["ASF"])
        except (ValueError, KeyError, TypeError):
            continue
        label = 1 if cdr > 0 else 0
        rows.append(([age, educ, mmse, nwbv, etiv, asf], label))
    wb.close()
    return rows


def _make_synthetic(n=240, seed=1):
    """
    Generate OASIS-like clinical rows. The signal mirrors the real dataset:
    demented subjects tend to be older, have lower MMSE, and lower nWBV.
    """
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        demented = rng.random() < 0.42
        if demented:
            age  = rng.gauss(76, 7)
            mmse = rng.gauss(22, 4)     # lower cognitive score
            nwbv = rng.gauss(0.71, 0.03)  # more atrophy
            educ = rng.gauss(12, 3)
        else:
            age  = rng.gauss(70, 9)
            mmse = rng.gauss(29, 1.2)   # near-perfect score
            nwbv = rng.gauss(0.76, 0.03)
            educ = rng.gauss(15, 3)
        etiv = rng.gauss(1480, 170)
        asf  = 1600.0 / etiv            # ASF is inversely related to eTIV, as in OASIS
        feats = [age, educ, mmse, nwbv, etiv, asf]
        rows.append((feats, 1 if demented else 0))
    return rows


def standardize(rows):
    """Scale each feature to mean 0, std 1. Neural nets train far better this way."""
    n_feats = len(rows[0][0])
    means, stds = [], []
    for j in range(n_feats):
        col = [r[0][j] for r in rows]
        m = sum(col) / len(col)
        var = sum((c - m) ** 2 for c in col) / len(col)
        means.append(m); stds.append(var ** 0.5 or 1.0)
    scaled = []
    for feats, y in rows:
        z = [(feats[j] - means[j]) / stds[j] for j in range(n_feats)]
        scaled.append((z, y))
    return scaled, means, stds


def load():
    rows = _load_real() if USE_REAL else _make_synthetic()
    rows, _, _ = standardize(rows)
    random.Random(0).shuffle(rows)
    split = int(0.8 * len(rows))
    return rows[:split], rows[split:]   # train, test
