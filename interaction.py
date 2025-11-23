"""
InteractionChecker
Loads interactions.csv and checks candidate meds pairwise.

Expected CSV columns (case-insensitive):
- drug_a (or drug1)
- drug_b (or drug2)
- severity (optional)
- note (optional)

Matching is case-insensitive and order-insensitive.
If CSV missing/invalid -> no-op.
"""

import os
from itertools import combinations
from typing import List, Optional, Tuple, Dict

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None


class InteractionChecker:
    def __init__(self, csv_path: Optional[str] = None):
        self.csv_path = csv_path
        self.map: Dict[Tuple[str, str], dict] = {}
        self.count = 0
        if csv_path:
            self._load(csv_path)

    def _load(self, csv_path: str):
        if pd is None:
            return
        if not os.path.exists(csv_path):
            return

        df = pd.read_csv(csv_path)
        cols = {c.lower(): c for c in df.columns}

        a_col = cols.get("drug_a") or cols.get("drug1")
        b_col = cols.get("drug_b") or cols.get("drug2")
        sev_col = cols.get("severity")
        note_col = cols.get("note") or cols.get("description")

        if not a_col or not b_col:
            return

        for _, row in df.iterrows():
            a = str(row[a_col]).strip()
            b = str(row[b_col]).strip()
            if not a or not b:
                continue
            key = tuple(sorted([a.lower(), b.lower()]))
            self.map[key] = {
                "drug_a": a,
                "drug_b": b,
                "severity": str(row.get(sev_col, "")).strip() if sev_col else "",
                "note": str(row.get(note_col, "")).strip() if note_col else "",
            }

        self.count = len(self.map)

    def check(self, meds: List[str]) -> Tuple[List[dict], int]:
        meds_l = [m.lower() for m in meds]
        dangers = []
        checked_pairs = 0
        for i, j in combinations(range(len(meds_l)), 2):
            checked_pairs += 1
            key = tuple(sorted([meds_l[i], meds_l[j]]))
            if key in self.map:
                dangers.append(self.map[key])
        return dangers, checked_pairs
