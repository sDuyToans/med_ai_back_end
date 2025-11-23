"""
DrugDB
Loads a CSV of known drugs and offers fuzzy matching / text scan.

Expected CSV columns (case-insensitive):
- name  (required)
- aliases (optional): comma separated aliases / brand names

If the CSV is missing or invalid, DrugDB becomes a no-op.
"""

import os
import re
from typing import List, Optional, Dict

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

try:
    from rapidfuzz import process, fuzz
except Exception:  # pragma: no cover
    process = None
    fuzz = None


class DrugDB:
    def __init__(self, csv_path: Optional[str] = None):
        self.csv_path = csv_path
        self.names: List[str] = []
        self.alias_to_name: Dict[str, str] = {}
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
        name_col = cols.get("name") or cols.get("drug") or cols.get("generic_name")
        if not name_col:
            return

        alias_col = cols.get("aliases") or cols.get("alias") or cols.get("brand_names")

        for _, row in df.iterrows():
            name = str(row[name_col]).strip()
            if not name:
                continue
            self.names.append(name)
            # map name to itself
            self.alias_to_name[name.lower()] = name
            if alias_col:
                aliases = str(row.get(alias_col, "")).strip()
                if aliases:
                    for a in re.split(r"[;,/|]\s*|\s{2,}", aliases):
                        a = a.strip()
                        if a:
                            self.alias_to_name[a.lower()] = name

        self.count = len(self.names)

    def normalize(self, drug: str) -> str:
        if not drug:
            return drug
        key = drug.lower().strip()
        if key in self.alias_to_name:
            return self.alias_to_name[key]

        # fuzzy normalize if rapidfuzz exists
        if process and self.names:
            match, score, _ = process.extractOne(
                drug, self.names, scorer=fuzz.WRatio
            )
            if score >= 88:
                return match
        return drug

    def normalize_many(self, drugs: List[str]) -> List[str]:
        out = []
        seen = set()
        for d in drugs:
            nd = self.normalize(d)
            lk = nd.lower()
            if lk not in seen:
                seen.add(lk)
                out.append(nd)
        return out

    def find_in_text(self, text: str) -> List[str]:
        """
        Heuristic scan for drug names inside raw OCR text.
        Uses aliases map + fuzzy extraction if available.
        """
        if not text or not self.names:
            return []

        # basic tokenization
        cleaned = re.sub(r"[^a-zA-Z0-9\s\-]", " ", text)
        tokens = [t for t in cleaned.split() if len(t) >= 3]

        found = []
        # exact alias matches
        for t in tokens:
            key = t.lower()
            if key in self.alias_to_name:
                found.append(self.alias_to_name[key])

        # fuzzy against longer n-grams (optional)
        if process:
            # create candidate phrases length 1-3
            grams = tokens[:]
            for i in range(len(tokens) - 1):
                grams.append(tokens[i] + " " + tokens[i + 1])
            for i in range(len(tokens) - 2):
                grams.append(tokens[i] + " " + tokens[i + 1] + " " + tokens[i + 2])

            for g in grams:
                match, score, _ = process.extractOne(
                    g, self.names, scorer=fuzz.WRatio
                )
                if score >= 90:
                    found.append(match)

        # unique keep order
        seen = set()
        out = []
        for f in found:
            k = f.lower()
            if k not in seen:
                seen.add(k)
                out.append(f)
        return out
