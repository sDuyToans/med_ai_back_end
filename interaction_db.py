import csv
from typing import List, Tuple, Dict

class InteractionDB:
    def __init__(self, csv_path="interactions_clean.csv"):
        self.rows = []
        try:
            with open(csv_path, newline='', encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    self.rows.append({
                        "drug1": r["drug1"].strip().lower(),
                        "drug2": r["drug2"].strip().lower(),
                        "interaction": r["interaction"].strip(),
                    })
        except Exception:
            self.rows = []

    def check_list(self, meds: List[str]) -> Tuple[int, List[Dict]]:
        meds_clean = [m.lower().strip() for m in meds if m.strip()]
        found = []
        checked = 0

        for i in range(len(meds_clean)):
            for j in range(i+1, len(meds_clean)):
                a = meds_clean[i]
                b = meds_clean[j]
                checked += 1

                for row in self.rows:
                    if {row["drug1"], row["drug2"]} == {a, b}:
                        found.append({
                            "drug_1": a,
                            "drug_2": b,
                            "interaction": row["interaction"]
                        })

        return checked, found