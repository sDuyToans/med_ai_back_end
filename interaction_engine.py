import pandas as pd
import os

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Load CSVs on startup
INTERACTIONS = pd.read_csv(os.path.join(DATA_DIR, "interactions_clean.csv"))

# Optional: load FDA data later
FDA_DATA = pd.read_csv(os.path.join(DATA_DIR, "fda_final_clean.csv"))


def find_interactions(drug_a: str, drug_b: str):
    """Return interactions between drug_a and drug_b using CSV lookup."""
    a = drug_a.lower().strip()
    b = drug_b.lower().strip()

    matches = INTERACTIONS[
        ((INTERACTIONS["drug_1_clean"] == a) & (INTERACTIONS["drug_2_clean"] == b)) |
        ((INTERACTIONS["drug_1_clean"] == b) & (INTERACTIONS["drug_2_clean"] == a))
    ]

    return matches.to_dict(orient="records")