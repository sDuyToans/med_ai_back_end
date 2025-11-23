def build_explanation(drugs, interactions, drugdb):
    """
    Generate explanation ONLY from your data.
    No AI, no Gemini, no hallucinations.
    """

    output = []

    # -----------------------
    # 1. Medication summaries
    # -----------------------
    for d in drugs:
        output.append(f"## {d}")
        rows = drugdb.fda_info(d)

        if not rows:
            output.append("No FDA details available.\n")
            continue

        first = rows[0]

        warnings = first.get("warnings", "")
        purpose = first.get("purpose", "")
        dosage = first.get("dosage", "")
        indications = first.get("indications", "")

        if purpose:
            output.append(f"**Purpose:** {purpose}")
        if indications:
            output.append(f"**Indications:** {indications}")
        if dosage:
            output.append(f"**Dosage:** {dosage}")
        if warnings:
            output.append(f"**Warnings:** {warnings}")

        output.append("")  # spacing

    # -----------------------
    # 2. Drug interactions
    # -----------------------
    if interactions:
        output.append("## Drug Interactions Found")
        for pair in interactions:
            a = pair["drug_1"]
            b = pair["drug_2"]
            sev = pair.get("severity", "")
            desc = pair.get("description", "")
            output.append(f"- **{a} + {b}** — {sev}\n  {desc}")
    else:
        output.append("## No known dangerous interactions found in database.")

    return "\n".join(output)