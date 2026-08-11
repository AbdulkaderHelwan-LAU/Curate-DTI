# -*- coding: cp1252 -*-
"""Audit potential duplicate evidence in the curated measurement table.

This script reports possible duplicates but does not modify the dataset.
"""

from pathlib import Path
import pandas as pd


PROJECT_DIR = Path(".")

MEASUREMENT_FILE = (
    PROJECT_DIR
    / "processed/BindingDB_2026-07-01/common"
    / "BindingDB_single_chain_measurements_all_endpoints.parquet"
)

REPORT_DIR = (
    PROJECT_DIR
    / "processed/BindingDB_2026-07-01/reports/multi_endpoint"
    / "duplicate_audit"
)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------------------
# Load only the fields needed for the audit
# -------------------------------------------------------------------------

available_columns = pd.read_parquet(
    MEASUREMENT_FILE,
    engine="fastparquet",
).columns.tolist()

print("Available columns:")
print("\n".join(available_columns))

candidate_columns = [
    "Measurement_ID",
    "BindingDB_Source_Row_ID",
    "BindingDB_Reactant_set_id",
    "endpoint_type",
    "Drug_ID",
    "Protein_Entity_ID",
    "Source_Document_ID",
    "measurement_relation",
    "measurement_boundary_nM",
    "pH_numeric",
    "temperature_C_numeric",
]

columns_to_load = [
    column for column in candidate_columns
    if column in available_columns
]

missing_columns = sorted(set(candidate_columns) - set(columns_to_load))

print("\nMissing optional audit columns:")
print(missing_columns if missing_columns else "None")

df = pd.read_parquet(
    MEASUREMENT_FILE,
    columns=columns_to_load,
    engine="fastparquet",
)

print(f"\nMeasurement rows loaded: {len(df):,}")


# -------------------------------------------------------------------------
# Audit 1: identifier uniqueness
# -------------------------------------------------------------------------

identifier_summary = []

for identifier in [
    "Measurement_ID",
    "BindingDB_Source_Row_ID",
]:
    if identifier not in df.columns:
        continue

    duplicated_mask = df[identifier].duplicated(keep=False)

    identifier_summary.append(
        {
            "identifier": identifier,
            "total_rows": len(df),
            "missing_values": int(df[identifier].isna().sum()),
            "duplicated_rows": int(duplicated_mask.sum()),
            "duplicated_values": int(
                df.loc[duplicated_mask, identifier].nunique(dropna=True)
            ),
        }
    )

identifier_summary = pd.DataFrame(identifier_summary)
identifier_summary.to_csv(
    REPORT_DIR / "identifier_duplicate_summary.csv",
    index=False,
)

print("\nIDENTIFIER AUDIT")
print(identifier_summary.to_string(index=False))


# -------------------------------------------------------------------------
# Audit 2: one endpoint observation per BindingDB source row
# -------------------------------------------------------------------------

source_endpoint_key = [
    "BindingDB_Source_Row_ID",
    "endpoint_type",
]

if all(column in df.columns for column in source_endpoint_key):
    source_endpoint_duplicate_mask = df.duplicated(
        source_endpoint_key,
        keep=False,
    )

    source_endpoint_summary = pd.DataFrame(
        [
            {
                "audit": "BindingDB source row and endpoint",
                "duplicated_rows": int(
                    source_endpoint_duplicate_mask.sum()
                ),
                "duplicated_groups": int(
                    df.loc[
                        source_endpoint_duplicate_mask,
                        source_endpoint_key,
                    ]
                    .drop_duplicates()
                    .shape[0]
                ),
            }
        ]
    )

    source_endpoint_summary.to_csv(
        REPORT_DIR / "source_row_endpoint_duplicate_summary.csv",
        index=False,
    )

    print("\nSOURCE-ROW–ENDPOINT AUDIT")
    print(source_endpoint_summary.to_string(index=False))


# -------------------------------------------------------------------------
# Audit 3: candidate repeated evidence
#
# Same endpoint, ligand, protein entity, source document, relation and value.
# These are candidates for inspection, not necessarily erroneous duplicates.
# -------------------------------------------------------------------------

evidence_key = [
    "endpoint_type",
    "Drug_ID",
    "Protein_Entity_ID",
    "Source_Document_ID",
    "measurement_relation",
    "measurement_boundary_nM",
]

missing_key_columns = [
    column for column in evidence_key
    if column not in df.columns
]

if missing_key_columns:
    raise KeyError(
        "Required duplicate-audit columns are missing: "
        + ", ".join(missing_key_columns)
    )

candidate_mask = df.duplicated(
    evidence_key,
    keep=False,
)

candidate_rows = df.loc[candidate_mask].copy()

candidate_groups = (
    candidate_rows
    .groupby(evidence_key, dropna=False)
    .size()
    .reset_index(name="rows_in_group")
    .sort_values(
        ["rows_in_group", "endpoint_type"],
        ascending=[False, True],
    )
)

candidate_groups["additional_rows_beyond_one"] = (
    candidate_groups["rows_in_group"] - 1
)

overall_summary = pd.DataFrame(
    [
        {
            "total_measurement_rows": len(df),
            "candidate_duplicate_groups": len(candidate_groups),
            "rows_in_candidate_groups": len(candidate_rows),
            "additional_rows_beyond_one_per_group": int(
                candidate_groups[
                    "additional_rows_beyond_one"
                ].sum()
            ),
            "percentage_of_measurements_in_candidate_groups": (
                100.0 * len(candidate_rows) / len(df)
            ),
        }
    ]
)

overall_summary.to_csv(
    REPORT_DIR / "candidate_duplicate_overall_summary.csv",
    index=False,
)

candidate_groups.to_csv(
    REPORT_DIR / "candidate_duplicate_groups.csv",
    index=False,
)


# -------------------------------------------------------------------------
# Endpoint-specific summary
# -------------------------------------------------------------------------

if candidate_groups.empty:
    endpoint_summary = pd.DataFrame(
        columns=[
            "endpoint_type",
            "candidate_duplicate_groups",
            "rows_in_candidate_groups",
            "additional_rows_beyond_one_per_group",
        ]
    )
else:
    endpoint_summary = (
        candidate_groups
        .groupby("endpoint_type", as_index=False)
        .agg(
            candidate_duplicate_groups=(
                "rows_in_group",
                "size",
            ),
            rows_in_candidate_groups=(
                "rows_in_group",
                "sum",
            ),
            additional_rows_beyond_one_per_group=(
                "additional_rows_beyond_one",
                "sum",
            ),
        )
    )

endpoint_summary.to_csv(
    REPORT_DIR / "candidate_duplicate_summary_by_endpoint.csv",
    index=False,
)


# -------------------------------------------------------------------------
# Save candidate records for inspection
# -------------------------------------------------------------------------

sort_columns = evidence_key.copy()

if "Measurement_ID" in candidate_rows.columns:
    sort_columns.append("Measurement_ID")

candidate_rows = candidate_rows.sort_values(sort_columns)

candidate_rows.to_parquet(
    REPORT_DIR / "candidate_duplicate_measurements.parquet",
    engine="fastparquet",
    compression="gzip",
    index=False,
)

candidate_rows.head(10_000).to_csv(
    REPORT_DIR / "candidate_duplicate_measurements_sample.csv",
    index=False,
)


# -------------------------------------------------------------------------
# Print results
# -------------------------------------------------------------------------

print("\nCANDIDATE DUPLICATE-EVIDENCE AUDIT")
print(overall_summary.to_string(index=False))

print("\nBY ENDPOINT")
print(
    endpoint_summary.to_string(index=False)
    if not endpoint_summary.empty
    else "No candidate duplicate groups detected."
)

print("\nLargest candidate groups:")
print(
    candidate_groups.head(20).to_string(index=False)
    if not candidate_groups.empty
    else "No candidate duplicate groups detected."
)

print(f"\nReports written to:\n{REPORT_DIR}")
print("\nNo dataset records were modified or removed.")