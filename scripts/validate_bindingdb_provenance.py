#!/usr/bin/env python3
"""
Validate provenance and source-document metadata in a Curate-DTI BindingDB release.

Default project layout
----------------------
PROJECT/
  processed/
    BindingDB_2026-07-01/
      benchmarks/single_chain/data/
      reports/

Example
-------
python validate_bindingdb_provenance.py \
    --project-dir . \
    --release BindingDB_2026-07-01

Outputs
-------
processed/<release>/reports/provenance_validation/
  provenance_field_coverage.csv
  source_identifier_consistency.csv
  source_document_summary.csv
  publication_year_summary.csv
  publication_year_counts.csv
  provenance_problem_rows.parquet
  provenance_validation_summary.json
  provenance_validation_summary.txt

The script does not modify any released data table.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_PROJECT_DIR = Path(".")
DEFAULT_RELEASE = "BindingDB_2026-07-01"


# First match wins. Add an alias here if a future release renames a field.
COLUMN_ALIASES = {
    "measurement_id": [
        "Measurement_ID",
        "measurement_id",
        "BindingDB_Source_Row_ID",
        "BindingDB Reactant_set_id",
    ],
    "source_document_id": [
        "Source_Document_ID",
        "source_document_id",
    ],
    "source_document_type": [
        "Source_Document_Type",
        "source_document_type",
    ],
    "article_doi": [
        "Article_DOI",
        "Article DOI",
        "article_doi",
    ],
    "bindingdb_entry_doi": [
        "BindingDB_Entry_DOI",
        "BindingDB Entry DOI",
        "bindingdb_entry_doi",
    ],
    "pmid": [
        "PMID",
        "PubMed_ID",
        "pubmed_id",
        "pmid",
    ],
    "patent_number": [
        "Patent_Number",
        "Patent Number",
        "patent_number",
    ],
    "authors": [
        "Authors",
        "authors",
    ],
    "institution": [
        "Institution",
        "institution",
    ],
    "publication_year": [
        "Publication_Year",
        "publication_year",
        "Year_of_Publication",
    ],
    "publication_date": [
        "Publication_Date_Raw",
        "Date_of_publication",
        "Date of publication",
        "Publication_Date",
        "publication_date",
    ],
    "bindingdb_date": [
        "BindingDB_Date_Raw",
        "Date_in_BindingDB",
        "Date in BindingDB",
        "bindingdb_date",
    ],
    "curation_source": [
        "Data_Source",
        "Curation_DataSource",
        "Curation/DataSource",
        "curation_source",
        "data_source",
    ],
    "pdb_complex_ids": [
        "Ligand_Target_Complex_PDB_IDs",
        "PDB_IDs_for_Ligand_Target_Complex",
        "PDB ID(s) for Ligand-Target Complex",
        "PDB_Complex_IDs",
        "pdb_complex_ids",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate provenance fields in a processed BindingDB release."
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=DEFAULT_PROJECT_DIR,
        help=f"Curate-DTI project directory (default: {DEFAULT_PROJECT_DIR})",
    )
    parser.add_argument(
        "--release",
        default=DEFAULT_RELEASE,
        help=f"Processed release directory name (default: {DEFAULT_RELEASE})",
    )
    parser.add_argument(
        "--measurement-file",
        type=Path,
        default=None,
        help="Optional explicit path to the primary curated measurement Parquet file.",
    )
    parser.add_argument(
        "--min-year",
        type=int,
        default=1800,
        help="Minimum accepted publication year.",
    )
    parser.add_argument(
        "--max-year",
        type=int,
        default=2026,
        help="Maximum accepted publication year.",
    )
    return parser.parse_args()


def find_measurement_file(release_dir: Path) -> Path:
    candidates = []
    for path in release_dir.rglob("*.parquet"):
        name = path.name.lower()
        if "measurement" not in name:
            continue
        if "all_target" in name or "multichain" in name or "multi_chain" in name:
            continue
        candidates.append(path)

    preferred = [
        p for p in candidates
        if "benchmark" in str(p).lower() or "single_chain" in str(p).lower()
    ]
    selected = preferred or candidates

    if len(selected) == 1:
        return selected[0]
    if not selected:
        raise FileNotFoundError(
            f"No primary measurement Parquet file was found under {release_dir}. "
            "Provide --measurement-file explicitly."
        )

    listing = "\n".join(f"  - {p}" for p in selected)
    raise RuntimeError(
        "Several possible measurement files were found. "
        "Provide --measurement-file explicitly:\n" + listing
    )


def resolve_columns(columns: Iterable[str]) -> dict[str, str | None]:
    available = set(columns)
    resolved: dict[str, str | None] = {}
    for logical_name, aliases in COLUMN_ALIASES.items():
        resolved[logical_name] = next(
            (candidate for candidate in aliases if candidate in available),
            None,
        )
    return resolved


def clean_text(series: pd.Series) -> pd.Series:
    result = series.astype("string").str.strip()
    return result.mask(result.eq("") | result.str.lower().isin({"nan", "none", "null"}))


def normalize_doi(series: pd.Series) -> pd.Series:
    result = clean_text(series).str.lower()
    result = result.str.replace(r"^https?://(?:dx\.)?doi\.org/", "", regex=True)
    result = result.str.replace(r"^doi\s*:\s*", "", regex=True)
    return result.str.strip().mask(result.eq(""))


def normalize_pmid(series: pd.Series) -> pd.Series:
    result = clean_text(series).str.replace(r"^pmid\s*:\s*", "", regex=True)
    result = result.str.replace(r"\.0$", "", regex=True)
    return result.mask(~result.str.fullmatch(r"\d+", na=False))


def normalize_patent(series: pd.Series) -> pd.Series:
    result = clean_text(series).str.upper()
    result = result.str.replace(r"^PATENT\s*:\s*", "", regex=True)
    result = result.str.replace(r"[\s\-_/]+", "", regex=True)
    return result.mask(result.eq(""))


def coverage_row(label: str, series: pd.Series, total: int) -> dict:
    present = int(clean_text(series).notna().sum())
    return {
        "provenance_field": label,
        "measurements_available": present,
        "measurements_missing": total - present,
        "coverage_percentage": round(100.0 * present / total, 6) if total else np.nan,
    }


def consistency_for_identifier(
    df: pd.DataFrame,
    normalized_identifier: pd.Series,
    source_ids: pd.Series,
    identifier_name: str,
) -> tuple[dict, pd.Series, pd.Series]:
    usable = normalized_identifier.notna()
    work = pd.DataFrame(
        {
            "identifier": normalized_identifier[usable],
            "source_document_id": source_ids[usable],
        }
    )

    with_missing_source = int(work["source_document_id"].isna().sum())
    mapped = work.dropna(subset=["source_document_id"])

    source_ids_per_identifier = (
        mapped.groupby("identifier", observed=True)["source_document_id"].nunique()
    )
    identifiers_per_source_id = (
        mapped.groupby("source_document_id", observed=True)["identifier"].nunique()
    )

    split_identifiers = source_ids_per_identifier[source_ids_per_identifier > 1]
    merged_identifiers = identifiers_per_source_id[identifiers_per_source_id > 1]

    summary = {
        "identifier_type": identifier_name,
        "rows_with_identifier": int(usable.sum()),
        "unique_normalized_identifiers": int(normalized_identifier.nunique(dropna=True)),
        "rows_missing_source_document_id": with_missing_source,
        "identifiers_mapped_to_multiple_source_ids": int(len(split_identifiers)),
        "source_ids_mapped_to_multiple_identifiers": int(len(merged_identifiers)),
    }
    return summary, split_identifiers, merged_identifiers


def main() -> None:
    args = parse_args()
    release_dir = args.project_dir / "processed" / args.release
    reports_dir = release_dir / "reports" / "provenance_validation"
    reports_dir.mkdir(parents=True, exist_ok=True)

    measurement_path = (
        args.measurement_file.resolve()
        if args.measurement_file is not None
        else find_measurement_file(release_dir)
    )

    print(f"Release directory: {release_dir}")
    print(f"Measurement file:  {measurement_path}")
    print(f"Reports directory: {reports_dir}")

    df = pd.read_parquet(measurement_path)
    total = len(df)
    columns = resolve_columns(df.columns)

    print(f"Measurement rows:  {total:,}")
    print("\nResolved provenance columns:")
    for logical_name, actual_name in columns.items():
        print(f"  {logical_name:24s}: {actual_name or 'NOT FOUND'}")

    required = ["measurement_id", "source_document_id"]
    absent_required = [name for name in required if columns[name] is None]
    if absent_required:
        raise KeyError(
            "Required provenance columns were not found: "
            + ", ".join(absent_required)
            + "\nAvailable columns:\n"
            + "\n".join(map(str, df.columns))
        )

    measurement_ids = clean_text(df[columns["measurement_id"]])
    source_ids = clean_text(df[columns["source_document_id"]])

    normalized: dict[str, pd.Series] = {}
    if columns["article_doi"]:
        normalized["Article DOI"] = normalize_doi(df[columns["article_doi"]])
    if columns["patent_number"]:
        normalized["Patent number"] = normalize_patent(df[columns["patent_number"]])
    if columns["pmid"]:
        normalized["PMID"] = normalize_pmid(df[columns["pmid"]])

    coverage = []
    coverage_fields = [
        ("Article DOI", "article_doi"),
        ("BindingDB entry DOI", "bindingdb_entry_doi"),
        ("PMID", "pmid"),
        ("Patent number", "patent_number"),
        ("Source document identifier", "source_document_id"),
        ("Authors", "authors"),
        ("Institution", "institution"),
        ("Publication year", "publication_year"),
        ("Curation or data source", "curation_source"),
        ("PDB complex identifier", "pdb_complex_ids"),
    ]
    for label, logical_name in coverage_fields:
        actual = columns[logical_name]
        if actual is None:
            coverage.append(
                {
                    "provenance_field": label,
                    "measurements_available": 0,
                    "measurements_missing": total,
                    "coverage_percentage": 0.0,
                    "column_found": False,
                }
            )
        else:
            row = coverage_row(label, df[actual], total)
            row["column_found"] = True
            coverage.append(row)

    coverage_df = pd.DataFrame(coverage)
    coverage_df.to_csv(reports_dir / "provenance_field_coverage.csv", index=False)

    consistency_rows = []
    split_details = {}
    merged_details = {}
    for identifier_name, values in normalized.items():
        summary, split_ids, merged_ids = consistency_for_identifier(
            df, values, source_ids, identifier_name
        )
        consistency_rows.append(summary)
        split_details[identifier_name] = split_ids
        merged_details[identifier_name] = merged_ids

    consistency_df = pd.DataFrame(consistency_rows)
    consistency_df.to_csv(
        reports_dir / "source_identifier_consistency.csv", index=False
    )

    any_available_source = pd.Series(False, index=df.index)
    for values in normalized.values():
        any_available_source |= values.notna()

    available_source_but_missing_document_id = (
        any_available_source & source_ids.isna()
    )

    duplicate_measurement_ids = measurement_ids.duplicated(keep=False)
    missing_measurement_ids = measurement_ids.isna()

    source_summary = pd.DataFrame(
        [
            {
                "quantity": "measurement_rows",
                "count": total,
            },
            {
                "quantity": "unique_measurement_ids",
                "count": int(measurement_ids.nunique(dropna=True)),
            },
            {
                "quantity": "missing_measurement_ids",
                "count": int(missing_measurement_ids.sum()),
            },
            {
                "quantity": "rows_with_duplicate_measurement_ids",
                "count": int(duplicate_measurement_ids.sum()),
            },
            {
                "quantity": "rows_with_source_document_id",
                "count": int(source_ids.notna().sum()),
            },
            {
                "quantity": "unique_source_document_ids",
                "count": int(source_ids.nunique(dropna=True)),
            },
            {
                "quantity": "rows_with_available_primary_source_but_missing_source_document_id",
                "count": int(available_source_but_missing_document_id.sum()),
            },
            {
                "quantity": "rows_without_doi_patent_or_pmid",
                "count": int((~any_available_source).sum()),
            },
        ]
    )
    source_summary.to_csv(
        reports_dir / "source_document_summary.csv", index=False
    )

    year_invalid = pd.Series(False, index=df.index)
    year_missing = pd.Series(True, index=df.index)
    year_numeric = pd.Series(np.nan, index=df.index, dtype="float64")
    publication_year_counts = pd.DataFrame(columns=["publication_year", "count"])

    if columns["publication_year"]:
        original_year = clean_text(df[columns["publication_year"]])
        year_missing = original_year.isna()
        year_numeric = pd.to_numeric(original_year, errors="coerce")
        year_integer = year_numeric.notna() & np.isclose(
            year_numeric, np.round(year_numeric)
        )
        year_in_range = year_numeric.between(args.min_year, args.max_year)
        year_invalid = (~year_missing) & (~year_integer | ~year_in_range)

        valid_years = year_numeric[~year_missing & ~year_invalid].astype("int64")
        publication_year_counts = (
            valid_years.value_counts()
            .sort_index()
            .rename_axis("publication_year")
            .reset_index(name="count")
        )
        publication_year_counts["percentage_of_measurements"] = (
            100.0 * publication_year_counts["count"] / total
        )

    publication_year_counts.to_csv(
        reports_dir / "publication_year_counts.csv", index=False
    )

    publication_year_summary = pd.DataFrame(
        [
            {"quantity": "measurement_rows", "count": total},
            {"quantity": "rows_with_publication_year", "count": int((~year_missing).sum())},
            {"quantity": "rows_missing_publication_year", "count": int(year_missing.sum())},
            {"quantity": "rows_with_invalid_publication_year", "count": int(year_invalid.sum())},
            {
                "quantity": "minimum_valid_publication_year",
                "count": (
                    int(year_numeric[~year_missing & ~year_invalid].min())
                    if (~year_missing & ~year_invalid).any()
                    else None
                ),
            },
            {
                "quantity": "maximum_valid_publication_year",
                "count": (
                    int(year_numeric[~year_missing & ~year_invalid].max())
                    if (~year_missing & ~year_invalid).any()
                    else None
                ),
            },
        ]
    )
    publication_year_summary.to_csv(
        reports_dir / "publication_year_summary.csv", index=False
    )

    problem_mask = (
        missing_measurement_ids
        | duplicate_measurement_ids
        | available_source_but_missing_document_id
        | year_invalid
    )

    problem_columns = [
        name for name in {
            columns["measurement_id"],
            columns["source_document_id"],
            columns["source_document_type"],
            columns["article_doi"],
            columns["pmid"],
            columns["patent_number"],
            columns["publication_year"],
            columns["publication_date"],
            columns["bindingdb_date"],
            columns["curation_source"],
        }
        if name is not None
    ]
    problems = df.loc[problem_mask, problem_columns].copy()
    problems["problem_missing_measurement_id"] = missing_measurement_ids[problem_mask]
    problems["problem_duplicate_measurement_id"] = duplicate_measurement_ids[problem_mask]
    problems["problem_source_id_missing_despite_primary_source"] = (
        available_source_but_missing_document_id[problem_mask]
    )
    problems["problem_invalid_publication_year"] = year_invalid[problem_mask]
    problems.to_parquet(
        reports_dir / "provenance_problem_rows.parquet", index=False
    )

    split_count = int(
        sum(len(series) for series in split_details.values())
    )
    merged_count = int(
        sum(len(series) for series in merged_details.values())
    )

    validation_passed = (
        int(missing_measurement_ids.sum()) == 0
        and int(duplicate_measurement_ids.sum()) == 0
        and int(available_source_but_missing_document_id.sum()) == 0
        and int(year_invalid.sum()) == 0
        and split_count == 0
    )

    summary = {
        "release": args.release,
        "measurement_file": str(measurement_path),
        "measurement_rows": total,
        "resolved_columns": columns,
        "unique_measurement_ids": int(measurement_ids.nunique(dropna=True)),
        "missing_measurement_ids": int(missing_measurement_ids.sum()),
        "rows_with_duplicate_measurement_ids": int(duplicate_measurement_ids.sum()),
        "rows_with_source_document_id": int(source_ids.notna().sum()),
        "unique_source_document_ids": int(source_ids.nunique(dropna=True)),
        "rows_with_available_primary_source_but_missing_source_document_id": int(
            available_source_but_missing_document_id.sum()
        ),
        "rows_with_invalid_publication_year": int(year_invalid.sum()),
        "normalized_identifiers_split_across_multiple_source_ids": split_count,
        "source_ids_linked_to_multiple_values_of_same_identifier_type": merged_count,
        "problem_rows_written": int(problem_mask.sum()),
        "validation_passed": validation_passed,
        "notes": [
            "Missing provenance is reported but is not itself a data error.",
            "A source ID linked to multiple PMID, DOI or patent values is flagged for inspection.",
            "PDB links are profiled as metadata and are not sequence-and-ligand verified.",
        ],
    }

    with (reports_dir / "provenance_validation_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)

    with (reports_dir / "provenance_validation_summary.txt").open(
        "w", encoding="utf-8"
    ) as handle:
        handle.write("BINDINGDB PROVENANCE VALIDATION\n")
        handle.write("=" * 72 + "\n")
        handle.write(f"Release: {args.release}\n")
        handle.write(f"Measurement file: {measurement_path}\n")
        handle.write(f"Measurement rows: {total:,}\n")
        handle.write(
            f"Rows with Source_Document_ID: {source_ids.notna().sum():,}\n"
        )
        handle.write(
            f"Unique Source_Document_ID values: "
            f"{source_ids.nunique(dropna=True):,}\n"
        )
        handle.write(
            "Available DOI/patent/PMID but missing Source_Document_ID: "
            f"{available_source_but_missing_document_id.sum():,}\n"
        )
        handle.write(
            f"Invalid publication-year rows: {year_invalid.sum():,}\n"
        )
        handle.write(
            "Normalized identifiers split across multiple Source_Document_ID "
            f"values: {split_count:,}\n"
        )
        handle.write(
            "Source_Document_ID values linked to multiple values of the same "
            f"identifier type: {merged_count:,}\n"
        )
        handle.write(f"Problem rows written: {problem_mask.sum():,}\n")
        handle.write(
            f"Core validation status: {'PASS' if validation_passed else 'REVIEW'}\n"
        )

    print("\nPROVENANCE COVERAGE")
    print(coverage_df.to_string(index=False))
    print("\nSOURCE IDENTIFIER CONSISTENCY")
    print(
        consistency_df.to_string(index=False)
        if not consistency_df.empty
        else "No DOI, PMID or patent columns were detected."
    )
    print("\nVALIDATION SUMMARY")
    for key, value in summary.items():
        if key not in {"resolved_columns", "notes"}:
            print(f"  {key}: {value}")
    print(f"\nReports saved to: {reports_dir}")


if __name__ == "__main__":
    main()
