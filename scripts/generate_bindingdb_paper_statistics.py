#!/usr/bin/env python3
"""Generate publication-ready statistics for the DTI-Curate BindingDB release.

Run this script after ``preprocess_bindingdb_benchmark.py``.
It does not modify the benchmark tables and does not calculate molecular or
protein embeddings.

Default input:
  ./
      processed/BindingDB_2026-07-01/benchmarks/single_chain/data/

Default output:
  .../processed/BindingDB_2026-07-01/reports/paper_statistics/

The output contains CSV tables, ``paper_statistics.json``, and
``paper_statistics.md``. Model performance is intentionally excluded because
it must be obtained from separately executed benchmark experiments.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_PROJECT = "."
DEFAULT_RELEASE = "BindingDB_2026-07-01"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate all data statistics used in the BindingDB paper."
    )
    parser.add_argument(
        "--project-dir",
        default=os.environ.get("DTI_PROJECT_DIR", DEFAULT_PROJECT),
    )
    parser.add_argument(
        "--release-id",
        default=os.environ.get("BINDINGDB_RELEASE_ID", DEFAULT_RELEASE),
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Override the benchmark single-chain data directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override the statistics output directory.",
    )
    return parser.parse_args()


def python_scalar(value: Any) -> Any:
    if value is pd.NA:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def numeric_summary(series: pd.Series) -> dict[str, Any]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {"count": 0}
    result = values.describe(
        percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    ).to_dict()
    return {str(key): python_scalar(value) for key, value in result.items()}


def frequency_table(
    series: pd.Series,
    column_name: str,
    total: int | None = None,
) -> pd.DataFrame:
    result = (
        series.value_counts(dropna=False)
        .rename_axis(column_name)
        .reset_index(name="count")
    )
    denominator = total if total is not None else int(result["count"].sum())
    result["percentage"] = (
        100.0 * result["count"] / denominator if denominator else np.nan
    )
    return result


def missingness_table(frame: pd.DataFrame, table_name: str) -> pd.DataFrame:
    missing = frame.isna().sum()
    result = pd.DataFrame(
        {
            "table": table_name,
            "column": frame.columns,
            "dtype": [str(dtype) for dtype in frame.dtypes],
            "missing": [int(missing[column]) for column in frame.columns],
            "missing_percentage": [
                100.0 * int(missing[column]) / len(frame) if len(frame) else np.nan
                for column in frame.columns
            ],
            "unique_nonmissing": [
                int(frame[column].nunique(dropna=True)) for column in frame.columns
            ],
        }
    )
    return result


def dataframe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {str(key): python_scalar(value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def save_csv(
    frame: pd.DataFrame,
    output_dir: Path,
    filename: str,
    tables: dict[str, dict[str, Any]],
) -> None:
    frame.to_csv(output_dir / filename, index=False)
    tables[Path(filename).stem] = {
        "filename": filename,
        "rows": int(len(frame)),
        "columns": list(map(str, frame.columns)),
    }


def markdown_table(frame: pd.DataFrame, max_rows: int = 25) -> str:
    shown = frame.head(max_rows).copy()
    if shown.empty:
        return "_No rows._"
    for column in shown.columns:
        if pd.api.types.is_float_dtype(shown[column]):
            shown[column] = shown[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.4f}"
            )
    header = "| " + " | ".join(map(str, shown.columns)) + " |"
    separator = "| " + " | ".join(["---"] * len(shown.columns)) + " |"
    rows = [
        "| "
        + " | ".join(
            str(value).replace("|", "\\|").replace("\n", " ")
            for value in row
        )
        + " |"
        for row in shown.itertuples(index=False, name=None)
    ]
    suffix = (
        f"\n\n_Showing {max_rows:,} of {len(frame):,} rows._"
        if len(frame) > max_rows
        else ""
    )
    return "\n".join([header, separator, *rows]) + suffix


def require_columns(
    frame: pd.DataFrame,
    table_name: str,
    required: list[str],
) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise KeyError(f"{table_name} is missing required columns: {missing}")


def main() -> None:
    args = parse_args()
    project_dir = Path(args.project_dir)
    release_dir = project_dir / "processed" / args.release_id
    data_dir = (
        Path(args.input_dir)
        if args.input_dir
        else release_dir / "benchmarks" / "single_chain" / "data"
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else release_dir / "reports" / "paper_statistics"
    )
    report_dir = release_dir / "reports" / "single_chain"
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "all_target_measurements": (
            release_dir / "data" / "BindingDB_ic50_measurements_all_targets.parquet"
        ),
        "multichain_chains": (
            release_dir / "data" / "BindingDB_multichain_target_chains.parquet"
        ),
        "measurements": data_dir / "BindingDB_single_chain_measurements.parquet",
        "interactions": data_dir / "BindingDB_single_chain_interactions.parquet",
        "drugs": data_dir / "BindingDB_single_chain_drug_structures.parquet",
        "proteins": data_dir / "BindingDB_single_chain_protein_sequences.parquet",
        "manifest": report_dir / "benchmark_manifest.json",
        "stages": report_dir / "preprocessing_stage_counts.csv",
        "construct_audit": report_dir / "construct_audit.json",
    }
    missing_files = [str(path) for path in paths.values() if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(
            "Run preprocessing first. Missing files:\n  "
            + "\n  ".join(missing_files)
        )

    print(f"Input data:       {data_dir}")
    print(f"Input reports:    {report_dir}")
    print(f"Output directory: {output_dir}")
    print("Loading final benchmark tables...", flush=True)
    all_target_measurements = pd.read_parquet(
        paths["all_target_measurements"],
        columns=[
            "BindingDB Reactant_set_id",
            "BindingDB_Source_Row_ID",
            "number_of_target_chains",
            "target_scope",
            "IC50_relation",
        ],
    )
    multichain_chains = pd.read_parquet(
        paths["multichain_chains"],
        columns=[
            "BindingDB Reactant_set_id",
            "BindingDB_Source_Row_ID",
            "number_of_target_chains",
            "chain_index",
            "chain_sequence_available",
            "all_chain_sequences_available",
            "Target_Entity_ID",
        ],
    )
    measurements = pd.read_parquet(paths["measurements"])
    interactions = pd.read_parquet(paths["interactions"])
    drugs = pd.read_parquet(paths["drugs"])
    proteins = pd.read_parquet(paths["proteins"])
    manifest = json.loads(paths["manifest"].read_text())
    construct_audit = json.loads(paths["construct_audit"].read_text())
    stages = pd.read_csv(paths["stages"])

    require_columns(
        measurements,
        "measurements",
        [
            "canonical_smiles",
            "Protein_Entity_ID",
            "UniProt_ID",
            "IC50_relation",
            "IC50_boundary_nM",
            "measurement_class",
            "Data_Source",
            "Source_Document_ID",
            "Source_Document_Type",
            "publication_year",
            "has_linked_complex_PDB",
            "target_name_construct_hint",
            "construct_annotation_uncertain",
        ],
    )
    require_columns(
        interactions,
        "interactions",
        [
            "canonical_smiles",
            "Protein_Entity_ID",
            "UniProt_ID",
            "classification_valid",
            "classification_conflict",
            "y_class",
            "regression_valid",
            "pIC50",
            "regression_quality",
            "n_measurements",
            "n_exact",
            "Organism",
            "sequence_status",
        ],
    )
    require_columns(
        proteins,
        "proteins",
        [
            "Protein_Entity_ID",
            "UniProt_ID",
            "sequence_hash",
            "sequence_length",
            "sequence_status",
            "sequence_source",
            "Organism",
        ],
    )
    require_columns(
        drugs,
        "drugs",
        [
            "canonical_smiles",
            "molecular_weight",
            "heavy_atom_count",
        ],
    )

    tables: dict[str, dict[str, Any]] = {}
    statistics: dict[str, Any] = {
        "release": manifest.get("bindingdb_release", {}),
        "scope": manifest.get("scope"),
        "generated_from": str(data_dir),
    }

    # ------------------------------------------------------------------
    # Dataset overview and preprocessing attrition
    # ------------------------------------------------------------------
    class_valid = interactions["classification_valid"].fillna(False)
    regression_valid = interactions["regression_valid"].fillna(False)
    class_counts = (
        interactions.loc[class_valid, "y_class"]
        .value_counts(dropna=False)
        .to_dict()
    )
    overview = {
        "raw_rows": manifest.get("input", {}).get("raw_rows"),
        "all_target_ic50_rows_retained": int(len(all_target_measurements)),
        "multichain_ic50_rows_retained": int(
            all_target_measurements["target_scope"].eq("multi_chain").sum()
        ),
        "multichain_chain_rows_retained": int(len(multichain_chains)),
        "multichain_targets_with_all_sequences": int(
            multichain_chains.loc[
                multichain_chains["all_chain_sequences_available"],
                "BindingDB_Source_Row_ID",
            ].nunique()
        ),
        "retained_measurement_rows": int(len(measurements)),
        "construct_aware_interaction_pairs": int(len(interactions)),
        "standardized_drugs": int(len(drugs)),
        "protein_entities": int(len(proteins)),
        "unique_uniprot_accessions": int(proteins["UniProt_ID"].nunique()),
        "unique_organisms": int(proteins["Organism"].nunique(dropna=True)),
        "unique_source_documents": int(
            measurements["Source_Document_ID"].nunique()
        ),
        "classification_valid_pairs": int(class_valid.sum()),
        "classification_class_1_pairs": int(class_counts.get(1, 0)),
        "classification_class_0_pairs": int(class_counts.get(0, 0)),
        "classification_conflict_pairs": int(
            interactions["classification_conflict"].fillna(False).sum()
        ),
        "regression_valid_pairs": int(regression_valid.sum()),
    }
    statistics["dataset_overview"] = overview
    save_csv(stages, output_dir, "preprocessing_attrition.csv", tables)
    all_target_scope = frequency_table(
        all_target_measurements["target_scope"], "target_scope"
    )
    save_csv(
        all_target_scope,
        output_dir,
        "all_target_scope_distribution.csv",
        tables,
    )
    multichain_count_distribution = frequency_table(
        multichain_chains.drop_duplicates("BindingDB_Source_Row_ID")[
            "number_of_target_chains"
        ],
        "number_of_target_chains",
    )
    save_csv(
        multichain_count_distribution,
        output_dir,
        "multichain_target_count_distribution.csv",
        tables,
    )

    # ------------------------------------------------------------------
    # Measurement relations and classification
    # ------------------------------------------------------------------
    relation_counts = frequency_table(
        measurements["IC50_relation"], "IC50_relation"
    )
    save_csv(relation_counts, output_dir, "ic50_relation_distribution.csv", tables)

    measurement_class = measurements["measurement_class"].map(
        {1: "threshold_positive", 0: "threshold_negative"}
    ).fillna("ambiguous")
    measurement_class_counts = frequency_table(
        measurement_class, "measurement_class"
    )
    save_csv(
        measurement_class_counts,
        output_dir,
        "measurement_class_distribution.csv",
        tables,
    )

    pair_class = interactions.loc[class_valid, "y_class"].map(
        {1: "threshold_positive", 0: "threshold_negative"}
    )
    pair_class_counts = frequency_table(pair_class, "pair_class")
    save_csv(
        pair_class_counts,
        output_dir,
        "classification_pair_distribution.csv",
        tables,
    )

    classification_status = np.select(
        [
            interactions["classification_conflict"].fillna(False),
            interactions["classification_valid"].fillna(False),
        ],
        ["conflict", "valid"],
        default="insufficient_or_ambiguous_evidence",
    )
    classification_status_counts = frequency_table(
        pd.Series(classification_status), "classification_status"
    )
    save_csv(
        classification_status_counts,
        output_dir,
        "classification_validity_distribution.csv",
        tables,
    )

    # ------------------------------------------------------------------
    # Regression and repeated-measurement uncertainty
    # ------------------------------------------------------------------
    pIC50 = interactions.loc[regression_valid, "pIC50"]
    statistics["regression"] = {
        "pIC50": numeric_summary(pIC50),
        "IC50_nM": numeric_summary(
            interactions.loc[regression_valid, "IC50_nM"]
        ),
        "pIC50_std_replicated_pairs": numeric_summary(
            interactions.loc[interactions["n_exact"].gt(1), "pIC50_std"]
        ),
        "exact_measurements_per_pair": numeric_summary(
            interactions.loc[regression_valid, "n_exact"]
        ),
    }
    regression_quality = frequency_table(
        interactions["regression_quality"], "regression_quality"
    )
    save_csv(
        regression_quality,
        output_dir,
        "regression_quality_distribution.csv",
        tables,
    )
    pIC50_bins = pd.cut(
        pIC50,
        bins=np.arange(3.0, 12.5, 0.5),
        right=False,
        include_lowest=True,
    )
    pIC50_histogram = frequency_table(
        pIC50_bins, "pIC50_interval", total=len(pIC50)
    )
    save_csv(
        pIC50_histogram,
        output_dir,
        "pIC50_histogram_0.5.csv",
        tables,
    )

    replication_bins = pd.cut(
        interactions["n_measurements"],
        bins=[0, 1, 2, 3, 5, 10, 20, 50, np.inf],
        labels=["1", "2", "3", "4-5", "6-10", "11-20", "21-50", ">50"],
    )
    replication_distribution = frequency_table(
        replication_bins, "measurements_per_pair"
    )
    save_csv(
        replication_distribution,
        output_dir,
        "measurement_replication_distribution.csv",
        tables,
    )

    # ------------------------------------------------------------------
    # Construct-aware protein statistics
    # ------------------------------------------------------------------
    sequence_status = frequency_table(
        proteins["sequence_status"], "sequence_status"
    )
    save_csv(
        sequence_status,
        output_dir,
        "protein_sequence_status_distribution.csv",
        tables,
    )
    sequence_source = frequency_table(
        proteins["sequence_source"], "sequence_source"
    )
    save_csv(
        sequence_source,
        output_dir,
        "protein_sequence_source_distribution.csv",
        tables,
    )

    entities_per_uniprot = (
        proteins.groupby("UniProt_ID", as_index=False)
        .agg(
            n_protein_entities=("Protein_Entity_ID", "nunique"),
            Organism=("Organism", "first"),
        )
        .sort_values(
            ["n_protein_entities", "UniProt_ID"],
            ascending=[False, True],
        )
    )
    entity_count_distribution = frequency_table(
        entities_per_uniprot["n_protein_entities"],
        "entities_per_uniprot",
    )
    save_csv(
        entity_count_distribution,
        output_dir,
        "protein_entities_per_uniprot_distribution.csv",
        tables,
    )
    save_csv(
        entities_per_uniprot[entities_per_uniprot["n_protein_entities"].gt(1)],
        output_dir,
        "uniprot_accessions_with_multiple_entities.csv",
        tables,
    )
    statistics["protein_constructs"] = {
        "sequence_length": numeric_summary(proteins["sequence_length"]),
        "construct_audit": construct_audit,
        "target_name_construct_hint_entities": int(
            proteins.get(
                "target_name_construct_hint",
                pd.Series(False, index=proteins.index),
            ).fillna(False).sum()
        ),
        "construct_annotation_uncertain_entities": int(
            proteins.get(
                "construct_annotation_uncertain",
                pd.Series(False, index=proteins.index),
            ).fillna(False).sum()
        ),
    }

    # ------------------------------------------------------------------
    # Organisms and data sources
    # ------------------------------------------------------------------
    organism_entities = (
        proteins.groupby("Organism", dropna=False, as_index=False)
        .agg(
            protein_entities=("Protein_Entity_ID", "nunique"),
            uniprot_accessions=("UniProt_ID", "nunique"),
        )
    )
    organism_pairs = (
        interactions.groupby("Organism", dropna=False, as_index=False)
        .size()
        .rename(columns={"size": "interaction_pairs"})
    )
    organism_measurements = (
        interactions.groupby("Organism", dropna=False, as_index=False)[
            "n_measurements"
        ]
        .sum()
        .rename(columns={"n_measurements": "measurement_rows"})
    )
    organism_summary = (
        organism_entities.merge(organism_pairs, on="Organism", how="outer")
        .merge(organism_measurements, on="Organism", how="outer")
        .sort_values("interaction_pairs", ascending=False)
    )
    organism_summary["pair_percentage"] = (
        100.0 * organism_summary["interaction_pairs"] / len(interactions)
    )
    save_csv(
        organism_summary,
        output_dir,
        "organism_summary.csv",
        tables,
    )

    data_sources = frequency_table(
        measurements["Data_Source"], "Data_Source"
    )
    save_csv(data_sources, output_dir, "data_source_distribution.csv", tables)

    # ------------------------------------------------------------------
    # Publications, patents, chronology, and chemical-series concentration
    # ------------------------------------------------------------------
    document_types = frequency_table(
        measurements["Source_Document_Type"], "Source_Document_Type"
    )
    save_csv(
        document_types,
        output_dir,
        "source_document_type_distribution.csv",
        tables,
    )
    publication_years = frequency_table(
        measurements["publication_year"], "publication_year"
    ).sort_values("publication_year")
    save_csv(
        publication_years,
        output_dir,
        "publication_year_distribution.csv",
        tables,
    )

    identified_document_measurements = measurements[
        measurements["Source_Document_Type"].isin(
            ["article_doi", "pmid", "patent"]
        )
    ].copy()
    document_summary = (
        identified_document_measurements.groupby(
            "Source_Document_ID", as_index=False
        )
        .agg(
            Source_Document_Type=("Source_Document_Type", "first"),
            publication_year=("publication_year", "min"),
            measurement_rows=("Source_Document_ID", "size"),
            unique_drugs=("canonical_smiles", "nunique"),
            protein_entities=("Protein_Entity_ID", "nunique"),
            uniprot_accessions=("UniProt_ID", "nunique"),
            Data_Sources=("Data_Source", lambda values: " | ".join(
                sorted({str(value) for value in values.dropna()})
            )),
        )
        .sort_values("measurement_rows", ascending=False)
    )
    save_csv(
        document_summary,
        output_dir,
        "source_document_summary.csv",
        tables,
    )
    document_size_bins = pd.cut(
        document_summary["unique_drugs"],
        bins=[0, 1, 5, 10, 20, 50, 100, 500, 1000, np.inf],
        labels=[
            "1",
            "2-5",
            "6-10",
            "11-20",
            "21-50",
            "51-100",
            "101-500",
            "501-1000",
            ">1000",
        ],
    )
    document_size_distribution = frequency_table(
        document_size_bins, "unique_drugs_per_document"
    )
    save_csv(
        document_size_distribution,
        output_dir,
        "document_compound_count_distribution.csv",
        tables,
    )
    documents_le_20 = document_summary["unique_drugs"].le(20)
    statistics["source_documents"] = {
        "identified_articles_or_patents": int(len(document_summary)),
        "measurement_rows_with_identified_article_or_patent": int(
            len(identified_document_measurements)
        ),
        "documents_with_20_or_fewer_unique_drugs": int(documents_le_20.sum()),
        "percentage_documents_with_20_or_fewer_unique_drugs": (
            100.0 * float(documents_le_20.mean())
            if len(document_summary)
            else None
        ),
        "measurement_rows_from_documents_with_20_or_fewer_unique_drugs": int(
            document_summary.loc[documents_le_20, "measurement_rows"].sum()
        ),
        "unique_drugs_per_document": numeric_summary(
            document_summary["unique_drugs"]
        ),
        "measurements_per_document": numeric_summary(
            document_summary["measurement_rows"]
        ),
    }

    # ------------------------------------------------------------------
    # Drug properties and structural links
    # ------------------------------------------------------------------
    drug_numeric_columns = [
        column
        for column in [
            "molecular_weight",
            "heavy_atom_count",
            "atom_count",
            "bond_count",
            "formal_charge",
            "n_measurement_rows",
            "n_protein_entities",
            "n_uniprot_ids",
        ]
        if column in drugs.columns
    ]
    statistics["drug_properties"] = {
        column: numeric_summary(drugs[column])
        for column in drug_numeric_columns
    }
    drug_property_summary = pd.DataFrame(
        [
            {"property": column, **numeric_summary(drugs[column])}
            for column in drug_numeric_columns
        ]
    )
    save_csv(
        drug_property_summary,
        output_dir,
        "drug_property_summary.csv",
        tables,
    )

    pdb_status = frequency_table(
        measurements["has_linked_complex_PDB"].fillna(False),
        "has_linked_complex_PDB",
    )
    save_csv(
        pdb_status,
        output_dir,
        "bindingdb_complex_pdb_link_distribution.csv",
        tables,
    )

    # ------------------------------------------------------------------
    # Missingness, integrity and consolidated outputs
    # ------------------------------------------------------------------
    missingness = pd.concat(
        [
            missingness_table(measurements, "measurements"),
            missingness_table(interactions, "interactions"),
            missingness_table(drugs, "drug_structures"),
            missingness_table(proteins, "protein_sequences"),
        ],
        ignore_index=True,
    )
    save_csv(
        missingness,
        output_dir,
        "final_table_missingness_and_uniques.csv",
        tables,
    )

    integrity = {
        "duplicate_measurement_reactant_ids": int(
            measurements["BindingDB_Reactant_Set_ID"].duplicated().sum()
        ),
        "duplicate_construct_aware_pairs": int(
            interactions.duplicated(
                ["canonical_smiles", "Protein_Entity_ID"]
            ).sum()
        ),
        "duplicate_drug_rows": int(
            drugs["canonical_smiles"].duplicated().sum()
        ),
        "duplicate_protein_entity_rows": int(
            proteins["Protein_Entity_ID"].duplicated().sum()
        ),
        "interaction_drugs_missing_from_drug_table": int(
            len(
                set(interactions["canonical_smiles"])
                - set(drugs["canonical_smiles"])
            )
        ),
        "interaction_entities_missing_from_protein_table": int(
            len(
                set(interactions["Protein_Entity_ID"])
                - set(proteins["Protein_Entity_ID"])
            )
        ),
    }
    statistics["integrity"] = integrity
    statistics["tables"] = tables
    (output_dir / "paper_statistics.json").write_text(
        json.dumps(statistics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    overview_frame = pd.DataFrame(
        [{"metric": key, "value": value} for key, value in overview.items()]
    )
    regression_frame = pd.DataFrame(
        [
            {"statistic": key, "value": value}
            for key, value in statistics["regression"]["pIC50"].items()
        ]
    )
    construct_frame = pd.DataFrame(
        [
            {"metric": key, "value": value}
            for key, value in construct_audit.items()
        ]
    )
    markdown = f"""# DTI-Curate BindingDB paper statistics

Release: **{args.release_id}**

These values were calculated from the final processed tables. Class 0 means
threshold-negative evidence at the selected IC50 cutoff; it must not be
described as a universally proven biological inactive. BindingDB PDB links are
reported as links, not as sequence-and-ligand-verified structures.

## Dataset overview

{markdown_table(overview_frame, 50)}

## Preprocessing attrition

{markdown_table(stages, 50)}

## IC50 relations

{markdown_table(relation_counts, 20)}

## Classification pairs

{markdown_table(pair_class_counts, 10)}

## Classification validity

{markdown_table(classification_status_counts, 10)}

## Regression pIC50

{markdown_table(regression_frame, 20)}

## Regression quality

{markdown_table(regression_quality, 10)}

## Construct-aware audit

{markdown_table(construct_frame, 30)}

## Protein sequence categories

{markdown_table(sequence_status, 30)}

## Main organisms

{markdown_table(organism_summary, 25)}

## Main data sources

{markdown_table(data_sources, 25)}

## Source-document types

{markdown_table(document_types, 15)}

## Drug properties

{markdown_table(drug_property_summary, 20)}

## Integrity checks

{markdown_table(pd.DataFrame(
    [{"check": key, "value": value} for key, value in integrity.items()]
), 20)}

## Interpretation boundary

These files report dataset composition and preprocessing outcomes. Predictive
performance, confidence intervals, split comparisons, scaffold leakage,
cold-drug, cold-protein, cold-UniProt, cold-construct and temporal evaluation
must be generated by the modeling experiment pipeline and must not be inferred
from these descriptive statistics.
"""
    (output_dir / "paper_statistics.md").write_text(
        markdown, encoding="utf-8"
    )

    print("\nCompleted.")
    print(f"CSV tables: {output_dir}")
    print(f"JSON:       {output_dir / 'paper_statistics.json'}")
    print(f"Markdown:   {output_dir / 'paper_statistics.md'}")


if __name__ == "__main__":
    main()
