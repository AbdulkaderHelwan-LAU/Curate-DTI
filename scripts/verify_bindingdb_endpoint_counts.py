#!/usr/bin/env python3
"""Verify source-row and endpoint counts in a BindingDB TSV export.

The file is processed in chunks, so this script can be used on the full
BindingDB export without loading the approximately 9 GB file into memory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path

import pandas as pd


ENDPOINT_ALIASES = {
    "IC50": ("IC50 (nM)", "IC50", "IC50_nM"),
    "Ki": ("Ki (nM)", "Ki", "Ki_nM"),
    "Kd": ("Kd (nM)", "Kd", "Kd_nM"),
    "EC50": ("EC50 (nM)", "EC50", "EC50_nM"),
}

VALUE_PATTERN = re.compile(
    r"^\s*(?P<relation><=|>=|[=<>~]|≤|≥|≈)?\s*"
    r"(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*$"
)


def sha256sum(path: Path, block_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def resolve_endpoint_columns(columns: list[str]) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for endpoint, aliases in ENDPOINT_ALIASES.items():
        matches = [name for name in aliases if name in columns]
        if not matches:
            raise SystemExit(
                f"Could not find the {endpoint} column. Tried: {', '.join(aliases)}"
            )
        resolved[endpoint] = matches[0]
    return resolved


def normalize_relation(raw: str) -> tuple[str | None, bool]:
    """Return normalized relation and whether the complete cell parsed."""
    match = VALUE_PATTERN.fullmatch(raw.replace(",", ""))
    if not match:
        return None, False

    relation = match.group("relation") or "="
    relation = {
        "≤": "<=",
        "≥": ">=",
        "~": "approximate",
        "≈": "approximate",
    }.get(relation, relation)

    value = float(match.group("value"))
    return relation, value > 0 and value < float("inf")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tsv", type=Path, help="Path to the BindingDB TSV export")
    parser.add_argument(
        "--chunksize", type=int, default=100_000, help="Rows per chunk (default: 100000)"
    )
    parser.add_argument(
        "--skip-sha256", action="store_true", help="Skip full-file SHA-256 calculation"
    )
    parser.add_argument(
        "--json-output", type=Path, help="Optional path for a machine-readable JSON report"
    )
    args = parser.parse_args()

    path = args.tsv.resolve()
    if not path.is_file():
        raise SystemExit(f"Input file not found: {path}")

    header = pd.read_csv(path, sep="\t", nrows=0, low_memory=False)
    endpoint_columns = resolve_endpoint_columns(list(header.columns))

    source_rows = 0
    endpoint_counts = Counter({endpoint: 0 for endpoint in endpoint_columns})
    endpoints_per_row = Counter()
    relation_counts = {
        endpoint: Counter() for endpoint in endpoint_columns
    }
    unparsable_nonempty = Counter({endpoint: 0 for endpoint in endpoint_columns})

    reader = pd.read_csv(
        path,
        sep="\t",
        usecols=list(endpoint_columns.values()),
        dtype="string",
        chunksize=args.chunksize,
        low_memory=False,
        on_bad_lines="error",
    )

    for chunk_number, chunk in enumerate(reader, start=1):
        source_rows += len(chunk)
        present_count = pd.Series(0, index=chunk.index, dtype="int8")

        for endpoint, column in endpoint_columns.items():
            values = chunk[column]
            present = values.notna() & values.str.strip().ne("")
            count = int(present.sum())
            endpoint_counts[endpoint] += count
            present_count += present.astype("int8")

            for raw in values[present].astype(str):
                relation, valid_positive = normalize_relation(raw)
                if relation is None:
                    unparsable_nonempty[endpoint] += 1
                elif valid_positive:
                    relation_counts[endpoint][relation] += 1
                else:
                    relation_counts[endpoint]["nonpositive_or_nonfinite"] += 1

        endpoints_per_row.update(
            {int(k): int(v) for k, v in present_count.value_counts().items()}
        )
        if chunk_number % 10 == 0:
            print(f"Processed {source_rows:,} source rows...", flush=True)

    endpoint_explicit_total = sum(endpoint_counts.values())
    rows_with_multiple = sum(
        count for number, count in endpoints_per_row.items() if number >= 2
    )
    extra_observations = endpoint_explicit_total - source_rows

    report = {
        "file": str(path),
        "file_size_bytes": os.path.getsize(path),
        "sha256": None if args.skip_sha256 else sha256sum(path),
        "resolved_columns": endpoint_columns,
        "source_rows": source_rows,
        "endpoint_counts": dict(endpoint_counts),
        "endpoint_explicit_total": endpoint_explicit_total,
        "extra_endpoint_observations_over_source_rows": extra_observations,
        "rows_by_number_of_reported_endpoints": {
            str(k): endpoints_per_row[k] for k in sorted(endpoints_per_row)
        },
        "rows_with_two_or_more_endpoints": rows_with_multiple,
        "positive_parsed_relation_counts": {
            endpoint: dict(counts) for endpoint, counts in relation_counts.items()
        },
        "unparsable_nonempty_cells": dict(unparsable_nonempty),
    }

    print("\n" + "=" * 72)
    print("BINDINGDB ENDPOINT COUNT VERIFICATION")
    print("=" * 72)
    print(f"File:                         {path}")
    print(f"File size:                    {report['file_size_bytes']:,} bytes")
    if report["sha256"]:
        print(f"SHA-256:                      {report['sha256']}")
    print(f"Source rows:                  {source_rows:,}")
    for endpoint in ("IC50", "Ki", "Kd", "EC50"):
        print(f"{endpoint:<5} observations:             {endpoint_counts[endpoint]:,}")
    print(f"Endpoint-explicit total:      {endpoint_explicit_total:,}")
    print(f"Additional observations:      {extra_observations:,}")
    print(f"Rows with >=2 endpoints:      {rows_with_multiple:,}")
    print("Rows by endpoint count:       " + ", ".join(
        f"{k} endpoint(s)={endpoints_per_row[k]:,}"
        for k in sorted(endpoints_per_row)
    ))

    print("\nPositive parsed relations:")
    for endpoint in ("IC50", "Ki", "Kd", "EC50"):
        counts = relation_counts[endpoint]
        rendered = ", ".join(f"{key}={value:,}" for key, value in sorted(counts.items()))
        print(f"  {endpoint}: {rendered or 'none'}")
        if unparsable_nonempty[endpoint]:
            print(f"    Unparsable nonempty cells: {unparsable_nonempty[endpoint]:,}")

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nJSON report written to: {args.json_output}")


if __name__ == "__main__":
    main()
