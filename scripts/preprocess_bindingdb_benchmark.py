"""Build the DTI-Curate BindingDB multi-endpoint data release.

Scope
-----
* IC50, Ki, Kd, and EC50 are retained as distinct endpoint types.
* Endpoint values are never pooled into one regression target.
* All selected endpoint records are retained in an all-target archive.
* A benchmark-ready subset is created for exactly one experimental chain.
* Reviewed Swiss-Prot primary accessions only.
* Molecular structures standardized with RDKit without size- or
  carbon-based exclusion.
* The experimental BindingDB target sequence defines the protein entity.
* Swiss-Prot is used only for validation, canonical metadata, and the extremely
  rare case in which BindingDB does not provide the target sequence.
* Exact endpoint measurements support endpoint-specific regression.
* Exact and logically conclusive inequalities support threshold classification.

Expected project layout
-----------------------
DTI-Curate/
    raw_inputs/       BindingDB_All.tsv and uniprot_sprot.fasta.gz
    preprocessing/    release-specific reusable preprocessing caches
    processed/
        BindingDB_<release>/
            data/                all-target and multi-chain archives
            common/              shared drug, protein, and long-form tables
            endpoints/           IC50, Ki, Kd, and EC50 benchmark views
            reports/             statistics, exclusions, and manifests

The raw BindingDB and Swiss-Prot files are read only. Existing feature files
and older processed datasets are not read, modified, or overwritten.
"""

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import gc
import gzip
import hashlib
import json
import os
import re
import warnings

import numpy as np
import pandas as pd
try:
    from Bio import BiopythonDeprecationWarning
    from Bio import __version__ as biopython_version
    from Bio.Align import PairwiseAligner, substitution_matrices
except ImportError as error:
    raise ImportError(
        "Biopython is required for construct-aware sequence alignment. "
        "Install it with `pip install biopython` in the preprocessing environment."
    ) from error
from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors, inchi
from rdkit.Chem.MolStandardize import rdMolStandardize


# =============================================================================
# 0. Configuration
# =============================================================================
PROJECT_DIR = Path(
    os.environ.get(
        "DTI_PROJECT_DIR",
        ".",
    )
)
RAW_INPUT_DIR = PROJECT_DIR / "raw_inputs"
PREPROCESSING_DIR = PROJECT_DIR / "preprocessing"
PROCESSED_DIR = PROJECT_DIR / "processed"

BINDINGDB_RELEASE_DATE = os.environ.get(
    "BINDINGDB_RELEASE_DATE", "2026-07-01"
)
BINDINGDB_RELEASE_ID = os.environ.get(
    "BINDINGDB_RELEASE_ID",
    f"BindingDB_{BINDINGDB_RELEASE_DATE}",
)

RAW_BINDINGDB = Path(
    os.environ.get(
        "BINDINGDB_TSV",
        str(RAW_INPUT_DIR / "bindingdb_july_2026.tsv"),
    )
)
SWISSPROT_FASTA = Path(
    os.environ.get(
        "SWISSPROT_FASTA",
        str(RAW_INPUT_DIR / "uniprot_sprot.fasta.gz"),
    )
)

# Accept a versioned filename when raw_inputs contains exactly one candidate.
if not RAW_BINDINGDB.is_file():
    bindingdb_candidates = sorted(
        path
        for path in RAW_INPUT_DIR.glob("*.tsv")
        if "bindingdb" in path.name.lower()
    )
    if len(bindingdb_candidates) == 1:
        RAW_BINDINGDB = bindingdb_candidates[0]
if not SWISSPROT_FASTA.is_file():
    swissprot_candidates = sorted(RAW_INPUT_DIR.glob("*sprot*.fasta.gz"))
    if len(swissprot_candidates) == 1:
        SWISSPROT_FASTA = swissprot_candidates[0]

RELEASE_DIR = PROCESSED_DIR / BINDINGDB_RELEASE_ID
ARCHIVE_DATA_DIR = RELEASE_DIR / "data"
REPORT_ROOT = RELEASE_DIR / "reports"
# Optional reference scope retained as descriptive metadata only. These
# thresholds never determine standardization validity or record inclusion.
SEQUENCE_CONTEXT_FLAG = 1024
DRUG_BATCH_SIZE = 10_000
STANDARDIZATION_VERSION = (
    "rdkit_cleanup_fragment_parent_uncharge_no_scope_exclusion_v2"
)
LEGACY_STANDARDIZATION_VERSION = (
    "rdkit_cleanup_fragment_parent_uncharge_v1"
)
ALIGNMENT_VERSION = "pairwise_blosum62_global_semiglobal_v3"
MIN_HOMOLOGY_IDENTITY = 0.80
MIN_INDEL_VARIANT_COVERAGE = 0.90
NEAR_FULL_LENGTH_RATIO = 0.90
DISPERSION_LOG10_THRESHOLD = 2.0  # 100-fold; retained and flagged, never discarded.
PARQUET_COMPRESSION = "zstd"


# Raw BindingDB column names
REACTANT_ID = "BindingDB Reactant_set_id"
SOURCE_ROW_ID = "BindingDB_Source_Row_ID"
RAW_SMILES = "Ligand SMILES"
RAW_INCHI = "Ligand InChI"
RAW_INCHI_KEY = "Ligand InChI Key"
MONOMER_ID = "BindingDB MonomerID"
LIGAND_NAME = "BindingDB Ligand Name"
TARGET_NAME = "Target Name"
BINDINGDB_ORGANISM = "Target Source Organism According to Curator or DataSource"
IC50 = "IC50 (nM)"
CHAIN_COUNT = "Number of Protein Chains in Target (>1 implies a multichain complex)"
TARGET_SEQUENCE = "BindingDB Target Chain Sequence 1"
TARGET_PDB = "PDB ID(s) of Target Chain 1"
SWISSPROT_RECOMMENDED = "UniProt (SwissProt) Recommended Name of Target Chain 1"
SWISSPROT_ENTRY = "UniProt (SwissProt) Entry Name of Target Chain 1"
UNIPROT_ID = "UniProt (SwissProt) Primary ID of Target Chain 1"
SWISSPROT_SECONDARY = "UniProt (SwissProt) Secondary ID(s) of Target Chain 1"


REQUIRED_COLUMNS = [
    REACTANT_ID,
    RAW_SMILES,
    RAW_INCHI,
    RAW_INCHI_KEY,
    MONOMER_ID,
    LIGAND_NAME,
    TARGET_NAME,
    BINDINGDB_ORGANISM,
    IC50,
    "pH",
    "Temp (C)",
    "Curation/DataSource",
    "Article DOI",
    "BindingDB Entry DOI",
    "PMID",
    "PubChem AID",
    "Patent Number",
    "Authors",
    "Date of publication",
    "Date in BindingDB",
    "Institution",
    "Ligand HET ID in PDB",
    "PDB ID(s) for Ligand-Target Complex",
    "PubChem CID",
    "PubChem SID",
    "ChEBI ID of Ligand",
    "ChEMBL ID of Ligand",
    "DrugBank ID of Ligand",
    "IUPHAR_GRAC ID of Ligand",
    "KEGG ID of Ligand",
    "ZINC ID of Ligand",
    CHAIN_COUNT,
    TARGET_SEQUENCE,
    TARGET_PDB,
    SWISSPROT_RECOMMENDED,
    SWISSPROT_ENTRY,
    UNIPROT_ID,
    SWISSPROT_SECONDARY,
]


for required_file in [RAW_BINDINGDB, SWISSPROT_FASTA]:
    if not required_file.is_file():
        raise FileNotFoundError(f"Required input not found: {required_file}")


# =============================================================================
# 1. Helpers
# =============================================================================
def clean_text(value):
    if pd.isna(value):
        return None
    value = str(value).strip()
    if not value or value.lower() in {"nan", "none", "null", "na", "n/a"}:
        return None
    # Pandas may infer integer identifiers with missing values as floats.
    if re.fullmatch(r"\d+\.0", value):
        value = value[:-2]
    return value


def normalize_text_series(series):
    result = series.astype("string").str.strip()
    return result.mask(
        result.isna()
        | result.eq("")
        | result.str.lower().isin(["nan", "none", "null", "na", "n/a"])
    )


def normalize_doi_series(series):
    result = normalize_text_series(series).str.lower()
    return result.str.replace(
        r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", regex=True
    )


def create_source_document_fields(frame):
    """
    Create stable source-document groups without splitting one PMID merely
    because DOI metadata are missing from some of its measurement rows.

    Policy
    ------
    1. Normalize DOI, PMID, patent and BindingDB-entry DOI values.
    2. When one PMID is associated with exactly one article DOI, propagate
       that DOI to every row carrying the PMID.
    3. When one PMID is associated with multiple article DOI values, group
       all of its rows conservatively under the PMID. This prevents one
       publication from entering more than one data partition.
    4. Apply the remaining fallbacks without ever mapping all missing
       provenance to one shared identifier.
    """
    article_doi = normalize_doi_series(frame["Article DOI"])
    pmid = normalize_text_series(frame["PMID"]).str.replace(r"\.0$", "", regex=True)
    patent = normalize_text_series(frame["Patent Number"]).str.upper()
    bindingdb_doi = normalize_doi_series(frame["BindingDB Entry DOI"])
    reactant_id = normalize_text_series(frame[REACTANT_ID])
    source_row_id = normalize_text_series(frame[SOURCE_ROW_ID])

    # Determine whether each PMID has zero, one or several associated DOIs.
    pmid_doi_pairs = pd.DataFrame(
        {
            "pmid": pmid,
            "article_doi": article_doi,
        },
        index=frame.index,
    ).dropna(subset=["pmid", "article_doi"])

    pmid_doi_counts = (
        pmid_doi_pairs.groupby("pmid", observed=True)["article_doi"].nunique()
    )
    pmids_with_one_doi = pmid_doi_counts[pmid_doi_counts.eq(1)].index
    pmids_with_multiple_dois = pmid_doi_counts[pmid_doi_counts.gt(1)].index

    # For an unambiguous PMID-to-DOI mapping, propagate the DOI to rows where
    # BindingDB provides the PMID but leaves the article DOI empty.
    pmid_to_doi = (
        pmid_doi_pairs[
            pmid_doi_pairs["pmid"].isin(pmids_with_one_doi)
        ]
        .drop_duplicates("pmid")
        .set_index("pmid")["article_doi"]
    )
    resolved_article_doi = article_doi.fillna(pmid.map(pmid_to_doi))

    source_id = pd.Series(pd.NA, index=frame.index, dtype="string")
    source_type = pd.Series(pd.NA, index=frame.index, dtype="string")

    # A PMID carrying several DOI values is treated as one conservative
    # publication group. It is assigned before the DOI candidate so that its
    # rows cannot be split between DOI- and PMID-based identifiers.
    multiple_doi_pmid = pmid.isin(pmids_with_multiple_dois)
    source_id.loc[multiple_doi_pmid] = (
        "PMID:" + pmid.loc[multiple_doi_pmid]
    )
    source_type.loc[multiple_doi_pmid] = "pmid_multiple_dois"

    candidates = [
        ("article_doi", "DOI:", resolved_article_doi),
        ("pmid", "PMID:", pmid),
        ("patent", "PATENT:", patent),
        ("bindingdb_entry_doi", "BDB_DOI:", bindingdb_doi),
        ("reactant_fallback", "BDB_REACTANT:", reactant_id),
        (
            "source_row_fallback",
            f"BDB_ROW:{BINDINGDB_RELEASE_ID}:",
            source_row_id,
        ),
    ]
    for document_type, prefix, values in candidates:
        missing = source_id.isna() & values.notna()
        source_id.loc[missing] = prefix + values.loc[missing]
        source_type.loc[missing] = document_type

    return source_id, source_type


def target_name_construct_hint(series):
    """Flag construct wording for audit only; never use names to define identity."""
    text = normalize_text_series(series).str.upper()
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    substitution = rf"\b[{amino_acids}]\d+[{amino_acids}]\b"
    construct_words = (
        r"\b(?:MUTANT|MUTATION|VARIANT|TRUNCAT(?:ED|ION)?|FRAGMENT|"
        r"CONSTRUCT|DOMAIN|DELETION|INSERTION|ISOFORM|RESIDUES?)\b"
    )
    return (
        text.str.contains(substitution, regex=True, na=False)
        | text.str.contains(construct_words, regex=True, na=False)
        | text.str.contains(r"\[\s*\d+\s*-\s*\d+\s*\]", regex=True, na=False)
    )


def join_unique(values):
    cleaned = sorted({v for v in (clean_text(x) for x in values) if v})
    return " | ".join(cleaned) if cleaned else None


def count_unique_nonmissing(values):
    return len({v for v in (clean_text(x) for x in values) if v})


def sha256_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_fingerprint(path, include_sha256=True):
    """Return release-audit metadata for an immutable raw input."""
    path = Path(path)
    stat = path.stat()
    result = {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "modified_utc": datetime.fromtimestamp(
            stat.st_mtime, tz=timezone.utc
        ).isoformat(),
    }
    if include_sha256:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
                digest.update(chunk)
        result["sha256"] = digest.hexdigest()
    return result


def normalize_sequence(value):
    value = clean_text(value)
    if value is None:
        return None
    return re.sub(r"\s+", "", value).upper()


def parse_swissprot_header(header):
    # >sp|P12345|ENTRY Protein name OS=Homo sapiens OX=9606 ...
    parts = header.rstrip("\n").split("|", 2)
    if len(parts) < 3:
        return None
    accession = parts[1].strip()
    remainder = parts[2].strip()
    description = remainder.split(" ", 1)[1] if " " in remainder else ""
    protein_name = description.split(" OS=", 1)[0].strip() or None
    organism_match = re.search(r"\sOS=(.*?)\sOX=(\d+)", " " + description)
    organism = organism_match.group(1).strip() if organism_match else None
    taxon_id = int(organism_match.group(2)) if organism_match else pd.NA
    entry_name = remainder.split(" ", 1)[0].strip() or None
    return accession, entry_name, protein_name, organism, taxon_id


def load_required_swissprot(required_ids):
    records = {}
    metadata = None
    sequence_parts = []

    def store_current():
        if metadata is None:
            return
        accession, entry_name, protein_name, organism, taxon_id = metadata
        if accession not in required_ids:
            return
        sequence = "".join(sequence_parts).upper()
        if sequence:
            records[accession] = {
                "SwissProt_Entry_Name": entry_name,
                "Protein_Name": protein_name,
                "Organism": organism,
                "Taxon_ID": taxon_id,
                "SwissProt_Sequence": sequence,
                "SwissProt_sequence_hash": sha256_text(sequence),
                "SwissProt_sequence_length": len(sequence),
            }

    with gzip.open(SWISSPROT_FASTA, "rt") as handle:
        for line in handle:
            if line.startswith(">"):
                store_current()
                metadata = parse_swissprot_header(line)
                sequence_parts = []
            else:
                sequence_parts.append(line.strip())
        store_current()
    return records


def parse_ic50_series(series):
    raw = series.astype("string").str.strip()
    parsed = raw.str.extract(
        r"^\s*(<=|>=|<|>|=|~|≈|≤|≥)?\s*"
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    )
    relation = parsed[0].fillna("=").replace(
        {"≤": "<=", "≥": ">=", "~": "approx", "≈": "approx"}
    )
    boundary = pd.to_numeric(parsed[1], errors="coerce")
    return relation, boundary


def extract_first_number(series):
    text = series.astype("string")
    number = text.str.extract(
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)",
        expand=False,
    )
    return pd.to_numeric(number, errors="coerce")


def parse_temperature_c(series):
    values = extract_first_number(series)
    text = series.astype("string").str.upper()
    is_kelvin = text.str.contains(r"(?:KELVIN|\bK\b)", regex=True, na=False)
    values.loc[is_kelvin & values.notna()] -= 273.15
    return values


SEQUENCE_CATEGORY_PRECEDENCE = [
    "swissprot_fallback",
    "exact_canonical",
    "exact_canonical_fragment",
    "canonical_with_terminal_extension",
    "extension_with_variants",
    "substitution_variant",
    "indel_variant",
    "fragment_with_variants",
    "complex_sequence_difference",
]


def build_construct_aligner(free_terminal_gaps):
    """Build a BLOSUM62 aligner with version-tolerant terminal-gap policy."""
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -11.0
    aligner.extend_gap_score = -1.0

    # Biopython renamed several terminal-gap attributes across releases.  Set
    # the most specific attributes available; free terminal gaps allow a true
    # experimental fragment to align to its canonical region without a large
    # artificial penalty.
    if not free_terminal_gaps:
        return aligner
    candidates = [
        "target_left_open_gap_score", "target_left_extend_gap_score",
        "target_right_open_gap_score", "target_right_extend_gap_score",
        "query_left_open_gap_score", "query_left_extend_gap_score",
        "query_right_open_gap_score", "query_right_extend_gap_score",
    ]
    set_specific = False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BiopythonDeprecationWarning)
        for attribute in candidates:
            try:
                setattr(aligner, attribute, 0.0)
                set_specific = True
            except (AttributeError, ValueError):
                continue
        if not set_specific:
            for attribute in ("target_end_gap_score", "query_end_gap_score"):
                try:
                    setattr(aligner, attribute, 0.0)
                except (AttributeError, ValueError):
                    continue
    return aligner


GLOBAL_CONSTRUCT_ALIGNER = build_construct_aligner(free_terminal_gaps=False)
SEMIGLOBAL_CONSTRUCT_ALIGNER = build_construct_aligner(free_terminal_gaps=True)


def alignment_safe_sequence(sequence):
    # BLOSUM62 accepts X but not every extended IUPAC residue code.
    return re.sub(r"[^ACDEFGHIKLMNPQRSTVWYBXZ]", "X", sequence)


def characterize_construct(experimental, canonical, source):
    """Return auditable alignment statistics; categories never define identity."""
    exp_len = len(experimental)
    can_len = len(canonical)
    base = {
        "alignment_version": ALIGNMENT_VERSION,
        "alignment_performed": False,
        "alignment_score": np.nan,
        "sequence_identity": np.nan,
        "identity_ungapped": np.nan,
        "canonical_coverage": np.nan,
        "construct_coverage": np.nan,
        "n_matches": pd.NA,
        "n_substitutions": pd.NA,
        "n_insertions": pd.NA,
        "n_deletions": pd.NA,
        "internal_insertions": pd.NA,
        "internal_deletions": pd.NA,
        "canonical_start": pd.NA,
        "canonical_end": pd.NA,
        "is_contiguous_fragment": False,
        "is_single_substitution_full_coverage": False,
        "has_terminal_extension": False,
        "has_polyhistidine_extension": False,
    }
    if source == "SwissProt_fallback_only" or experimental == canonical:
        base.update({
            "sequence_status": (
                "swissprot_fallback" if source == "SwissProt_fallback_only"
                else "exact_canonical"
            ),
            "sequence_identity": exp_len / max(exp_len, can_len),
            "identity_ungapped": 1.0,
            "canonical_coverage": 1.0,
            "construct_coverage": 1.0,
            "n_matches": can_len,
            "n_substitutions": 0,
            "n_insertions": 0,
            "n_deletions": 0,
            "internal_insertions": 0,
            "internal_deletions": 0,
            "canonical_start": 0,
            "canonical_end": can_len,
            "is_contiguous_fragment": False,
        })
        return base

    # Exact substring cases are deterministic and need no dynamic programming.
    if experimental in canonical:
        start = canonical.index(experimental)
        base.update({
            "sequence_status": "exact_canonical_fragment",
            "sequence_identity": exp_len / max(exp_len, can_len),
            "identity_ungapped": 1.0,
            "canonical_coverage": exp_len / can_len,
            "construct_coverage": 1.0,
            "n_matches": exp_len,
            "n_substitutions": 0,
            "n_insertions": 0,
            "n_deletions": can_len - exp_len,
            "internal_insertions": 0,
            "internal_deletions": 0,
            "canonical_start": start,
            "canonical_end": start + exp_len,
            "is_contiguous_fragment": True,
        })
        return base
    if canonical in experimental:
        start = experimental.index(canonical)
        prefix = experimental[:start]
        suffix = experimental[start + can_len:]
        extension = prefix + suffix
        base.update({
            "sequence_status": "canonical_with_terminal_extension",
            "sequence_identity": can_len / max(exp_len, can_len),
            "identity_ungapped": 1.0,
            "canonical_coverage": 1.0,
            "construct_coverage": can_len / exp_len,
            "n_matches": can_len,
            "n_substitutions": 0,
            "n_insertions": len(extension),
            "n_deletions": 0,
            "internal_insertions": 0,
            "internal_deletions": 0,
            "canonical_start": 0,
            "canonical_end": can_len,
            "is_contiguous_fragment": False,
            "has_terminal_extension": True,
            "has_polyhistidine_extension": bool(re.search(r"H{5,}", extension)),
        })
        return base

    # Near-full-length constructs use a fully global alignment so terminal
    # substitutions cannot be hidden as free end gaps. Shorter constructs use
    # a semiglobal alignment so canonical flanks do not dominate the score.
    length_ratio = min(exp_len, can_len) / max(exp_len, can_len)
    aligner = (
        GLOBAL_CONSTRUCT_ALIGNER
        if length_ratio >= NEAR_FULL_LENGTH_RATIO
        else SEMIGLOBAL_CONSTRUCT_ALIGNER
    )
    alignment = aligner.align(
        alignment_safe_sequence(canonical),
        alignment_safe_sequence(experimental),
    )[0]
    target_blocks, query_blocks = alignment.aligned
    matches = substitutions = 0
    for (t0, t1), (q0, q1) in zip(target_blocks, query_blocks):
        target_part = canonical[int(t0):int(t1)]
        query_part = experimental[int(q0):int(q1)]
        block_matches = sum(a == b for a, b in zip(target_part, query_part))
        matches += block_matches
        substitutions += len(target_part) - block_matches

    aligned_target = int(sum(int(end) - int(start) for start, end in target_blocks))
    aligned_query = int(sum(int(end) - int(start) for start, end in query_blocks))
    # Total unaligned residues already include terminal and internal gaps.
    insertions = max(0, exp_len - aligned_query)
    deletions = max(0, can_len - aligned_target)
    internal_insertions = internal_deletions = 0
    for index in range(1, len(target_blocks)):
        internal_deletions += max(
            0, int(target_blocks[index][0]) - int(target_blocks[index - 1][1])
        )
        internal_insertions += max(
            0, int(query_blocks[index][0]) - int(query_blocks[index - 1][1])
        )
    aligned_columns = matches + substitutions
    identity_ungapped = matches / aligned_columns if aligned_columns else 0.0
    identity = matches / max(exp_len, can_len) if max(exp_len, can_len) else 0.0
    canonical_coverage = aligned_target / can_len if can_len else np.nan
    construct_coverage = aligned_query / exp_len if exp_len else np.nan
    canonical_start = int(target_blocks[0][0]) if len(target_blocks) else pd.NA
    canonical_end = int(target_blocks[-1][1]) if len(target_blocks) else pd.NA
    query_start = int(query_blocks[0][0]) if len(query_blocks) else 0
    query_end = int(query_blocks[-1][1]) if len(query_blocks) else 0
    terminal_extension_sequence = experimental[:query_start] + experimental[query_end:]
    has_terminal_extension = bool(terminal_extension_sequence)
    has_polyhistidine_extension = bool(
        re.search(r"H{5,}", terminal_extension_sequence)
    )
    full_coverage = canonical_coverage >= 0.999 and construct_coverage >= 0.999

    if identity_ungapped < MIN_HOMOLOGY_IDENTITY:
        status = "complex_sequence_difference"
    elif has_terminal_extension and canonical_coverage >= MIN_INDEL_VARIANT_COVERAGE:
        status = "extension_with_variants"
    elif full_coverage and insertions == 0 and deletions == 0:
        status = "substitution_variant"
    elif (
        canonical_coverage >= MIN_INDEL_VARIANT_COVERAGE
        and construct_coverage >= MIN_INDEL_VARIANT_COVERAGE
        and (insertions > 0 or deletions > 0)
    ):
        status = "indel_variant"
    elif canonical_coverage < MIN_INDEL_VARIANT_COVERAGE:
        status = "fragment_with_variants"
    else:
        status = "complex_sequence_difference"

    base.update({
        "sequence_status": status,
        "alignment_performed": True,
        "alignment_score": float(alignment.score),
        "sequence_identity": identity,
        "identity_ungapped": identity_ungapped,
        "canonical_coverage": canonical_coverage,
        "construct_coverage": construct_coverage,
        "n_matches": matches,
        "n_substitutions": substitutions,
        "n_insertions": insertions,
        "n_deletions": deletions,
        "internal_insertions": internal_insertions,
        "internal_deletions": internal_deletions,
        "canonical_start": canonical_start,
        "canonical_end": canonical_end,
        "is_contiguous_fragment": (
            status in {"exact_canonical_fragment", "fragment_with_variants"}
            and
            internal_insertions == 0 and internal_deletions == 0
            and canonical_coverage < 0.999
        ),
        "is_single_substitution_full_coverage": (
            full_coverage and substitutions == 1 and insertions == 0 and deletions == 0
        ),
        "has_terminal_extension": has_terminal_extension,
        "has_polyhistidine_extension": has_polyhistidine_extension,
    })
    return base


def atomic_parquet_write(frame, path):
    temp = Path(str(path) + ".tmp")
    frame.to_parquet(temp, index=False, compression=PARQUET_COMPRESSION)
    os.replace(temp, path)


def report_stage(stage_rows, stage, before, after, note=""):
    stage_rows.append(
        {
            "stage": stage,
            "rows_before": int(before),
            "rows_after": int(after),
            "rows_removed": int(before - after),
            "note": note,
        }
    )
    print(
        f"{stage:<48} before={before:,} after={after:,} "
        f"removed={before-after:,}",
        flush=True,
    )


# =============================================================================
# 2. Multi-endpoint configuration
# =============================================================================

# The four endpoints are processed independently. They share chemical and
# protein-entity definitions but are never pooled into one regression target.
ENDPOINT_COLUMNS = {
    "IC50": "IC50 (nM)",
    "Ki": "Ki (nM)",
    "Kd": "Kd (nM)",
    "EC50": "EC50 (nM)",
}

# A common operational threshold is exposed explicitly rather than hidden in
# the label code. The biological interpretation remains endpoint-specific.
ENDPOINT_THRESHOLDS_NM = {
    "IC50": 1000.0,
    "Ki": 1000.0,
    "Kd": 1000.0,
    "EC50": 1000.0,
}

# Exact values outside 10^-3--10^6 nM are excluded as operational outliers.
# Inequality boundaries are not range-filtered because they can still provide
# logically conclusive threshold evidence.
ENDPOINT_P_RANGES = {
    endpoint: (3.0, 12.0) for endpoint in ENDPOINT_COLUMNS
}

ENDPOINT_P_NAMES = {
    "IC50": "pIC50",
    "Ki": "pKi",
    "Kd": "pKd",
    "EC50": "pEC50",
}

ENDPOINT_SEMANTICS = {
    "IC50": "inhibitory_potency",
    "Ki": "inhibition_constant",
    "Kd": "equilibrium_dissociation_affinity",
    "EC50": "functional_response_potency",
}

MULTI_ENDPOINT_RELEASE_DIR = RELEASE_DIR
COMMON_DATA_DIR = MULTI_ENDPOINT_RELEASE_DIR / "common"
ENDPOINT_ROOT = MULTI_ENDPOINT_RELEASE_DIR / "endpoints"
MULTI_REPORT_DIR = REPORT_ROOT / "multi_endpoint"
MULTI_CACHE_ROOT = (
    PREPROCESSING_DIR / BINDINGDB_RELEASE_ID / "multi_endpoint_cache"
)
MULTI_DRUG_CACHE_DIR = MULTI_CACHE_ROOT / (
    f"drug_standardization_{STANDARDIZATION_VERSION}_rdkit_{rdBase.rdkitVersion}"
)
LEGACY_MULTI_DRUG_CACHE_DIR = MULTI_CACHE_ROOT / (
    "drug_standardization_"
    f"{LEGACY_STANDARDIZATION_VERSION}_rdkit_{rdBase.rdkitVersion}"
)
MULTI_ALIGNMENT_CACHE = (
    MULTI_CACHE_ROOT / "protein_construct_alignments_union.parquet"
)
MULTI_ALIGNMENT_META = (
    MULTI_CACHE_ROOT / "protein_construct_alignments_union.json"
)

ALL_ENDPOINT_ARCHIVE_OUT = (
    ARCHIVE_DATA_DIR / "BindingDB_measurements_all_endpoints.parquet"
)
MULTICHAIN_ENDPOINT_CHAINS_OUT = (
    ARCHIVE_DATA_DIR / "BindingDB_multichain_target_chains.parquet"
)
MEASUREMENTS_LONG_OUT = (
    COMMON_DATA_DIR / "BindingDB_single_chain_measurements_all_endpoints.parquet"
)
INTERACTIONS_LONG_OUT = (
    COMMON_DATA_DIR / "BindingDB_single_chain_interactions_all_endpoints.parquet"
)
DRUGS_COMMON_OUT = (
    COMMON_DATA_DIR / "BindingDB_drug_structures.parquet"
)
PROTEINS_COMMON_OUT = (
    COMMON_DATA_DIR / "BindingDB_protein_entities.parquet"
)

for directory in [
    ARCHIVE_DATA_DIR,
    COMMON_DATA_DIR,
    ENDPOINT_ROOT,
    MULTI_REPORT_DIR,
    MULTI_DRUG_CACHE_DIR,
    MULTI_ALIGNMENT_CACHE.parent,
]:
    directory.mkdir(parents=True, exist_ok=True)
for endpoint in ENDPOINT_COLUMNS:
    (ENDPOINT_ROOT / endpoint / "data").mkdir(parents=True, exist_ok=True)
    (MULTI_REPORT_DIR / endpoint).mkdir(parents=True, exist_ok=True)


def endpoint_data_path(endpoint, table):
    return (
        ENDPOINT_ROOT
        / endpoint
        / "data"
        / f"BindingDB_{endpoint.lower()}_{table}.parquet"
    )


def report_endpoint_stage(stage_rows, frame, stage, before_counts, note=""):
    after_counts = frame["endpoint_type"].value_counts().to_dict()
    for endpoint in ENDPOINT_COLUMNS:
        before = int(before_counts.get(endpoint, 0))
        after = int(after_counts.get(endpoint, 0))
        stage_rows.append(
            {
                "endpoint_type": endpoint,
                "stage": stage,
                "rows_before": before,
                "rows_after": after,
                "rows_removed": before - after,
                "note": note,
            }
        )
        print(
            f"{endpoint:<5} | {stage:<42} "
            f"before={before:,} after={after:,} removed={before-after:,}",
            flush=True,
        )


def endpoint_counts(frame):
    return frame["endpoint_type"].value_counts().to_dict()


def save_value_counts(series, value_name, path):
    table = (
        series.value_counts(dropna=False)
        .rename_axis(value_name)
        .reset_index(name="count")
    )
    if table["count"].sum():
        table["percentage"] = 100.0 * table["count"] / table["count"].sum()
    else:
        table["percentage"] = np.nan
    table.to_csv(path, index=False)


# =============================================================================
# 3. Chemical standardization
# =============================================================================

cleanup = rdMolStandardize.Cleanup
fragment_parent = rdMolStandardize.FragmentParent
uncharger = rdMolStandardize.Uncharger()


def standardize_smiles(raw_smiles):
    """Return one deterministic RDKit parent structure or a true failure.

    Molecular size and carbon content are recorded as descriptors. They do
    not determine validity or inclusion in the curated release.
    """
    try:
        mol = Chem.MolFromSmiles(raw_smiles)
        if mol is None:
            return {
                "standardization_valid": False,
                "exclusion_reason": "rdkit_parse_failed",
            }
        mol = cleanup(mol)
        mol = fragment_parent(mol)
        mol = uncharger.uncharge(mol)
        Chem.SanitizeMol(mol)
        if mol.GetNumAtoms() == 0:
            return {
                "standardization_valid": False,
                "exclusion_reason": "empty_after_standardization",
            }
        canonical_smiles = Chem.MolToSmiles(
            mol, canonical=True, isomericSmiles=True
        )
        if "." in canonical_smiles:
            return {
                "standardization_valid": False,
                "exclusion_reason": "unresolved_mixture",
            }
        molecular_weight = float(Descriptors.MolWt(mol))
        heavy_atoms = int(mol.GetNumHeavyAtoms())
        contains_carbon = any(
            atom.GetAtomicNum() == 6 for atom in mol.GetAtoms()
        )
        standardized_inchi = inchi.MolToInchi(mol)
        standardized_inchi_key = inchi.InchiToInchiKey(standardized_inchi)
        return {
            "standardization_valid": True,
            "exclusion_reason": None,
            "canonical_smiles": canonical_smiles,
            "standardized_inchi": standardized_inchi,
            "standardized_inchi_key": standardized_inchi_key,
            "molecular_weight": molecular_weight,
            "heavy_atom_count": heavy_atoms,
            "contains_carbon": contains_carbon,
            "atom_count": int(mol.GetNumAtoms()),
            "bond_count": int(mol.GetNumBonds()),
            "formal_charge": int(
                sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
            ),
        }
    except Exception as error:
        return {
            "standardization_valid": False,
            "exclusion_reason": (
                f"standardization_exception:{type(error).__name__}"
            ),
        }


def build_drug_standardization_map(raw_smiles_values):
    """Standardize structures once, migrating the compatible legacy cache.

    Previously valid standardized structures can be reused. Only rows rejected
    solely by the former carbon, size, or molecular-weight rules are
    recalculated under the new inclusion policy.
    """
    unique_raw_smiles = sorted(
        {str(value) for value in raw_smiles_values if clean_text(value)}
    )
    part_paths = []
    n_batches = (
        len(unique_raw_smiles) + DRUG_BATCH_SIZE - 1
    ) // DRUG_BATCH_SIZE
    for start in range(0, len(unique_raw_smiles), DRUG_BATCH_SIZE):
        batch_index = start // DRUG_BATCH_SIZE
        batch = unique_raw_smiles[start : start + DRUG_BATCH_SIZE]
        part_path = (
            MULTI_DRUG_CACHE_DIR
            / f"drug_standardization_part_{batch_index:05d}.parquet"
        )
        part_paths.append(part_path)
        reuse = False
        if part_path.exists():
            existing = pd.read_parquet(part_path)
            reuse = (
                len(existing) == len(batch)
                and existing[RAW_SMILES].astype(str).tolist() == batch
                and existing["standardization_version"]
                .eq(STANDARDIZATION_VERSION)
                .all()
                and existing["rdkit_version"].eq(rdBase.rdkitVersion).all()
            )
        if not reuse:
            legacy_part_path = (
                LEGACY_MULTI_DRUG_CACHE_DIR
                / f"drug_standardization_part_{batch_index:05d}.parquet"
            )
            legacy = None
            if legacy_part_path.exists():
                candidate = pd.read_parquet(legacy_part_path)
                legacy_matches = (
                    len(candidate) == len(batch)
                    and candidate[RAW_SMILES].astype(str).tolist() == batch
                    and candidate["standardization_version"]
                    .eq(LEGACY_STANDARDIZATION_VERSION)
                    .all()
                    and candidate["rdkit_version"]
                    .eq(rdBase.rdkitVersion)
                    .all()
                )
                if legacy_matches:
                    legacy = candidate

            records = []
            legacy_scope_exclusions = {
                "no_carbon",
                "too_few_heavy_atoms",
                "molecular_weight_too_low",
                "molecular_weight_too_high",
            }
            for position, raw_smiles in enumerate(batch):
                if legacy is None:
                    record = {RAW_SMILES: raw_smiles}
                    record.update(standardize_smiles(raw_smiles))
                else:
                    legacy_record = legacy.iloc[position].to_dict()
                    reason = legacy_record.get("exclusion_reason")
                    if reason in legacy_scope_exclusions:
                        record = {RAW_SMILES: raw_smiles}
                        record.update(standardize_smiles(raw_smiles))
                    else:
                        record = legacy_record
                        if bool(record.get("standardization_valid")):
                            # Every structure accepted by v1 contained carbon.
                            record["contains_carbon"] = True
                        else:
                            record["contains_carbon"] = None

                record["standardization_version"] = STANDARDIZATION_VERSION
                record["rdkit_version"] = rdBase.rdkitVersion
                records.append(record)
            atomic_parquet_write(pd.DataFrame(records), part_path)
            action = "Migrated" if legacy is not None else "Standardized"
            print(
                f"{action} drug batch {batch_index + 1}/{n_batches}",
                flush=True,
            )
        else:
            print(
                f"Reusing drug batch {batch_index + 1}/{n_batches}",
                flush=True,
            )
    if not part_paths:
        return pd.DataFrame()
    result = pd.concat(
        [pd.read_parquet(path) for path in part_paths],
        ignore_index=True,
    )
    if len(result) != len(unique_raw_smiles):
        raise RuntimeError(
            "Drug standardization cache does not cover every raw SMILES."
        )
    return result


# =============================================================================
# 4. Load BindingDB and create endpoint-explicit measurement rows
# =============================================================================

print("=" * 100)
print("STEP 1: LOAD BINDINGDB AND SEPARATE ENDPOINT TYPES")
print("=" * 100)
print(f"Project directory:      {PROJECT_DIR}")
print(f"BindingDB release:      {BINDINGDB_RELEASE_ID}")
print(f"BindingDB TSV:          {RAW_BINDINGDB}")
print(f"Swiss-Prot FASTA:       {SWISSPROT_FASTA}")
print(f"Shared data directory:  {COMMON_DATA_DIR}")
print(f"Endpoint directory:     {ENDPOINT_ROOT}")
print(f"Reports:                {MULTI_REPORT_DIR}")

header = pd.read_csv(RAW_BINDINGDB, sep="\t", nrows=0).columns
required_columns = list(
    dict.fromkeys(
        REQUIRED_COLUMNS
        + list(ENDPOINT_COLUMNS.values())
    )
)
missing_columns = sorted(set(required_columns) - set(header))
if missing_columns:
    raise KeyError(
        f"Required BindingDB columns are missing: {missing_columns}"
    )

chain_specific_columns = [
    column
    for column in header
    if re.search(r"Target Chain(?: Sequence)? \d+$", str(column))
]
chain_indices = sorted(
    {
        int(
            re.search(
                r"Target Chain(?: Sequence)? (\d+)$", column
            ).group(1)
        )
        for column in chain_specific_columns
    }
)
if not chain_indices or chain_indices[0] != 1:
    raise RuntimeError(
        "Could not discover target-chain columns from the BindingDB header."
    )

raw = pd.read_csv(
    RAW_BINDINGDB,
    sep="\t",
    usecols=list(
        dict.fromkeys(required_columns + chain_specific_columns)
    ),
    low_memory=False,
)
raw_rows = len(raw)
raw[SOURCE_ROW_ID] = np.arange(1, raw_rows + 1, dtype=np.int64)
raw["number_of_target_chains"] = pd.to_numeric(
    raw[CHAIN_COUNT], errors="coerce"
).astype("Int64")

base_columns = [
    column
    for column in raw.columns
    if column not in ENDPOINT_COLUMNS.values()
]
endpoint_parts = []
for endpoint, source_column in ENDPOINT_COLUMNS.items():
    present = normalize_text_series(raw[source_column]).notna()
    part = raw.loc[present, base_columns].copy()
    part["endpoint_type"] = endpoint
    part["measurement_raw"] = raw.loc[present, source_column].astype("string")
    part["Measurement_ID"] = (
        BINDINGDB_RELEASE_ID
        + "__ROW_"
        + part[SOURCE_ROW_ID].astype(str)
        + "__"
        + endpoint
    )
    endpoint_parts.append(part)

all_endpoint_measurements = pd.concat(endpoint_parts, ignore_index=True)
if all_endpoint_measurements["Measurement_ID"].duplicated().any():
    raise RuntimeError("Endpoint-explicit Measurement_ID is not unique.")
print(f"Raw BindingDB rows:             {raw_rows:,}")
print(
    "Endpoint-explicit measurements: "
    f"{len(all_endpoint_measurements):,}"
)
print(
    all_endpoint_measurements["endpoint_type"]
    .value_counts()
    .reindex(ENDPOINT_COLUMNS)
    .to_string()
)

stage_rows = []
initial_counts = endpoint_counts(all_endpoint_measurements)
for endpoint, count in initial_counts.items():
    stage_rows.append(
        {
            "endpoint_type": endpoint,
            "stage": "Raw records reporting endpoint",
            "rows_before": raw_rows,
            "rows_after": int(count),
            "rows_removed": raw_rows - int(count),
            "note": "Endpoint fields are not mutually exclusive.",
        }
    )


# =============================================================================
# 5. Parse endpoint relations and save the all-target archive
# =============================================================================

print("\n" + "=" * 100)
print("STEP 2: PARSE RELATIONS AND SAVE ALL-TARGET ARCHIVE")
print("=" * 100)

(
    all_endpoint_measurements["measurement_relation"],
    all_endpoint_measurements["measurement_boundary_nM"],
) = parse_ic50_series(all_endpoint_measurements["measurement_raw"])
valid_archive_boundary = (
    all_endpoint_measurements["measurement_boundary_nM"].notna()
    & np.isfinite(all_endpoint_measurements["measurement_boundary_nM"])
    & all_endpoint_measurements["measurement_boundary_nM"].gt(0)
)
all_endpoint_measurements["p_measurement_boundary"] = np.nan
all_endpoint_measurements.loc[
    valid_archive_boundary, "p_measurement_boundary"
] = 9.0 - np.log10(
    all_endpoint_measurements.loc[
        valid_archive_boundary, "measurement_boundary_nM"
    ].astype(float)
)
all_endpoint_measurements["endpoint_semantics"] = (
    all_endpoint_measurements["endpoint_type"].map(ENDPOINT_SEMANTICS)
)
all_endpoint_measurements["target_scope"] = np.select(
    [
        all_endpoint_measurements["number_of_target_chains"]
        .eq(1)
        .fillna(False),
        all_endpoint_measurements["number_of_target_chains"]
        .gt(1)
        .fillna(False),
    ],
    ["single_chain", "multi_chain"],
    default="missing_or_invalid_chain_count",
)

archive_columns = list(
    dict.fromkeys(
        [
            "Measurement_ID",
            SOURCE_ROW_ID,
            REACTANT_ID,
            "endpoint_type",
            "endpoint_semantics",
            "measurement_raw",
            "measurement_relation",
            "measurement_boundary_nM",
            "p_measurement_boundary",
            RAW_SMILES,
            RAW_INCHI,
            RAW_INCHI_KEY,
            MONOMER_ID,
            LIGAND_NAME,
            TARGET_NAME,
            BINDINGDB_ORGANISM,
            CHAIN_COUNT,
            "number_of_target_chains",
            "target_scope",
            "pH",
            "Temp (C)",
            "Curation/DataSource",
            "Article DOI",
            "BindingDB Entry DOI",
            "PMID",
            "Patent Number",
            "Date of publication",
            "Date in BindingDB",
        ]
        + chain_specific_columns
    )
)
atomic_parquet_write(
    all_endpoint_measurements[archive_columns],
    ALL_ENDPOINT_ARCHIVE_OUT,
)
print(f"Saved all-target endpoint archive: {ALL_ENDPOINT_ARCHIVE_OUT}")

chain_stats = (
    all_endpoint_measurements.groupby(
        ["endpoint_type", "number_of_target_chains"],
        dropna=False,
    )
    .size()
    .reset_index(name="measurement_rows")
)
chain_stats["percentage_within_endpoint"] = (
    100.0
    * chain_stats["measurement_rows"]
    / chain_stats.groupby("endpoint_type")["measurement_rows"].transform("sum")
)
chain_stats.to_csv(
    MULTI_REPORT_DIR / "endpoint_chain_count_distribution.csv",
    index=False,
)


# =============================================================================
# 6. Preserve multi-chain targets once per BindingDB source row
# =============================================================================

print("\n" + "=" * 100)
print("STEP 3: BUILD NORMALIZED MULTI-CHAIN ARCHIVE")
print("=" * 100)

multichain_endpoint_rows = all_endpoint_measurements[
    all_endpoint_measurements["number_of_target_chains"].gt(1)
]
multichain_source_ids = set(multichain_endpoint_rows[SOURCE_ROW_ID])
multichain_source = raw[
    raw[SOURCE_ROW_ID].isin(multichain_source_ids)
].copy()
endpoint_membership = (
    multichain_endpoint_rows.groupby(SOURCE_ROW_ID)["endpoint_type"]
    .agg(join_unique)
    .rename("endpoint_types")
)
multichain_source = multichain_source.merge(
    endpoint_membership,
    left_on=SOURCE_ROW_ID,
    right_index=True,
    how="left",
    validate="one_to_one",
)

multichain_parts = []
for chain_index in chain_indices:
    expected = multichain_source["number_of_target_chains"].ge(chain_index)
    if not expected.any():
        continue
    part = multichain_source.loc[
        expected,
        [
            REACTANT_ID,
            SOURCE_ROW_ID,
            TARGET_NAME,
            BINDINGDB_ORGANISM,
            "number_of_target_chains",
            "endpoint_types",
        ],
    ].copy()
    part["chain_index"] = chain_index
    chain_field_map = {
        "BindingDB_Target_Sequence": (
            f"BindingDB Target Chain Sequence {chain_index}"
        ),
        "UniProt_SwissProt_Primary_ID": (
            f"UniProt (SwissProt) Primary ID of Target Chain {chain_index}"
        ),
        "UniProt_TrEMBL_Primary_ID": (
            f"UniProt (TrEMBL) Primary ID of Target Chain {chain_index}"
        ),
        "Target_Chain_PDB_IDs": (
            f"PDB ID(s) of Target Chain {chain_index}"
        ),
    }
    for output_column, source_column in chain_field_map.items():
        if source_column in multichain_source:
            part[output_column] = multichain_source.loc[
                expected, source_column
            ].values
        else:
            part[output_column] = pd.NA
    part["Sequence"] = part["BindingDB_Target_Sequence"].map(
        normalize_sequence
    )
    part["sequence_hash"] = part["Sequence"].map(
        lambda value: sha256_text(value) if value else None
    )
    part["Chain_Entity_ID"] = part["sequence_hash"].map(
        lambda value: f"CHAIN__{value[:16]}" if value else None
    )
    multichain_parts.append(part)

multichain_chains = pd.concat(multichain_parts, ignore_index=True)
multichain_chains["chain_sequence_available"] = (
    multichain_chains["Sequence"].notna()
)
chain_completeness = (
    multichain_chains.groupby(SOURCE_ROW_ID, as_index=False)
    .agg(
        expected_chains=("number_of_target_chains", "first"),
        represented_chain_rows=("chain_index", "size"),
        chains_with_sequence=("chain_sequence_available", "sum"),
    )
)
chain_completeness["all_chain_sequences_available"] = (
    chain_completeness["represented_chain_rows"].eq(
        chain_completeness["expected_chains"]
    )
    & chain_completeness["chains_with_sequence"].eq(
        chain_completeness["expected_chains"]
    )
)

complete_ids = set(
    chain_completeness.loc[
        chain_completeness["all_chain_sequences_available"], SOURCE_ROW_ID
    ]
)
target_entity_map = {}
for source_row_id, group in multichain_chains[
    multichain_chains[SOURCE_ROW_ID].isin(complete_ids)
].groupby(SOURCE_ROW_ID, sort=False):
    hashes = sorted(group["sequence_hash"].astype(str).tolist())
    target_entity_map[source_row_id] = (
        "TARGET__" + sha256_text("|".join(hashes))[:20]
    )
multichain_chains["Target_Entity_ID"] = multichain_chains[
    SOURCE_ROW_ID
].map(target_entity_map)
multichain_chains = multichain_chains.merge(
    chain_completeness[
        [SOURCE_ROW_ID, "all_chain_sequences_available"]
    ],
    on=SOURCE_ROW_ID,
    how="left",
    validate="many_to_one",
)
atomic_parquet_write(
    multichain_chains,
    MULTICHAIN_ENDPOINT_CHAINS_OUT,
)
print(f"Saved multi-chain archive: {MULTICHAIN_ENDPOINT_CHAINS_OUT}")

# The wide raw table and chain fields beyond those required for the
# single-chain workflow are no longer needed. Releasing them here materially
# reduces peak memory before chemical and protein processing.
archive_only_chain_columns = [
    column
    for column in chain_specific_columns
    if column not in required_columns
]
all_endpoint_measurements = all_endpoint_measurements.drop(
    columns=archive_only_chain_columns,
    errors="ignore",
)
del raw, multichain_source, multichain_parts
gc.collect()


# =============================================================================
# 7. Define single-chain endpoint candidates
# =============================================================================

print("\n" + "=" * 100)
print("STEP 4: DEFINE SINGLE-CHAIN ENDPOINT CANDIDATES")
print("=" * 100)

before_counts = endpoint_counts(all_endpoint_measurements)
df = all_endpoint_measurements[
    all_endpoint_measurements["number_of_target_chains"].eq(1)
].copy()
del all_endpoint_measurements
gc.collect()
report_endpoint_stage(
    stage_rows,
    df,
    "Require exactly one target chain",
    before_counts,
    "Multi-chain rows remain in the archival tables.",
)

df[RAW_SMILES] = normalize_text_series(df[RAW_SMILES])
df[UNIPROT_ID] = normalize_text_series(df[UNIPROT_ID])
before_counts = endpoint_counts(df)
df = df[df[RAW_SMILES].notna() & df[UNIPROT_ID].notna()].copy()
report_endpoint_stage(
    stage_rows,
    df,
    "Require SMILES and Swiss-Prot primary accession",
    before_counts,
)

valid_numeric = (
    df["measurement_boundary_nM"].notna()
    & np.isfinite(df["measurement_boundary_nM"])
    & df["measurement_boundary_nM"].gt(0)
)
before_counts = endpoint_counts(df)
df = df[valid_numeric].copy()
report_endpoint_stage(
    stage_rows,
    df,
    "Require positive finite endpoint boundary",
    before_counts,
)

df["p_measurement_boundary"] = (
    9.0 - np.log10(df["measurement_boundary_nM"])
)
is_exact = df["measurement_relation"].eq("=")
p_min = df["endpoint_type"].map(
    {key: value[0] for key, value in ENDPOINT_P_RANGES.items()}
)
p_max = df["endpoint_type"].map(
    {key: value[1] for key, value in ENDPOINT_P_RANGES.items()}
)
exact_out_of_range = (
    is_exact
    & (
        df["p_measurement_boundary"].lt(p_min)
        | df["p_measurement_boundary"].gt(p_max)
    )
)
atomic_parquet_write(
    df.loc[
        exact_out_of_range,
        [
            "Measurement_ID",
            SOURCE_ROW_ID,
            "endpoint_type",
            RAW_SMILES,
            UNIPROT_ID,
            "measurement_raw",
            "measurement_boundary_nM",
            "p_measurement_boundary",
        ],
    ],
    MULTI_REPORT_DIR / "excluded_exact_values_out_of_range.parquet",
)
before_counts = endpoint_counts(df)
df = df[~exact_out_of_range].copy()
report_endpoint_stage(
    stage_rows,
    df,
    "Require exact transformed value in [3, 12]",
    before_counts,
    "The range is not applied to inequality boundaries.",
)


# =============================================================================
# 8. Validate reviewed Swiss-Prot accessions and create protein entities
# =============================================================================

print("\n" + "=" * 100)
print("STEP 5: CREATE SHARED CONSTRUCT-AWARE PROTEIN ENTITIES")
print("=" * 100)

required_ids = set(df[UNIPROT_ID].dropna().astype(str).str.strip())
swiss = load_required_swissprot(required_ids)
matched_ids = set(swiss)
unmatched_ids = required_ids - matched_ids
print(f"Required accessions:           {len(required_ids):,}")
print(f"Found in reviewed Swiss-Prot: {len(matched_ids):,}")
print(f"Not found:                     {len(unmatched_ids):,}")

unmatched_report = (
    df[df[UNIPROT_ID].isin(unmatched_ids)]
    .groupby(["endpoint_type", UNIPROT_ID], as_index=False)
    .agg(
        measurement_rows=("Measurement_ID", "size"),
        unique_raw_drugs=(RAW_SMILES, "nunique"),
        BindingDB_Target_Names=(TARGET_NAME, join_unique),
        BindingDB_Organisms=(BINDINGDB_ORGANISM, join_unique),
    )
)
unmatched_report.to_csv(
    MULTI_REPORT_DIR / "excluded_unmatched_swissprot_ids.csv",
    index=False,
)
before_counts = endpoint_counts(df)
df = df[df[UNIPROT_ID].isin(matched_ids)].copy()
report_endpoint_stage(
    stage_rows,
    df,
    "Require accession in reviewed Swiss-Prot",
    before_counts,
)

df["BindingDB_Target_Sequence_Normalized"] = df[TARGET_SEQUENCE].map(
    normalize_sequence
)
df["measurement_sequence_source"] = np.where(
    df["BindingDB_Target_Sequence_Normalized"].notna(),
    "BindingDB_experimental",
    "SwissProt_fallback",
)
swiss_sequence_map = {
    uid: record["SwissProt_Sequence"] for uid, record in swiss.items()
}
fallback = df["BindingDB_Target_Sequence_Normalized"].isna()
df.loc[fallback, "BindingDB_Target_Sequence_Normalized"] = (
    df.loc[fallback, UNIPROT_ID].map(swiss_sequence_map)
)

entity_source = (
    df.groupby(
        [UNIPROT_ID, "BindingDB_Target_Sequence_Normalized"],
        as_index=False,
    )["measurement_sequence_source"]
    .agg(
        lambda values: (
            "BindingDB_experimental"
            if (values == "BindingDB_experimental").any()
            else "SwissProt_fallback_only"
        )
    )
    .rename(columns={"measurement_sequence_source": "sequence_source"})
)
entity_keys = entity_source.rename(
    columns={"BindingDB_Target_Sequence_Normalized": "Sequence"}
).copy()
entity_keys["sequence_hash"] = entity_keys["Sequence"].map(sha256_text)
entity_keys["Protein_Entity_ID"] = (
    entity_keys[UNIPROT_ID].astype(str)
    + "__"
    + entity_keys["sequence_hash"].str[:16]
)
if entity_keys["Protein_Entity_ID"].duplicated().any():
    collisions = entity_keys.loc[
        entity_keys["Protein_Entity_ID"].duplicated(keep=False),
        ["Protein_Entity_ID", UNIPROT_ID, "sequence_hash"],
    ]
    if (
        collisions.groupby("Protein_Entity_ID")["sequence_hash"]
        .nunique()
        .gt(1)
        .any()
    ):
        raise RuntimeError("Truncated Protein_Entity_ID hash collision detected.")
entity_keys["sequence_length"] = entity_keys["Sequence"].str.len().astype(
    "Int64"
)
entity_keys["seq_truncated"] = entity_keys["sequence_length"].gt(
    SEQUENCE_CONTEXT_FLAG
)

swiss_fingerprint = file_fingerprint(
    SWISSPROT_FASTA, include_sha256=True
)
alignment_cache_valid = False
if MULTI_ALIGNMENT_CACHE.exists() and MULTI_ALIGNMENT_META.exists():
    with MULTI_ALIGNMENT_META.open("r", encoding="utf-8") as handle:
        alignment_meta = json.load(handle)
    cached_alignments = pd.read_parquet(MULTI_ALIGNMENT_CACHE)
    alignment_cache_valid = (
        set(cached_alignments.get("Protein_Entity_ID", []))
        == set(entity_keys["Protein_Entity_ID"])
        and alignment_meta.get("alignment_version") == ALIGNMENT_VERSION
        and alignment_meta.get("swissprot_sha256")
        == swiss_fingerprint["sha256"]
    )
if alignment_cache_valid:
    print(f"Reusing protein alignment cache: {MULTI_ALIGNMENT_CACHE}")
else:
    records = []
    for row in entity_keys[
        [UNIPROT_ID, "Sequence", "sequence_source", "Protein_Entity_ID"]
    ].itertuples(index=False):
        result = characterize_construct(
            row.Sequence,
            swiss[row[0]]["SwissProt_Sequence"],
            row.sequence_source,
        )
        result["Protein_Entity_ID"] = row.Protein_Entity_ID
        records.append(result)
    cached_alignments = pd.DataFrame(records)
    atomic_parquet_write(cached_alignments, MULTI_ALIGNMENT_CACHE)
    with MULTI_ALIGNMENT_META.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "alignment_version": ALIGNMENT_VERSION,
                "swissprot_sha256": swiss_fingerprint["sha256"],
                "entity_count": len(entity_keys),
            },
            handle,
            indent=2,
        )
    print(f"Saved protein alignment cache: {MULTI_ALIGNMENT_CACHE}")

entity_keys = entity_keys.merge(
    cached_alignments,
    on="Protein_Entity_ID",
    how="left",
    validate="one_to_one",
)
entity_keys["SwissProt_sequence_length"] = pd.array(
    [
        swiss[uid]["SwissProt_sequence_length"]
        for uid in entity_keys[UNIPROT_ID]
    ],
    dtype="Int64",
)
entity_keys["SwissProt_sequence_hash"] = [
    swiss[uid]["SwissProt_sequence_hash"]
    for uid in entity_keys[UNIPROT_ID]
]
entity_keys["SwissProt_Sequence"] = [
    swiss[uid]["SwissProt_Sequence"] for uid in entity_keys[UNIPROT_ID]
]
entity_keys["Protein_Name"] = [
    swiss[uid]["Protein_Name"] for uid in entity_keys[UNIPROT_ID]
]
entity_keys["Organism"] = [
    swiss[uid]["Organism"] for uid in entity_keys[UNIPROT_ID]
]
entity_keys["Taxon_ID"] = pd.array(
    [swiss[uid]["Taxon_ID"] for uid in entity_keys[UNIPROT_ID]],
    dtype="Int64",
)
entity_keys["SwissProt_Entry_Name"] = [
    swiss[uid]["SwissProt_Entry_Name"] for uid in entity_keys[UNIPROT_ID]
]
entity_keys["valid_amino_acid_alphabet"] = entity_keys[
    "Sequence"
].str.fullmatch(r"[ACDEFGHIKLMNPQRSTVWYBXZJUO]+")

df = df.merge(
    entity_keys[
        [UNIPROT_ID, "Sequence", "Protein_Entity_ID"]
    ].rename(columns={"Sequence": "BindingDB_Target_Sequence_Normalized"}),
    on=[UNIPROT_ID, "BindingDB_Target_Sequence_Normalized"],
    how="left",
    validate="many_to_one",
)
if df["Protein_Entity_ID"].isna().any():
    raise RuntimeError("Protein entity assignment produced missing IDs.")

invalid_entities = entity_keys[
    ~entity_keys["valid_amino_acid_alphabet"]
].copy()
atomic_parquet_write(
    invalid_entities,
    MULTI_REPORT_DIR / "excluded_invalid_protein_sequences.parquet",
)
valid_entity_ids = set(
    entity_keys.loc[
        entity_keys["valid_amino_acid_alphabet"], "Protein_Entity_ID"
    ]
)
before_counts = endpoint_counts(df)
df = df[df["Protein_Entity_ID"].isin(valid_entity_ids)].copy()
report_endpoint_stage(
    stage_rows,
    df,
    "Require valid amino-acid alphabet",
    before_counts,
)


# =============================================================================
# 9. Standardize the union of retained drug structures
# =============================================================================

print("\n" + "=" * 100)
print("STEP 6: CREATE SHARED STANDARDIZED DRUG STRUCTURES")
print("=" * 100)

drug_map = build_drug_standardization_map(df[RAW_SMILES])
rejected_drugs = drug_map[~drug_map["standardization_valid"]].copy()
atomic_parquet_write(
    rejected_drugs,
    MULTI_REPORT_DIR / "excluded_drug_structures.parquet",
)
save_value_counts(
    rejected_drugs["exclusion_reason"],
    "exclusion_reason",
    MULTI_REPORT_DIR / "drug_exclusion_reason_counts.csv",
)
valid_drug_map = drug_map[drug_map["standardization_valid"]].copy()
valid_drug_map["structure_hash"] = valid_drug_map[
    "canonical_smiles"
].map(sha256_text)
valid_drug_map["Drug_ID"] = (
    "DRUG__" + valid_drug_map["structure_hash"].str[:20]
)
if valid_drug_map["Drug_ID"].duplicated().any():
    collisions = valid_drug_map.loc[
        valid_drug_map["Drug_ID"].duplicated(keep=False),
        ["Drug_ID", "canonical_smiles"],
    ]
    if collisions.groupby("Drug_ID")["canonical_smiles"].nunique().gt(1).any():
        raise RuntimeError("Truncated Drug_ID hash collision detected.")
df = df.merge(
    valid_drug_map,
    on=RAW_SMILES,
    how="left",
    validate="many_to_one",
)
before_counts = endpoint_counts(df)
df = df[df["canonical_smiles"].notna()].copy()
report_endpoint_stage(
    stage_rows,
    df,
    "Require valid standardized molecular structure",
    before_counts,
)


# =============================================================================
# 10. Create endpoint-specific labels and provenance
# =============================================================================

print("\n" + "=" * 100)
print("STEP 7: CREATE ENDPOINT-SPECIFIC LABELS AND PROVENANCE")
print("=" * 100)

relation = df["measurement_relation"]
value = df["measurement_boundary_nM"]
threshold = df["endpoint_type"].map(ENDPOINT_THRESHOLDS_NM).astype(float)
exact = relation.eq("=")
upper = relation.isin(["<", "<="])
strict_lower = relation.eq(">")
inclusive_lower = relation.eq(">=")

df["classification_threshold_nM"] = threshold
df["measurement_class"] = pd.Series(
    pd.NA, index=df.index, dtype="Int8"
)
df.loc[exact & value.le(threshold), "measurement_class"] = 1
df.loc[exact & value.gt(threshold), "measurement_class"] = 0
df.loc[upper & value.le(threshold), "measurement_class"] = 1
df.loc[strict_lower & value.ge(threshold), "measurement_class"] = 0
df.loc[inclusive_lower & value.gt(threshold), "measurement_class"] = 0
df["measurement_class_ambiguous"] = df["measurement_class"].isna()
df["is_exact_measurement"] = exact
df["endpoint_semantics"] = df["endpoint_type"].map(ENDPOINT_SEMANTICS)
df["p_measurement_name"] = df["endpoint_type"].map(ENDPOINT_P_NAMES)

df["pH_numeric"] = extract_first_number(df["pH"])
df["temperature_C_numeric"] = parse_temperature_c(df["Temp (C)"])
df["publication_date_parsed"] = pd.to_datetime(
    df["Date of publication"], errors="coerce"
)
df["publication_year"] = (
    df["publication_date_parsed"].dt.year.astype("Int64")
)
df["bindingdb_date_parsed"] = pd.to_datetime(
    df["Date in BindingDB"], errors="coerce"
)
df["Source_Document_ID"], df["Source_Document_Type"] = (
    create_source_document_fields(df)
)
df["Article_Source_Document_ID"] = df["Source_Document_ID"].where(
    df["Source_Document_Type"].isin(
        ["article_doi", "pmid", "pmid_multiple_dois"]
    )
)
df["Patent_Source_Document_ID"] = df["Source_Document_ID"].where(
    df["Source_Document_Type"].eq("patent")
)
df["Document_Source_Document_ID"] = df["Source_Document_ID"].where(
    df["Source_Document_Type"].isin(
        [
            "article_doi",
            "pmid",
            "pmid_multiple_dois",
            "patent",
            "bindingdb_entry_doi",
        ]
    )
)
df["Fallback_Source_Group_ID"] = df["Source_Document_ID"].where(
    df["Source_Document_Type"].isin(
        [
            "reactant_fallback",
            "source_row_fallback",
        ]
    )
)
if df["Source_Document_ID"].isna().any():
    raise RuntimeError("Source_Document_ID assignment produced missing IDs.")
save_value_counts(
    df["Source_Document_Type"],
    "source_document_type",
    MULTI_REPORT_DIR / "source_document_type_counts.csv",
)
df["target_name_construct_hint"] = target_name_construct_hint(df[TARGET_NAME])
df["construct_annotation_uncertain"] = (
    df["target_name_construct_hint"]
    & df["measurement_sequence_source"].eq("SwissProt_fallback")
)
df["has_linked_complex_PDB"] = normalize_text_series(
    df["PDB ID(s) for Ligand-Target Complex"]
).notna()
df["complex_PDB_match_status"] = np.where(
    df["has_linked_complex_PDB"],
    "unverified_bindingdb_link",
    "no_linked_complex",
)

df = df.rename(
    columns={
        REACTANT_ID: "BindingDB_Reactant_Set_ID",
        MONOMER_ID: "BindingDB_Monomer_ID",
        LIGAND_NAME: "BindingDB_Ligand_Name",
        TARGET_NAME: "BindingDB_Target_Name",
        BINDINGDB_ORGANISM: "BindingDB_Organism",
        UNIPROT_ID: "UniProt_ID",
        "pH": "pH_raw",
        "Temp (C)": "temperature_raw",
        "Curation/DataSource": "Data_Source",
        "Article DOI": "Article_DOI",
        "BindingDB Entry DOI": "BindingDB_Entry_DOI",
        "PubChem AID": "PubChem_AID",
        "Patent Number": "Patent_Number",
        "Date of publication": "Publication_Date_Raw",
        "Date in BindingDB": "BindingDB_Date_Raw",
        "Ligand HET ID in PDB": "Ligand_PDB_HET_ID",
        "PDB ID(s) for Ligand-Target Complex": (
            "Ligand_Target_Complex_PDB_IDs"
        ),
        "PubChem CID": "PubChem_CID",
        "PubChem SID": "PubChem_SID",
        "ChEBI ID of Ligand": "ChEBI_ID",
        "ChEMBL ID of Ligand": "ChEMBL_ID",
        "DrugBank ID of Ligand": "DrugBank_ID",
        "IUPHAR_GRAC ID of Ligand": "IUPHAR_GRAC_ID",
        "KEGG ID of Ligand": "KEGG_ID",
        "ZINC ID of Ligand": "ZINC_ID",
        TARGET_PDB: "Target_Chain_PDB_IDs",
        SWISSPROT_RECOMMENDED: (
            "BindingDB_SwissProt_Recommended_Name"
        ),
        SWISSPROT_ENTRY: "BindingDB_SwissProt_Entry_Name",
        SWISSPROT_SECONDARY: "BindingDB_SwissProt_Secondary_IDs",
    }
)


# =============================================================================
# 11. Save the long and endpoint-specific measurement tables
# =============================================================================

print("\n" + "=" * 100)
print("STEP 8: SAVE MEASUREMENT TABLES")
print("=" * 100)

measurement_columns = [
    "Measurement_ID",
    SOURCE_ROW_ID,
    "BindingDB_Reactant_Set_ID",
    "endpoint_type",
    "endpoint_semantics",
    "Drug_ID",
    "canonical_smiles",
    "standardized_inchi_key",
    "BindingDB_Monomer_ID",
    "BindingDB_Ligand_Name",
    "Protein_Entity_ID",
    "UniProt_ID",
    "measurement_sequence_source",
    "BindingDB_Target_Name",
    "BindingDB_Organism",
    "measurement_raw",
    "measurement_relation",
    "measurement_boundary_nM",
    "p_measurement_name",
    "p_measurement_boundary",
    "is_exact_measurement",
    "classification_threshold_nM",
    "measurement_class",
    "measurement_class_ambiguous",
    "target_name_construct_hint",
    "construct_annotation_uncertain",
    "pH_raw",
    "pH_numeric",
    "temperature_raw",
    "temperature_C_numeric",
    "Data_Source",
    "Article_DOI",
    "BindingDB_Entry_DOI",
    "Source_Document_ID",
    "Source_Document_Type",
    "PMID",
    "PubChem_AID",
    "Patent_Number",
    "Authors",
    "Publication_Date_Raw",
    "publication_year",
    "BindingDB_Date_Raw",
    "Institution",
    "Ligand_PDB_HET_ID",
    "Ligand_Target_Complex_PDB_IDs",
    "Target_Chain_PDB_IDs",
    "has_linked_complex_PDB",
    "complex_PDB_match_status",
]
measurements = df[measurement_columns].copy()
atomic_parquet_write(measurements, MEASUREMENTS_LONG_OUT)
for endpoint in ENDPOINT_COLUMNS:
    endpoint_measurements = measurements[
        measurements["endpoint_type"].eq(endpoint)
    ].copy()
    p_name = ENDPOINT_P_NAMES[endpoint]
    endpoint_measurements[f"{endpoint}_raw"] = endpoint_measurements[
        "measurement_raw"
    ]
    endpoint_measurements[f"{endpoint}_relation"] = endpoint_measurements[
        "measurement_relation"
    ]
    endpoint_measurements[f"{endpoint}_boundary_nM"] = (
        endpoint_measurements["measurement_boundary_nM"]
    )
    endpoint_measurements[f"{p_name}_boundary"] = endpoint_measurements[
        "p_measurement_boundary"
    ]
    atomic_parquet_write(
        endpoint_measurements,
        endpoint_data_path(endpoint, "measurements"),
    )
    print(
        f"Saved {endpoint} measurements: "
        f"{endpoint_data_path(endpoint, 'measurements')}"
    )


# =============================================================================
# 12. Build shared drug and protein-entity tables
# =============================================================================

print("\n" + "=" * 100)
print("STEP 9: SAVE SHARED DRUG AND PROTEIN TABLES")
print("=" * 100)

drug_metadata_columns = [
    "canonical_smiles",
    RAW_SMILES,
    RAW_INCHI_KEY,
    "BindingDB_Monomer_ID",
    "BindingDB_Ligand_Name",
    "PubChem_CID",
    "PubChem_SID",
    "ChEBI_ID",
    "ChEMBL_ID",
    "DrugBank_ID",
    "IUPHAR_GRAC_ID",
    "KEGG_ID",
    "ZINC_ID",
    "Ligand_PDB_HET_ID",
]
drug_metadata = (
    df[drug_metadata_columns]
    .drop_duplicates()
    .groupby("canonical_smiles", as_index=False)
    .agg(
        n_raw_smiles=(RAW_SMILES, "nunique"),
        BindingDB_Original_InChI_Keys=(RAW_INCHI_KEY, join_unique),
        BindingDB_Monomer_IDs=("BindingDB_Monomer_ID", join_unique),
        BindingDB_Ligand_Names=("BindingDB_Ligand_Name", join_unique),
        n_ligand_names=("BindingDB_Ligand_Name", count_unique_nonmissing),
        PubChem_CIDs=("PubChem_CID", join_unique),
        PubChem_SIDs=("PubChem_SID", join_unique),
        ChEBI_IDs=("ChEBI_ID", join_unique),
        ChEMBL_IDs=("ChEMBL_ID", join_unique),
        DrugBank_IDs=("DrugBank_ID", join_unique),
        IUPHAR_GRAC_IDs=("IUPHAR_GRAC_ID", join_unique),
        KEGG_IDs=("KEGG_ID", join_unique),
        ZINC_IDs=("ZINC_ID", join_unique),
        Ligand_PDB_HET_IDs=("Ligand_PDB_HET_ID", join_unique),
    )
)
structure_columns = [
    "Drug_ID",
    "canonical_smiles",
    "structure_hash",
    "standardized_inchi",
    "standardized_inchi_key",
    "molecular_weight",
    "heavy_atom_count",
    "contains_carbon",
    "atom_count",
    "bond_count",
    "formal_charge",
    "standardization_version",
    "rdkit_version",
]
structure_data = valid_drug_map[structure_columns].drop_duplicates(
    "canonical_smiles"
)
drug_structures = structure_data.merge(
    drug_metadata,
    on="canonical_smiles",
    how="inner",
    validate="one_to_one",
)
drug_counts = (
    df.groupby("canonical_smiles", as_index=False)
    .agg(
        n_measurement_rows=("Measurement_ID", "size"),
        n_endpoint_types=("endpoint_type", "nunique"),
        endpoint_types=("endpoint_type", join_unique),
        n_protein_entities=("Protein_Entity_ID", "nunique"),
        n_uniprot_ids=("UniProt_ID", "nunique"),
    )
)
drug_structures = drug_structures.merge(
    drug_counts,
    on="canonical_smiles",
    how="left",
    validate="one_to_one",
).sort_values("canonical_smiles").reset_index(drop=True)
atomic_parquet_write(drug_structures, DRUGS_COMMON_OUT)

protein_metadata = (
    df[
        [
            "Protein_Entity_ID",
            "BindingDB_Target_Name",
            "target_name_construct_hint",
            "construct_annotation_uncertain",
            "BindingDB_Organism",
            "BindingDB_SwissProt_Recommended_Name",
            "BindingDB_SwissProt_Entry_Name",
            "BindingDB_SwissProt_Secondary_IDs",
            "Target_Chain_PDB_IDs",
        ]
    ]
    .drop_duplicates()
    .groupby("Protein_Entity_ID", as_index=False)
    .agg(
        BindingDB_Target_Names=("BindingDB_Target_Name", join_unique),
        n_target_names=("BindingDB_Target_Name", count_unique_nonmissing),
        target_name_construct_hint=("target_name_construct_hint", "max"),
        construct_annotation_uncertain=(
            "construct_annotation_uncertain", "max"
        ),
        BindingDB_Organisms=("BindingDB_Organism", join_unique),
        BindingDB_SwissProt_Recommended_Names=(
            "BindingDB_SwissProt_Recommended_Name", join_unique
        ),
        BindingDB_SwissProt_Entry_Names=(
            "BindingDB_SwissProt_Entry_Name", join_unique
        ),
        BindingDB_SwissProt_Secondary_IDs=(
            "BindingDB_SwissProt_Secondary_IDs", join_unique
        ),
        Target_Chain_PDB_IDs=("Target_Chain_PDB_IDs", join_unique),
    )
)
protein_counts = (
    df.groupby("Protein_Entity_ID", as_index=False)
    .agg(
        n_measurement_rows=("Measurement_ID", "size"),
        n_endpoint_types=("endpoint_type", "nunique"),
        endpoint_types=("endpoint_type", join_unique),
        n_unique_drugs=("canonical_smiles", "nunique"),
    )
)
protein_entities = (
    entity_keys.merge(
        protein_metadata,
        on="Protein_Entity_ID",
        how="inner",
        validate="one_to_one",
    )
    .merge(
        protein_counts,
        on="Protein_Entity_ID",
        how="inner",
        validate="one_to_one",
    )
)
protein_entities = protein_entities[
    protein_entities["Protein_Entity_ID"].isin(set(df["Protein_Entity_ID"]))
].copy()
protein_entities = protein_entities.rename(
    columns={UNIPROT_ID: "UniProt_ID"}
).sort_values(
    ["UniProt_ID", "Protein_Entity_ID"]
).reset_index(drop=True)
atomic_parquet_write(protein_entities, PROTEINS_COMMON_OUT)
print(f"Saved shared drugs:    {DRUGS_COMMON_OUT}")
print(f"Saved shared proteins: {PROTEINS_COMMON_OUT}")


# =============================================================================
# 13. Aggregate endpoint-specific drug-protein-entity interactions
# =============================================================================

print("\n" + "=" * 100)
print("STEP 10: AGGREGATE ENDPOINT-SPECIFIC INTERACTIONS")
print("=" * 100)

df["threshold_positive_evidence"] = (
    df["measurement_class"].eq(1).fillna(False).astype("int32")
)
df["threshold_negative_evidence"] = (
    df["measurement_class"].eq(0).fillna(False).astype("int32")
)
df["ambiguous_evidence"] = (
    df["measurement_class"].isna().astype("int32")
)
df["exact_evidence"] = df["measurement_relation"].eq("=").astype("int32")
df["lower_than_evidence"] = (
    df["measurement_relation"].isin(["<", "<="]).astype("int32")
)
df["greater_than_evidence"] = (
    df["measurement_relation"].isin([">", ">="]).astype("int32")
)
df["approximate_evidence"] = (
    df["measurement_relation"].eq("approx").astype("int32")
)
df["exact_value_nM"] = df["measurement_boundary_nM"].where(
    df["measurement_relation"].eq("=")
)
df["exact_p_value"] = df["p_measurement_boundary"].where(
    df["measurement_relation"].eq("=")
)

pair_keys = ["endpoint_type", "canonical_smiles", "Protein_Entity_ID"]
interactions = (
    df.groupby(pair_keys, as_index=False)
    .agg(
        endpoint_semantics=("endpoint_semantics", "first"),
        classification_threshold_nM=(
            "classification_threshold_nM", "first"
        ),
        UniProt_ID=("UniProt_ID", "first"),
        n_measurements=("Measurement_ID", "size"),
        n_exact=("exact_evidence", "sum"),
        n_lower_than=("lower_than_evidence", "sum"),
        n_greater_than=("greater_than_evidence", "sum"),
        n_approximate=("approximate_evidence", "sum"),
        n_threshold_positive=("threshold_positive_evidence", "sum"),
        n_threshold_negative=("threshold_negative_evidence", "sum"),
        n_ambiguous=("ambiguous_evidence", "sum"),
        median_exact_value_nM=("exact_value_nM", "median"),
        min_exact_value_nM=("exact_value_nM", "min"),
        max_exact_value_nM=("exact_value_nM", "max"),
        p_value_std=("exact_p_value", "std"),
        p_value_min=("exact_p_value", "min"),
        p_value_max=("exact_p_value", "max"),
        pH_median=("pH_numeric", "median"),
        temperature_C_median=("temperature_C_numeric", "median"),
        publication_year_min=("publication_year", "min"),
        publication_year_max=("publication_year", "max"),
        n_data_sources=("Data_Source", "nunique"),
        n_source_groups=("Source_Document_ID", "nunique"),
        n_source_documents=("Document_Source_Document_ID", "nunique"),
        n_article_documents=("Article_Source_Document_ID", "nunique"),
        n_patent_documents=("Patent_Source_Document_ID", "nunique"),
        n_fallback_source_groups=("Fallback_Source_Group_ID", "nunique"),
        target_name_construct_hint=("target_name_construct_hint", "max"),
        construct_annotation_uncertain=(
            "construct_annotation_uncertain", "max"
        ),
        has_linked_complex_PDB=("has_linked_complex_PDB", "max"),
    )
)
exact_values = df[df["measurement_relation"].eq("=")][
    pair_keys + ["exact_value_nM"]
]
quartiles = (
    exact_values.groupby(pair_keys)["exact_value_nM"]
    .quantile([0.25, 0.75])
    .unstack()
    .reset_index()
    .rename(
        columns={
            0.25: "q1_exact_value_nM",
            0.75: "q3_exact_value_nM",
        }
    )
)
interactions = interactions.merge(
    quartiles,
    on=pair_keys,
    how="left",
    validate="one_to_one",
)
interactions["classification_conflict"] = (
    interactions["n_threshold_positive"].gt(0)
    & interactions["n_threshold_negative"].gt(0)
)
interactions["classification_valid"] = (
    interactions["n_threshold_positive"]
    .add(interactions["n_threshold_negative"])
    .gt(0)
    & ~interactions["classification_conflict"]
)
interactions["y_class"] = pd.Series(
    pd.NA, index=interactions.index, dtype="Int8"
)
interactions.loc[
    interactions["classification_valid"]
    & interactions["n_threshold_positive"].gt(0),
    "y_class",
] = 1
interactions.loc[
    interactions["classification_valid"]
    & interactions["n_threshold_negative"].gt(0),
    "y_class",
] = 0

interactions["regression_valid"] = interactions["n_exact"].gt(0)
interactions["p_value"] = (
    9.0 - np.log10(interactions["median_exact_value_nM"])
)
interactions["exact_value_log10_range"] = np.log10(
    interactions["max_exact_value_nM"]
    / interactions["min_exact_value_nM"]
)
interactions["measurement_dispersed"] = (
    interactions["n_exact"].gt(1)
    & interactions["exact_value_log10_range"].gt(
        DISPERSION_LOG10_THRESHOLD
    )
)
interactions["regression_quality"] = np.select(
    [
        interactions["n_exact"].eq(0),
        interactions["n_exact"].eq(1),
        interactions["measurement_dispersed"],
    ],
    [
        "no_exact_measurement",
        "single_measurement",
        "dispersed",
    ],
    default="replicated_consistent",
)

interactions = interactions.merge(
    protein_entities[
        [
            "Protein_Entity_ID",
            "Protein_Name",
            "Organism",
            "Taxon_ID",
            "sequence_status",
            "sequence_source",
            "sequence_length",
        ]
    ],
    on="Protein_Entity_ID",
    how="left",
    validate="many_to_one",
).merge(
    drug_structures[
        [
            "Drug_ID",
            "canonical_smiles",
            "BindingDB_Ligand_Names",
            "standardized_inchi_key",
        ]
    ],
    on="canonical_smiles",
    how="left",
    validate="many_to_one",
)
interaction_key_text = (
    interactions["endpoint_type"].astype(str)
    + "|"
    + interactions["Drug_ID"].astype(str)
    + "|"
    + interactions["Protein_Entity_ID"].astype(str)
)
interactions["interaction_hash"] = interaction_key_text.map(sha256_text)
interactions["Interaction_ID"] = (
    "INTERACTION__" + interactions["interaction_hash"].str[:20]
)
first_columns = [
    "Interaction_ID",
    "endpoint_type",
    "endpoint_semantics",
    "Drug_ID",
    "canonical_smiles",
    "standardized_inchi_key",
    "BindingDB_Ligand_Names",
    "Protein_Entity_ID",
    "UniProt_ID",
    "Protein_Name",
    "Organism",
    "Taxon_ID",
    "sequence_status",
    "sequence_source",
    "sequence_length",
]
interactions = interactions[
    first_columns
    + [
        column
        for column in interactions.columns
        if column not in first_columns
    ]
].sort_values(pair_keys).reset_index(drop=True)
atomic_parquet_write(interactions, INTERACTIONS_LONG_OUT)

for endpoint in ENDPOINT_COLUMNS:
    view = interactions[
        interactions["endpoint_type"].eq(endpoint)
    ].copy()
    p_name = ENDPOINT_P_NAMES[endpoint]
    view[f"{endpoint}_nM"] = view["median_exact_value_nM"]
    view[f"{endpoint}_min_nM"] = view["min_exact_value_nM"]
    view[f"{endpoint}_max_nM"] = view["max_exact_value_nM"]
    view[f"{endpoint}_q1_nM"] = view["q1_exact_value_nM"]
    view[f"{endpoint}_q3_nM"] = view["q3_exact_value_nM"]
    view[p_name] = view["p_value"]
    view[f"{p_name}_std"] = view["p_value_std"]
    view[f"{p_name}_min"] = view["p_value_min"]
    view[f"{p_name}_max"] = view["p_value_max"]
    atomic_parquet_write(
        view,
        endpoint_data_path(endpoint, "interactions"),
    )
    print(
        f"Saved {endpoint} interactions: "
        f"{endpoint_data_path(endpoint, 'interactions')}"
    )


# =============================================================================
# 14. Construct-aware and cross-endpoint audits
# =============================================================================

print("\n" + "=" * 100)
print("STEP 11: RUN CONSTRUCT AND CROSS-ENDPOINT AUDITS")
print("=" * 100)

audit_rows = []
construct_examples = []
for endpoint in ENDPOINT_COLUMNS:
    endpoint_pairs = interactions[
        interactions["endpoint_type"].eq(endpoint)
    ].copy()
    valid_pairs = endpoint_pairs[
        endpoint_pairs["classification_valid"]
        & endpoint_pairs["y_class"].notna()
    ]
    accession_groups = (
        valid_pairs.groupby(
            ["canonical_smiles", "UniProt_ID"],
            as_index=False,
        )
        .agg(
            n_entities=("Protein_Entity_ID", "nunique"),
            n_labels=("y_class", "nunique"),
            n_construct_pairs=("Protein_Entity_ID", "size"),
        )
    )
    dependent = accession_groups[
        accession_groups["n_entities"].gt(1)
        & accession_groups["n_labels"].gt(1)
    ]
    if not dependent.empty:
        detail = valid_pairs.merge(
            dependent[["canonical_smiles", "UniProt_ID"]],
            on=["canonical_smiles", "UniProt_ID"],
            how="inner",
        )
        construct_examples.append(detail)

    uniprot_level = (
        df[df["endpoint_type"].eq(endpoint)]
        .groupby(["canonical_smiles", "UniProt_ID"], as_index=False)
        .agg(
            n_positive=("threshold_positive_evidence", "sum"),
            n_negative=("threshold_negative_evidence", "sum"),
        )
    )
    uniprot_level["classification_conflict"] = (
        uniprot_level["n_positive"].gt(0)
        & uniprot_level["n_negative"].gt(0)
    )
    audit_rows.append(
        {
            "endpoint_type": endpoint,
            "construct_aware_pairs": len(endpoint_pairs),
            "uniprot_level_pairs": len(uniprot_level),
            "construct_aware_classification_conflicts": int(
                endpoint_pairs["classification_conflict"].sum()
            ),
            "uniprot_level_classification_conflicts": int(
                uniprot_level["classification_conflict"].sum()
            ),
            "drug_uniprot_groups_with_multiple_entities": int(
                accession_groups["n_entities"].gt(1).sum()
            ),
            "construct_dependent_classification_groups": len(dependent),
            "dispersed_exact_pairs": int(
                endpoint_pairs["measurement_dispersed"].sum()
            ),
        }
    )

construct_audit = pd.DataFrame(audit_rows)
construct_audit.to_csv(
    MULTI_REPORT_DIR / "construct_audit_by_endpoint.csv",
    index=False,
)
if construct_examples:
    atomic_parquet_write(
        pd.concat(construct_examples, ignore_index=True),
        MULTI_REPORT_DIR
        / "construct_dependent_classification_pairs.parquet",
    )

endpoint_pair_membership = (
    interactions.groupby(
        ["canonical_smiles", "Protein_Entity_ID"],
        as_index=False,
    )
    .agg(
        n_endpoint_types=("endpoint_type", "nunique"),
        endpoint_types=("endpoint_type", join_unique),
    )
)
save_value_counts(
    endpoint_pair_membership["n_endpoint_types"],
    "n_endpoint_types",
    MULTI_REPORT_DIR / "pair_endpoint_overlap_counts.csv",
)

# A compact boolean presence matrix avoids materializing millions of Python
# tuple objects solely to count cross-endpoint pair overlap.
pair_presence = (
    interactions[
        ["canonical_smiles", "Protein_Entity_ID", "endpoint_type"]
    ]
    .assign(present=np.uint8(1))
    .pivot_table(
        index=["canonical_smiles", "Protein_Entity_ID"],
        columns="endpoint_type",
        values="present",
        aggfunc="max",
        fill_value=0,
    )
    .reindex(columns=list(ENDPOINT_COLUMNS), fill_value=0)
    .astype("int64")
)
overlap_matrix = pair_presence.T.dot(pair_presence)
overlap_matrix.to_csv(
    MULTI_REPORT_DIR / "endpoint_pair_overlap_matrix.csv"
)
del pair_presence
gc.collect()


# =============================================================================
# 15. Integrity checks, statistics, and manifest
# =============================================================================

print("\n" + "=" * 100)
print("STEP 12: VALIDATE AND SAVE RELEASE REPORTS")
print("=" * 100)

if measurements["Measurement_ID"].isna().any():
    raise RuntimeError("Measurement_ID contains missing values.")
if measurements["Measurement_ID"].duplicated().any():
    raise RuntimeError("Measurement_ID contains duplicates.")
if measurements["Source_Document_ID"].isna().any():
    raise RuntimeError("Measurement table contains missing source groups.")
if interactions[pair_keys].duplicated().any():
    raise RuntimeError("Endpoint interaction keys contain duplicates.")
if interactions["Interaction_ID"].isna().any():
    raise RuntimeError("Interaction_ID contains missing values.")
if interactions["Interaction_ID"].duplicated().any():
    raise RuntimeError("Truncated Interaction_ID hash collision detected.")
if drug_structures["Drug_ID"].isna().any():
    raise RuntimeError("Drug_ID contains missing values.")
if drug_structures["Drug_ID"].duplicated().any():
    raise RuntimeError("Drug_ID contains duplicates.")
if drug_structures["canonical_smiles"].duplicated().any():
    raise RuntimeError("Shared drug table contains duplicate structures.")
if protein_entities["Protein_Entity_ID"].duplicated().any():
    raise RuntimeError("Shared protein table contains duplicate entities.")
if not set(interactions["canonical_smiles"]).issubset(
    set(drug_structures["canonical_smiles"])
):
    raise RuntimeError("Interaction table references missing drugs.")
if not set(measurements["Drug_ID"]).issubset(set(drug_structures["Drug_ID"])):
    raise RuntimeError("Measurement table references missing Drug_ID values.")
if not set(interactions["Drug_ID"]).issubset(set(drug_structures["Drug_ID"])):
    raise RuntimeError("Interaction table references missing Drug_ID values.")
if not set(measurements["Protein_Entity_ID"]).issubset(
    set(protein_entities["Protein_Entity_ID"])
):
    raise RuntimeError("Measurement table references missing proteins.")
if not set(interactions["Protein_Entity_ID"]).issubset(
    set(protein_entities["Protein_Entity_ID"])
):
    raise RuntimeError("Interaction table references missing proteins.")
if interactions.loc[
    interactions["classification_valid"], "y_class"
].isna().any():
    raise RuntimeError("Valid classification rows have missing labels.")
if interactions.loc[
    interactions["regression_valid"], "p_value"
].isna().any():
    raise RuntimeError("Valid regression rows have missing targets.")

stage_counts = pd.DataFrame(stage_rows)
stage_counts.to_csv(
    MULTI_REPORT_DIR / "preprocessing_stage_counts_by_endpoint.csv",
    index=False,
)

summary_rows = []
for endpoint in ENDPOINT_COLUMNS:
    endpoint_measurements = measurements[
        measurements["endpoint_type"].eq(endpoint)
    ]
    endpoint_interactions = interactions[
        interactions["endpoint_type"].eq(endpoint)
    ]
    summary_rows.append(
        {
            "endpoint_type": endpoint,
            "measurement_rows": len(endpoint_measurements),
            "unique_standardized_drugs": (
                endpoint_measurements["canonical_smiles"].nunique()
            ),
            "unique_protein_entities": (
                endpoint_measurements["Protein_Entity_ID"].nunique()
            ),
            "unique_uniprot_accessions": (
                endpoint_measurements["UniProt_ID"].nunique()
            ),
            "interaction_pairs": len(endpoint_interactions),
            "valid_classification_pairs": int(
                endpoint_interactions["classification_valid"].sum()
            ),
            "class_1_pairs": int(
                endpoint_interactions["y_class"].eq(1).sum()
            ),
            "class_0_pairs": int(
                endpoint_interactions["y_class"].eq(0).sum()
            ),
            "classification_conflicts": int(
                endpoint_interactions["classification_conflict"].sum()
            ),
            "valid_regression_pairs": int(
                endpoint_interactions["regression_valid"].sum()
            ),
            "dispersed_regression_pairs": int(
                endpoint_interactions["measurement_dispersed"].sum()
            ),
        }
    )
summary = pd.DataFrame(summary_rows)
summary.to_csv(
    MULTI_REPORT_DIR / "final_endpoint_statistics.csv",
    index=False,
)

for endpoint in ENDPOINT_COLUMNS:
    endpoint_report_dir = MULTI_REPORT_DIR / endpoint
    endpoint_measurements = measurements[
        measurements["endpoint_type"].eq(endpoint)
    ]
    endpoint_interactions = interactions[
        interactions["endpoint_type"].eq(endpoint)
    ]
    save_value_counts(
        endpoint_measurements["measurement_relation"],
        "measurement_relation",
        endpoint_report_dir / "measurement_relation_counts.csv",
    )
    save_value_counts(
        endpoint_interactions["regression_quality"],
        "regression_quality",
        endpoint_report_dir / "regression_quality_counts.csv",
    )
    save_value_counts(
        endpoint_interactions["y_class"],
        "y_class",
        endpoint_report_dir / "classification_label_counts.csv",
    )

input_fingerprints = {
    "bindingdb_tsv": file_fingerprint(
        RAW_BINDINGDB, include_sha256=True
    ),
    "swissprot_fasta": swiss_fingerprint,
}
manifest = {
    "release_id": BINDINGDB_RELEASE_ID,
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "scope": (
        "Construct-aware, single-chain, endpoint-explicit BindingDB "
        "benchmark with separate IC50, Ki, Kd, and EC50 targets."
    ),
    "endpoint_columns": ENDPOINT_COLUMNS,
    "endpoint_semantics": ENDPOINT_SEMANTICS,
    "endpoint_thresholds_nM": ENDPOINT_THRESHOLDS_NM,
    "endpoint_p_ranges": ENDPOINT_P_RANGES,
    "regression_policy": (
        "Only exact positive measurements provide conventional regression "
        "targets; endpoints are never pooled."
    ),
    "classification_policy": (
        "Exact and logically conclusive qualified measurements provide "
        "threshold evidence. Labels are endpoint-specific and do not prove "
        "physical interaction or non-interaction."
    ),
    "dispersion_log10_threshold": DISPERSION_LOG10_THRESHOLD,
    "standardization_version": STANDARDIZATION_VERSION,
    "identifier_policy": {
        "Measurement_ID": (
            "BindingDB release ID, one-based source-row number, and endpoint "
            "type. It is unique within and across versioned releases."
        ),
        "Drug_ID": (
            "DRUG__ followed by the first 20 hexadecimal characters of the "
            "SHA-256 hash of the standardized canonical isomeric SMILES."
        ),
        "Protein_Entity_ID": (
            "Reviewed UniProt accession followed by the first 16 hexadecimal "
            "characters of the SHA-256 hash of the normalized sequence."
        ),
        "Interaction_ID": (
            "INTERACTION__ followed by the first 20 hexadecimal characters "
            "of the SHA-256 hash of endpoint type, Drug_ID, and "
            "Protein_Entity_ID."
        ),
        "Source_Document_ID": (
            "Hierarchical DOI, PMID, patent, BindingDB-entry DOI, reactant-set, "
            "or release-specific source-row grouping identifier."
        ),
    },
    "molecular_validity_policy": (
        "A structure is excluded only when parsing, cleanup, parent "
        "selection, sanitization, or generation of one unambiguous parent "
        "structure fails. Carbon content, heavy-atom count, and molecular "
        "weight do not determine inclusion."
    ),
    "rdkit_version": rdBase.rdkitVersion,
    "alignment_version": ALIGNMENT_VERSION,
    "sequence_category_precedence": SEQUENCE_CATEGORY_PRECEDENCE,
    "input_fingerprints": input_fingerprints,
    "outputs": {
        "shared_drugs": str(DRUGS_COMMON_OUT.relative_to(RELEASE_DIR)),
        "shared_proteins": str(PROTEINS_COMMON_OUT.relative_to(RELEASE_DIR)),
        "all_measurements_long": str(
            MEASUREMENTS_LONG_OUT.relative_to(RELEASE_DIR)
        ),
        "all_interactions_long": str(
            INTERACTIONS_LONG_OUT.relative_to(RELEASE_DIR)
        ),
        "all_target_archive": str(
            ALL_ENDPOINT_ARCHIVE_OUT.relative_to(RELEASE_DIR)
        ),
        "multichain_archive": str(
            MULTICHAIN_ENDPOINT_CHAINS_OUT.relative_to(RELEASE_DIR)
        ),
        "source_document_type_report": str(
            (
                MULTI_REPORT_DIR / "source_document_type_counts.csv"
            ).relative_to(RELEASE_DIR)
        ),
        "endpoint_measurements": {
            endpoint: str(
                endpoint_data_path(endpoint, "measurements").relative_to(
                    RELEASE_DIR
                )
            )
            for endpoint in ENDPOINT_COLUMNS
        },
        "endpoint_interactions": {
            endpoint: str(
                endpoint_data_path(endpoint, "interactions").relative_to(
                    RELEASE_DIR
                )
            )
            for endpoint in ENDPOINT_COLUMNS
        },
    },
    "final_endpoint_statistics": summary.to_dict(orient="records"),
}
with (MULTI_REPORT_DIR / "benchmark_manifest.json").open(
    "w", encoding="utf-8"
) as handle:
    json.dump(manifest, handle, indent=2)

with (MULTI_REPORT_DIR / "summary.txt").open(
    "w", encoding="utf-8"
) as handle:
    handle.write("BINDINGDB MULTI-ENDPOINT CONSTRUCT-AWARE RELEASE\n")
    handle.write("=" * 80 + "\n\n")
    handle.write(summary.to_string(index=False))
    handle.write("\n\nCONSTRUCT AUDIT\n")
    handle.write(construct_audit.to_string(index=False))
    handle.write("\n")

print("\nFINAL ENDPOINT STATISTICS")
print(summary.to_string(index=False))
print("\nIntegrity checks: PASS")
print(f"Release directory: {RELEASE_DIR}")
print(f"Reports:           {MULTI_REPORT_DIR}")
