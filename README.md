# Curate-DTI

Reproducible preprocessing and validation code for **Curate-DTI**, a
construct-aware, multi-endpoint drug--protein bioactivity dataset derived from
the 1 July 2026 BindingDB export.

Curate-DTI processes IC50, Ki, Kd and EC50 independently, preserves exact and
inequality-qualified measurements, standardizes ligand structures, and defines
protein entities using both a reviewed UniProt accession and the experimental
target sequence reported by BindingDB.

## Dataset

The Curate-DTI v1.0.0 data release is archived in Zenodo:

<https://doi.org/10.5281/zenodo.21879265>

The dataset contains:

| Endpoint | Measurements | Interactions |
|---|---:|---:|
| IC50 | 1,982,041 | 1,548,382 |
| Ki | 565,541 | 445,298 |
| Kd | 124,325 | 86,858 |
| EC50 | 247,810 | 195,145 |
| **Total** | **2,919,717** | **2,275,683** |

The data archive, column dictionary, validation reports and file checksums are
provided in the Zenodo release rather than this code repository.

## Repository contents

```text
Curate-DTI/
├── scripts/
│   ├── preprocess_bindingdb_benchmark.py
│   ├── verify_bindingdb_endpoint_counts.py
│   ├── validate_bindingdb_provenance.py
│   ├── audit_duplicate_measurements.py
│   ├── generate_bindingdb_paper_statistics.py
│   └── generate_curate_dti_column_dictionary.py
├── environment.yml
├── requirements.txt
├── CITATION.cff
├── LICENSE
└── README.md
```

## Input files

The source files are not redistributed through this repository. Download them
from their official providers and place them in `raw_inputs/`:

```text
raw_inputs/
├── bindingdb_july_2026.tsv
└── uniprot_sprot.fasta.gz
```

The released dataset was generated from the BindingDB export dated 1 July
2026 and a reviewed UniProtKB/Swiss-Prot FASTA file downloaded on 23 July 2026.
The dataset manifest records the filenames and checksums of the frozen inputs.

## Installation

Using Conda is recommended:

```bash
conda env create -f environment.yml
conda activate curate-dti
```

Alternatively, use Python 3.12.4 and install the pinned packages:

```bash
python -m pip install -r requirements.txt
```

## Run the preprocessing pipeline

From the repository root:

```bash
export DTI_PROJECT_DIR="$PWD"
export BINDINGDB_RELEASE_DATE="2026-07-01"
export BINDINGDB_RELEASE_ID="BindingDB_2026-07-01"

python scripts/preprocess_bindingdb_benchmark.py
```

The pipeline reads the frozen files from `raw_inputs/`, writes reusable caches
to `preprocessing/`, and creates the versioned release under
`processed/BindingDB_2026-07-01/`.

The input locations can be overridden explicitly:

```bash
export BINDINGDB_TSV="/path/to/bindingdb_july_2026.tsv"
export SWISSPROT_FASTA="/path/to/uniprot_sprot.fasta.gz"
```

## Validation and reporting

The remaining scripts support independent endpoint-count verification,
provenance validation, candidate duplicate-evidence auditing, manuscript
statistics and construction of the released column dictionary. Run them from
the repository root after preprocessing. Scripts exposing command-line options
can be inspected with `--help`.

Example:

```bash
python scripts/generate_bindingdb_paper_statistics.py \
  --project-dir . \
  --release-id BindingDB_2026-07-01
```

The duplicate audit identifies candidate repeated evidence but does not remove
or modify dataset records.

## Important interpretation notes

- IC50, Ki, Kd and EC50 are separate endpoint-specific tasks and should not be
  pooled into one regression target.
- Conventional regression targets use exact measurements only.
- Inequality-qualified measurements are retained for relation-aware
  classification and possible censored-regression analyses.
- The supplied classification labels use an operational threshold of 1,000 nM
  and are not universal definitions of binding or biological inactivity.
- Protein entities distinguish reported experimental sequences associated with
  the same reviewed UniProt accession.
- The multi-chain archive is not part of the model-ready single-chain dataset.

Consult the dataset README and column dictionary before selecting modeling
fields or constructing evaluation splits.

## Citation

Please cite the Zenodo dataset record and the associated Data Descriptor. A
machine-readable software citation is provided in `CITATION.cff`.

## Licence

The original code in this repository is released under the MIT License. The
Curate-DTI dataset and documentation are distributed separately under CC
BY-SA 4.0, subject to the upstream source licences described in the dataset
release.

