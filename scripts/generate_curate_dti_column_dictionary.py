#!/usr/bin/env python3
"""Generate the public Curate-DTI column dictionary from its schema inventory."""

from pathlib import Path
import re
import pandas as pd


RELEASE = Path("release/Curate-DTI_v1.0.0")
INPUT = RELEASE / "documentation" / "parquet_schema_inventory.csv"
OUTPUT = RELEASE / "documentation" / "column_dictionary.csv"


DESCRIPTIONS = {
    "Measurement_ID": "Stable identifier for one endpoint-specific measurement record.",
    "Interaction_ID": "Stable identifier for one endpoint-specific standardized drug--protein-entity pair.",
    "Drug_ID": "Stable drug identifier derived from the standardized canonical isomeric SMILES.",
    "Protein_Entity_ID": "Construct-aware protein identifier derived from the reviewed UniProt accession and normalized experimental sequence.",
    "UniProt_ID": "Reviewed Swiss-Prot primary accession associated with the protein entity.",
    "BindingDB_Source_Row_ID": "One-based identifier assigned to the original BindingDB source row during preprocessing.",
    "BindingDB_Reactant_Set_ID": "BindingDB reactant-set identifier reported in the source export.",
    "BindingDB Reactant_set_id": "BindingDB reactant-set identifier reported in the source export.",
    "endpoint_type": "Endpoint represented by the record.",
    "endpoint_semantics": "Endpoint-specific biological interpretation used by Curate-DTI.",
    "canonical_smiles": "RDKit-standardized canonical isomeric SMILES defining drug identity.",
    "structure_hash": "Complete SHA-256 hash of the standardized canonical isomeric SMILES.",
    "standardized_inchi": "Standardized InChI generated from the retained parent structure.",
    "standardized_inchi_key": "InChIKey generated from the standardized InChI.",
    "molecular_weight": "RDKit molecular weight of the standardized parent structure.",
    "heavy_atom_count": "Number of non-hydrogen atoms in the standardized parent structure.",
    "contains_carbon": "Whether the standardized parent structure contains at least one carbon atom; descriptive only.",
    "atom_count": "Total number of explicit atoms in the standardized parent structure.",
    "bond_count": "Number of bonds in the standardized parent structure.",
    "formal_charge": "Sum of atomic formal charges in the standardized parent structure.",
    "standardization_version": "Version label identifying the molecular-standardization procedure.",
    "rdkit_version": "RDKit version used for molecular parsing and standardization.",
    "n_raw_smiles": "Number of distinct reported ligand SMILES mapping to the standardized drug.",
    "BindingDB_Original_InChI_Keys": "Distinct original BindingDB InChIKeys associated with the standardized drug.",
    "BindingDB_Monomer_IDs": "Distinct BindingDB monomer identifiers associated with the standardized drug.",
    "BindingDB_Ligand_Names": "Distinct reported BindingDB ligand names associated with the standardized drug or interaction.",
    "BindingDB_Ligand_Name": "BindingDB ligand name reported for the measurement.",
    "n_ligand_names": "Number of distinct reported ligand names associated with the standardized drug.",
    "PubChem_CIDs": "Distinct PubChem compound identifiers associated with the standardized drug.",
    "PubChem_SIDs": "Distinct PubChem substance identifiers associated with the standardized drug.",
    "ChEBI_IDs": "Distinct ChEBI identifiers associated with the standardized drug.",
    "ChEMBL_IDs": "Distinct ChEMBL identifiers associated with the standardized drug.",
    "DrugBank_IDs": "Distinct DrugBank identifiers associated with the standardized drug.",
    "IUPHAR_GRAC_IDs": "Distinct IUPHAR/BPS Guide to PHARMACOLOGY identifiers associated with the standardized drug.",
    "KEGG_IDs": "Distinct KEGG identifiers associated with the standardized drug.",
    "ZINC_IDs": "Distinct ZINC identifiers associated with the standardized drug.",
    "Ligand_PDB_HET_IDs": "Distinct PDB ligand component identifiers associated with the standardized drug.",
    "n_measurement_rows": "Number of retained measurement rows associated with the drug or protein entity.",
    "n_endpoint_types": "Number of distinct endpoint types associated with the drug or protein entity.",
    "endpoint_types": "Distinct endpoint types associated with the record.",
    "n_protein_entities": "Number of distinct construct-aware protein entities associated with the standardized drug.",
    "n_uniprot_ids": "Number of distinct reviewed UniProt accessions associated with the standardized drug.",
    "Sequence": "Normalized experimental amino-acid sequence defining the protein entity.",
    "sequence_source": "Source from which the sequence defining the protein entity was obtained.",
    "measurement_sequence_source": "Source of the protein sequence assigned to the measurement.",
    "sequence_hash": "Complete SHA-256 hash of the normalized amino-acid sequence.",
    "sequence_length": "Length of the normalized experimental amino-acid sequence.",
    "seq_truncated": "Whether sequence truncation was detected or recorded during preprocessing.",
    "alignment_version": "Version label identifying the sequence-comparison procedure and parameters.",
    "alignment_performed": "Whether alignment-based comparison with the canonical Swiss-Prot sequence was performed.",
    "alignment_score": "Score returned by the configured pairwise sequence alignment.",
    "sequence_identity": "Fraction of identical residues under the reported alignment definition.",
    "identity_ungapped": "Fraction of identical residues among aligned non-gap residue pairs.",
    "canonical_coverage": "Fraction of the canonical Swiss-Prot sequence covered by the experimental sequence alignment.",
    "construct_coverage": "Fraction of the experimental sequence covered by the alignment.",
    "n_matches": "Number of identical aligned residue pairs.",
    "n_substitutions": "Number of aligned residue substitutions.",
    "n_insertions": "Number of residues classified as insertions relative to the canonical sequence.",
    "n_deletions": "Number of residues classified as deletions relative to the canonical sequence.",
    "internal_insertions": "Number of insertions occurring internally rather than at sequence termini.",
    "internal_deletions": "Number of deletions occurring internally rather than at sequence termini.",
    "canonical_start": "Zero-based inclusive start coordinate of the aligned experimental region in the canonical sequence.",
    "canonical_end": "Zero-based exclusive end coordinate of the aligned experimental region in the canonical sequence.",
    "is_contiguous_fragment": "Whether the experimental sequence is an exact contiguous fragment of the canonical sequence.",
    "is_single_substitution_full_coverage": "Whether the sequence covers the full canonical sequence and differs by one substitution.",
    "has_terminal_extension": "Whether the experimental sequence contains residues extending beyond a canonical terminus.",
    "has_polyhistidine_extension": "Whether a terminal extension contains a possible polyhistidine affinity tag.",
    "sequence_status": "Deterministic descriptive category assigned by comparison with the canonical Swiss-Prot sequence.",
    "SwissProt_sequence_length": "Length of the canonical Swiss-Prot amino-acid sequence.",
    "SwissProt_sequence_hash": "Complete SHA-256 hash of the canonical Swiss-Prot sequence.",
    "SwissProt_Sequence": "Canonical amino-acid sequence from the frozen reviewed Swiss-Prot reference.",
    "Protein_Name": "Reviewed protein name obtained from Swiss-Prot.",
    "Organism": "Canonical organism associated with the reviewed Swiss-Prot accession.",
    "Taxon_ID": "NCBI taxonomy identifier associated with the Swiss-Prot organism.",
    "SwissProt_Entry_Name": "Swiss-Prot entry name associated with the reviewed accession.",
    "valid_amino_acid_alphabet": "Whether the normalized experimental sequence passed the configured amino-acid alphabet validation.",
    "BindingDB_Target_Names": "Distinct BindingDB target names associated with the protein entity.",
    "n_target_names": "Number of distinct BindingDB target names associated with the protein entity.",
    "target_name_construct_hint": "Whether a BindingDB target name contains wording suggestive of a construct, domain, mutation or tag.",
    "construct_annotation_uncertain": "Whether construct interpretation remains uncertain from the available name and sequence metadata.",
    "BindingDB_Organisms": "Distinct organism descriptions reported by BindingDB for the protein entity.",
    "BindingDB_SwissProt_Recommended_Names": "Distinct BindingDB-reported Swiss-Prot recommended names.",
    "BindingDB_SwissProt_Entry_Names": "Distinct BindingDB-reported Swiss-Prot entry names.",
    "BindingDB_SwissProt_Secondary_IDs": "Distinct BindingDB-reported Swiss-Prot secondary accessions.",
    "Target_Chain_PDB_IDs": "Available PDB identifiers reported for the target chain.",
    "n_unique_drugs": "Number of distinct standardized drugs associated with the protein entity.",
    "measurement_raw": "Original endpoint text retained for auditability.",
    "measurement_relation": "Normalized relation associated with the endpoint boundary.",
    "measurement_boundary_nM": "Positive finite numerical endpoint boundary expressed in nanomolar units.",
    "p_measurement_name": "Name of the endpoint-specific negative-logarithmic value.",
    "p_measurement_boundary": "Negative base-10 logarithm of the molar measurement boundary.",
    "is_exact_measurement": "Whether the normalized measurement relation is exact.",
    "classification_threshold_nM": "Operational endpoint threshold used to generate the optional benchmark classification view.",
    "measurement_class": "Measurement-level threshold evidence: 1 positive, 0 negative, or missing when ambiguous.",
    "measurement_class_ambiguous": "Whether the measurement does not determine one side of the operational threshold.",
    "BindingDB_Target_Name": "Target name reported by BindingDB for the measurement.",
    "BindingDB_Organism": "Target organism description reported by BindingDB for the measurement.",
    "BindingDB_Monomer_ID": "BindingDB monomer identifier reported for the measurement ligand.",
    "pH_raw": "Original BindingDB pH value.",
    "pH_numeric": "First valid numerical pH value extracted from the reported field.",
    "temperature_raw": "Original BindingDB temperature value.",
    "temperature_C_numeric": "Parsed assay temperature in degrees Celsius.",
    "Data_Source": "BindingDB curation or data-source designation.",
    "Article_DOI": "Normalized article digital object identifier, when available.",
    "BindingDB_Entry_DOI": "BindingDB entry DOI, when available.",
    "Source_Document_ID": "Stable source-group identifier derived from available DOI, PMID, patent or fallback metadata.",
    "Source_Document_Type": "Provenance field used to construct Source_Document_ID.",
    "PMID": "PubMed identifier, when available.",
    "PubChem_AID": "PubChem assay identifier, when available.",
    "Patent_Number": "Patent number, when available.",
    "Authors": "Authors reported in the BindingDB source metadata.",
    "Publication_Date_Raw": "Original publication-date value from BindingDB.",
    "publication_year": "Parsed publication year.",
    "BindingDB_Date_Raw": "Original date-in-BindingDB value.",
    "Institution": "Institution reported in the BindingDB source metadata.",
    "Ligand_PDB_HET_ID": "PDB ligand component identifier reported for the measurement.",
    "Ligand_Target_Complex_PDB_IDs": "PDB identifiers reported for ligand--target complexes.",
    "has_linked_complex_PDB": "Whether at least one linked ligand--target complex PDB identifier is available.",
    "complex_PDB_match_status": "Status describing agreement among available ligand and target PDB identifiers.",
    "n_measurements": "Number of measurement records aggregated into the interaction.",
    "n_exact": "Number of exact measurements contributing to the interaction.",
    "n_lower_than": "Number of normalized less-than measurements contributing to the interaction.",
    "n_greater_than": "Number of normalized greater-than measurements contributing to the interaction.",
    "n_approximate": "Number of approximate measurements contributing to the interaction.",
    "n_threshold_positive": "Number of measurements providing conclusive threshold-positive evidence.",
    "n_threshold_negative": "Number of measurements providing conclusive threshold-negative evidence.",
    "n_ambiguous": "Number of measurements whose relation and boundary do not determine one side of the threshold.",
    "median_exact_value_nM": "Median of contributing exact endpoint values, calculated in nanomolar space.",
    "min_exact_value_nM": "Minimum contributing exact endpoint value in nanomolar units.",
    "max_exact_value_nM": "Maximum contributing exact endpoint value in nanomolar units.",
    "q1_exact_value_nM": "First quartile of contributing exact endpoint values in nanomolar units.",
    "q3_exact_value_nM": "Third quartile of contributing exact endpoint values in nanomolar units.",
    "p_value": "Negative-logarithmic regression target calculated from the median exact endpoint value.",
    "p_value_std": "Standard deviation of contributing exact negative-logarithmic endpoint values.",
    "p_value_min": "Minimum contributing exact negative-logarithmic endpoint value.",
    "p_value_max": "Maximum contributing exact negative-logarithmic endpoint value.",
    "pH_median": "Median parsed pH among contributing measurements.",
    "temperature_C_median": "Median parsed temperature in degrees Celsius among contributing measurements.",
    "publication_year_min": "Earliest parsed publication year among contributing measurements.",
    "publication_year_max": "Latest parsed publication year among contributing measurements.",
    "n_data_sources": "Number of distinct BindingDB data-source values among contributing measurements.",
    "n_source_groups": "Number of distinct source groups among contributing measurements.",
    "n_source_documents": "Number of distinct nonmissing source-document identifiers among contributing measurements.",
    "n_article_documents": "Number of distinct article-derived source documents among contributing measurements.",
    "n_patent_documents": "Number of distinct patent-derived source documents among contributing measurements.",
    "n_fallback_source_groups": "Number of source groups formed using the fallback identifier policy.",
    "classification_conflict": "Whether both threshold-positive and threshold-negative evidence occur for the interaction.",
    "classification_valid": "Whether the interaction has a nonconflicting binary benchmark label.",
    "y_class": "Optional interaction-level benchmark label: 1 threshold-positive, 0 threshold-negative, missing when invalid.",
    "regression_valid": "Whether at least one exact measurement supports a conventional regression target.",
    "exact_value_log10_range": "Log10 span between the maximum and minimum contributing exact endpoint values.",
    "measurement_dispersed": "Whether contributing exact measurements span at least the configured dispersion threshold.",
    "regression_quality": "Categorical summary of exact-measurement support and dispersion.",
    "interaction_hash": "Complete hash underlying the stable interaction identifier.",
    "number_of_target_chains": "Reported number of protein chains in the target.",
    "chain_index": "One-based position of the expected target chain within the source measurement.",
    "BindingDB_Target_Sequence": "Target-chain sequence reported by BindingDB for the normalized chain record.",
    "UniProt_SwissProt_Primary_ID": "Reviewed Swiss-Prot primary accession reported for the chain.",
    "UniProt_TrEMBL_Primary_ID": "UniProtKB/TrEMBL primary accession reported for the chain.",
    "Chain_Entity_ID": "Stable identifier for the normalized chain record when sufficient information is available.",
    "chain_sequence_available": "Whether a sequence is available for the expected chain.",
    "Target_Entity_ID": "Stable identifier linking chain rows belonging to the same reported multi-chain target.",
    "all_chain_sequences_available": "Whether sequences are available for every expected chain in the multi-chain target.",
}


ALLOWED = {
    "endpoint_type": "IC50; Ki; Kd; EC50",
    "measurement_relation": "=; <; >",
    "is_exact_measurement": "true; false",
    "measurement_class": "1; 0; missing",
    "measurement_class_ambiguous": "true; false",
    "classification_valid": "true; false",
    "classification_conflict": "true; false",
    "y_class": "1; 0; missing",
    "regression_valid": "true; false",
    "contains_carbon": "true; false",
    "alignment_performed": "true; false",
    "has_terminal_extension": "true; false",
    "has_polyhistidine_extension": "true; false",
    "chain_sequence_available": "true; false",
    "all_chain_sequences_available": "true; false; missing",
}


UNITS = {
    "molecular_weight": "Da",
    "measurement_boundary_nM": "nM",
    "classification_threshold_nM": "nM",
    "median_exact_value_nM": "nM",
    "min_exact_value_nM": "nM",
    "max_exact_value_nM": "nM",
    "q1_exact_value_nM": "nM",
    "q3_exact_value_nM": "nM",
    "pH_numeric": "dimensionless",
    "pH_median": "dimensionless",
    "temperature_C_numeric": "degrees Celsius",
    "temperature_C_median": "degrees Celsius",
    "p_measurement_boundary": "negative log10 molar",
    "p_value": "negative log10 molar",
    "p_value_std": "log10 units",
    "p_value_min": "negative log10 molar",
    "p_value_max": "negative log10 molar",
    "exact_value_log10_range": "log10 units",
}


def endpoint_alias(column):
    match = re.fullmatch(r"(IC50|Ki|Kd|EC50)_(raw|relation|boundary_nM)", column)
    if match:
        endpoint, suffix = match.groups()
        meanings = {
            "raw": f"Original reported {endpoint} text retained for auditability.",
            "relation": f"Normalized relation for the reported {endpoint} boundary.",
            "boundary_nM": f"Positive finite {endpoint} boundary expressed in nanomolar units.",
        }
        return meanings[suffix]
    match = re.fullmatch(r"p(IC50|Ki|Kd|EC50)_boundary", column)
    if match:
        return f"Negative base-10 logarithm of the molar {match.group(1)} boundary."
    match = re.fullmatch(r"(IC50|Ki|Kd|EC50)_(nM|min_nM|max_nM|q1_nM|q3_nM)", column)
    if match:
        endpoint, suffix = match.groups()
        label = {
            "nM": "median",
            "min_nM": "minimum",
            "max_nM": "maximum",
            "q1_nM": "first quartile",
            "q3_nM": "third quartile",
        }[suffix]
        return f"{label.capitalize()} exact {endpoint} value contributing to the interaction, in nanomolar units."
    match = re.fullmatch(r"p(IC50|Ki|Kd|EC50)(|_std|_min|_max)", column)
    if match:
        endpoint, suffix = match.groups()
        label = {"": "regression target", "_std": "standard deviation", "_min": "minimum", "_max": "maximum"}[suffix]
        return f"Endpoint-specific {label} for exact {endpoint} evidence on the negative-log10 molar scale."
    return None


def group_for(column, file_name):
    if "archive/BindingDB_measurements_all_endpoints" in file_name:
        return "source archive"
    if column.endswith("_ID") or column in {"Drug_ID", "Protein_Entity_ID", "UniProt_ID", "Measurement_ID", "Interaction_ID"}:
        return "identifier"
    if column in {"y_class", "p_value", "regression_valid", "classification_valid", "classification_conflict"} or endpoint_alias(column):
        return "task target"
    if column.startswith("n_") or column in {"measurement_dispersed", "regression_quality", "construct_annotation_uncertain"}:
        return "quality summary"
    if any(token in column.lower() for token in ["doi", "pmid", "patent", "publication", "authors", "institution", "source"]):
        return "provenance"
    if any(token in column.lower() for token in ["sequence", "alignment", "coverage", "insertion", "deletion", "substitution", "organism", "taxon", "protein"]):
        return "protein"
    if any(token in column.lower() for token in ["smiles", "inchi", "molecular", "atom", "bond", "charge", "ligand", "pubchem", "chembl", "drugbank", "zinc", "kegg"]):
        return "ligand"
    if any(token in column.lower() for token in ["measurement", "threshold", "exact", "p_value", "ph_", "temperature"]):
        return "measurement"
    return "metadata"


def description_for(column, file_name):
    if column in DESCRIPTIONS:
        return DESCRIPTIONS[column]
    alias = endpoint_alias(column)
    if alias:
        return alias
    if "archive/BindingDB_measurements_all_endpoints" in file_name:
        return "Field retained from the endpoint-explicit BindingDB source archive; consult the BindingDB field documentation and original source metadata."
    return "Released metadata field; consult the preprocessing code and release manifest for its exact derivation."


def units_for(column):
    if column in UNITS:
        return UNITS[column]
    if column.endswith("_nM"):
        return "nM"
    if re.fullmatch(r"p(IC50|Ki|Kd|EC50)(|_boundary|_min|_max)", column):
        return "negative log10 molar"
    if re.fullmatch(r"p(IC50|Ki|Kd|EC50)_std", column):
        return "log10 units"
    if any(token in column for token in ["identity", "coverage"]):
        return "fraction from 0 to 1"
    return "not applicable"


def missing_meaning(column):
    if column in {"y_class", "measurement_class"}:
        return "Missing when threshold evidence is ambiguous, conflicting or unavailable."
    if column == "p_value" or re.fullmatch(r"p(IC50|Ki|Kd|EC50)", column):
        return "Missing when no exact measurement supports a conventional regression target."
    if column == "classification_threshold_nM":
        return "Not expected to be missing in endpoint-specific curated records."
    if column == "measurement_boundary_nM" or re.fullmatch(r"(IC50|Ki|Kd|EC50)_boundary_nM", column):
        return "Missing when the reported endpoint value cannot be resolved to a positive finite numerical boundary."
    if column == "p_measurement_boundary" or re.fullmatch(r"p(IC50|Ki|Kd|EC50)_boundary", column):
        return "Missing when a valid positive numerical endpoint boundary is unavailable."
    if column.endswith("_nM") or "exact_value" in column or column.startswith("p_value_"):
        return "Missing when the required exact measurement evidence is unavailable."
    if any(token in column.lower() for token in ["doi", "pmid", "patent", "pdb", "ph", "temperature", "institution", "authors"]):
        return "Missing when the source export does not provide the metadata or it cannot be parsed."
    return "See the field description and source table; missing generally indicates unavailable or inapplicable information."


def recommended_use(column, group):
    if column == "regression_valid":
        return "Filter to true before using the released conventional regression target."
    if column == "classification_valid":
        return "Filter to true before using the optional binary benchmark label."
    if column == "y_class":
        return "Optional threshold-based classification target; do not interpret as universal binding/non-binding truth."
    if column == "p_value" or re.fullmatch(r"p(IC50|Ki|Kd|EC50)", column):
        return "Continuous endpoint-specific regression target."
    if column in {"Drug_ID", "Protein_Entity_ID", "Measurement_ID", "Interaction_ID", "UniProt_ID"}:
        return "Joining, grouping or split construction, as appropriate."
    if group == "provenance":
        return "Source auditing and leakage-aware source-document or temporal splitting."
    if group == "quality summary":
        return "Quality assessment, filtering or stratified analysis; not a model target unless explicitly justified."
    return "Interpret according to the source table and analysis objective."


if not INPUT.is_file():
    raise FileNotFoundError(f"Schema inventory not found: {INPUT}")

schema = pd.read_csv(INPUT)
rows = []

for item in schema.itertuples(index=False):
    column = item.column_name
    file_name = item.file
    group = group_for(column, file_name)
    rows.append(
        {
            "file": file_name,
            "row_count": item.row_count,
            "column_position": item.column_position,
            "column_name": column,
            "data_type": item.data_type,
            "field_group": group,
            "description": description_for(column, file_name),
            "units": units_for(column),
            "allowed_values": ALLOWED.get(column, "not restricted or field-specific"),
            "missing_value_meaning": missing_meaning(column),
            "recommended_use": recommended_use(column, group),
        }
    )

dictionary = pd.DataFrame(rows)
dictionary.to_csv(OUTPUT, index=False)

generic = dictionary["description"].str.startswith("Released metadata field")
print(f"Written: {OUTPUT}")
print(f"Rows: {len(dictionary):,}")
print(f"Files: {dictionary['file'].nunique()}")
print(f"Columns with generic curated descriptions: {generic.sum()}")
