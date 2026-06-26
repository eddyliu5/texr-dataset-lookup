# TEXR Dataset Lookup

TEXR Dataset Lookup is a lightweight, standard-library-only utility for
searching the metadata indexes used by TEXR. It includes 36,115 synthetic
tabular datasets from
[TEXR](https://huggingface.co/datasets/eddyliu-hf/texr) and 2,096
records from the OpenTabs release used in our experiments.

`DatasetLookup()` searches TEXR by default. OpenTabs remains available as
an explicit alternative:

```python
from lookup import DatasetLookup

texr = DatasetLookup()
opentabs = DatasetLookup(catalog="opentabs")
custom = DatasetLookup("path/to/custom_index.json")
```

TEXR records include the CSV and metadata paths, dataset names,
description, source URL, generation model and configuration, sample and
feature counts, and feature names. OpenTabs records include the original path,
dataset names, source provenance, confidence annotation, notes, and feature
names.

## Demo

```python
from lookup import DatasetLookup, is_in_opentabs, is_in_texr

db = DatasetLookup()
print(db.size)
print(is_in_texr("5G cellular networks"))
print(is_in_opentabs("adult"))

record = db.get_dataset("5G cellular networks")
print(record["original_path"])
print(record["metadata_path"])
print(record["generation_model"])

hits = db.search("cellular network", limit=3)
for hit in hits:
    print(hit["canonical_name"], hit["generation_model"])

feature_hits = db.search_by_feature("network type", limit=3)
print([hit["canonical_name"] for hit in feature_hits])
```

Example output:

```text
36115
True
True
texr/Meta-Llama-3.1-8B-Instruct/.../synthetic_5G_cellular_networks_2000.csv
texr/Meta-Llama-3.1-8B-Instruct/.../metadata_5G_cellular_networks_2000.json
Meta-Llama-3.1-8B-Instruct
```

Main outputs:

- `has_dataset(query)`: whether the selected catalog contains a likely match
- `is_in_texr(query)`: TEXR-specific membership check
- `is_in_opentabs(query)`: OpenTabs-specific membership check
- `get_dataset(query)`: best matching record, or `None`
- `exact_matches(query)`: exact normalized matches
- `search(query)`: fuzzy search over names, paths, metadata, and feature names
- `search_by_feature(query)`: datasets with similar feature names

Module-level helpers accept `catalog="texr"` or `catalog="opentabs"`.

## OpenTabs source matching

We credit the original OpenTabs authors for the real-world tables used by the
OpenTabs catalog. The dataset can be downloaded
[here](https://mega.nz/file/oqUlgbCa#AwNrJD6RDTIroZbJhMUIe5hS2y_DpBGMPLnsutZcAL0),
and the original OpenTabs/CM2 implementation is available from
[Chao-Ye/CM2](https://github.com/Chao-Ye/CM2).

As described by the original authors, OpenTabs draws from four source
families: OpenML, UCI, Kaggle, and Data.gov. We curated the bundled source
index by normalizing dataset names and filenames, checking dataset pages and
source URLs where available, and validating ambiguous matches against feature
schemas.
