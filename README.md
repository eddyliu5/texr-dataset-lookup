# TEXR Dataset Lookup

TEXR Dataset Lookup is a lightweight, standard-library-only utility for
searching the metadata indexes used by TEXR. It includes 36,115 synthetic
tabular datasets from
[TEXR-36K](https://huggingface.co/datasets/eddyliu-hf/texr-36k) and 2,096
records from the OpenTabs release used in our experiments.

`DatasetLookup()` searches TEXR-36K by default. OpenTabs remains available as
an explicit alternative:

```python
from lookup import DatasetLookup

texr = DatasetLookup()
opentabs = DatasetLookup(catalog="opentabs")
custom = DatasetLookup("path/to/custom_index.json")
```

TEXR-36K records include the CSV and metadata paths, dataset names,
description, source URL, generation model and configuration, sample and
feature counts, and feature names. OpenTabs records include the original path,
dataset names, source provenance, confidence annotation, notes, and feature
names.

## Demo

```python
from lookup import DatasetLookup, is_in_opentabs, is_in_texr_36k

db = DatasetLookup()
print(db.size)
print(is_in_texr_36k("5G cellular networks"))
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
texr-36k/Meta-Llama-3.1-8B-Instruct/.../synthetic_5G_cellular_networks_2000.csv
texr-36k/Meta-Llama-3.1-8B-Instruct/.../metadata_5G_cellular_networks_2000.json
Meta-Llama-3.1-8B-Instruct
```

Main outputs:

- `has_dataset(query)`: whether the selected catalog contains a likely match
- `is_in_texr_36k(query)`: TEXR-36K-specific membership check
- `is_in_opentabs(query)`: OpenTabs-specific membership check
- `get_dataset(query)`: best matching record, or `None`
- `exact_matches(query)`: exact normalized matches
- `search(query)`: fuzzy search over names, paths, metadata, and feature names
- `search_by_feature(query)`: datasets with similar feature names

Module-level helpers accept `catalog="texr-36k"` or `catalog="opentabs"`.
