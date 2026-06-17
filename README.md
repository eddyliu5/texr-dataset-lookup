# OpenTabs Lookup

OpenTabs Lookup is a lightweight utility for searching OpenTabs dataset metadata
and source provenance. This repository provides a curated metadata index; the full OpenTabs
dataset can be found [here](https://mega.nz/file/oqUlgbCa#AwNrJD6RDTIroZbJhMUIe5hS2y_DpBGMPLnsutZcAL0),
and the original OpenTabs/CM2 repository is
[Chao-Ye/CM2](https://github.com/Chao-Ye/CM2).

The bundled index covers all 2,096 files in
the OpenTabs release used in our experiments and maps each file to standardized names, extracted
feature columns, source URLs, and confidence annotations. Duplicate dataset
variants and benchmark resamples are preserved, while repeated source-resolution
work was consolidated through schema and feature matching.

## Source matching

As described by the original OpenTabs authors, dataset sources were drawn from four
source families: OpenML, UCI, Kaggle, and Data.gov. We curated the source index
by normalizing dataset names and filenames, checking exact names and dataset page
URLs where possible, using source URL/domain evidence, and validating
ambiguous matches against extracted feature schemas.

## Usage

The input is a dataset query string, usually an OpenTabs dataset name, file name,
path fragment, URL, or feature name. By default, `DatasetLookup()` reads the
bundled `opentabs_source_matches.json`, which currently contains 2,096 dataset
records. You can also pass a custom JSON path:

```python
db = DatasetLookup("path/to/opentabs_source_matches.json")
```

Each returned record is a Python dictionary with these fields:

- `original_path`: original OpenTabs file path
- `input_name`: dataset name used in OpenTabs
- `canonical_name`: normalized dataset name
- `source`: curated source label
- `url`: source URL, when available
- `confidence`: match quality, one of `high`, `medium`, or `low`
- `notes`: short note about the match
- `feature_names`: list of column names

## Demo

```python
from lookup import DatasetLookup

db = DatasetLookup()

print(db.size)
print(db.has_dataset("adult"))

record = db.get_dataset("churn")
print(record["canonical_name"], record["source"], record["url"])

hits = db.search("census income", limit=3)
for hit in hits:
    print(hit["input_name"], hit["source"], hit["confidence"])

feature_hits = db.search_by_feature("sepal length", limit=3)
print([hit["canonical_name"] for hit in feature_hits])
```

Main outputs:

- `has_dataset(query)`: `True` or `False`
- `get_dataset(query)`: the best matching record, or `None`
- `exact_matches(query)`: exact normalized matches as a list of records
- `search(query, limit=10)`: fuzzy matches over names, paths, URLs, notes, and feature names
- `search_by_feature(query, limit=10)`: datasets with similar feature names
