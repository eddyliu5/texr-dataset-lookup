# OpenTabs Lookup

Lookup helper for OpenTabs dataset metadata and source.

## Demo

```python
from lookup import DatasetLookup

db = DatasetLookup()

print(db.size)
print(db.has_dataset("adult"))
print(db.get_dataset("churn"))
print(db.search("census income", limit=3))
print(db.search_by_feature("sepal length", limit=3))
```
