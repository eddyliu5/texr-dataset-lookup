"""Simple lookup helpers for TEXR and OpenTabs dataset metadata.

This module has no third-party dependencies. Keep it next to the bundled JSON
indexes and import ``DatasetLookup`` from research code.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CATALOG = "texr"
TEXR_DATA_FILE = BASE_DIR / "texr_source_matches.json"
OPENTABS_DATA_FILE = BASE_DIR / "opentabs_source_matches.json"
CATALOG_FILES: Dict[str, Path] = {
    "texr": TEXR_DATA_FILE,
    "opentabs": OPENTABS_DATA_FILE,
}
# Backward-compatible name for the current default index.
DATA_FILE = CATALOG_FILES[DEFAULT_CATALOG]

DEFAULT_FIELDS: Tuple[str, ...] = (
    "input_name",
    "canonical_name",
    "original_path",
    "metadata_path",
    "source",
    "notes",
    "url",
    "description",
    "model_family",
    "generation_model",
    "generation_config",
)

CONFIDENCE_RANK = {"high": 3, "medium": 2, "low": 1}
NON_ALNUM = re.compile(r"[^a-z0-9]+")
CANDIDATE_MIN_SCORE = 0.45


def clean_string(value: Any) -> str:
    return "" if value is None else str(value).strip()


def normalize_key(text: str) -> str:
    return NON_ALNUM.sub("-", text.lower()).strip("-")


def normalize_text(text: str) -> str:
    return NON_ALNUM.sub(" ", text.lower()).strip()


def tokens(text: str) -> Set[str]:
    return {part for part in text.split() if part}


def normalized_feature_names(raw: Any) -> List[str]:
    if not isinstance(raw, list):
        return []

    out: List[str] = []
    seen: Set[str] = set()
    for item in raw:
        name = clean_string(item)
        key = normalize_key(name)
        if key and key not in seen:
            seen.add(key)
            out.append(name)
    return out


def normalize_catalog(catalog: str) -> str:
    key = normalize_key(clean_string(catalog))
    if key not in CATALOG_FILES:
        available = ", ".join(sorted(CATALOG_FILES))
        raise ValueError(f"Unknown catalog {catalog!r}; choose one of: {available}")
    return key


class DatasetLookup:
    """Lookup helper over a bundled catalog or a custom JSON index."""

    def __init__(
        self,
        json_path: Optional[str | Path] = None,
        *,
        catalog: Optional[str] = None,
    ) -> None:
        if json_path is not None and catalog is not None:
            raise ValueError("Pass either json_path or catalog, not both")

        if json_path is not None:
            self.catalog = "custom"
            self.json_path = Path(json_path)
        else:
            self.catalog = normalize_catalog(catalog or DEFAULT_CATALOG)
            self.json_path = CATALOG_FILES[self.catalog]

        self.records = self._load_records()

        self._exact_index: Dict[str, Set[int]] = defaultdict(set)
        self._feature_index: Dict[str, Set[int]] = defaultdict(set)
        self._text_token_index: Dict[str, Set[int]] = defaultdict(set)
        self._feature_token_index: Dict[str, Set[int]] = defaultdict(set)
        self._normalized_fields: List[Dict[str, str]] = []
        self._field_tokens: List[Dict[str, Set[str]]] = []
        self._feature_text: List[str] = []
        self._feature_tokens: List[Set[str]] = []

        self._build_indexes()

    @property
    def size(self) -> int:
        return len(self.records)

    def _load_records(self) -> List[Dict[str, Any]]:
        rows = json.loads(self.json_path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError(f"{self.json_path} must contain a JSON array")

        records: List[Dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            record = dict(row)
            record["feature_names"] = normalized_feature_names(record.get("feature_names"))
            records.append(record)
        return records

    def _candidate_exact_keys(self, record: Dict[str, Any]) -> Iterable[str]:
        for field in (
            "input_name",
            "canonical_name",
            "original_path",
            "metadata_path",
            "url",
        ):
            value = clean_string(record.get(field))
            if value:
                yield value

        for field in ("original_path", "metadata_path"):
            path_value = clean_string(record.get(field))
            if path_value:
                path = Path(path_value)
                if path.name:
                    yield path.name
                if path.stem:
                    yield path.stem

    def _build_indexes(self) -> None:
        for idx, record in enumerate(self.records):
            normalized: Dict[str, str] = {}
            token_map: Dict[str, Set[str]] = {}

            for field in DEFAULT_FIELDS:
                text = normalize_text(clean_string(record.get(field)))
                normalized[field] = text
                token_map[field] = tokens(text)

            feature_text = normalize_text(" ".join(record["feature_names"]))
            self._feature_text.append(feature_text)
            feature_tokens = tokens(feature_text)
            self._feature_tokens.append(feature_tokens)

            for key in self._candidate_exact_keys(record):
                key = normalize_key(key)
                if key:
                    self._exact_index[key].add(idx)

            for feature in record["feature_names"]:
                key = normalize_key(feature)
                if key:
                    self._feature_index[key].add(idx)
                    for token in tokens(key.replace("-", " ")):
                        self._feature_token_index[token].add(idx)

            text_tokens: Set[str] = set()
            for field_token_set in token_map.values():
                text_tokens.update(field_token_set)
            for token in text_tokens:
                self._text_token_index[token].add(idx)

            self._normalized_fields.append(normalized)
            self._field_tokens.append(token_map)

    def _sort_key(self, idx: int) -> Tuple[int, str, str, str, str]:
        record = self.records[idx]
        return (
            -CONFIDENCE_RANK.get(clean_string(record.get("confidence")).lower(), 0),
            clean_string(record.get("split")),
            clean_string(record.get("canonical_name")),
            clean_string(record.get("input_name")),
            clean_string(record.get("original_path")),
        )

    def _score_text(self, query: str, query_tokens: Set[str], text: str, text_tokens: Set[str]) -> float:
        if not query or not text:
            return 0.0
        if query == text:
            return 1.0
        if query in text:
            return 0.95
        if text in query and len(text) >= 4:
            return 0.90

        overlap = 0.0
        if query_tokens and text_tokens:
            overlap = len(query_tokens & text_tokens) / len(query_tokens)

        ratio = SequenceMatcher(None, query, text).ratio()
        return max(overlap * 0.90, ratio * 0.80)

    def _candidate_record_ids(
        self,
        query_tokens: Set[str],
        exact_ids: Set[int],
        *,
        include_feature_names: bool,
        min_score: float,
    ) -> Iterable[int]:
        if min_score < CANDIDATE_MIN_SCORE:
            return range(len(self.records))

        candidate_ids = set(exact_ids)
        for token in query_tokens:
            candidate_ids.update(self._text_token_index.get(token, set()))
            if include_feature_names:
                candidate_ids.update(self._feature_token_index.get(token, set()))

        if candidate_ids:
            return candidate_ids

        # Typo-only queries may not share any exact token with the index.
        return range(len(self.records))

    def exact_matches(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Return records whose normalized name, path, or URL matches query."""

        key = normalize_key(clean_string(query))
        if not key:
            return []

        ids = sorted(self._exact_index.get(key, set()), key=self._sort_key)
        return [dict(self.records[idx]) for idx in ids[: max(limit, 0)]]

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        min_score: float = 0.45,
        fields: Optional[Sequence[str]] = None,
        include_feature_names: bool = True,
    ) -> List[Dict[str, Any]]:
        """Return fuzzy matches across catalog metadata and feature names."""

        raw_query = clean_string(query)
        if not raw_query:
            return []

        query_key = normalize_key(raw_query)
        query_text = normalize_text(raw_query)
        query_tokens = tokens(query_text)
        fields_to_search = tuple(fields) if fields else DEFAULT_FIELDS
        exact_ids = self._exact_index.get(query_key, set())
        candidate_ids = self._candidate_record_ids(
            query_tokens,
            exact_ids,
            include_feature_names=include_feature_names,
            min_score=min_score,
        )

        scored: List[Tuple[int, float]] = []
        for idx in candidate_ids:
            best = 1.0 if idx in exact_ids else 0.0

            for field in fields_to_search:
                text = self._normalized_fields[idx].get(field, "")
                score = self._score_text(query_text, query_tokens, text, self._field_tokens[idx].get(field, set()))
                best = max(best, score)

            if include_feature_names:
                score = self._score_text(query_text, query_tokens, self._feature_text[idx], self._feature_tokens[idx])
                best = max(best, score)

            if best >= min_score:
                scored.append((idx, best))

        scored.sort(key=lambda item: (-item[1], self._sort_key(item[0])))
        return [dict(self.records[idx]) for idx, _ in scored[: max(limit, 0)]]

    def search_by_feature(
        self,
        feature_query: str,
        *,
        limit: int = 10,
        min_score: float = 0.78,
    ) -> List[Dict[str, Any]]:
        """Return datasets with feature names similar to feature_query."""

        query_key = normalize_key(clean_string(feature_query))
        if not query_key:
            return []

        query_text = query_key.replace("-", " ")
        query_tokens = tokens(query_text)
        scored_by_record: Dict[int, float] = {}

        exact_feature_ids = set(self._feature_index.get(query_key, set()))
        candidate_ids = set(exact_feature_ids)
        if len(candidate_ids) < max(limit, 0):
            for token in query_tokens:
                candidate_ids.update(self._feature_token_index.get(token, set()))

        if candidate_ids:
            for idx in candidate_ids:
                for feature in self.records[idx]["feature_names"]:
                    feature_text = normalize_text(feature)
                    score = self._score_text(
                        query_text,
                        query_tokens,
                        feature_text,
                        tokens(feature_text),
                    )
                    if score >= min_score:
                        scored_by_record[idx] = max(
                            scored_by_record.get(idx, 0.0),
                            score,
                        )
        else:
            # Preserve fuzzy typo matching when no exact feature token exists.
            for feature_key, ids in self._feature_index.items():
                feature_text = feature_key.replace("-", " ")
                score = self._score_text(
                    query_text,
                    query_tokens,
                    feature_text,
                    tokens(feature_text),
                )
                if score < min_score:
                    continue
                for idx in ids:
                    scored_by_record[idx] = max(
                        scored_by_record.get(idx, 0.0),
                        score,
                    )

        scored = sorted(scored_by_record.items(), key=lambda item: (-item[1], self._sort_key(item[0])))
        return [dict(self.records[idx]) for idx, _ in scored[: max(limit, 0)]]

    def get_dataset(self, query: str, *, strict: bool = False) -> Optional[Dict[str, Any]]:
        """Return the best matching dataset record, or None."""

        if strict:
            exact = self.exact_matches(query, limit=1)
            return exact[0] if exact else None

        hits = self.search(query, limit=1)
        return hits[0] if hits else None

    def has_dataset(self, query: str, *, strict: bool = False, min_score: float = 0.88) -> bool:
        """Return True when query appears to match this catalog."""

        if self.exact_matches(query, limit=1):
            return True
        if strict:
            return False
        return bool(self.search(query, limit=1, min_score=min_score, include_feature_names=False))


_CATALOG_LOOKUPS: Dict[str, DatasetLookup] = {}


def get_catalog_lookup(catalog: str = DEFAULT_CATALOG) -> DatasetLookup:
    key = normalize_catalog(catalog)
    if key not in _CATALOG_LOOKUPS:
        _CATALOG_LOOKUPS[key] = DatasetLookup(catalog=key)
    return _CATALOG_LOOKUPS[key]


def get_default_lookup() -> DatasetLookup:
    return get_catalog_lookup(DEFAULT_CATALOG)


def exact_matches(
    query: str,
    *,
    catalog: str = DEFAULT_CATALOG,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    return get_catalog_lookup(catalog).exact_matches(query, **kwargs)


def search(
    query: str,
    *,
    catalog: str = DEFAULT_CATALOG,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    return get_catalog_lookup(catalog).search(query, **kwargs)


def search_by_feature(
    query: str,
    *,
    catalog: str = DEFAULT_CATALOG,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    return get_catalog_lookup(catalog).search_by_feature(query, **kwargs)


def get_dataset(
    query: str,
    *,
    catalog: str = DEFAULT_CATALOG,
    **kwargs: Any,
) -> Optional[Dict[str, Any]]:
    return get_catalog_lookup(catalog).get_dataset(query, **kwargs)


def has_dataset(
    query: str,
    *,
    catalog: str = DEFAULT_CATALOG,
    **kwargs: Any,
) -> bool:
    return get_catalog_lookup(catalog).has_dataset(query, **kwargs)


def is_in_texr(query: str, **kwargs: Any) -> bool:
    """Return True when query appears in the TEXR catalog."""

    return get_catalog_lookup("texr").has_dataset(query, **kwargs)


def is_in_opentabs(query: str, **kwargs: Any) -> bool:
    """Return True when query appears in the OpenTabs catalog."""

    return get_catalog_lookup("opentabs").has_dataset(query, **kwargs)


# Backward-compatible aliases from the earlier script.
get_dataset_info = get_dataset
