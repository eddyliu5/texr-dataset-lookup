"""Simple lookup helpers for OpenTabs dataset metadata.

This module has no third-party dependencies. Keep it next to
``opentabs_source_matches.json`` and import ``DatasetLookup`` from research code.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


DATA_FILE = Path(__file__).resolve().with_name("opentabs_source_matches.json")

DEFAULT_FIELDS: Tuple[str, ...] = (
    "input_name",
    "canonical_name",
    "original_path",
    "source",
    "notes",
    "url",
)

CONFIDENCE_RANK = {"high": 3, "medium": 2, "low": 1}
NON_ALNUM = re.compile(r"[^a-z0-9]+")


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


class DatasetLookup:
    """Lookup helper over the bundled OpenTabs source-match JSON file."""

    def __init__(self, json_path: Optional[str | Path] = None) -> None:
        self.json_path = Path(json_path) if json_path else DATA_FILE
        self.records = self._load_records()

        self._exact_index: Dict[str, Set[int]] = defaultdict(set)
        self._feature_index: Dict[str, Set[int]] = defaultdict(set)
        self._feature_display: Dict[str, Set[str]] = defaultdict(set)
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
        for field in ("input_name", "canonical_name", "original_path", "url"):
            value = clean_string(record.get(field))
            if value:
                yield value

        path_value = clean_string(record.get("original_path"))
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
            self._feature_tokens.append(tokens(feature_text))

            for key in self._candidate_exact_keys(record):
                key = normalize_key(key)
                if key:
                    self._exact_index[key].add(idx)

            for feature in record["feature_names"]:
                key = normalize_key(feature)
                if key:
                    self._feature_index[key].add(idx)
                    self._feature_display[key].add(feature)

            self._normalized_fields.append(normalized)
            self._field_tokens.append(token_map)

    def _sort_key(self, idx: int) -> Tuple[int, str, str, str]:
        record = self.records[idx]
        return (
            -CONFIDENCE_RANK.get(clean_string(record.get("confidence")).lower(), 0),
            clean_string(record.get("split")),
            clean_string(record.get("canonical_name")),
            clean_string(record.get("input_name")),
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

    def exact_matches(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Return records whose normalized name/path/url exactly matches query."""

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
        """Return fuzzy text matches across dataset names, paths, URLs, and notes."""

        raw_query = clean_string(query)
        if not raw_query:
            return []

        query_key = normalize_key(raw_query)
        query_text = normalize_text(raw_query)
        query_tokens = tokens(query_text)
        fields_to_search = tuple(fields) if fields else DEFAULT_FIELDS
        exact_ids = self._exact_index.get(query_key, set())

        scored: List[Tuple[int, float]] = []
        for idx in range(len(self.records)):
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

        for feature_key, ids in self._feature_index.items():
            feature_text = feature_key.replace("-", " ")
            score = self._score_text(query_text, query_tokens, feature_text, tokens(feature_text))
            if score < min_score:
                continue
            for idx in ids:
                scored_by_record[idx] = max(scored_by_record.get(idx, 0.0), score)

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
        """Return True when query appears to match an OpenTabs dataset."""

        if self.exact_matches(query, limit=1):
            return True
        if strict:
            return False
        return bool(self.search(query, limit=1, min_score=min_score, include_feature_names=False))


_DEFAULT_LOOKUP: Optional[DatasetLookup] = None


def get_default_lookup() -> DatasetLookup:
    global _DEFAULT_LOOKUP
    if _DEFAULT_LOOKUP is None:
        _DEFAULT_LOOKUP = DatasetLookup()
    return _DEFAULT_LOOKUP


def exact_matches(query: str, **kwargs: Any) -> List[Dict[str, Any]]:
    return get_default_lookup().exact_matches(query, **kwargs)


def search(query: str, **kwargs: Any) -> List[Dict[str, Any]]:
    return get_default_lookup().search(query, **kwargs)


def search_by_feature(query: str, **kwargs: Any) -> List[Dict[str, Any]]:
    return get_default_lookup().search_by_feature(query, **kwargs)


def get_dataset(query: str, **kwargs: Any) -> Optional[Dict[str, Any]]:
    return get_default_lookup().get_dataset(query, **kwargs)


def has_dataset(query: str, **kwargs: Any) -> bool:
    return get_default_lookup().has_dataset(query, **kwargs)


# Backward-compatible aliases from the earlier script.
get_dataset_info = get_dataset
is_in_opentabs = has_dataset
