from __future__ import annotations

import os
from typing import Optional

from .map_downloader import download_pbf
from .poi_index import POI, POIExtractor, POISpatialIndex

_TOPIC_TO_CATEGORIES: dict[str, list[str]] = {
    "accessibility": ["school", "hospital", "market", "transit"],
    "school": ["school"],
    "hospital": ["hospital"],
    "market": ["market"],
    "shopping": ["market", "shop"],
    "transit": ["transit"],
    "commute": ["transit", "workplace"],
    "work": ["workplace", "transit"],
    "food": ["restaurant", "market"],
    "restaurant": ["restaurant"],
    "recreation": ["park", "restaurant"],
    "park": ["park"],
    "housing": ["housing"],
}

_DEFAULT_RADIUS_M = 1500
_DEFAULT_KNN = 20


def _detect_topic(question_text: str, question_desc: str) -> list[str]:
    raw = (question_text + " " + question_desc).lower()
    matched: list[str] = []
    for topic, cats in _TOPIC_TO_CATEGORIES.items():
        if topic in raw:
            matched.append(topic)
    if not matched:
        for kw in ["nearby", "close", "distance", "access", "reachable"]:
            if kw in raw:
                matched = ["accessibility"]
                break
    if not matched:
        matched = ["accessibility"]
    return matched


def _format_poi_list(pois: list[POI], max_entries: int = 15) -> str:
    if not pois:
        return "  No POIs found in the area."
    lines = []
    for i, p in enumerate(pois[:max_entries], 1):
        name = p.name if p.name else f"Unnamed {p.category}"
        lines.append(f"  {i}. {name} ({p.category}) — {p.lat:.5f}, {p.lon:.5f}")
    return "\n".join(lines)


def _poi_summary(pois: list[POI]) -> str:
    counts: dict[str, int] = {}
    for p in pois:
        counts[p.category] = counts.get(p.category, 0) + 1
    parts = []
    for cat, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        parts.append(f"{cnt} {cat}(s)")
    return ", ".join(parts) if parts else "no POIs"


class MapContext:
    """Unified map context manager.

    Downloads PBF, builds POI spatial index, and provides dynamic
    POI context for agent survey questions.
    """

    def __init__(
        self,
        dataset_name: str,
        data_dir: str,
        cache_dir: Optional[str] = None,
        default_radius_m: float = _DEFAULT_RADIUS_M,
        default_knn: int = _DEFAULT_KNN,
    ):
        self._dataset_name = dataset_name
        self._data_dir = data_dir
        self._cache_dir = cache_dir
        self._default_radius_m = default_radius_m
        self._default_knn = default_knn

        self._pbf_path: Optional[str] = None
        self._extractor = POIExtractor()
        self._index = POISpatialIndex()
        self._ready = False

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def pbf_path(self) -> Optional[str]:
        return self._pbf_path

    @property
    def poi_count(self) -> int:
        return self._index.size

    def initialize(self) -> bool:
        print(f"[MAP_CONTEXT] Initializing for dataset: {self._dataset_name}")

        self._pbf_path = download_pbf(
            self._dataset_name,
            self._data_dir,
            cache_dir=self._cache_dir,
        )
        if not self._pbf_path:
            print("[MAP_CONTEXT] No PBF available. POI index will be disabled.")
            return False

        print(f"[MAP_CONTEXT] Extracting POIs from {self._pbf_path}...")
        pois = self._extractor.extract(self._pbf_path)
        print(f"[MAP_CONTEXT] Found {len(pois)} POIs. Building spatial index...")
        self._index.build(pois)
        self._ready = True

        summary = _poi_summary(pois)
        print(f"[MAP_CONTEXT] Index ready. Categories: {summary}")
        return True

    def get_poi_context(
        self,
        lat: float,
        lon: float,
        question_text: str = "",
        question_desc: str = "",
        radius_m: Optional[float] = None,
    ) -> str:
        if not self._ready:
            return ""

        r = radius_m or self._default_radius_m
        topics = _detect_topic(question_text, question_desc)
        categories: list[str] = []
        for t in topics:
            categories.extend(_TOPIC_TO_CATEGORIES.get(t, []))
        categories = list(set(categories))

        pois = self._index.query_radius(lat, lon, radius_m=r, categories=categories or None)

        if not pois:
            pois = self._index.query_knn(lat, lon, k=self._default_knn, categories=categories or None)

        if not pois:
            return ""

        lines = [
            "### Nearby Points of Interest",
            f"Location: ({lat:.5f}, {lon:.5f})",
            f"Search radius: {r}m",
            f"Topics: {', '.join(topics)}",
            f"Categories queried: {', '.join(categories) if categories else 'all'}",
            "",
            "POIs found:",
            _format_poi_list(pois),
            "",
            f"Summary: {_poi_summary(pois)}",
        ]
        return "\n".join(lines)

    def get_accessibility_summary(
        self,
        lat: float,
        lon: float,
        radius_m: Optional[float] = None,
    ) -> str:
        if not self._ready:
            return ""

        r = radius_m or self._default_radius_m
        key_cats = ["school", "hospital", "market", "transit"]
        lines = ["### Accessibility Summary"]
        lines.append(f"Location: ({lat:.5f}, {lon:.5f})")
        lines.append(f"Radius: {r}m")
        lines.append("")

        for cat in key_cats:
            pois = self._index.query_radius(lat, lon, radius_m=r, categories=[cat])
            if pois:
                nearest = min(pois, key=lambda p: p.distance_to(lat, lon))
                dist = nearest.distance_to(lat, lon)
                lines.append(f"  {cat.capitalize()}: {len(pois)} found, nearest at {dist:.0f}m ({nearest.name or 'unnamed'})")
            else:
                lines.append(f"  {cat.capitalize()}: none within {r}m")

        return "\n".join(lines)

    def reload(self) -> bool:
        self._ready = False
        return self.initialize()
