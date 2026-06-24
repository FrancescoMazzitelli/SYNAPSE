from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class POI:
    lat: float
    lon: float
    name: str
    category: str
    tags: dict = field(default_factory=dict)

    def distance_to(self, lat: float, lon: float) -> float:
        return _haversine(self.lat, self.lon, lat, lon)


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


_CATEGORY_MAP: dict[str, list[str]] = {
    "school": ["school", "kindergarten", "university", "college", "education"],
    "hospital": ["hospital", "clinic", "doctors", "pharmacy", "healthcare"],
    "market": ["marketplace", "supermarket", "convenience", "grocery", "shop"],
    "restaurant": ["restaurant", "cafe", "fast_food", "food_court", "bar"],
    "park": ["park", "garden", "nature_reserve", "playground"],
    "transit": ["bus_station", "bus_stop", "subway_entrance", "train_station", "tram_stop"],
    "workplace": ["office", "company", "coworking_space", "factory"],
    "housing": ["apartments", "residential", "house"],
}


def _classify_poi(tags: dict) -> Optional[str]:
    name = tags.get("name", "").lower()
    amenity = tags.get("amenity", "").lower()
    shop = tags.get("shop", "").lower()
    leisure = tags.get("leisure", "").lower()
    landuse = tags.get("landuse", "").lower()
    highway = tags.get("highway", "").lower()
    building = tags.get("building", "").lower()
    raw = f"{name} {amenity} {shop} {leisure} {landuse} {highway} {building}"
    for category, keywords in _CATEGORY_MAP.items():
        if any(kw in raw for kw in keywords):
            return category
    return None


class POIExtractor:
    """Extract POIs from an OSM PBF file using pyosmium."""

    def __init__(self):
        self.pois: list[POI] = []

    def extract(self, pbf_path: str) -> list[POI]:
        try:
            import osmium
        except ImportError:
            raise RuntimeError("osmium is required. Install with: pip install osmium")

        class POIHandler(osmium.SimpleHandler):
            def __init__(self):
                super().__init__()
                self.pois = []

            def node(self, n):
                tags = dict(n.tags)
                if not tags:
                    return
                category = _classify_poi(tags)
                if category is None:
                    return
                self.pois.append(POI(
                    lat=n.location.lat,
                    lon=n.location.lon,
                    name=tags.get("name", ""),
                    category=category,
                    tags=tags,
                ))

            def way(self, w):
                tags = dict(w.tags)
                if not tags:
                    return
                if not w.nodes:
                    return
                centroid_lat = sum(nd.location.lat for nd in w.nodes) / len(w.nodes)
                centroid_lon = sum(nd.location.lon for nd in w.nodes) / len(w.nodes)
                category = _classify_poi(tags)
                if category is None:
                    return
                self.pois.append(POI(
                    lat=centroid_lat,
                    lon=centroid_lon,
                    name=tags.get("name", ""),
                    category=category,
                    tags=tags,
                ))

        handler = POIHandler()
        handler.apply_file(pbf_path, locations=True, idx="sparse_mem_array")
        self.pois = handler.pois
        return self.pois


class POISpatialIndex:
    """Fast spatial index for POI queries using cKDTree."""

    def __init__(self):
        self._pois: list[POI] = []
        self._tree: Optional[cKDTree] = None
        self._coords: Optional[np.ndarray] = None

    def build(self, pois: list[POI]) -> None:
        self._pois = pois
        if not pois:
            self._tree = None
            self._coords = None
            return
        self._coords = np.array([[p.lat, p.lon] for p in pois])
        self._tree = cKDTree(self._coords)

    def query_radius(
        self,
        lat: float,
        lon: float,
        radius_meters: float = 1000,
        categories: Optional[list[str]] = None,
    ) -> list[POI]:
        if self._tree is None or self._coords is None:
            return []
        degrees = radius_meters / 111320.0
        idxs = self._tree.query_ball_point([lat, lon], r=degrees)
        results = []
        for i in idxs:
            poi = self._pois[i]
            if categories and poi.category not in categories:
                continue
            if _haversine(poi.lat, poi.lon, lat, lon) <= radius_meters:
                results.append(poi)
        return results

    def query_knn(
        self,
        lat: float,
        lon: float,
        k: int = 10,
        categories: Optional[list[str]] = None,
    ) -> list[POI]:
        if self._tree is None or self._coords is None:
            return []
        dists, idxs = self._tree.query([lat, lon], k=min(k, len(self._pois)))
        results = []
        for i in np.atleast_1d(idxs):
            poi = self._pois[int(i)]
            if categories and poi.category not in categories:
                continue
            results.append(poi)
        return results

    def count_by_category(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for poi in self._pois:
            counts[poi.category] = counts.get(poi.category, 0) + 1
        return counts

    @property
    def size(self) -> int:
        return len(self._pois)
