from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Optional


DATASET_REGION_MAP: dict[str, tuple[str, str, str]] = {
    "data_hanoi": ("vietnam", "hanoi", "bbbike"),
    "data_lyon_EDGT": ("france", "rhone-alpes", "geofabrik"),
    "data_NYC_mobility": ("north-america", "us/new-york", "geofabrik"),
    "data_pmus_yaounde": ("africa", "cameroon", "geofabrik"),
    "data_MyDailyTravelData": ("north-america", "us/new-york", "geofabrik"),
    "data_VTC_survey": ("france", "france", "geofabrik"),
}


def _find_existing_pbf(data_dir: str) -> Optional[str]:
    pattern = os.path.join(data_dir, "*.osm.pbf")
    files = sorted(glob.glob(pattern))
    if files:
        return files[0]
    pattern = os.path.join(data_dir, "**", "*.osm.pbf")
    files = sorted(glob.glob(pattern, recursive=True))
    return files[0] if files else None


def _download_geofabrik(region: str, dest_dir: str) -> str:
    from pydriosm.downloader import GeofabrikDownloader

    dl = GeofabrikDownloader(cdd=dest_dir)
    subregions = dl.get_valid_subregion_names()
    for name in subregions:
        if region.lower().replace("-", "").replace("/", "").replace(" ", "") in name.lower().replace("-", "").replace("/", "").replace(" ", ""):
            print(f"[MAP] Downloading PBF for region: {name}")
            dl.download_data(name)
            pbf_path = dl.get_default_filename(name)
            return pbf_path

    print(f"[MAP] Region '{region}' not found in Geofabrik. Available: {list(subregions)[:10]}...")
    return ""


def _download_bbbike(city: str, dest_dir: str) -> str:
    from pydriosm.downloader import BBBikeDownloader

    dl = BBBikeDownloader(cdd=dest_dir)
    cities = dl.get_bbbike_cities()
    city_lower = city.lower()
    for c in cities:
        if city_lower in c.lower():
            print(f"[MAP] Downloading PBF for city: {c}")
            dl.download_data(c)
            pbf_path = dl.get_default_filename(c)
            return pbf_path

    print(f"[MAP] City '{city}' not found in BBBike. Available: {cities[:10]}...")
    return ""


def download_pbf(dataset_name: str, data_dir: str, cache_dir: Optional[str] = None) -> Optional[str]:
    existing = _find_existing_pbf(data_dir)
    if existing:
        print(f"[MAP] PBF already found: {existing}")
        return existing

    if dataset_name not in DATASET_REGION_MAP:
        print(f"[MAP] No download mapping for '{dataset_name}'. Looking for local PBF files...")
        return _find_existing_pbf(data_dir)

    region, subregion, source = DATASET_REGION_MAP[dataset_name]
    dest = cache_dir or data_dir
    Path(dest).mkdir(parents=True, exist_ok=True)

    if source == "bbbike":
        pbf_path = _download_bbbike(subregion, dest)
    else:
        pbf_path = _download_geofabrik(subregion, dest)

    if pbf_path:
        full_path = os.path.join(dest, pbf_path) if not os.path.isabs(pbf_path) else pbf_path
        if os.path.exists(full_path):
            print(f"[MAP] PBF downloaded: {full_path}")
            return full_path

    return None
