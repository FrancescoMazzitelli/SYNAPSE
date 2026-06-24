from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Optional


def _run(cmd: list[str], cwd: Optional[str] = None) -> None:
    subprocess.run(cmd, check=True, cwd=cwd, capture_output=True)


class ValhallaManager:
    """Manages Valhalla routing engine as an in-process component.

    Builds graph tiles from OSM PBF files and provides routing via
    pyvalhalla Actor, avoiding the need for a separate HTTP server.
    """

    def __init__(
        self,
        tile_dir: str = "/tmp/valhalla_tiles",
        costing: str = "auto",
    ):
        self._tile_dir = Path(tile_dir)
        self._costing = costing
        self._lock = threading.Lock()
        self._actor = None
        self._config: dict = {}
        self._current_map: Optional[str] = None

    @property
    def is_ready(self) -> bool:
        return self._actor is not None

    @property
    def current_map(self) -> Optional[str]:
        return self._current_map

    # ── graph building ────────────────────────────────────────────────────

    def _find_pbf_files(self, data_dir: str) -> list[str]:
        pattern = os.path.join(data_dir, "*.osm.pbf")
        files = sorted(glob.glob(pattern))
        if not files:
            pattern = os.path.join(data_dir, "**", "*.osm.pbf")
            files = sorted(glob.glob(pattern, recursive=True))
        return files

    def _build_config(self) -> dict:
        tile_dir_str = str(self._tile_dir)
        return {
            "mjolnir": {
                "tile_dir": tile_dir_str,
                "tile_extract": os.path.join(tile_dir_str, "valhalla_tiles.tar"),
            },
            "service_limits": {
                "auto": {"max_distance": 5000000},
                "bicycle": {"max_distance": 500000},
                "pedestrian": {"max_distance": 200000},
            },
            "midgard": {
                "data_dir": tile_dir_str,
            },
        }

    def _write_config(self) -> str:
        self._config = self._build_config()
        cfg_path = self._tile_dir / "valhalla.json"
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(json.dumps(self._config, indent=2))
        return str(cfg_path)

    def build_graph(self, data_dir: str) -> bool:
        """Build Valhalla graph tiles from OSM PBF files in *data_dir*.

        Returns True on success, False if no PBF files found or build fails.
        """
        pbf_files = self._find_pbf_files(data_dir)
        if not pbf_files:
            print(f"[VALHALLA] No .osm.pbf files found in {data_dir}")
            return False

        with self._lock:
            self._shutdown_actor()

            self._tile_dir.mkdir(parents=True, exist_ok=True)

            cfg_path = self._write_config()

            cmd = [
                "python", "-m", "valhalla",
                "valhalla_build_tiles",
                "-c", cfg_path,
                *pbf_files,
            ]
            print(f"[VALHALLA] Building graph from {len(pbf_files)} PBF file(s)...")
            try:
                _run(cmd)
            except subprocess.CalledProcessError as exc:
                print(f"[VALHALLA] Build failed: {exc.stderr.decode()[:500]}")
                return False

            tar_path = self._tile_dir / "valhalla_tiles.tar"
            if not tar_path.exists():
                extract_cmd = [
                    "python", "-m", "valhalla",
                    "valhalla_build_extract",
                    "-c", cfg_path,
                ]
                try:
                    _run(extract_cmd)
                except subprocess.CalledProcessError as exc:
                    print(f"[VALHALLA] Extract build failed: {exc.stderr.decode()[:500]}")
                    return False

            self._load_actor()
            self._current_map = data_dir
            print(f"[VALHALLA] Graph ready ({len(pbf_files)} PBF(s) from {data_dir})")
            return True

    # ── actor lifecycle ───────────────────────────────────────────────────

    def _load_actor(self):
        try:
            from valhalla import Actor, get_config
        except ImportError:
            raise RuntimeError(
                "pyvalhalla is required. Install with: pip install pyvalhalla"
            )

        tile_tar = self._tile_dir / "valhalla_tiles.tar"
        if not tile_tar.exists():
            raise RuntimeError(f"Tile extract not found: {tile_tar}")

        config = get_config(tile_extract=str(tile_tar))
        self._actor = Actor(config)

    def _shutdown_actor(self):
        self._actor = None
        self._current_map = None

    # ── routing ───────────────────────────────────────────────────────────

    def route(
        self,
        origin: tuple[float, float],
        destination: tuple[float, float],
        costing: Optional[str] = None,
        **kwargs,
    ) -> dict:
        """Compute a route between origin and destination.

        Args:
            origin: (lat, lon) tuple
            destination: (lat, lon) tuple
            costing: Valhalla costing model (auto, bicycle, pedestrian, etc.)
            **kwargs: Additional Valhalla route parameters

        Returns:
            Dict with Valhalla route response (trip, legs, distance, duration, etc.)
        """
        if self._actor is None:
            raise RuntimeError("Valhalla graph not loaded. Call build_graph() first.")

        request = {
            "locations": [
                {"lat": origin[0], "lon": origin[1]},
                {"lat": destination[0], "lon": destination[1]},
            ],
            "costing": costing or self._costing,
        }
        request.update(kwargs)

        return self._actor.route(request)

    def matrix(
        self,
        sources: list[tuple[float, float]],
        targets: list[tuple[float, float]],
        costing: Optional[str] = None,
        **kwargs,
    ) -> dict:
        """Compute a time-distance matrix.

        Args:
            sources: List of (lat, lon) tuples
            targets: List of (lat, lon) tuples
            costing: Valhalla costing model
            **kwargs: Additional Valhalla matrix parameters

        Returns:
            Dict with Valhalla matrix response
        """
        if self._actor is None:
            raise RuntimeError("Valhalla graph not loaded. Call build_graph() first.")

        request = {
            "sources": [{"lat": lat, "lon": lon} for lat, lon in sources],
            "targets": [{"lat": lat, "lon": lon} for lat, lon in targets],
            "costing": costing or self._costing,
        }
        request.update(kwargs)

        return self._actor.matrix(request)

    def isochrone(
        self,
        location: tuple[float, float],
        contours: list[dict],
        costing: Optional[str] = None,
        **kwargs,
    ) -> dict:
        """Compute isochrones from a location.

        Args:
            location: (lat, lon) tuple
            contours: List of contour dicts, e.g. [{"time": 10}, {"time": 20}]
            costing: Valhalla costing model
            **kwargs: Additional Valhalla isochrone parameters

        Returns:
            Dict with Valhalla isochrone response
        """
        if self._actor is None:
            raise RuntimeError("Valhalla graph not loaded. Call build_graph() first.")

        request = {
            "locations": [{"lat": location[0], "lon": location[1]}],
            "costing": costing or self._costing,
            "contours": contours,
        }
        request.update(kwargs)

        return self._actor.isochrone(request)

    # ── cleanup ───────────────────────────────────────────────────────────

    def cleanup(self):
        """Shut down actor and remove tile directory."""
        with self._lock:
            self._shutdown_actor()
            if self._tile_dir.exists():
                shutil.rmtree(self._tile_dir)

    def switch_map(self, data_dir: str) -> bool:
        """Switch to a different map dataset.

        Rebuilds the graph from OSM PBF files in the new directory.
        """
        if self._current_map == data_dir:
            return True
        return self.build_graph(data_dir)
