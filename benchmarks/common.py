"""Shared helpers for ObjectNat benchmarks: timing, machine info, an independent visibility oracle."""

import json
import os
import platform
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import shapely
from shapely import Point, Polygon

from objectnat import config

RESULTS_DIR = Path(__file__).parent / "results"
EXAMPLES_DATA = Path(__file__).parent.parent / "docs" / "methods" / "examples" / "examples_data"


def machine_info() -> dict:
    try:
        import objectnat_core  # pylint: disable=import-outside-toplevel

        core_version = objectnat_core.__version__
    except ImportError:
        core_version = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "shapely": shapely.__version__,
        "geos": shapely.geos_version_string,
        "objectnat_core": core_version,
    }


@contextmanager
def rust_core(enabled: bool):
    previous = config.enable_rust_core
    config.set_enable_rust_core(enabled)
    try:
        yield
    finally:
        config.set_enable_rust_core(previous)


def timed(func, *args, **kwargs):
    start = time.perf_counter()
    result = func(*args, **kwargs)
    return result, time.perf_counter() - start


def iou(a, b) -> float:
    union = shapely.union(a, b)
    area = shapely.area(union)
    if area == 0:
        return 1.0
    return float(shapely.area(shapely.intersection(a, b)) / area)


def iou_many(left, right) -> np.ndarray:
    return np.array([iou(a, b) for a, b in zip(left, right)])


def sym_diff_area(left, right) -> np.ndarray:
    return np.array([shapely.area(shapely.symmetric_difference(a, b)) for a, b in zip(left, right)])


def oracle_visibility(point: Point, obstacle_geoms, radius: float, quad_segs: int = 16) -> Polygon:
    """
    Independent brute-force reference: view disc minus the union of every wall's shadow and the obstacles.

    The shadow of a wall segment is the region behind it as seen from the observer; the union of all
    wall shadows is exactly the occluded region. Each segment is split so that no piece subtends more
    than 60 degrees, which keeps its shadow quadrilateral convex and reaching beyond the disc.
    No pruning, no nearest-wall ordering: slow, but shares no logic with either implementation.
    """
    disc = point.buffer(radius, quad_segs=quad_segs)
    geoms = [g for g in obstacle_geoms if g is not None and not g.is_empty and g.intersects(disc)]
    if not geoms:
        return disc
    o = np.array([point.x, point.y])
    far = 4.0 * radius
    shadows = []
    for geom in geoms:
        boundary = geom.boundary if geom.geom_type in ("Polygon", "MultiPolygon") else geom
        for line in shapely.get_parts(boundary):
            coords = np.asarray(line.coords)
            for a, b in zip(coords[:-1], coords[1:]):
                va, vb = a - o, b - o
                if np.hypot(*va) == 0 or np.hypot(*vb) == 0:
                    continue
                angle = abs(np.arctan2(va[0] * vb[1] - va[1] * vb[0], va @ vb))
                pieces = max(1, int(np.ceil(angle / (np.pi / 3))))
                for k in range(pieces):
                    p = a + (b - a) * (k / pieces)
                    q = a + (b - a) * ((k + 1) / pieces)
                    vp, vq = p - o, q - o
                    p_far = o + vp / np.hypot(*vp) * far
                    q_far = o + vq / np.hypot(*vq) * far
                    quad = Polygon([p, q, q_far, p_far])
                    if not quad.is_valid:
                        quad = quad.buffer(0)
                    if not quad.is_empty and quad.area > 0:
                        shadows.append(quad)
    hidden = shapely.union_all(shadows + geoms)
    visible = disc.difference(hidden)
    parts = [g for g in shapely.get_parts(visible) if g.geom_type == "Polygon"]
    containing = [g for g in parts if g.distance(point) < 1e-6]
    if containing:
        return max(containing, key=lambda g: g.area)
    return Polygon()


def write_results(name: str, payload: dict, markdown: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{name}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (RESULTS_DIR / f"{name}.md").write_text(markdown, encoding="utf-8")


def markdown_table(rows: list[dict], columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(title for _, title in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, sep]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key, "")) for key, _ in columns) + " |")
    return "\n".join(lines)
