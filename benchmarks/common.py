"""Shared helpers for ObjectNat benchmarks: timing, machine info, accuracy metrics."""

import json
import os
import platform
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import shapely

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from legacy_visibility import (  # noqa: E402  pylint: disable=wrong-import-position,wrong-import-order
    legacy_get_visibility,
    legacy_visibility_accurate,
)

from tests.visibility_oracle import (  # noqa: E402,F401  pylint: disable=wrong-import-position,unused-import
    oracle_visibility,
)

RESULTS_DIR = Path(__file__).parent / "results"
EXAMPLES_DATA = Path(__file__).parent.parent / "docs" / "methods" / "examples" / "examples_data"


def machine_info() -> dict:
    import objectnat  # pylint: disable=import-outside-toplevel

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "shapely": shapely.__version__,
        "geos": shapely.geos_version_string,
        "objectnat": objectnat.__version__,
    }


@contextmanager
def legacy_visibility_in_noise():
    """
    Run the noise functions on the frozen 2.1.0 Python visibility instead of the Rust kernel.

    Patches module attributes, so it only reaches workers that inherit the parent's memory: use the
    ``fork`` start method (Linux/macOS) for ``simulate_noise`` with ``use_parallel=True``.
    """
    from objectnat.methods.noise import (  # pylint: disable=import-outside-toplevel
        noise_simulation,
        noise_simulation_simplified,
    )

    saved = noise_simulation._visibility_accurate, noise_simulation_simplified.get_visibility
    noise_simulation._visibility_accurate = legacy_visibility_accurate
    noise_simulation_simplified.get_visibility = legacy_get_visibility
    try:
        yield
    finally:
        noise_simulation._visibility_accurate, noise_simulation_simplified.get_visibility = saved


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
