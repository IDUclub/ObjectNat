"""
Visibility benchmark: pure-Python ``get_visibility`` vs the Rust core (``objectnat-core``).

For every scene and view radius it measures, on the same observer points:

* wall time of ``get_visibility(method="accurate")`` — Python sequential, Python ``parallel=True``,
  Rust core (all threads) — and of ``method="simple"`` for context;
* accuracy against an independent brute-force oracle (view disc minus the union of all wall shadows,
  see ``common.oracle_visibility``): IoU and symmetric-difference area;
* agreement between the Rust core and the current Python ``accurate`` implementation.

A separate throughput run feeds many more points to the Rust core only.

Usage::

    uv run python benchmarks/bench_visibility.py            # full run, writes benchmarks/results/visibility.*
    uv run python benchmarks/bench_visibility.py --quick    # small smoke run
"""

import argparse
import os
import sys

import geopandas as gpd
import numpy as np
import shapely
from shapely import box

sys.path.insert(0, os.path.dirname(__file__))

from common import (  # noqa: E402  pylint: disable=wrong-import-position
    EXAMPLES_DATA,
    iou_many,
    machine_info,
    markdown_table,
    oracle_visibility,
    rust_core,
    sym_diff_area,
    timed,
    write_results,
)

from objectnat import config, get_visibility  # noqa: E402  pylint: disable=wrong-import-position
from objectnat.methods.visibility._native import (  # noqa: E402  pylint: disable=wrong-import-position
    geometries_to_segments,
    native_visibility_polygons,
)

CRS = 32636
X0, Y0 = 400_000.0, 6_600_000.0  # inside UTM zone 36N, so get_visibility keeps the same CRS


def scene_wall(rng, n):
    obstacles = gpd.GeoDataFrame(geometry=[box(X0 + 40, Y0 - 1000, X0 + 42, Y0 + 1000)], crs=CRS)
    pts = np.column_stack([X0 + rng.uniform(-60, 35, n), Y0 + rng.uniform(-200, 200, n)])
    return obstacles, pts


def scene_blocks(rng, n):
    size, street = 40.0, 20.0
    pitch = size + street
    blocks = [
        box(X0 + i * pitch, Y0 + j * pitch, X0 + i * pitch + size, Y0 + j * pitch + size)
        for i in range(-20, 20)
        for j in range(-20, 20)
    ]
    obstacles = gpd.GeoDataFrame(geometry=blocks, crs=CRS)
    union = obstacles.union_all()
    pts = []
    while len(pts) < n:
        p = (X0 + rng.uniform(-600, 600), Y0 + rng.uniform(-600, 600))
        if not union.contains(shapely.Point(p)):
            pts.append(p)
    return obstacles, np.array(pts)


def scene_real(rng, n):
    buildings = gpd.read_parquet(EXAMPLES_DATA / "buildings.parquet")[["geometry"]]
    minx, miny, maxx, maxy = buildings.total_bounds
    union = buildings.union_all()
    pts = []
    while len(pts) < n:
        p = (rng.uniform(minx + 1500, maxx - 1500), rng.uniform(miny + 1000, maxy - 1000))
        if not union.contains(shapely.Point(p)):
            pts.append(p)
    return buildings, np.array(pts)


SCENES = {
    "wall": (scene_wall, [100]),
    "blocks": (scene_blocks, [150, 400]),
    "real": (scene_real, [100, 300, 600]),
}


def points_gdf(xy, crs):
    return gpd.GeoDataFrame(geometry=gpd.points_from_xy(xy[:, 0], xy[:, 1]), crs=crs)


def run(quick: bool) -> None:
    rng = np.random.default_rng(42)
    n_accuracy = 6 if quick else 25
    n_throughput = 200 if quick else 5000
    config.set_enable_tqdm(False)
    rows, throughput_rows, raw = [], [], []

    # warm up the native core and the process pool machinery
    warm_obstacles, warm_pts = scene_wall(rng, 2)
    with rust_core(True):
        get_visibility(points_gdf(warm_pts, CRS), warm_obstacles, 50)

    for scene_name, (factory, radii) in SCENES.items():
        obstacles, xy_all = factory(rng, max(n_accuracy, n_throughput))
        crs = obstacles.crs
        xy = xy_all[:n_accuracy]
        pts = points_gdf(xy, crs)
        for radius in radii:
            print(f"[{scene_name} r={radius}] oracle...", flush=True)
            oracle = np.array(
                [oracle_visibility(p, obstacles.geometry.values, radius) for p in pts.geometry], dtype=object
            )
            with rust_core(False):
                print(f"[{scene_name} r={radius}] python accurate...", flush=True)
                py_acc, t_py = timed(get_visibility, pts, obstacles, radius, "accurate")
                print(f"[{scene_name} r={radius}] python accurate parallel...", flush=True)
                _, t_py_par = timed(get_visibility, pts, obstacles, radius, "accurate", parallel=True)
                print(f"[{scene_name} r={radius}] python simple...", flush=True)
                py_simple, t_simple = timed(get_visibility, pts, obstacles, radius, "simple")
            with rust_core(True):
                rs_acc, t_rs = timed(get_visibility, pts, obstacles, radius, "accurate")

            py_g, rs_g, simple_g = (g.to_crs(crs).geometry.values for g in (py_acc, rs_acc, py_simple))
            iou_py, iou_rs, iou_simple = (iou_many(g, oracle) for g in (py_g, rs_g, simple_g))
            iou_rs_py = iou_many(rs_g, py_g)
            sd_py, sd_rs = sym_diff_area(py_g, oracle), sym_diff_area(rs_g, oracle)
            n = len(pts)
            row = {
                "scene": scene_name,
                "radius_m": radius,
                "points": n,
                "py_ms": round(1000 * t_py / n, 1),
                "py_par_ms": round(1000 * t_py_par / n, 1),
                "simple_ms": round(1000 * t_simple / n, 1),
                "rust_ms": round(1000 * t_rs / n, 2),
                "speedup": round(t_py / t_rs, 1),
                "speedup_vs_par": round(t_py_par / t_rs, 1),
                "iou_py_min": round(float(iou_py.min()), 7),
                "iou_rust_min": round(float(iou_rs.min()), 7),
                "iou_simple_mean": round(float(iou_simple.mean()), 4),
                "iou_rust_vs_py_min": round(float(iou_rs_py.min()), 7),
                "symdiff_py_m2": f"{float(sd_py.max()):.2e}",
                "symdiff_rust_m2": f"{float(sd_rs.max()):.2e}",
            }
            rows.append(row)
            raw.append({**row, "iou_py": iou_py.tolist(), "iou_rust": iou_rs.tolist()})
            print(row, flush=True)

            big = points_gdf(xy_all[:n_throughput], crs)
            with rust_core(True):
                _, t_all = timed(get_visibility, big, obstacles, radius, "accurate")
                _, t_one = timed(get_visibility, big, obstacles, radius, "accurate", max_workers=1)
            segments, one_sided = geometries_to_segments(obstacles.geometry.values)
            big_xy, big_r = xy_all[:n_throughput], np.full(n_throughput, float(radius))
            _, k_one = timed(native_visibility_polygons, big_xy, big_r, segments, one_sided, n_threads=1)
            _, k_all = timed(native_visibility_polygons, big_xy, big_r, segments, one_sided)
            throughput_rows.append(
                {
                    "scene": scene_name,
                    "radius_m": radius,
                    "points": len(big),
                    "rust_1_thread_s": round(t_one, 2),
                    "rust_all_threads_s": round(t_all, 2),
                    "points_per_s": int(len(big) / t_all),
                    "kernel_1_thread_ms": round(1000 * k_one / len(big), 3),
                    "kernel_all_threads_ms": round(1000 * k_all / len(big), 3),
                    "py_estimate_s": round(row["py_ms"] * len(big) / 1000, 0),
                }
            )
            print(throughput_rows[-1], flush=True)

    info = machine_info()
    md = [
        "# Visibility benchmark",
        "",
        "Generated by `benchmarks/bench_visibility.py`" + (" (quick mode)" if quick else "") + ".",
        "",
        "Machine: " + ", ".join(f"{k} {v}" for k, v in info.items()),
        "",
        "## Speed and accuracy on the same points",
        "",
        "Times are per observer, end to end through `get_visibility` (reprojection included). "
        "IoU is against the independent brute-force oracle (disc minus the union of every wall shadow); "
        "1.0 means identical polygons.",
        "",
        markdown_table(
            rows,
            [
                ("scene", "Scene"),
                ("radius_m", "Radius, m"),
                ("points", "Points"),
                ("py_ms", "Python accurate, ms"),
                ("py_par_ms", "Python accurate parallel, ms"),
                ("simple_ms", "Python simple, ms"),
                ("rust_ms", "Rust, ms"),
                ("speedup", "Speed-up vs Python"),
                ("speedup_vs_par", "Speed-up vs parallel"),
                ("iou_py_min", "IoU Python (min)"),
                ("iou_rust_min", "IoU Rust (min)"),
                ("iou_simple_mean", "IoU simple (mean)"),
                ("iou_rust_vs_py_min", "IoU Rust vs Python (min)"),
                ("symdiff_py_m2", "Max sym. diff. Python, m²"),
                ("symdiff_rust_m2", "Max sym. diff. Rust, m²"),
            ],
        ),
        "",
        "## Rust core throughput",
        "",
        "End to end through `get_visibility`, plus the bare kernel (`native_visibility_polygons` on prepared "
        "segments) to separate the algorithm from GeoPandas overhead (reprojection, spatial index, conversion).",
        "",
        markdown_table(
            throughput_rows,
            [
                ("scene", "Scene"),
                ("radius_m", "Radius, m"),
                ("points", "Points"),
                ("rust_1_thread_s", "1 thread, s"),
                ("rust_all_threads_s", "All threads, s"),
                ("points_per_s", "Points / s"),
                ("kernel_1_thread_ms", "Kernel, ms/point, 1 thread"),
                ("kernel_all_threads_ms", "Kernel, ms/point, all threads"),
                ("py_estimate_s", "Python accurate (estimated), s"),
            ],
        ),
        "",
    ]
    write_results(
        "visibility" + ("_quick" if quick else ""),
        {"machine": info, "rows": raw, "throughput": throughput_rows},
        "\n".join(md),
    )
    print("\n".join(md))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="small smoke run")
    run(parser.parse_args().quick)
