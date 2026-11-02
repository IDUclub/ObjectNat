"""
Noise benchmark: ``simulate_noise`` and ``calculate_simplified_noise_frame`` with the pure-Python
visibility vs the Rust core (``objectnat-core``).

The noise model itself is unchanged; only the visibility step (``_visibility_accurate`` for every source
and reflection point, and ``get_visibility`` for the noise frame) switches implementation. Agreement is
measured on the dissolved noise layers: the share of the union area where both runs give the same
noise level, plus the largest per-level area difference.

``simulate_noise`` with reflections is not deterministic in its default parallel mode: which reflection
points survive the dead-area filter depends on the order in which worker processes finish. Each case is
therefore run twice per implementation, and the Python-vs-Python agreement is reported as the baseline
the Python-vs-Rust agreement should be read against.

Usage::

    uv run python benchmarks/bench_noise.py            # writes benchmarks/results/noise.*
    uv run python benchmarks/bench_noise.py --quick
"""

import argparse
import os
import sys

import geopandas as gpd
import numpy as np
import shapely
from shapely import Point

sys.path.insert(0, os.path.dirname(__file__))

from common import (  # noqa: E402  pylint: disable=wrong-import-position
    EXAMPLES_DATA,
    machine_info,
    markdown_table,
    rust_core,
    timed,
    write_results,
)

from objectnat import (  # noqa: E402  pylint: disable=wrong-import-position
    calculate_simplified_noise_frame,
    config,
    simulate_noise,
)


def agreement(left: gpd.GeoDataFrame, right: gpd.GeoDataFrame) -> dict:
    left = left.to_crs(left.estimate_utm_crs())
    right = right.to_crs(left.crs)
    total = shapely.union_all(np.concatenate([left.geometry.values, right.geometry.values])).area
    same = 0.0
    worst_level_diff = 0.0
    levels = sorted(set(left["noise_level"]) | set(right["noise_level"]))
    for level in levels:
        a = shapely.union_all(left.loc[left["noise_level"] == level].geometry.values)
        b = shapely.union_all(right.loc[right["noise_level"] == level].geometry.values)
        same += a.intersection(b).area
        worst_level_diff = max(worst_level_diff, abs(a.area - b.area))
    return {
        "levels": len(levels),
        "area_m2": round(total, 1),
        "same_level_share": round(same / total, 6) if total else 1.0,
        "max_level_area_diff_m2": round(worst_level_diff, 3),
    }


def sources_points():
    return gpd.GeoDataFrame(
        data=[[90, 3000], [80, 200]],
        geometry=[Point(30.27060176, 59.93546846), Point(30.27213917, 59.93575345)],
        columns=["source_noise_db", "geometric_mean_freq_hz"],
        crs=4326,
    )


def run(quick: bool) -> None:
    config.set_enable_tqdm(False)
    buildings = gpd.read_parquet(EXAMPLES_DATA / "buildings.parquet")
    trees = gpd.read_parquet(EXAMPLES_DATA / "trees.parquet")
    rows = []

    cases = [("simulate_noise", 1), ("simulate_noise", 2)]
    if not quick:
        cases.append(("simulate_noise", 3))
    cases.append(("noise_frame", None))

    for name, reflections in cases:
        if name == "simulate_noise":

            def call(reflections=reflections):
                return simulate_noise(
                    source_points=sources_points(),
                    obstacles=buildings,
                    trees=trees,
                    standart_absorb_ratio=0.05,
                    tree_resolution=4,
                    air_temperature=20,
                    target_noise_db=40,
                    db_sim_step=1,
                    reflection_n=reflections,
                    dead_area_r=5,
                )

        else:
            local = buildings.to_crs(32636)
            centre = sources_points().to_crs(32636).geometry.iloc[0]
            area = centre.buffer(400)
            near = local[local.intersects(area)]
            noisy = near.sample(n=max(1, len(near) // 50), random_state=42)[["geometry"]].copy()
            noisy["source_noise_db"] = 90
            noisy["geometric_mean_freq_hz"] = 2000
            points = sources_points().to_crs(32636)
            frame_sources = gpd.GeoDataFrame(
                gpd.pd.concat([noisy, points], ignore_index=True), geometry="geometry", crs=32636
            )

            def call(frame_sources=frame_sources, near=near):
                return calculate_simplified_noise_frame(
                    frame_sources.copy(), near.copy(), 20, target_noise_db=40, db_sim_step=5
                )

        label = name if reflections is None else f"{name} (reflection_n={reflections})"
        print(f"[{label}] python...", flush=True)
        with rust_core(False):
            py_result, t_py = timed(call)
            py_again, t_py2 = timed(call)
        print(f"[{label}] rust...", flush=True)
        with rust_core(True):
            rs_result, t_rs = timed(call)
            rs_again, t_rs2 = timed(call)
        t_py, t_rs = (t_py + t_py2) / 2, (t_rs + t_rs2) / 2
        row = {"case": label, "python_s": round(t_py, 2), "rust_s": round(t_rs, 2), "speedup": round(t_py / t_rs, 1)}
        row.update(agreement(py_result, rs_result))
        baseline_py = agreement(py_result, py_again)
        baseline_rs = agreement(rs_result, rs_again)
        row["py_vs_py_same_level_share"] = baseline_py["same_level_share"]
        row["py_vs_py_max_level_area_diff_m2"] = baseline_py["max_level_area_diff_m2"]
        row["rust_vs_rust_same_level_share"] = baseline_rs["same_level_share"]
        rows.append(row)
        print(row, flush=True)

    info = machine_info()
    md = [
        "# Noise benchmark",
        "",
        "Generated by `benchmarks/bench_noise.py`" + (" (quick mode)" if quick else "") + ".",
        "",
        "Machine: " + ", ".join(f"{k} {v}" for k, v in info.items()),
        "",
        "Same inputs as the noise example notebook (two point sources, example buildings and trees). "
        "Only the visibility step changes between the runs. *Same level share* is the fraction of the "
        "noise area where both runs assign the same level. Times are the mean of two runs. With "
        "reflections the default parallel mode is not deterministic, so the run-to-run agreement of the "
        "same implementation is shown as the baseline.",
        "",
        markdown_table(
            rows,
            [
                ("case", "Case"),
                ("python_s", "Python visibility, s"),
                ("rust_s", "Rust core, s"),
                ("speedup", "Speed-up"),
                ("levels", "Noise levels"),
                ("area_m2", "Noise area, m²"),
                ("same_level_share", "Same level share, Python vs Rust"),
                ("max_level_area_diff_m2", "Max per-level area diff, m²"),
                ("py_vs_py_same_level_share", "Baseline: Python vs Python"),
                ("py_vs_py_max_level_area_diff_m2", "Baseline max diff, m²"),
                ("rust_vs_rust_same_level_share", "Rust vs Rust"),
            ],
        ),
        "",
    ]
    write_results("noise" + ("_quick" if quick else ""), {"machine": info, "rows": rows}, "\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="skip the slowest case")
    run(parser.parse_args().quick)
