# Benchmarks

Reproducible speed and accuracy comparisons between ObjectNat's Rust kernels ([`rust/`](../rust)) and
the pure-Python implementations they replaced (frozen copy of ObjectNat 2.1.0 in
[`legacy_visibility.py`](legacy_visibility.py), not part of the package). Results from the last run are
committed under [`results/`](results).

| Script | What it compares | Result |
| --- | --- | --- |
| `bench_visibility.py` | `method="accurate"`: legacy Python sequential and `parallel=True` vs Rust; `method="simple"` for context | [`results/visibility.md`](results/visibility.md) |
| `bench_noise.py` | `simulate_noise` (reflection_n = 1, 2, 3) and `calculate_simplified_noise_frame` with legacy Python vs Rust visibility | [`results/noise.md`](results/noise.md) |

## Accuracy reference

`tests/visibility_oracle.py` is an independent brute-force visibility polygon: the view disc minus the
union of every wall's shadow quadrilateral and the obstacles themselves. It has no pruning and no
nearest-wall ordering, so it shares no logic with either implementation. Both implementations are
scored against it by IoU and by the largest symmetric-difference area.

Scenes:

- `wall` — one long wall; the exact answer is the view disc cut by a half-plane;
- `blocks` — a 40 × 40 grid of 40 m blocks with 20 m streets;
- `real` — the 4 101 example buildings in `docs/methods/examples/examples_data/buildings.parquet`.

## Running

```bash
uv sync --all-groups                     # builds the Rust extension; needs a Rust toolchain
uv run python benchmarks/bench_visibility.py
uv run python benchmarks/bench_noise.py
```

`bench_noise.py` swaps the legacy visibility into forked workers, so it runs on Linux and macOS only.
Add `--quick` for a short smoke run (written to `results/*_quick.*`, not committed).
Timings depend on the machine; the machine description is recorded at the top of each result.
