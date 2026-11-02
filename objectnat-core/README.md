# objectnat-core

Optional native (Rust) computational core for [ObjectNat](https://github.com/IDUclub/ObjectNat).

ObjectNat picks it up automatically when it is installed and falls back to its pure-Python
implementations otherwise. Results are the same up to floating point; see
`benchmarks/README.md` in the ObjectNat repository for the accuracy and speed comparison.

## What is inside

- `visibility_polygons` — exact batch visibility polygons (isovists) inside the same regular view
  polygon that `shapely` builds for `Point.buffer(r, quad_segs=q)`. Used by
  `objectnat.get_visibility(method="accurate")` and by the noise simulation.

Algorithm: obstacle segments near each observer are taken from a uniform grid, clipped to the view
polygon, and every segment endpoint and segment crossing becomes a critical direction. Between two
neighbouring critical directions the nearest wall cannot change, so one ray per angular interval
(grid traversal) gives the exact boundary. Observers run in parallel on all cores (rayon), without
copying the obstacles between processes.

## Install from source

Requires a Rust toolchain (`rustup`), Python 3.11+.

```bash
pip install ./objectnat-core           # from the ObjectNat repository root
# or, for development
cd objectnat-core && maturin develop --release
```

## Tests

```bash
cd objectnat-core && cargo test --release
```

## Disabling it

```python
from objectnat import config
config.set_enable_rust_core(False)   # force the pure-Python implementations
```
