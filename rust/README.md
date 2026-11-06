# objectnat-rs

Rust part of [ObjectNat](https://github.com/IDUclub/ObjectNat), compiled with
[maturin](https://www.maturin.rs/) into the extension module `objectnat._rust`. It is not a separate
package: `pip install ObjectNat` ships it inside the wheels, and building ObjectNat from source builds it.

## Contents

- `src/visibility.rs` — exact batch visibility polygons (isovists) inside the same regular view polygon
  that shapely builds for `Point.buffer(r, quad_segs=q)`. Backs `get_visibility(method="accurate")` and
  the visibility step of the noise simulation.
- `src/lib.rs` — PyO3 bindings (`visibility_polygons`), abi3 for CPython 3.11+.

Visibility algorithm: obstacle walls near each observer come from a uniform grid and are clipped to the
view polygon; walls of valid polygons are oriented and culled when seen from the inside. Every wall
endpoint and wall crossing becomes a critical direction. Between two neighbouring critical directions
the nearest wall cannot change, so one grid-traversal ray per angular interval gives the exact boundary.
Observers run in parallel with rayon.

## Development

```bash
uv sync --all-groups                       # builds the extension into the editable install
(cd rust && cargo fmt --check && cargo clippy --release --all-targets -- -D warnings && cargo test --release)
uv run maturin build --release             # wheel into rust/target/wheels
```

The crate version is kept in sync with `pyproject.toml` by python-semantic-release; never bump it by hand.
