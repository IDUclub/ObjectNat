"""Regular-grid results as rasters: arrays with a transform and a CRS, for plots, GeoTIFF files and web maps."""

from typing import Any, NamedTuple

import geopandas as gpd
import numpy as np
import shapely
from pyproj import CRS, Transformer

from objectnat._rust import label_rings  # pylint: disable=no-name-in-module

GRID_ATTR = "grid"
MAX_LABEL = 255  # label_rings takes uint8 labels


def resolve_output(output: str, points: str) -> str:
    """Check ``output`` of a grid method: ``"zones"``, ``points`` (``"receivers"`` or ``"cells"``) or ``"both"``."""
    if output not in ("zones", points, "both"):
        raise ValueError(f'output must be "zones", "{points}" or "both"; got {output!r}')
    return output


def label_zones(labels: np.ndarray, origin: tuple[float, float], cell_size: float) -> dict[int, object]:
    """One (Multi)Polygon per non-zero label of the grid, keyed by label."""
    coords, offsets, ring_labels = label_rings(labels, origin, cell_size)
    ring_index = np.repeat(np.arange(len(ring_labels)), np.diff(offsets))
    rings = shapely.linearrings(coords, indices=ring_index)
    is_exterior = shapely.is_ccw(rings)
    zones = {}
    for label in np.unique(ring_labels):
        shells = rings[(ring_labels == label) & is_exterior]
        holes = rings[(ring_labels == label) & ~is_exterior]
        shell_polys = shapely.polygons(shells)
        hole_lists = [[] for _ in range(len(shells))]
        if len(holes):
            hole_polys = shapely.polygons(holes)
            hole_ids, shell_ids = shapely.STRtree(shell_polys).query(hole_polys, predicate="within")
            # nested shells: a hole belongs to the smallest shell around it
            order = np.lexsort((shapely.area(shell_polys)[shell_ids], hole_ids))
            hole_ids, shell_ids = hole_ids[order], shell_ids[order]
            first = np.r_[True, hole_ids[1:] != hole_ids[:-1]]
            for h, s in zip(hole_ids[first], shell_ids[first]):
                hole_lists[s].append(holes[h])
        polygons = [shapely.Polygon(shell, hole_list) for shell, hole_list in zip(shells, hole_lists)]
        zone = shapely.MultiPolygon(polygons) if len(polygons) > 1 else polygons[0]
        if not zone.is_valid:
            zone = shapely.make_valid(zone, method="structure", keep_collapsed=False)
        zones[int(label)] = zone
    return zones


def grid_zones(xy: np.ndarray, values: np.ndarray, cell_size: float, cut=None) -> list[tuple[Any, Any]]:
    """
    ``(value, (Multi)Polygon)`` for every distinct value of grid points ``xy`` (cell centres ``cell_size`` apart):
    the union of their cells minus ``cut``, ordered by value. Rings are traced on the label grid in Rust.
    """
    if len(xy) == 0:
        return []
    lo = xy.min(axis=0)
    i, j = np.round((xy - lo) / cell_size).astype(np.int64).T
    origin = (float(lo[0] - cell_size / 2), float(lo[1] - cell_size / 2))
    keys, codes = np.unique(values, return_inverse=True)
    zones = []
    for start in range(0, len(keys), MAX_LABEL):
        labels = np.zeros((j.max() + 1, i.max() + 1), dtype=np.uint8)
        chunk = (codes >= start) & (codes < start + MAX_LABEL)
        labels[j[chunk], i[chunk]] = codes[chunk] - start + 1
        for label, geometry in label_zones(labels, origin, cell_size).items():
            if cut is not None and not cut.is_empty:
                geometry = geometry.difference(cut)
            if not geometry.is_empty:
                zones.append((keys[start + label - 1], geometry))
    return zones


def grid_attrs(local_crs, cell_size: float, origin: tuple[float, float]) -> dict:
    """``attrs`` entry that lets :func:`rasterize` place grid points exactly: CRS, cell size, one cell centre."""
    return {
        GRID_ATTR: {
            "crs": CRS.from_user_input(local_crs).to_wkt(),
            "cell_size": float(cell_size),
            "origin": (float(origin[0]), float(origin[1])),
        }
    }


class Raster(NamedTuple):
    """
    Values on a north-up grid of square cells, as returned by :func:`rasterize`.

    It is the tuple ``(values, transform, crs)`` that ``terrain`` arguments accept, so it can be passed back as
    terrain, written with rasterio, or drawn with matplotlib.

    Attributes:
        values: ``(rows, cols)`` float array; row 0 is the northern edge, ``NaN`` marks cells without a value.
        transform: Affine coefficients ``(a, b, c, d, e, f)`` as in rasterio and GDAL: the upper-left corner of
            cell ``(row, col)`` is ``x = c + a * col``, ``y = f + e * row`` (``b = d = 0``, ``e = -a``).
        crs: Projected CRS of the grid (metres).
    """

    values: np.ndarray
    transform: tuple
    crs: Any

    @property
    def cell_size(self) -> float:
        """Cell side in metres."""
        return float(self.transform[0])

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """``(minx, miny, maxx, maxy)`` of the grid."""
        a, _, c, _, e, f = self.transform
        rows, cols = self.values.shape
        return c, f + e * rows, c + a * cols, f

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """``(left, right, bottom, top)``, the ``extent`` argument of ``matplotlib.pyplot.imshow``."""
        minx, miny, maxx, maxy = self.bounds
        return minx, maxx, miny, maxy

    def plot(self, ax=None, **kwargs):
        """
        Draw the raster with ``ax.imshow`` in the CRS of the grid (cells without a value stay transparent).

        Keyword arguments go to ``imshow`` (``cmap``, ``vmin``, ``vmax``, ``alpha``, ...). Layers drawn on the same
        axes must be in :attr:`crs`, e.g. ``buildings.to_crs(raster.crs).plot(ax=ax)``. Requires matplotlib.

        Returns:
            The ``AxesImage``, e.g. for ``fig.colorbar``.
        """
        import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

        if ax is None:
            ax = plt.gca()
        kwargs.setdefault("interpolation", "nearest")
        image = ax.imshow(self.values, extent=self.extent, origin="upper", **kwargs)
        ax.set_aspect("equal")
        return image

    def to_geotiff(self, path, *, nodata: float = np.nan) -> None:
        """Write a single-band float32 GeoTIFF; cells without a value get ``nodata``. Requires rasterio."""
        try:
            import rasterio  # pylint: disable=import-outside-toplevel
            from rasterio.transform import Affine  # pylint: disable=import-outside-toplevel
        except ImportError as exc:
            raise ImportError("Writing GeoTIFF needs rasterio: pip install rasterio") from exc
        values = np.where(np.isnan(self.values), nodata, self.values).astype("float32")
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=values.shape[0],
            width=values.shape[1],
            count=1,
            dtype="float32",
            crs=CRS.from_user_input(self.crs).to_wkt(),
            transform=Affine(*self.transform[:6]),
            nodata=nodata,
            compress="deflate",
        ) as dst:
            dst.write(values, 1)

    def to_folium(
        self,
        *,
        cmap="viridis",
        vmin: float | None = None,
        vmax: float | None = None,
        opacity: float = 0.75,
        name: str | None = None,
        **kwargs,
    ):
        """
        Colour the raster and return it as a ``folium.raster_layers.ImageOverlay`` for a web map.

        Web maps show images in Web Mercator, so the grid is resampled (nearest cell) to a Mercator grid with
        about the same number of cells before colouring; the overlay then lines up with the base map. Add it
        with ``.add_to(m)``, e.g. to a map from ``GeoDataFrame.explore()``. Requires folium and matplotlib.

        Args:
            cmap: Matplotlib colormap or its name.
            vmin, vmax: Colour scale limits (default: minimum and maximum of the values).
            opacity: Opacity of the overlay, 0-1. Cells without a value are transparent.
            name: Layer name in ``folium.LayerControl``.
            **kwargs: Passed to ``ImageOverlay``.
        """
        try:
            import folium  # pylint: disable=import-outside-toplevel
            import matplotlib as mpl  # pylint: disable=import-outside-toplevel
        except ImportError as exc:
            raise ImportError("to_folium needs folium and matplotlib: pip install folium matplotlib") from exc

        mercator, (south, west, north, east) = self._to_mercator()
        finite = np.isfinite(mercator)
        lo = np.nanmin(self.values) if vmin is None else vmin
        hi = np.nanmax(self.values) if vmax is None else vmax
        norm = mpl.colors.Normalize(lo, hi if hi > lo else lo + 1)
        colormap = mpl.colormaps[cmap] if isinstance(cmap, str) else cmap
        rgba = colormap(norm(np.where(finite, mercator, lo)))
        rgba[..., 3] = np.where(finite, rgba[..., 3], 0.0)
        return folium.raster_layers.ImageOverlay(
            image=np.round(rgba * 255).astype("uint8"),
            bounds=[[south, west], [north, east]],
            opacity=opacity,
            name=name,
            mercator_project=False,
            **kwargs,
        )

    def explore(
        self,
        m=None,
        *,
        cmap="viridis",
        vmin: float | None = None,
        vmax: float | None = None,
        opacity: float = 0.75,
        name: str | None = None,
        legend: bool = True,
        caption: str | None = None,
        tiles: str = "OpenStreetMap",
        **kwargs,
    ):
        """
        Show the raster on an interactive folium map, like ``GeoDataFrame.explore()``.

        Args:
            m: Existing ``folium.Map`` to add the layer to, e.g. ``buildings.explore()``; a new map centred on the
                raster when ``None``.
            cmap, vmin, vmax, opacity, name: As in :meth:`to_folium`.
            legend: Add a colour bar.
            caption: Colour bar caption (default ``name``).
            tiles: Base map of a new map.
            **kwargs: Passed to ``folium.Map`` when a new map is created.

        Returns:
            The ``folium.Map``. Requires folium and matplotlib.
        """
        import branca.colormap  # pylint: disable=import-outside-toplevel
        import folium  # pylint: disable=import-outside-toplevel
        import matplotlib as mpl  # pylint: disable=import-outside-toplevel

        lo = float(np.nanmin(self.values)) if vmin is None else float(vmin)
        hi = float(np.nanmax(self.values)) if vmax is None else float(vmax)
        overlay = self.to_folium(cmap=cmap, vmin=lo, vmax=hi, opacity=opacity, name=name)
        if m is None:
            (south, west), (north, east) = overlay.bounds
            m = folium.Map(tiles=tiles, **kwargs)
            m.fit_bounds([[south, west], [north, east]])
        overlay.add_to(m)
        if legend:
            colormap = mpl.colormaps[cmap] if isinstance(cmap, str) else cmap
            colors = [mpl.colors.to_hex(colormap(v)) for v in np.linspace(0, 1, 11)]
            colour_bar = branca.colormap.LinearColormap(colors, vmin=lo, vmax=hi if hi > lo else lo + 1)
            colour_bar.caption = caption if caption is not None else (name or "")
            colour_bar.add_to(m)
        return m

    def _to_mercator(self) -> tuple[np.ndarray, tuple[float, float, float, float]]:
        """Nearest-cell resampling to a north-up EPSG:3857 grid; also its bounds in degrees."""
        minx, miny, maxx, maxy = self.bounds
        # the outline, densified: straight grid edges are curves in Mercator
        ring = shapely.get_coordinates(
            shapely.segmentize(shapely.box(*self.bounds), max(maxx - minx, maxy - miny) / 32)
        )
        mx, my = Transformer.from_crs(self.crs, 3857, always_xy=True).transform(ring[:, 0], ring[:, 1])
        mx0, mx1, my0, my1 = min(mx), max(mx), min(my), max(my)
        rows, cols = self.values.shape
        step = np.sqrt((mx1 - mx0) * (my1 - my0) / max(rows * cols, 1))
        out_cols, out_rows = max(int(np.ceil((mx1 - mx0) / step)), 1), max(int(np.ceil((my1 - my0) / step)), 1)
        cx = mx0 + (np.arange(out_cols) + 0.5) * (mx1 - mx0) / out_cols
        cy = my1 - (np.arange(out_rows) + 0.5) * (my1 - my0) / out_rows
        gx, gy = np.meshgrid(cx, cy)
        x, y = Transformer.from_crs(3857, self.crs, always_xy=True).transform(gx.ravel(), gy.ravel())
        a, _, c, _, e, f = self.transform
        col = np.floor((np.asarray(x) - c) / a).astype(np.int64)
        row = np.floor((np.asarray(y) - f) / e).astype(np.int64)
        inside = (col >= 0) & (col < cols) & (row >= 0) & (row < rows)
        out = np.full(out_rows * out_cols, np.nan)
        out[inside] = self.values[row[inside], col[inside]]
        to_deg = Transformer.from_crs(3857, 4326, always_xy=True)
        west, south = to_deg.transform(mx0, my0)
        east, north = to_deg.transform(mx1, my1)
        return out.reshape(out_rows, out_cols), (south, west, north, east)


def rasterize(points: gpd.GeoDataFrame, column: str, *, cell_size: float | None = None) -> Raster:
    """
    Turn grid points with values into a :class:`Raster`, one cell per point.

    Grid points are what the grid methods return: receivers of :func:`get_insolation` and
    :func:`simulate_noise_grid` (``output="receivers"`` or ``"both"``), cells of :func:`get_visibility_catchment`
    and :func:`get_viewshed_catchment` (``output="cells"`` or ``"both"``) and the points of
    :func:`get_isovist_grid`. Their
    ``attrs`` record the grid (CRS, cell size, position), so the raster reproduces it exactly in the local metric
    CRS of the calculation, whatever CRS the points were returned in, and also after filtering rows.

    Points without that record (for example, read from a file) require ``cell_size``; they are then placed in their
    own CRS if it is projected, or in an estimated UTM zone, with the first point as a cell centre.

    Args:
        points: Point GeoDataFrame on a regular grid.
        column: Numeric (or boolean) column to put into the cells.
        cell_size: Cell side in metres, for points without grid ``attrs``.

    Returns:
        :class:`Raster` covering the points; cells without a point are ``NaN``.

    Examples:
        >>> receivers = get_insolation(buildings, "2026-03-21", output="receivers")
        >>> raster = rasterize(receivers, "sun_hours")
        >>> raster.plot(cmap="inferno", vmin=0, vmax=14)
        >>> raster.explore(cmap="inferno", vmin=0, vmax=14, name="sun hours")
        >>> raster.to_geotiff("sun_hours.tif")
    """
    if column not in points.columns:
        raise ValueError(f"{column!r} is not a column of points")
    if points.crs is None:
        raise ValueError("points must have a CRS")
    grid = points.attrs.get(GRID_ATTR)
    if grid is not None and cell_size is None:
        crs = CRS.from_wkt(grid["crs"])
        step = float(grid["cell_size"])
        origin = grid["origin"]
    elif cell_size is not None:
        if cell_size <= 0:
            raise ValueError("cell_size must be > 0")
        crs = points.crs if points.crs.is_projected else points.estimate_utm_crs()
        step = float(cell_size)
        origin = None
    else:
        raise ValueError("points carry no grid record in attrs; pass cell_size")
    if len(points) == 0:
        raise ValueError("points is empty")
    xy = shapely.get_coordinates(points.geometry.to_crs(crs).values)
    if len(xy) != len(points):
        raise ValueError("points must hold one point per row")
    if origin is None:
        origin = xy[0]
    i = np.round((xy[:, 0] - origin[0]) / step).astype(np.int64)
    j = np.round((xy[:, 1] - origin[1]) / step).astype(np.int64)
    i0, j1 = i.min(), j.max()
    values = np.full((j1 - j.min() + 1, i.max() - i0 + 1), np.nan)
    values[j1 - j, i - i0] = points[column].to_numpy(dtype=float)
    left = origin[0] + (i0 - 0.5) * step
    top = origin[1] + (j1 + 0.5) * step
    return Raster(values, (step, 0.0, float(left), 0.0, -step, float(top)), crs)
