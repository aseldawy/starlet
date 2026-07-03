"""starlet — spatial tiling, MVT generation, and tile serving for geospatial data."""
from __future__ import annotations

from importlib.metadata import version
__version__ = version("starlet")

from starlet._types import TileResult, MVTResult, Dataset

__all__ = [
    "tile",
    "generate_mvt",
    "build",
    "create_app",
    "export_pmtiles",
    "TileResult",
    "MVTResult",
    "Dataset",
]

_GEOJSON_DEFAULT_PARTITION_SIZE = 512 * 1024 * 1024
_GEOPARQUET_DEFAULT_PARTITION_SIZE = 128 * 1024 * 1024


def tile(
    input: str,
    outdir: str,
    *,
    partition_size: int | None = None,
    sort: str = "zorder",
    compression: str = "zstd",
    sample_cap: int | None = 10_000,
    sample_ratio: float = 1.0,
    seed: int = 42,
    geom_col: str = "geometry",
    sfc_bits: int = 16,
    max_parallel_files: int = 64,
    covering_bbox: bool = False,
    csv_x_col: str | None = None,
    csv_y_col: str | None = None,
    csv_wkt_col: str | None = None,
    csv_split_size: int = 32 * 1024 * 1024,
    src_crs: str = "EPSG:4326",
    geojson_executor: str = "process",
    orchestrator: str = "two-stage",
    two_stage_executor: str = "process",
    two_stage_assignment_workers: int | None = None,
    two_stage_write_workers: int | None = None,
    two_stage_reducers: int | None = None,
    temp_dir: str | None = None,
) -> TileResult:
    """Partition a GeoParquet/GeoJSON dataset into spatially-tiled Parquet files.

    Parameters
    ----------
    input : str
        Path to a GeoParquet, GeoJSON, or GeoJSON-Lines file.
    outdir : str
        Output directory. Tiled files go into ``<outdir>/parquet_tiles/``
        and histograms into ``<outdir>/histograms/``.
    partition_size : int | None
        Target partition size in bytes. When omitted, defaults to 512 MiB for
        GeoJSON and 128 MiB for GeoParquet. The number of partitions is
        derived from the input file size.
    sort : str
        Row sort order within each tile: ``"zorder"``, ``"hilbert"``,
        ``"columns"``, or ``"none"``.
    compression : str
        Parquet compression codec (default ``"zstd"``).
    sample_cap : int | None
        Reservoir sampling cap for centroid sampling.
    sample_ratio : float
        Bernoulli sampling ratio for centroids (0 < r <= 1).
    seed : int
        Random seed for RSGrove partitioner.
    geom_col : str
        Name of the geometry column.
    sfc_bits : int
        Bits per axis for Z-order / Hilbert key.
    max_parallel_files : int
        Maximum concurrent tile files during write.
    covering_bbox : bool
        Opt-in read-time pruning. If True, write four per-row bbox covering
        columns plus bounded, spatially-coherent row groups so the on-demand
        tile server can skip row groups/rows at read time. Off by default
        (faster batch tiling, smaller files); enable when serving tiles
        on the fly from these partitions.
    csv_x_col, csv_y_col : str | None
        Column names containing x/y coordinates for CSV inputs. Provide both,
        or provide ``csv_wkt_col`` instead.
    csv_wkt_col : str | None
        Column name containing WKT geometry for CSV inputs.
    csv_split_size : int
        Target byte length for each CSV source split.
    src_crs : str
        CRS hint for CSV inputs and other sources without embedded CRS.
    geojson_executor : str
        Executor used for GeoJSON spatial sampling: ``"process"`` for
        production CPU parallelism or ``"thread"`` for small inputs / test
        environments (avoids process-pool spawn overhead).
    orchestrator : str
        Tiling orchestrator to use: ``"round"`` or ``"two-stage"``.
    two_stage_executor : str
        Executor used by the two-stage orchestrator: ``"process"`` or ``"thread"``.
    two_stage_assignment_workers : int | None
        Number of workers for two-stage split assignment.
    two_stage_write_workers : int | None
        Number of workers for two-stage tile writes.
    two_stage_reducers : int | None
        Number of hash-shuffle reducers for the two-stage orchestrator.
    temp_dir : str | None
        Parent directory for two-stage temporary shard files. Defaults to
        ``./tmp`` under the current working directory.

    Returns
    -------
    TileResult
    """
    import logging
    import math
    from pathlib import Path

    from starlet._internal.tiling.datasource import read_spatial_sample, source_for_path
    from starlet._internal.tiling.geojson_source import is_geojson_path
    from starlet._internal.tiling.assigner import RSGroveAssigner
    from starlet._internal.tiling.orchestrator import RoundOrchestrator
    from starlet._internal.tiling.two_stage_orchestrator import TwoStageOrchestrator
    from starlet._internal.tiling.writer_pool import SortMode
    from starlet._internal.histogram.hist_pyramid import build_histograms_for_dir

    logger = logging.getLogger("starlet.tile")

    # Parse sort mode
    _sort_map = {
        "none": SortMode.NONE,
        "columns": SortMode.COLUMNS,
        "zorder": SortMode.ZORDER,
        "hilbert": SortMode.HILBERT,
    }
    sort_mode = _sort_map.get(sort.strip().lower(), SortMode.ZORDER)

    # Build data source and choose a format-appropriate default size.
    source = source_for_path(
        input,
        geom_col=geom_col,
        csv_x_col=csv_x_col,
        csv_y_col=csv_y_col,
        csv_wkt_col=csv_wkt_col,
        csv_split_size=csv_split_size,
        src_crs=src_crs,
    )
    geom_col = getattr(source, "geom_col", geom_col)
    if partition_size is None:
        partition_size = (
            _GEOJSON_DEFAULT_PARTITION_SIZE
            if is_geojson_path(input)
            else _GEOPARQUET_DEFAULT_PARTITION_SIZE
        )

    # Determine partition count
    if partition_size <= 0:
        raise ValueError("partition_size must be greater than zero")
    input_size_bytes = source.input_size_bytes()
    target_partitions = max(1, math.ceil(input_size_bytes / partition_size))
    logger.info(
        "Target partitions: %d (input=%d bytes, target_partition_size=%d bytes)",
        target_partitions,
        input_size_bytes,
        partition_size,
    )

    # Build assigner
    spatial_sample = read_spatial_sample(
        input,
        geom_col=geom_col,
        seed=seed,
        sample_ratio=sample_ratio,
        sample_cap=sample_cap,
        csv_x_col=csv_x_col,
        csv_y_col=csv_y_col,
        csv_wkt_col=csv_wkt_col,
        csv_split_size=csv_split_size,
        src_crs=src_crs,
        geojson_executor=geojson_executor,
    )
    assigner = RSGroveAssigner.from_sample_and_mbr(
        sample_points=spatial_sample.sample_points,
        mbr=spatial_sample.mbr,
        num_partitions=target_partitions,
        geom_col=geom_col,
    )

    tiles_dir = str(Path(outdir) / "parquet_tiles")
    hist_dir = str(Path(outdir) / "histograms")

    orchestrator_name = orchestrator.strip().lower().replace("_", "-")
    if orchestrator_name == "round":
        tiling_orchestrator = RoundOrchestrator(
            source=source,
            assigner=assigner,
            outdir=tiles_dir,
            geom_col=geom_col,
            max_parallel_files=max_parallel_files,
            compression=compression,
            sort_mode=sort_mode,
            sfc_bits=sfc_bits,
            covering_bbox=covering_bbox,
        )
    elif orchestrator_name in {"two-stage", "twostage"}:
        tiling_orchestrator = TwoStageOrchestrator(
            source=source,
            assigner=assigner,
            outdir=tiles_dir,
            geom_col=geom_col,
            compression=compression,
            sort_mode=sort_mode,
            sfc_bits=sfc_bits,
            executor=two_stage_executor,
            assignment_workers=two_stage_assignment_workers,
            write_workers=two_stage_write_workers,
            num_reducers=two_stage_reducers,
            temp_dir=temp_dir,
            covering_bbox=covering_bbox,
        )
    else:
        raise ValueError("orchestrator must be 'round' or 'two-stage'")
    tiling_orchestrator.run()

    logger.info("Tiling complete. Building histograms.")
    build_histograms_for_dir(
        tiles_dir=tiles_dir,
        outdir=hist_dir,
        geom_col=geom_col,
        grid_size=4096,
        dtype="float64",
        hist_max_parallel=8,
        hist_rg_parallel=4,
    )

    # Gather result metadata
    tile_files = list(Path(tiles_dir).glob("*.parquet"))
    total_rows = 0
    for tf in tile_files:
        import pyarrow.parquet as pq
        meta = pq.read_metadata(str(tf))
        total_rows += meta.num_rows

    ds = Dataset(outdir)
    result_bbox = ds.bbox or (0.0, 0.0, 0.0, 0.0)

    return TileResult(
        outdir=outdir,
        num_files=len(tile_files),
        total_rows=total_rows,
        bbox=result_bbox,
        histogram_path=str(Path(hist_dir) / "global_prefix.npy"),
    )


def generate_mvt(
    tile_dir: str,
    *,
    zoom: int = 7,
    threshold: float = 0,
    outdir: str | None = None,
    auto_zoom: bool = True,
    occupancy_threshold: float = 0.01,
    max_workers: int | None = None,
) -> MVTResult:
    """Generate Mapbox Vector Tiles from a tiled dataset.

    Parameters
    ----------
    tile_dir : str
        Dataset directory containing ``parquet_tiles/`` and ``histograms/``.
    zoom : int
        Maximum zoom level.
    threshold : float
        Minimum feature count per tile.
    outdir : str | None
        MVT output directory. Defaults to ``<tile_dir>/mvt/``.
    auto_zoom : bool
        Automatically detect maximum useful zoom level from histogram density.
        If True and data becomes sparse before ``zoom``, generation stops early.
        Default True.
    occupancy_threshold : float
        Minimum tile occupancy (nonempty_tiles / total_tiles) for auto-zoom detection.
        Default 0.01 (1% occupancy).

    Returns
    -------
    MVTResult
    """
    from pathlib import Path
    from starlet._internal.mvt.generator import BucketMVTGenerator

    parquet_dir = str(Path(tile_dir) / "parquet_tiles")
    # Prefer the precomputed integral image written by the tiling stage; fall
    # back to the raw histogram (recomputing the prefix sum) for older datasets.
    hist_dir = Path(tile_dir) / "histograms"
    prefix_path = hist_dir / "global_prefix.npy"
    hist_path = str(prefix_path if prefix_path.exists() else hist_dir / "global.npy")
    mvt_outdir = outdir or str(Path(tile_dir) / "mvt")

    gen = BucketMVTGenerator(
        parquet_dir=parquet_dir,
        hist_path=hist_path,
        outdir=mvt_outdir,
        last_zoom=zoom,
        threshold=threshold,
        auto_zoom=auto_zoom,
        occupancy_threshold=occupancy_threshold,
        max_workers=max_workers,
    )
    gen.run()

    # Count generated tiles
    mvt_path = Path(mvt_outdir)
    tile_count = len(list(mvt_path.rglob("*.mvt")))
    zoom_levels = sorted(
        int(d.name) for d in mvt_path.iterdir()
        if d.is_dir() and d.name.isdigit()
    ) if mvt_path.exists() else []

    return MVTResult(
        outdir=mvt_outdir,
        zoom_levels=zoom_levels,
        tile_count=tile_count,
    )


def build(
    input: str,
    outdir: str,
    *,
    zoom: int = 7,
    partition_size: int | None = None,
    threshold: float = 100_000,
    pmtiles: bool = False,
    pmtiles_compression: str = "gzip",
    **tile_kwargs,
) -> tuple[TileResult, MVTResult, str | None]:
    """Run the full pipeline: tile then generate MVTs.

    Parameters
    ----------
    input : str
        Path to source GeoParquet or GeoJSON file.
    outdir : str
        Output dataset directory.
    zoom : int
        Maximum zoom level for MVT generation.
    partition_size : int | None
        Target partition size in bytes (forwarded to :func:`tile`). When
        omitted, a format-appropriate default is used.
    threshold : float
        Minimum feature count per MVT tile.
    pmtiles : bool
        If True, export MVT tiles to a PMTiles archive after generation.
        Default False.
    pmtiles_compression : str
        Compression for PMTiles export: "gzip", "brotli", "zstd", "none".
        Default "gzip". Only used if pmtiles=True.
    **tile_kwargs
        Additional keyword arguments forwarded to :func:`tile`
        (e.g. ``covering_bbox=True``, ``orchestrator="round"``,
        ``geojson_executor="thread"``).

    Returns
    -------
    tuple[TileResult, MVTResult, str | None]
        Returns (tile_result, mvt_result, pmtiles_path).
        pmtiles_path is None if pmtiles=False.
    """
    from pathlib import Path

    tile_result = tile(
        input=input, outdir=outdir, partition_size=partition_size, **tile_kwargs
    )
    mvt_result = generate_mvt(tile_dir=outdir, zoom=zoom, threshold=threshold)

    pmtiles_path = None
    if pmtiles:
        from starlet._internal.pmtiles.exporter import export_to_pmtiles

        dataset_name = Path(outdir).name
        pmtiles_path = str(Path(outdir).parent / f"{dataset_name}.pmtiles")

        export_to_pmtiles(
            mvt_dir=str(Path(outdir) / "mvt"),
            output_path=pmtiles_path,
            tile_type="mvt",
            compression=pmtiles_compression,
        )

    return tile_result, mvt_result, pmtiles_path


def export_pmtiles(
    mvt_dir: str,
    output_path: str,
    tile_type: str = "mvt",
    compression: str = "gzip",
) -> str:
    """Export MVT tiles to PMTiles archive format.

    Parameters
    ----------
    mvt_dir : str
        Directory containing MVT tiles in z/x/y.mvt structure.
        Typically ``<dataset>/mvt/``.
    output_path : str
        Path to output .pmtiles file.
    tile_type : str
        Tile type: "mvt" (vector), "png", "jpg", "webp" (raster).
        Default "mvt".
    compression : str
        Compression: "gzip", "none", "brotli", "zstd".
        Default "gzip".

    Returns
    -------
    str
        Path to created PMTiles file.

    Examples
    --------
    >>> # After running build/generate_mvt
    >>> export_pmtiles(
    ...     mvt_dir="datasets/mydata/mvt",
    ...     output_path="datasets/mydata.pmtiles"
    ... )
    """
    from starlet._internal.pmtiles.exporter import export_to_pmtiles
    return export_to_pmtiles(mvt_dir, output_path, tile_type, compression)


def create_app(data_dir: str, cache_size: int = 256):
    """Create a Flask tile server application.

    Parameters
    ----------
    data_dir : str
        Root directory containing dataset subdirectories.
    cache_size : int
        Number of tiles in the in-memory LRU cache.

    Returns
    -------
    Flask
        Configured Flask application.
    """
    from starlet._internal.server.app import create_app as _create_app
    return _create_app(data_dir=data_dir, cache_size=cache_size)
