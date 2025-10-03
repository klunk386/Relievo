"""
RELIEVO - DEM-based 3D Terrain Model Generator

Author: Valerio Poggi
License: GNU Affero General Public License v3.0 (AGPL-3.0)

Description
-----------
Relievo is a Python module for generating watertight 3D terrain models 
from one or more GeoTIFF Digital Elevation Models (DEMs). Models are 
clipped to a user-defined geographic region and exported as STL meshes 
ready for physical fabrication or scientific visualization.

The terrain model includes:
- a topographic surface from sampled DEM points,
- vertical side walls matching the polygon boundary,
- a flat base at user-defined depth.

The module supports:
- automatic merging and projection of DEM tiles,
- input geometries as bounding boxes, shapely polygons, or GeoJSON,
- uniform resampling or densification of polygon boundaries,
- optional subdivision into square tiles with individual STL output.

Typical applications include:
- 3D printing for outreach and education,
- scaled terrain modeling for geoscience research,
- topography visualization for exhibitions or technical design.

Main Features
-------------
- DEM reprojection to projected (metric) CRS (e.g., UTM)
- Polygon boundary handling and optional densification
- DEM sampling with interior point filtering
- Triangle-based surface meshing with polygon constraints
- Generation of side walls and flat base meshes
- Export of STL file(s), with optional tiling

Intended for use in research, teaching, and outreach.
"""
import json
from pathlib import Path

import numpy as np
import rasterio
import triangle
import trimesh
from pyproj import CRS, Transformer
from rasterio.merge import merge
from rasterio.transform import array_bounds, rowcol
from rasterio.warp import (
    Resampling,
    calculate_default_transform,
    reproject,
)
from scipy.spatial import cKDTree
from shapely import vectorized
from shapely.geometry import (
    LinearRing,
    MultiPolygon,
    Point,
    Polygon,
    shape,
)
from shapely.ops import transform as shapely_transform
from shapely.validation import explain_validity


def relievo(
    tif_paths,
    geometry,
    resolution_m=100,
    base_depth=-1000.0,
    xy_scale=1 / 750,
    vertical_exaggeration=1.5,
    tile_size_m=None,
    tile_nx=None,
    tile_ny=None,
    utm_crs="EPSG:32633",
    out_prefix="model_tile",
    property_key=None,
    property_value=None,
    verbose=True
):
    """
    Generate a watertight 3D STL terrain model from DEM and polygon geometry.

    Depending on the input, the output can be a single STL file or multiple
    tiles covering the selected region. The model includes topography, side
    walls, and a flat base.

    Parameters
    ----------
    tif_paths : str or list of str
        Path or list of paths to GeoTIFF DEM files or a directory.
    geometry : list, str, or shapely.geometry.Polygon
        Region of interest. Accepts:
        - A bounding box list ((lon_min, lat_min), (lon_max, lat_max))
        - A file path to a GeoJSON file
        - A Shapely Polygon or MultiPolygon
    resolution_m : float, default 100
        Horizontal sampling resolution in meters.
    base_depth : float, default -1000.0
        Elevation of the flat base in meters.
    xy_scale : float, default 1/750
        Scaling factor for horizontal axes (X, Y).
    vertical_exaggeration : float, default 1.5
        Additional vertical exaggeration multiplier (applied on Z).
    tile_size_m : float or None, default None
        Optional tile size in meters. If set, export multiple STL tiles.
    tile_size_m : float or None, default None
        Optional square tile size in meters. Mutually exclusive with
        (tile_nx, tile_ny). If set, multiple STL tiles are exported.
    tile_nx, tile_ny : int or None, default None
        Number of tiles along X (longitude) and Y (latitude).
        Mutually exclusive with tile_size_m. If both are provided,
        the DEM is subdivided into a grid of tile_nx × tile_ny.
    utm_crs : str, default "EPSG:32633"
        EPSG code of the projected CRS used.
    out_prefix : str, default "model_tile"
        Path and prefix name for the output STL file(s).
    property_key : str, optional
        GeoJSON property key used for selecting features.
    property_value : str or int, optional
        GeoJSON property value to match for feature selection.
    verbose : bool, default True
        If True, print progress messages during processing.

    Returns
    -------
    None
        STL file(s) are written to disk.
    """
    if verbose:
        print("[1] Loading and merging DEM tiles...")

    dem_array, transform, crs_dem = load_dem_from_geotiff(tif_paths)

    if verbose:
        print(f"[2] Reprojecting DEM to {utm_crs}...")

    crs_utm = CRS.from_string(utm_crs)
    dem_proj, transform_proj = project_dem_to_meters(
        dem_array=dem_array,
        transform=transform,
        source_crs=crs_dem,
        target_crs=crs_utm,
        resolution=resolution_m
    )

    if verbose:
        print("[3] Interpreting and projecting geometry...")

    if (isinstance(geometry, (list, tuple)) and len(geometry) == 2):
        if  all(isinstance(pt, (list, tuple)) for pt in geometry):
            (lon_min, lat_min), (lon_max, lat_max) = geometry
            polygon_proj = create_rectangle_polygon_utm(
                lon_min, lon_max, lat_min, lat_max,
                target_crs=crs_utm,
                resolution=resolution_m
            )

    elif isinstance(geometry, (str, Path)):
        polygon = load_polygon_from_geojson(
            geometry,
            property_key=property_key,
            property_value=property_value
        )
        polygon_proj = project_polygon_to_meters(
            polygon, source_crs=crs_dem, target_crs=crs_utm
        )
        polygon_proj = densify_polygon_boundary(
            polygon_proj,
            target_resolution=resolution_m
        )

    elif isinstance(geometry, (Polygon, MultiPolygon)):
        polygon_proj = project_polygon_to_meters(
            geometry, source_crs=crs_dem, target_crs=crs_utm
        )
        polygon_proj = densify_polygon_boundary(
            polygon_proj,
            target_resolution=resolution_m
        )

    else:
        raise TypeError("Unsupported geometry format.")

    if tile_size_m is not None:
        if (tile_nx is not None or tile_ny is not None):
            raise ValueError(
                "Use either tile_size_m OR (tile_nx and tile_ny), not both."
            )
    if (tile_nx is None) ^ (tile_ny is None):
        raise ValueError("Provide both tile_nx and tile_ny, or neither.")

    if tile_nx is not None and tile_ny is not None:
        if verbose:
            print("[4] Subdividing model into a fixed grid of tiles...")
        return build_model_tiles(
            dem_array=dem_proj,
            transform=transform_proj,
            crs=crs_utm,
            polygon=polygon_proj,
            tile_size=("auto_by_count", (tile_nx, tile_ny)),
            base_depth=base_depth,
            xy_scale=xy_scale,
            z_exag=vertical_exaggeration,
            out_prefix=out_prefix,
            verbose=verbose
        )

    if tile_size_m is not None:
        if verbose:
            print("[4] Subdividing model into tiles...")
        return build_model_tiles(
            dem_array=dem_proj,
            transform=transform_proj,
            crs=crs_utm,
            polygon=polygon_proj,
            tile_size=tile_size_m,
            base_depth=base_depth,
            xy_scale=xy_scale,
            z_exag=vertical_exaggeration,
            out_prefix=out_prefix,
            verbose=verbose
        )

    if verbose:
        print("[4] Building full 3D model (no tiling)...")
    return build_single_mesh(
        dem_array=dem_proj,
        transform=transform_proj,
        crs=crs_utm,
        polygon_tile=polygon_proj,
        base_depth=base_depth,
        xy_scale=xy_scale,
        vertical_exaggeration=vertical_exaggeration,
        out_prefix=out_prefix,
        verbose=verbose
    )


def build_single_mesh(
    dem_array,
    transform,
    crs,
    polygon_tile,
    base_depth,
    xy_scale,
    vertical_exaggeration,
    out_prefix,
    verbose=True
):
    """
    Build and export a watertight 3D terrain mesh from a DEM and a polygon.

    This function performs all the steps needed to convert a projected DEM 
    and a polygon region into a scaled STL file, including sampling, mesh 
    generation, and export.

    Parameters
    ----------
    dem_array : np.ndarray
        DEM elevation data in projected coordinates.
    transform : affine.Affine
        Affine transform corresponding to the DEM.
    crs : rasterio.crs.CRS
        Projected CRS (e.g., UTM).
    polygon_tile : shapely.geometry.Polygon or MultiPolygon
        Region of interest, already projected.
    base_depth : float
        Elevation of the flat base in meters.
    xy_scale : float
        Scaling factor for horizontal axes (X and Y).
    vertical_exaggeration : float
        Multiplier applied to vertical axis (Z).
    out_prefix : str
        Path and prefix for the STL output file.
    verbose : bool, default True
        If True, print status messages during processing.

    Returns
    -------
    None
    """
    polygon_tile = densify_polygon_boundary(
        polygon_tile,
        target_resolution=transform.a
    )

    if verbose:
        print("  [5.1] Sampling DEM within polygon...")

    sampled_points_m = sample_points_from_dem(
        dem_array=dem_array,
        transform=transform,
        polygon=polygon_tile,
        resolution=transform.a
    )

    if verbose:
        print("  [5.2] Elevating polygon boundary...")

    polygon_3d = set_polygon_elevation(
        polygon=polygon_tile,
        dem_array=dem_array,
        transform=transform,
        crs_dem=crs,
        crs_polygon=crs
    )

    if verbose:
        print("  [5.3] Generating topographic surface mesh...")

    surface_mesh = build_surface_mesh_constrained(
        points_3d=sampled_points_m,
        polygon=polygon_3d
    )
    scale_mesh(
        surface_mesh,
        xy_scale=xy_scale,
        z_scale=vertical_exaggeration
    )

    if verbose:
        print("  [5.4] Generating flat base mesh...")

    base_mesh = build_flat_base_mesh(
        polygon=polygon_tile,
        base_z=base_depth
    )
    scale_mesh(
        base_mesh,
        xy_scale=xy_scale,
        z_scale=vertical_exaggeration
    )

    if verbose:
        print("  [5.5] Generating vertical wall mesh...")

    wall_mesh = build_vertical_walls(
        polygon_3d=polygon_3d,
        base_z=base_depth
    )
    scale_mesh(
        wall_mesh,
        xy_scale=xy_scale,
        z_scale=vertical_exaggeration
    )

    if verbose:
        print("  [5.6] Assembling full 3D mesh...")

    full_mesh = build_closed_mesh(
        top_mesh=surface_mesh,
        base_mesh=base_mesh,
        wall_mesh=wall_mesh
    )

    if verbose:
        print(f"  [5.7] Exporting STL: {out_prefix}.stl")

    full_mesh.export(f"{out_prefix}.stl")


def build_model_tiles(
    dem_array,
    transform,
    crs,
    polygon,
    tile_size,
    base_depth,
    xy_scale,
    z_exag,
    out_prefix,
    verbose=True
):
    """
    Subdivide a projected polygon area into tiles and generate one STL
    per tile.

    This function divides the bounding box of the polygon into square tiles 
    of the given size (in meters). Only tiles intersecting the polygon are 
    processed and exported.

    Parameters
    ----------
    dem_array : np.ndarray
        Projected DEM data (in meters).
    transform : affine.Affine
        Affine transform of the DEM.
    crs : rasterio.crs.CRS
        CRS of the DEM and polygon (projected).
    polygon : shapely.Polygon or MultiPolygon
        Projected polygon defining the full model extent.
    tile_size : float or tuple
        - If float: side length (meters) of square tiles.
        - If ("auto_by_count", (nx, ny)): subdivides into a grid
          of nx × ny tiles.
    base_depth : float
        Elevation of the flat base (in meters).
    xy_scale : float
        Scaling factor for horizontal axes.
    z_exag : float
        Vertical exaggeration factor.
    out_prefix : str
        Base path/prefix for tile STL files.
    verbose : bool, default True
        Whether to print progress messages.

    Returns
    -------
    None
    """
    minx, miny, maxx, maxy = polygon.bounds

    if (isinstance(tile_size, tuple)
            and tile_size
            and tile_size[0] == "auto_by_count"):
        nx, ny = tile_size[1]
        if not (
            isinstance(nx, int) and nx > 0
            and isinstance(ny, int) and ny > 0
        ):
            raise ValueError(
                "tile_nx and tile_ny must be positive integers."
            )
        sx = (maxx - minx) / nx
        sy = (maxy - miny) / ny
    else:
        if not (isinstance(tile_size, (int, float)) and tile_size > 0):
            raise ValueError("tile_size must be a positive number.")
        sx = float(tile_size)
        sy = float(tile_size)
        nx = int(np.ceil((maxx - minx) / sx))
        ny = int(np.ceil((maxy - miny) / sy))

    if verbose:
        print(f"  [6.1] Tiling {nx} x {ny} regions...")

    for i in range(nx):
        for j in range(ny):
            x0 = minx + i * sx
            x1 = min(x0 + sx, maxx)
            y0 = miny + j * sy
            y1 = min(y0 + sy, maxy)

            tile_box = Polygon([
                (x0, y0), (x1, y0),
                (x1, y1), (x0, y1),
                (x0, y0)
            ])
            tile_poly = polygon.intersection(tile_box)

            if tile_poly.is_empty or tile_poly.area < 1.0:
                continue

            tile_name = f"{out_prefix}_{i}_{j}"

            if verbose:
                print(f"    [6.2] Tile ({i},{j}) → {tile_name}.stl")

            build_single_mesh(
                dem_array=dem_array,
                transform=transform,
                crs=crs,
                polygon_tile=tile_poly,
                base_depth=base_depth,
                xy_scale=xy_scale,
                vertical_exaggeration=z_exag,
                out_prefix=tile_name,
                verbose=verbose
            )

    if verbose:
        print("[7] All tiles successfully exported.")


def load_dem_from_geotiff(tiff_paths):
    """
    Load and merge one or more GeoTIFF DEM files into a single elevation grid.

    The function reprojects all input rasters to EPSG:4326 (WGS84) if needed,
    merges them into a single array, and returns the merged DEM, the affine
    transform, and the output CRS.

    Parameters
    ----------
    tiff_paths : str, Path, or list of str/Path
        Path(s) to one or more GeoTIFF DEM files.

    Returns
    -------
    dem_array : np.ndarray
        2D elevation array in meters (WGS84).
    transform : affine.Affine
        Affine transform for geographic coordinates (EPSG:4326).
    crs : rasterio.crs.CRS
        Coordinate reference system (WGS84).
    """
    if isinstance(tiff_paths, (str, Path)):
        tiff_paths = [tiff_paths]

    target_crs = CRS.from_epsg(4326)
    src_files = []

    for path in tiff_paths:
        with rasterio.open(path) as src:
            if src.crs != target_crs:
                transform, width, height = calculate_default_transform(
                    src.crs, target_crs, src.width, src.height, *src.bounds
                )
                kwargs = src.meta.copy()
                kwargs.update({
                    "crs": target_crs,
                    "transform": transform,
                    "width": width,
                    "height": height
                })
                data = np.empty(
                    (src.count, height, width),
                    dtype=src.dtypes[0]
                )
                for band in range(src.count):
                    reproject(
                        source=rasterio.band(src, band + 1),
                        destination=data[band],
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=target_crs,
                        resampling=Resampling.bilinear
                    )
                memfile = rasterio.io.MemoryFile()
                dataset = memfile.open(**kwargs)
                dataset.write(data)
                src_files.append(dataset)
            else:
                src_files.append(rasterio.open(path))

    merged, transform = merge(src_files)
    dem_array = merged[0]

    for src in src_files:
        src.close()

    return dem_array, transform, target_crs


def load_polygon_from_geojson(
    path,
    property_key=None,
    property_value=None
):
    """
    Load a polygon or multipolygon from a GeoJSON file.

    Optionally filters features by matching a property key and value. 
    Returns the first valid polygon if no filter is specified.

    Parameters
    ----------
    path : str or Path
        Path to the GeoJSON file.
    property_key : str, optional
        Property name to use for filtering.
    property_value : str or int, optional
        Value to match in the specified property.
    verbose : bool, default False
        If True, print selected feature properties.

    Returns
    -------
    polygon : shapely.geometry.Polygon or MultiPolygon
        Geometry of the selected feature in geographic coordinates.

    Raises
    ------
    FileNotFoundError
        If the GeoJSON file does not exist.
    ValueError
        If no features or geometry are found.
    TypeError
        If the geometry is not a polygon or multipolygon.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"GeoJSON file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        geojson = json.load(f)

    features = geojson.get("features")
    if not features:
        raise ValueError("No features found in GeoJSON.")

    selected = None
    if property_key is not None and property_value is not None:
        for feature in features:
            props = feature.get("properties", {})
            if props.get(property_key) == property_value:
                selected = feature
                break
        if selected is None:
            raise ValueError(
                f"No feature found with {property_key} = {property_value}"
            )
    else:
        selected = features[0]

    geom = selected.get("geometry")
    if geom is None:
        raise ValueError("Selected feature has no geometry.")

    polygon = shape(geom)
    if not isinstance(polygon, (Polygon, MultiPolygon)):
        raise TypeError("Geometry must be a Polygon or MultiPolygon.")

    if not polygon.is_valid:
        reason = explain_validity(polygon)
        print(f"Warning: invalid geometry ({reason})")

    return polygon


def project_dem_to_meters(
    dem_array,
    transform,
    source_crs,
    target_crs,
    resolution=None
):
    """
    Reproject a DEM from geographic to projected (metric) CRS.

    Parameters
    ----------
    dem_array : np.ndarray
        DEM elevation array.
    transform : affine.Affine
        Affine transform of the original DEM.
    source_crs : rasterio.crs.CRS
        CRS of the original DEM (e.g., EPSG:4326).
    target_crs : rasterio.crs.CRS
        Target projected CRS (e.g., EPSG:32633).
    resolution : float or tuple of float, optional
        Desired output resolution in meters. If None, it's inferred.

    Returns
    -------
    dem_projected : np.ndarray
        DEM array in the projected CRS.
    transform_projected : affine.Affine
        Affine transform in the projected CRS.

    Raises
    ------
    ValueError
        If the computed output dimensions are invalid (zero width or height).
    """
    height, width = dem_array.shape

    if isinstance(resolution, (int, float)):
        resolution = (resolution, resolution)

    bounds = array_bounds(height, width, transform)

    transform_proj, width_proj, height_proj = calculate_default_transform(
        src_crs=source_crs,
        dst_crs=target_crs,
        width=width,
        height=height,
        left=bounds[0],
        bottom=bounds[1],
        right=bounds[2],
        top=bounds[3],
        resolution=resolution
    )

    if width_proj == 0 or height_proj == 0:
        raise ValueError("Invalid projected dimensions.")

    dem_projected = np.empty(
        (height_proj, width_proj),
        dtype=dem_array.dtype
    )

    reproject(
        source=dem_array,
        destination=dem_projected,
        src_transform=transform,
        src_crs=source_crs,
        dst_transform=transform_proj,
        dst_crs=target_crs,
        resampling=Resampling.bilinear
    )

    return dem_projected, transform_proj


def project_polygon_to_meters(
    polygon,
    source_crs,
    target_crs
):
    """
    Project a polygon or multipolygon to a target projected CRS.

    Parameters
    ----------
    polygon : shapely.geometry.Polygon or MultiPolygon
        Geometry in source CRS.
    source_crs : rasterio.crs.CRS
        CRS of the input polygon.
    target_crs : rasterio.crs.CRS
        Target projected CRS (e.g., UTM).

    Returns
    -------
    shapely.geometry.Polygon or MultiPolygon
        Projected geometry in meters.

    Raises
    ------
    TypeError
        If the input is not a Polygon or MultiPolygon.
    """
    if not isinstance(polygon, (Polygon, MultiPolygon)):
        raise TypeError("Input must be a Polygon or MultiPolygon.")

    transformer = Transformer.from_crs(
        source_crs, target_crs, always_xy=True
    )

    return shapely_transform(transformer.transform, polygon)


def project_points_to_meters(
    points_3d,
    source_crs,
    target_crs
):
    """
    Project 3D points from geographic to projected (metric) coordinates.

    Parameters
    ----------
    points_3d : np.ndarray
        Array of shape (N, 3) with (lon, lat, elev).
    source_crs : rasterio.crs.CRS
        Original CRS (e.g., EPSG:4326).
    target_crs : rasterio.crs.CRS
        Target projected CRS (e.g., EPSG:32633).

    Returns
    -------
    np.ndarray
        Array of shape (N, 3) with (x, y, z) in meters.
    """
    lon, lat = points_3d[:, 0], points_3d[:, 1]
    z = points_3d[:, 2]

    transformer = Transformer.from_crs(
        source_crs, target_crs, always_xy=True
    )
    x, y = transformer.transform(lon, lat)

    return np.column_stack((x, y, z))


def create_rectangle_polygon(
    lon_min,
    lon_max,
    lat_min,
    lat_max,
    resolution=None
):
    """
    Create a rectangular polygon in geographic coordinates.

    Optionally resamples the boundary at uniform spacing.

    Parameters
    ----------
    lon_min, lon_max : float
        Longitude bounds in degrees.
    lat_min, lat_max : float
        Latitude bounds in degrees.
    resolution : float, optional
        Spacing between points in degrees. If None, only corners are used.

    Returns
    -------
    shapely.geometry.Polygon
        Rectangle polygon in geographic coordinates (EPSG:4326).

    Raises
    ------
    ValueError
        If bounds are invalid.
    """
    if lon_min >= lon_max:
        raise ValueError("lon_min must be smaller than lon_max")
    if lat_min >= lat_max:
        raise ValueError("lat_min must be smaller than lat_max")

    if resolution is None:
        corners = [
            (lon_min, lat_min),
            (lon_max, lat_min),
            (lon_max, lat_max),
            (lon_min, lat_max),
            (lon_min, lat_min)
        ]
        return Polygon(corners)

    # Resample edges
    left = np.column_stack([
        np.full_like(np.arange(lat_min, lat_max, resolution), lon_min),
        np.arange(lat_min, lat_max, resolution)
    ])
    top = np.column_stack([
        np.arange(lon_min, lon_max, resolution),
        np.full_like(np.arange(lon_min, lon_max, resolution), lat_max)
    ])
    right = np.column_stack([
        np.full_like(np.arange(lat_max, lat_min, -resolution), lon_max),
        np.arange(lat_max, lat_min, -resolution)
    ])
    bottom = np.column_stack([
        np.arange(lon_max, lon_min, -resolution),
        np.full_like(np.arange(lon_max, lon_min, -resolution), lat_min)
    ])

    ring = np.vstack([left, top, right, bottom])
    ring = np.vstack([ring, ring[0]])

    return Polygon(LinearRing(ring))


def create_rectangle_polygon_utm(
    lon_min,
    lon_max,
    lat_min,
    lat_max,
    target_crs,
    resolution=None
):
    """
    Create a rectangular polygon in projected UTM coordinates.

    Optionally resamples the boundary at a given spacing.

    Parameters
    ----------
    lon_min, lon_max : float
        Longitude bounds in degrees.
    lat_min, lat_max : float
        Latitude bounds in degrees.
    target_crs : pyproj.CRS
        Target projected CRS (e.g., EPSG:32633).
    resolution : float, optional
        Spacing between points (in meters). If None, only corners are used.

    Returns
    -------
    shapely.geometry.Polygon
        Projected rectangle polygon in meters.

    Raises
    ------
    ValueError
        If geographic bounds are invalid.
    """
    if not (lon_min < lon_max and lat_min < lat_max):
        raise ValueError("Invalid geographic bounds.")

    transformer = Transformer.from_crs(
        "EPSG:4326", target_crs, always_xy=True
    )

    x0, y0 = transformer.transform(lon_min, lat_min)
    x1, y1 = transformer.transform(lon_max, lat_min)
    x2, y2 = transformer.transform(lon_max, lat_max)
    x3, y3 = transformer.transform(lon_min, lat_max)

    if resolution is None:
        coords = [(x0, y0), (x1, y1), (x2, y2), (x3, y3), (x0, y0)]
        return Polygon(coords)

    def resample(p0, p1):
        vec = np.array(p1) - np.array(p0)
        dist = np.linalg.norm(vec)
        n = max(2, int(np.ceil(dist / resolution)))
        return [
            tuple(p0 + t * vec) for t in np.linspace(0, 1, n, endpoint=False)
        ]

    points = []
    points += resample((x0, y0), (x1, y1))
    points += resample((x1, y1), (x2, y2))
    points += resample((x2, y2), (x3, y3))
    points += resample((x3, y3), (x0, y0))
    points.append((x0, y0))

    return Polygon(LinearRing(points))


def densify_polygon_boundary(
    polygon,
    target_resolution=50.0
):
    """
    Densify the exterior boundary of a polygon while preserving
    original vertices.

    New points are added to edges that exceed the specified spacing.

    Parameters
    ----------
    polygon : Polygon or MultiPolygon
        Projected geometry (in meters).
    target_resolution : float
        Maximum allowed spacing between adjacent points (in meters).

    Returns
    -------
    shapely.geometry.Polygon
        Polygon with densified exterior boundary.
    """
    if isinstance(polygon, MultiPolygon):
        polygon = max(polygon.geoms, key=lambda p: p.area)

    coords = np.array(polygon.exterior.coords)
    densified = []

    for i in range(len(coords) - 1):
        p0 = coords[i]
        p1 = coords[i + 1]
        vec = p1 - p0
        dist = np.linalg.norm(vec)

        densified.append(p0)

        if dist > target_resolution:
            n_segments = int(np.floor(dist / target_resolution))
            for j in range(1, n_segments + 1):
                pt = p0 + (j / (n_segments + 1)) * vec
                densified.append(pt)

    densified.append(coords[-1])
    return Polygon(LinearRing(densified))


def set_polygon_elevation(
    polygon,
    dem_array,
    transform,
    crs_dem,
    crs_polygon
):
    """
    Add elevation (Z) to a polygon by sampling the DEM.

    Parameters
    ----------
    polygon : shapely.geometry.Polygon or MultiPolygon
        Projected polygon to be enriched with elevation.
    dem_array : np.ndarray
        DEM elevation grid.
    transform : affine.Affine
        Affine transform of the DEM.
    crs_dem : rasterio.crs.CRS
        CRS of the DEM.
    crs_polygon : rasterio.crs.CRS
        CRS of the polygon (should match DEM CRS).

    Returns
    -------
    Polygon or MultiPolygon
        Geometry with elevation (Z) added.
    """
    transformer = Transformer.from_crs(
        crs_polygon, crs_dem, always_xy=True
    )

    def add_z_to_coords(coords_2d):
        coords_2d = np.asarray(coords_2d)
        lonlat = np.array([
            transformer.transform(x, y) for x, y in coords_2d
        ])
        rows, cols = rowcol(
            transform, lonlat[:, 0], lonlat[:, 1], op=np.floor
        )
        rows = np.asarray(
            np.clip(rows, 0, dem_array.shape[0] - 1), dtype=int
        )
        cols = np.asarray(
            np.clip(cols, 0, dem_array.shape[1] - 1), dtype=int
        )
        zs = dem_array[rows, cols]
        return [(x, y, z) for (x, y), z in zip(coords_2d, zs)]

    def elevate_polygon(poly):
        exterior_z = add_z_to_coords(poly.exterior.coords)
        interiors_z = [
            add_z_to_coords(ring.coords) for ring in poly.interiors
        ]
        return Polygon(
            LinearRing(exterior_z),
            [LinearRing(ring) for ring in interiors_z]
        )

    if isinstance(polygon, Polygon):
        return elevate_polygon(polygon)

    if isinstance(polygon, MultiPolygon):
        return MultiPolygon([elevate_polygon(p) for p in polygon.geoms])

    raise TypeError("Input must be a Polygon or MultiPolygon.")


def sample_points_from_dem(
    dem_array,
    transform,
    polygon,
    resolution=50.0
):
    """
    Sample 3D points from a DEM within a polygon mask.

    Parameters
    ----------
    dem_array : np.ndarray
        DEM elevation array.
    transform : affine.Affine
        Affine transform in projected CRS.
    polygon : shapely.geometry.Polygon or MultiPolygon
        Area to sample (must be in same CRS as DEM).
    resolution : float, default 50.0
        Grid spacing in meters.

    Returns
    -------
    points_3d : np.ndarray
        Array of shape (N, 3) with (x, y, z) points.

    Raises
    ------
    ValueError
        If no points fall inside the polygon.
    """
    if not isinstance(polygon, (Polygon, MultiPolygon)):
        raise TypeError("Input must be a Polygon or MultiPolygon.")

    minx, miny, maxx, maxy = polygon.bounds
    x_vals = np.arange(minx, maxx, resolution)
    y_vals = np.arange(miny, maxy, resolution)
    xx, yy = np.meshgrid(x_vals, y_vals)
    flat_xy = np.column_stack((xx.ravel(), yy.ravel()))

    mask = vectorized.contains(polygon, flat_xy[:, 0], flat_xy[:, 1])
    inside_points = flat_xy[mask]

    if inside_points.size == 0:
        raise ValueError("No DEM points inside polygon.")

    rows, cols = rowcol(
        transform, inside_points[:, 0], inside_points[:, 1], op=np.floor
    )
    rows = np.asarray(np.clip(rows, 0, dem_array.shape[0] - 1), dtype=int)
    cols = np.asarray(np.clip(cols, 0, dem_array.shape[1] - 1), dtype=int)
    zs = dem_array[rows, cols]

    return np.column_stack((inside_points, zs))


def scale_mesh(
    mesh,
    xy_scale,
    z_scale=None
):
    """
    Apply scaling to a mesh in XY and optionally in Z (vertical exaggeration).

    Parameters
    ----------
    mesh : trimesh.Trimesh
        The input mesh to scale.
    xy_scale : float
        Scaling factor for X and Y coordinates.
    z_scale : float, optional
        If given, multiplies Z-axis. If None, same as xy_scale.

    Returns
    -------
    trimesh.Trimesh
        The scaled mesh (modified in-place).
    """
    if not isinstance(xy_scale, (int, float)) or xy_scale <= 0.0:
        raise ValueError("xy_scale must be a positive number.")

    if z_scale is None:
        z_scale = xy_scale
    elif not isinstance(z_scale, (int, float)) or z_scale <= 0.0:
        raise ValueError("z_scale must be a positive number.")
    else:
        z_scale = xy_scale * z_scale

    mesh.vertices[:, 0:2] *= xy_scale
    mesh.vertices[:, 2] *= z_scale

    return mesh


def build_surface_mesh_constrained(
    points_3d,
    polygon
):
    """
    Build a constrained topographic mesh from points and a polygon boundary.

    Parameters
    ----------
    points_3d : np.ndarray
        (x, y, z) coordinates sampled from the DEM.
    polygon : Polygon or MultiPolygon
        Projected geometry with elevation (Z) values.

    Returns
    -------
    trimesh.Trimesh
        Surface mesh bounded by the polygon.
    """
    if isinstance(polygon, MultiPolygon):
        polygon = max(polygon.geoms, key=lambda p: p.area)

    ring_coords = np.array(polygon.exterior.coords)
    if ring_coords.shape[1] != 3:
        raise ValueError("Polygon must have Z coordinates.")

    ring_xy = ring_coords[:, :2]
    ring_z = ring_coords[:, 2]
    num_ring_pts = len(ring_xy)

    if np.allclose(ring_xy[0], ring_xy[-1]):
        ring_xy = ring_xy[:-1]
        ring_z = ring_z[:-1]
        num_ring_pts -= 1

    inner_xy = points_3d[:, :2]
    inner_z = points_3d[:, 2]

    all_vertices_xy = np.vstack([ring_xy, inner_xy])
    all_z = np.concatenate([ring_z, inner_z])

    segments = [[i, (i + 1) % num_ring_pts] for i in range(num_ring_pts)]

    t_input = {
        "vertices": all_vertices_xy,
        "segments": segments
    }

    t_output = triangle.triangulate(t_input, "p")

    if "triangles" not in t_output or "vertices" not in t_output:
        raise RuntimeError("Triangulation failed.")

    vertex_xy = t_output["vertices"]
    faces = t_output["triangles"]

    tree = cKDTree(all_vertices_xy)
    _, indices = tree.query(vertex_xy)
    vertex_z = all_z[indices]

    vertices = np.column_stack([vertex_xy, vertex_z])
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def build_flat_base_mesh(
    polygon,
    base_z=0.0
):
    """
    Build a flat triangulated base from a polygon at a constant elevation.

    Parameters
    ----------
    polygon : Polygon or MultiPolygon
        Geometry in projected coordinates.
    base_z : float, default 0.0
        Elevation (Z) of the flat base.

    Returns
    -------
    trimesh.Trimesh
        Flat triangulated mesh at base_z.
    """
    if isinstance(polygon, MultiPolygon):
        polygon = max(polygon.geoms, key=lambda p: p.area)

    ring_coords = np.array(polygon.exterior.coords)[:, :2]
    if np.allclose(ring_coords[0], ring_coords[-1]):
        ring_coords = ring_coords[:-1]

    _, idx = np.unique(
        ring_coords.round(decimals=6),
        axis=0,
        return_index=True
    )
    ring_2d = ring_coords[np.sort(idx)]

    if len(ring_2d) < 3:
        raise ValueError("Not enough unique points to form polygon.")

    num_pts = len(ring_2d)
    segments = [[i, (i + 1) % num_pts] for i in range(num_pts)]

    t_input = {
        "vertices": ring_2d,
        "segments": segments
    }

    t_output = triangle.triangulate(t_input, "p")
    if "triangles" not in t_output:
        raise RuntimeError("Base triangulation failed.")

    verts_2d = t_output["vertices"]
    faces = t_output["triangles"]
    verts_3d = np.column_stack([verts_2d, np.full(len(verts_2d), base_z)])

    return trimesh.Trimesh(vertices=verts_3d, faces=faces, process=False)


def build_vertical_walls(
    polygon_3d,
    base_z=0.0
):
    """
    Build vertical side walls from polygon boundary down to a base level.

    Parameters
    ----------
    polygon_3d : Polygon or MultiPolygon
        Projected geometry with Z values.
    base_z : float
        Elevation of the base.

    Returns
    -------
    trimesh.Trimesh
        Mesh of vertical walls.
    """
    if isinstance(polygon_3d, MultiPolygon):
        polygon_3d = max(polygon_3d.geoms, key=lambda p: p.area)

    coords = np.array(polygon_3d.exterior.coords)
    if coords.shape[1] != 3:
        raise ValueError("Polygon must have Z coordinates.")

    top_vertices = coords[:-1]
    base_vertices = np.column_stack([
        top_vertices[:, 0],
        top_vertices[:, 1],
        np.full(len(top_vertices), base_z)
    ])

    wall_vertices = []
    wall_faces = []

    n = len(top_vertices)
    for i in range(n):
        v0 = base_vertices[i]
        v1 = base_vertices[(i + 1) % n]
        v2 = top_vertices[i]
        v3 = top_vertices[(i + 1) % n]

        idx = len(wall_vertices)
        wall_vertices.extend([v0, v1, v2, v3])

        wall_faces.append([idx, idx + 1, idx + 2])
        wall_faces.append([idx + 1, idx + 3, idx + 2])

    return trimesh.Trimesh(
        vertices=np.array(wall_vertices),
        faces=np.array(wall_faces),
        process=False
    )


def build_closed_mesh(
    top_mesh,
    base_mesh,
    wall_mesh
):
    """
    Combine top, base, and wall meshes into a single watertight model.

    Parameters
    ----------
    top_mesh : trimesh.Trimesh
        Surface mesh (e.g. from DEM).
    base_mesh : trimesh.Trimesh
        Flat base mesh.
    wall_mesh : trimesh.Trimesh
        Vertical side walls.

    Returns
    -------
    trimesh.Trimesh
        Combined 3D mesh.
    """
    return trimesh.util.concatenate([
        top_mesh,
        base_mesh,
        wall_mesh
    ])


def read_file_list(
    file_path
):
    """
    Read a text file containing one file path per line and return a list.

    Parameters
    ----------
    file_path : str or Path
        Path to the input text file.

    Returns
    -------
    list of str
        List of file paths.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines
