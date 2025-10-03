"""
Command-line interface for RELIEVO - 3D Terrain Model Generator

Author: Valerio Poggi
License: AGPL-3.0
"""

import argparse
import sys
from pathlib import Path

from relievo import relievo


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a 3D terrain STL model from DEM and region geometry."
    )

    parser.add_argument(
        "--dem", nargs="*", help="Path(s) to GeoTIFF DEM files", default=[]
    )
    parser.add_argument(
        "--dem-list", type=str,
        help="Text file with a list of DEM GeoTIFF paths (one per line)"
    )

    parser.add_argument(
        "--geometry", required=True,
        help=(
            "Geometry input: either "
            "'lon_min,lat_min,lon_max,lat_max' (BBOX) or a GeoJSON file path"
        )
    )
    parser.add_argument("--property-key", help="GeoJSON property key for selection")
    parser.add_argument("--property-value", help="GeoJSON property value to match")

    parser.add_argument("--resolution", type=float, default=100.0,
                        help="Sampling resolution in meters (default: 100)")
    parser.add_argument("--base-depth", type=float, default=-1000.0,
                        help="Base depth in meters (default: -1000.0)")
    parser.add_argument("--xy-scale", type=float, default=1 / 750,
                        help="Horizontal scale factor (default: 1/750)")
    parser.add_argument("--z-exag", type=float, default=1.5,
                        help="Vertical exaggeration factor (default: 1.5)")
    parser.add_argument("--tile-size", type=float,
                        help="Tile size in meters for STL tiling")
    parser.add_argument("--utm-crs", type=str, default="EPSG:32633",
                        help="Projected CRS to use (default: EPSG:32633)")

    parser.add_argument("--out-prefix", required=True,
                        help="Output STL prefix (path and filename prefix)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    return parser.parse_args()


def main():
    args = parse_args()

    dem_paths = list(args.dem)
    if args.dem_list:
        try:
            with open(args.dem_list, "r") as f:
                lines = [line.strip() for line in f if line.strip()]
                dem_paths.extend(lines)
        except FileNotFoundError:
            print(f"ERROR: DEM list file not found: {args.dem_list}", file=sys.stderr)
            sys.exit(1)

    if not dem_paths:
        print("ERROR: No DEM files provided (--dem or --dem-list required).", file=sys.stderr)
        sys.exit(1)

    geometry = args.geometry
    if "," in geometry:
        try:
            lon_min, lat_min, lon_max, lat_max = map(float, geometry.split(","))
            geometry = [(lon_min, lat_min), (lon_max, lat_max)]
        except ValueError:
            print("ERROR: Invalid BBOX format for --geometry.", file=sys.stderr)
            sys.exit(1)
    else:
        geometry = Path(geometry)

    relievo(
        tif_paths=dem_paths,
        geometry=geometry,
        resolution_m=args.resolution,
        base_depth=args.base_depth,
        xy_scale=args.xy_scale,
        vertical_exaggeration=args.z_exag,
        tile_size_m=args.tile_size,
        utm_crs=args.utm_crs,
        out_prefix=args.out_prefix,
        property_key=args.property_key,
        property_value=args.property_value,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
