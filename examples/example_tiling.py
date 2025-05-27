from relievo import relievo

relievo(
    tif_paths=[
        "data/N045E012/ALPSMLC30_N045E012_DSM.tif",
        "data/N045E013/ALPSMLC30_N045E013_DSM.tif"
    ],
    geometry="data/test_polygon.geojson",
    resolution_m=100,
    base_depth=-1000.0,
    tile_size_m=50000,
    xy_scale=1 / 700,
    vertical_exaggeration=2.0,
    out_prefix="output/terrain_tiles",
    verbose=True
)
