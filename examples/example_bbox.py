from relievo import relievo

relievo(
    tif_paths=[
        "data/N045E012/ALPSMLC30_N045E012_DSM.tif",
        "data/N045E013/ALPSMLC30_N045E013_DSM.tif"
    ],
    geometry=[12.3, 13.9, 45.6, 46.7],  # lon_min, lon_max, lat_min, lat_max
    resolution_m=100,
    base_depth=-1000.0,
    xy_scale=1 / 750,
    vertical_exaggeration=1.5,
    out_prefix="output/terrain_bbox",
    verbose=True
)
