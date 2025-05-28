from relievo import relievo

relievo(
    tif_paths=[
        "data/N045E012/ALPSMLC30_N045E012_DSM.tif",
        "data/N045E013/ALPSMLC30_N045E013_DSM.tif",
        "data/N046E012/ALPSMLC30_N046E012_DSM.tif",
        "data/N046E013/ALPSMLC30_N046E013_DSM.tif"
    ],
    geometry=((12.3, 45.6), (13.9, 46.7)),
    resolution_m=100,
    base_depth=-1000.0,
    xy_scale=1/700,
    vertical_exaggeration=2.0,
    utm_crs="EPSG:32633",
    out_prefix="output/relief_box",
    verbose=True
)
