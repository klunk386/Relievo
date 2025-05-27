from relievo import relievo

relievo(
    tif_paths="data/",
    geometry="data/georef-italy-regione.geojson",
    property_key="reg_code",
    property_value="6",  # e.g. Friuli Venezia Giulia
    resolution_m=100,
    base_depth=-1000.0,
    xy_scale=1 / 750,
    vertical_exaggeration=2.0,
    out_prefix="output/terrain_region",
    verbose=True
)
