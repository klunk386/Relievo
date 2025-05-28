relievo \
  --dem data/N045E012/ALPSMLC30_N045E012_DSM.tif \
        data/N045E013/ALPSMLC30_N045E013_DSM.tif \
        data/N046E012/ALPSMLC30_N046E012_DSM.tif \
        data/N046E013/ALPSMLC30_N046E013_DSM.tif \
  --geometry data/georef-italy-regione.geojson \
  --property-key reg_code \
  --property-value 6 \
  --resolution 100 \
  --base-depth -1000 \
  --tile-size 65000 \
  --xy-scale 0.00142857 \
  --z-exag 2.0 \
  --utm-crs EPSG:32633 \
  --out-prefix output/relief_polygon_1 \
  --verbose