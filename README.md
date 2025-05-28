![Relievo Logo](logo/relievo_logo.png)

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python >=3.8](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Status: Beta](https://img.shields.io/badge/status-beta-yellow)]()
[![GitHub Repo](https://img.shields.io/badge/github-relievo-lightgrey?logo=github)](https://github.com/yourname/relievo)

# Relievo - 3D Terrain Model Generator

**Relievo** is a Python tool for generating watertight 3D terrain models 
from DEM (Digital Elevation Model) tiles and user-defined geographic regions. 
It supports topographic mesh generation, vertical exaggeration, 
side walls and base extrusion, and optional tiling for large-scale output.

STL files can be exported directly for 3D printing or scientific visualization.

---

## Installation

### Using pip (standard)

```bash
pip install .
```

### Editable install (for development)

```bash
pip install -e .
```

> After installation, the command-line tool `relievo` will be available in your system path.
> If you receive a warning about the script location, consider adding it to your `$PATH`.

---

## How to Use

You can use **Relievo** either as a **Python module** or from the **command line**.

---

### ✅ 1. As a Python Module

Import the function and call it directly:

```python
from relievo import relievo
```

#### Example A – Bounding box

```python
relievo(
    tif_paths=["data/dem1.tif", "data/dem2.tif"],
    geometry=((12.3, 45.6), (13.9, 46.7)),
    resolution_m=100,
    base_depth=-1000.0,
    xy_scale=1/700,
    vertical_exaggeration=2.0,
    out_prefix="output/relief_box"
)
```

#### Example B – GeoJSON region with property filter

```python
relievo(
    tif_paths="data/",
    geometry="data/regions.geojson",
    property_key="reg_code",
    property_value="6",
    resolution_m=100,
    base_depth=-1000.0,
    xy_scale=1/700,
    vertical_exaggeration=2.0,
    out_prefix="output/relief_region"
)
```

#### Example C – Tiled STL output

```python
relievo(
    tif_paths="data/",
    geometry="data/polygon.geojson",
    resolution_m=100,
    base_depth=-1000.0,
    xy_scale=1/700,
    vertical_exaggeration=2.0,
    tile_size_m=50000,
    out_prefix="output/relief_tiles"
)
```

---

### ✅ 2. As a Command-Line Tool

After installation, you can run the tool via terminal:

```bash
relievo --dem data/dem1.tif data/dem2.tif \
        --geometry 12.3,45.6,13.9,46.7 \
        --resolution 100 \
        --base-depth -1000 \
        --xy-scale 0.00142857 \
        --z-exag 2.0 \
        --tile-size 65000 \
        --utm-crs EPSG:32633 \
        --out-prefix output/relief_box \
        --verbose
```

#### Example using GeoJSON with attribute filtering:

```bash
relievo --dem data/*.tif \
        --geometry data/regions.geojson \
        --property-key reg_code \
        --property-value 6 \
        --out-prefix output/relief_region \
        --verbose
```

#### Example using a DEM file list:

Create a text file `dem_list.txt`:

```
data/dem1.tif
data/dem2.tif
data/dem3.tif
```

Then run:

```bash
relievo --dem-list dem_list.txt \
        --geometry data/polygon.geojson \
        --out-prefix output/relief_from_list \
        --verbose
```

---

## License

This project is distributed under the GNU Affero General Public License v3.0.

---

## Author

**Valerio Poggi**  
Istituto Nazionale di Oceanografia e di Geofisica Sperimentale (OGS)  
vpoggi@ogs.it
