Sentinel-2 Explorer

A Streamlit-based web application for exploring and analyzing Sentinel-2 satellite imagery.
The tool allows users to upload an Area of Interest (AOI) shapefile, filter imagery by date range, compute key spectral indices (NDVI, NDWI, SAVI, etc.), and export clipped GeoTIFFs in an intuitive, interactive interface.

Features

Upload AOI shapefile (.zip) with required components (.shp, .dbf, .shx, .prj)

Select a date range for imagery acquisition

Cloud filtering to exclude low-quality scenes

True-color visualization using Sentinel-2 RGB bands (B04, B03, B02)

Computation of spectral indices: NDVI, NDWI, MNDWI, SAVI, EVI, GCI, BSI, NDBI

Export processed results as GeoTIFFs

Interactive mapping with AOI preview and visualization support

Installation and Usage

Clone the repository and run the application locally:

git clone https://github.com/<your-username>/sentinel2-explorer.git
cd sentinel2-explorer
pip install -r requirements.txt
streamlit run app.py

Requirements

Key Python dependencies include:

Streamlit
 – interactive UI framework

GeoPandas
 – shapefile and vector data handling

Rasterio
 – raster data processing

Leafmap
 – interactive mapping integration

Pystac-client
 – STAC API queries

Planetary Computer
 – Sentinel-2 data access

Matplotlib
 – data visualization

The full list of dependencies is provided in requirements.txt.

Workflow

Upload an AOI shapefile (.zip containing .shp, .dbf, .shx, .prj)

Select the desired date range for imagery

The application queries Sentinel-2 L2A imagery via the Microsoft Planetary Computer STAC API

Relevant bands are clipped to the AOI, resampled, and aligned

Spectral indices are computed (e.g., NDVI, NDWI, etc.)

Results are visualized within the browser, with an option to export as GeoTIFFs

Screenshots

(Optional: add screenshots or GIFs of the application interface here)

License

This project is released under the MIT License. Users are free to use, modify, and distribute with proper attribution.

Acknowledgements

Sentinel-2 imagery provided by Copernicus Open Access Hub
 through Microsoft Planetary Computer

Developed using open-source libraries including Streamlit, Rasterio, GeoPandas, Leafmap, and NumPy

Recommended Repository Structure
sentinel2-explorer/
│── app.py
│── requirements.txt
│── README.md
│── LICENSE
│── .gitignore
