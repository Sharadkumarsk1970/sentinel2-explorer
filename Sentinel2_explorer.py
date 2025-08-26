import os
import io
import zipfile
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.warp import transform_bounds, reproject, Resampling
from shapely.geometry import box, shape
from pystac_client import Client
import planetary_computer
import leafmap.foliumap as leafmap

# ---------------- Streamlit Page Setup ----------------
st.set_page_config(
    page_title="Sentinel-2 Viewer & GeoTIFF Exporter",
    layout="wide",
    page_icon="🛰️"
)

# 🌟 Custom Banner with Smooth Scroll + Clean Professional Style
st.markdown(
    """
    <style>
    html {
        scroll-behavior: smooth;
    }
    .banner {
        background: linear-gradient(135deg, #0F4C75, #3282B8);
        padding: 35px 20px;
        border-radius: 15px;
        text-align: center;
        color: white;
        box-shadow: 0px 6px 15px rgba(0,0,0,0.35);
        animation: fadeIn 1.5s ease-in-out;
    }
    .banner h1 {
        font-size: 42px;
        margin-bottom: 12px;
        font-weight: bold;
    }
    .banner p {
        font-size: 20px;
        margin: 0;
        opacity: 0.95;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .features {
        display: flex;
        justify-content: space-around;
        margin-top: 25px;
        gap: 20px;
        flex-wrap: wrap;
    }
    .card {
        background: white;
        color: #0F4C75;
        padding: 20px;
        border-radius: 12px;
        width: 250px;
        text-align: center;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        cursor: pointer;
        text-decoration: none;
    }
    .card:hover {
        transform: translateY(-8px);
        box-shadow: 0px 8px 20px rgba(0,0,0,0.2);
    }
    .card h3 {
        margin-bottom: 10px;
        font-size: 20px;
    }
    </style>

    <div class="banner">
        <h1>🛰️ Sentinel-2 Explorer</h1>
        <p>A modern tool to visualize <b>Sentinel-2 imagery</b>, compute <b>spectral indices</b>, 
        and export <b>GeoTIFFs</b> for your Area of Interest (AOI).</p>
    </div>

    <div class="features">
        <a href="#upload" class="card">📂 <h3>Upload Shapefile</h3><p>Upload AOI (.zip) with .shp, .dbf, .shx, .prj</p></a>
        <a href="#date-range" class="card">📅 <h3>Date Range</h3><p>Filter scenes by time period</p></a>
        <a href="#rgb" class="card">🖼️ <h3>True Color</h3><p>View natural RGB composites</p></a>
        <a href="#indices" class="card">🌈 <h3>Indices</h3><p>NDVI, NDWI, SAVI, MNDWI & more</p></a>
        <a href="#export" class="card">💾 <h3>Export</h3><p>Download clipped GeoTIFFs</p></a>
    </div>
    <hr>
    """,
    unsafe_allow_html=True
)

# ---------------- Upload Shapefile ----------------
shapefile = st.file_uploader("Upload AOI Shapefile (ZIP with .shp, .dbf, .shx, .prj)", type=["zip"])

# ---------------- Date Range ----------------
c1, c2 = st.columns(2)
with c1:
    start_date = st.date_input("Start Date")
with c2:
    end_date = st.date_input("End Date")

# ---------------- Index Options ----------------
all_indices = ["NDVI", "NDWI", "MNDWI", "NDBI", "SAVI", "EVI", "GCI", "BSI"]
selected_indices = st.multiselect("Choose indices to compute", all_indices, default=["NDVI", "NDWI"])

export_rgb = st.checkbox("Export clipped RGB GeoTIFF (B04,B03,B02)", value=True)
export_indices = st.checkbox("Export selected indices as GeoTIFFs", value=True)

# ---------------- Custom Colormaps ----------------
index_cmaps = {
    "NDVI": "YlGn",
    "NDWI": "Blues",
    "MNDWI": "PuBu",
    "NDBI": "Oranges",
    "SAVI": "YlGnBu",
    "EVI": "BrBG",
    "GCI": "Greens",
    "BSI": "RdYlBu",
}

# ---------------- Helpers ----------------
def geotiff_bytes(array, meta, count=1):
    meta_out = meta.copy()
    meta_out.update({"driver": "GTiff", "count": count, "dtype": rasterio.float32, "nodata": -9999.0})

    if array.dtype != np.float32:
        array = array.astype(np.float32)
    data = np.where(np.isfinite(array), array, -9999.0).astype(np.float32)

    buf = io.BytesIO()
    with rasterio.MemoryFile() as memfile:
        with memfile.open(**meta_out) as dst:
            if count == 1:
                dst.write(data, 1)
            else:
                for i in range(count):
                    dst.write(data[:, :, i], i + 1)
        buf.write(memfile.read())
    buf.seek(0)
    return buf

def raster_overlaps_aoi(src, aoi_gdf_ll):
    minx, miny, maxx, maxy = transform_bounds(src.crs, "EPSG:4326", *src.bounds, densify_pts=21)
    raster_poly_ll = box(minx, miny, maxx, maxy)
    return aoi_gdf_ll.unary_union.intersects(raster_poly_ll)

def load_clip_band(url, aoi_gdf_ll):
    with rasterio.open(url) as src:
        if not raster_overlaps_aoi(src, aoi_gdf_ll):
            raise ValueError("AOI does not overlap raster scene.")
        aoi_in_src = aoi_gdf_ll.to_crs(src.crs)
        geoms = [geom.__geo_interface__ for geom in aoi_in_src.geometry]
        out_img, out_trans = mask(src, geoms, crop=True)
        meta = src.meta.copy()
        meta.update({"height": out_img.shape[1], "width": out_img.shape[2], "transform": out_trans})
        return out_img[0].astype(np.float32), meta

def resample_to_match(src_arr, src_meta, ref_meta):
    dst_arr = np.empty((ref_meta["height"], ref_meta["width"]), dtype=np.float32)
    reproject(
        source=src_arr,
        destination=dst_arr,
        src_transform=src_meta["transform"],
        src_crs=src_meta["crs"],
        dst_transform=ref_meta["transform"],
        dst_crs=ref_meta["crs"],
        resampling=Resampling.bilinear,
    )
    return dst_arr

def compute_index(name, bands):
    eps = 1e-6
    if name == "NDVI":
        return (bands["B08"] - bands["B04"]) / (bands["B08"] + bands["B04"] + eps)
    if name == "NDWI":
        return (bands["B03"] - bands["B08"]) / (bands["B03"] + bands["B08"] + eps)
    if name == "MNDWI":
        return (bands["B03"] - bands["B11"]) / (bands["B03"] + bands["B11"] + eps)
    if name == "NDBI":
        return (bands["B11"] - bands["B08"]) / (bands["B11"] + bands["B08"] + eps)
    if name == "SAVI":
        return 1.5 * (bands["B08"] - bands["B04"]) / (bands["B08"] + bands["B04"] + 0.5)
    if name == "EVI":
        return 2.5 * (bands["B08"] - bands["B04"]) / (bands["B08"] + 6*bands["B04"] - 7.5*bands["B02"] + 1 + eps)
    if name == "GCI":
        return (bands["B08"] / (bands["B03"] + eps)) - 1.0
    if name == "BSI":
        num = (bands["B11"] + bands["B02"]) - (bands["B08"] + bands["B04"])
        den = (bands["B11"] + bands["B02"]) + (bands["B08"] + bands["B04"] + eps)
        return num / den
    raise ValueError(f"Unknown index: {name}")

# ---------- Unified Plotting Helper ----------
def plot_array(arr, title, cmap="viridis", vmin=None, vmax=None, add_cbar=False):
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis("off")
    if add_cbar:
        cbar = plt.colorbar(im, ax=ax, fraction=0.036, pad=0.04)
        cbar.set_label(f"{title} value")
    st.pyplot(fig)

# ---------------- Main Workflow ----------------
if shapefile and start_date and end_date:
    tmpdir = tempfile.mkdtemp()
    zip_path = os.path.join(tmpdir, shapefile.name)
    with open(zip_path, "wb") as f:
        f.write(shapefile.getbuffer())
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(tmpdir)

    shp_files = [os.path.join(tmpdir, f) for f in os.listdir(tmpdir) if f.lower().endswith(".shp")]
    if not shp_files:
        st.error("❌ No .shp file found in the uploaded ZIP!")
        st.stop()

    os.environ["SHAPE_RESTORE_SHX"] = "YES"
    gdf = gpd.read_file(shp_files[0]).to_crs("EPSG:4326")
    bbox = gdf.total_bounds

    st.success("✅ AOI loaded.")
    st.subheader("📍 AOI Preview")
    m = leafmap.Map(center=[(bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2], zoom=10)
    m.add_gdf(gdf, "AOI Boundary")
    m.to_streamlit(height=400)

    # --- STAC Search ---
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox.tolist(),
        datetime=f"{start_date}/{end_date}",
        query={"eo:cloud_cover": {"lt": 30}},
    )
    items = list(search.items())
    if not items:
        st.warning("No scenes found for this AOI & date range.")
        st.stop()

    # Filter overlapping scenes
    valid_items = []
    for it in items:
        try:
            geom = it.geometry or it.to_dict().get("geometry")
            if geom and gpd.GeoDataFrame(geometry=[shape(geom)], crs="EPSG:4326").intersects(gdf.unary_union).any():
                valid_items.append(it)
        except Exception:
            continue
    if not valid_items:
        st.warning("No Sentinel-2 scenes overlap your AOI.")
        st.stop()

    valid_items.sort(key=lambda it: it.properties.get("eo:cloud_cover", 1000))
    scene_labels = [
        f'{it.id} | {it.properties.get("datetime","")} | clouds: {it.properties.get("eo:cloud_cover","?")}%'
        for it in valid_items
    ]
    choice = st.selectbox("Choose scene:", scene_labels, index=0)
    item = valid_items[scene_labels.index(choice)]
    st.write(f"Selected scene: **{item.id}**")

    item_signed = planetary_computer.sign(item)
    need_bands = {b: item_signed.assets[b].href for b in ["B02", "B03", "B04", "B08", "B11"]}

    # ---------------- Band Download + Resample ----------------
    bands, base_meta = {}, None
    process_order = ["B04", "B03", "B02", "B08", "B11"]

    st.info("Downloading, clipping, and aligning bands…")
    for b in process_order:
        url = need_bands[b]
        arr, meta = load_clip_band(url, gdf)
        arr = (arr / 10000.0).astype(np.float32)
        if base_meta is None:
            base_meta = meta
        if (arr.shape[0] != base_meta["height"]) or (arr.shape[1] != base_meta["width"]):
            arr = resample_to_match(arr, meta, base_meta)
        bands[b] = arr

    # --- RGB Visualization ---
    rgb = np.dstack((bands["B04"], bands["B03"], bands["B02"]))
    stretch = np.percentile(rgb, 98)
    rgb_disp = np.clip(rgb / (stretch + 1e-6), 0, 1)
    st.subheader("🖼️ True Color Composite")
    plot_array(rgb_disp, "True Color RGB", cmap=None)

    if export_rgb:
        rgb_buf = geotiff_bytes(rgb.astype(np.float32), base_meta, count=3)
        st.download_button(
            "⬇️ Download RGB GeoTIFF",
            data=rgb_buf,
            file_name=f"{item.id}_RGB.tif",
            mime="image/tiff",
        )

    # --- Indices Visualization ---
    if selected_indices:
        st.subheader("🌈 Indices")
        for idx in selected_indices:
            try:
                idx_arr = compute_index(idx, bands)
            except Exception as e:
                st.warning(f"Could not compute {idx}: {e}")
                continue

            vmin, vmax = -1.0, 1.0
            if idx in ("GCI", "BSI"):
                vmin = float(np.nanpercentile(idx_arr, 2))
                vmax = float(np.nanpercentile(idx_arr, 98))

            cmap = index_cmaps.get(idx, "viridis")
            plot_array(idx_arr, idx, cmap=cmap, vmin=vmin, vmax=vmax, add_cbar=True)

            if export_indices:
                idx_buf = geotiff_bytes(idx_arr, base_meta, count=1)
                st.download_button(
                    f"⬇️ Download {idx} GeoTIFF",
                    data=idx_buf,
                    file_name=f"{item.id}_{idx}.tif",
                    mime="image/tiff",
                )

else:
    st.info("👆 Upload an AOI shapefile (.zip) and choose a start & end date to begin.")

