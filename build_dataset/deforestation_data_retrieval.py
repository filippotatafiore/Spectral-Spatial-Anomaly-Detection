############################################################
######################################### ANOMALOUS DATA
############################################################


import os, random
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.affinity import translate
from shapely.strtree import STRtree
from rasterio.features import rasterize
import ee
import wxee
from dotenv import load_dotenv
load_dotenv()
ee.Authenticate()
ee.Initialize(project=os.getenv("EAI_PROJECT_ID"))

BANDS_TO_USE = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
TARGET_SIZE = 120
OUT = "./new_data_v3"
SHP_PATH = '../datasets/Mapbiomas_alerta/alerts_with_intersections.shp'

WINDOW_DAYS = 60                  # search window after the reference date
MAX_TRIES = 3                     # candidate images downloaded per ROI
MAX_BAD_FRAC = 0.01               # max fraction of cloud / shadow / cirrus / no-data pixels INSIDE the patch
SCL_BAD = [0, 1, 3, 8, 9, 10]     # SCL: no data, defective, cloud shadow, cloud medium/high prob., thin cirrus
SAVE_PRE_EVENT = False            # True: also save the image before the reference alert (bi-temporal data)

import time
N_RETRIES = 4
TRANSIENT = ("429", "Too Many Requests", "timed out", "Timeout", "Connection", "500", "502", "503", "Internal error")

def with_retries(fn):
    """
        Retry transient Earth Engine errors (quota, timeouts, server errors) with exponential backoff.
    """
    for attempt in range(N_RETRIES):
        try:
            return fn()
        except Exception as e:
            if attempt == N_RETRIES - 1 or not any(t in str(e) for t in TRANSIENT):
                raise
            time.sleep(10 * 2 ** attempt)

os.makedirs(f"{OUT}/metadata", exist_ok=True)

# --------------- Alerts (both dates parsed once)
gdf_all = gpd.read_file(SHP_PATH, engine="pyogrio")                             # ALL alerts: used for masks and normal checks
gdf_all["after"] = pd.to_datetime(gdf_all["AFTERIMGDT"], errors="coerce")
gdf_all["before"] = pd.to_datetime(gdf_all["BEFORIMGDT"], errors="coerce")
gdf_full = gdf_all.dropna(subset=["after", "before"]).reset_index(drop=True)    # only dated alerts are used as patch centers
native_crs = gdf_all.crs
print(f"{len(gdf_all) - len(gdf_full)} alerts without dates (-> 'ignore' in masks, excluded from normals)")


def non_overlapping(geoms):
    """
        Greedy selection of non-overlapping boxes (same result as the pairwise loop, but with a spatial index).
    """
    geoms = list(geoms)
    tree = STRtree(geoms)
    kept, blocked = [], set()
    for i, g in enumerate(geoms):
        if i in blocked:
            continue
        kept.append(i)
        blocked.update(tree.query(g, predicate="intersects").tolist())
    return kept


def fetch_patch(roi, start, end, ascending=True):
    """
        First (ascending=True: earliest, else latest) Sentinel-2 L2A image in [start, end) that fully covers the ROI
        and is clean INSIDE the patch (SCL clouds/shadows/cirrus + no-data <= MAX_BAD_FRAC).
        Returns (image (12, H, W) float32 reflectance, acquisition date, xarray dataset) or (None, None, None).
    """
    col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
           .filterDate(start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))
           .filterBounds(roi)
           .filter(ee.Filter.contains(leftField='.geo', rightValue=roi))   # full footprint coverage (no partial swath)
           .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50))             # coarse prefilter, the real check is per patch
           .sort('system:time_start', ascending))
    n = min(with_retries(lambda: col.size().getInfo()), MAX_TRIES)
    candidates = col.toList(max(n, 1))
    for k in range(n):
        img = ee.Image(candidates.get(k))
        sel = img.select(BANDS_TO_USE + ['SCL']).toUint16()     # SCL is uint8: one dtype for the download
        ds = with_retries(lambda: sel.wx.to_xarray(region=roi, scale=10, crs="EPSG:3857", progress=False))
        data = np.stack([np.squeeze(ds[b].values) for b in BANDS_TO_USE], axis=0).astype(np.float32) / 10000.0
        scl = np.nan_to_num(np.squeeze(ds['SCL'].values), nan=0).astype(int)
        if data.shape[1] < TARGET_SIZE or data.shape[2] < TARGET_SIZE:
            continue
        crop = data[:, :TARGET_SIZE, :TARGET_SIZE]
        nodata = ~np.isfinite(crop).all(axis=0) | (np.nan_to_num(crop).sum(axis=0) == 0)
        bad = nodata | np.isin(scl[:TARGET_SIZE, :TARGET_SIZE], SCL_BAD)
        if bad.mean() > MAX_BAD_FRAC:
            continue
        if np.nanmax(crop) > 2.0:     # fill values / saturation: never a valid reflectance
            continue
        date = pd.Timestamp(with_retries(lambda: img.date().format('YYYY-MM-dd').getInfo()))
        return np.nan_to_num(data), date, ds.rio.write_crs("EPSG:3857")
    return None, None, None



RECENT_DAYS = 365        # an alert is "anomalous" in an image if confirmed (AFTERIMGDT) within the previous year
MAX_IGNORE_FRAC = 0.05   # skip patches where ambiguous alerts cover more than this fraction
MIN_ANOM_PX = 20
N_SAMPLE, N_TARGET = 8000, 2000

dirs = {k: f"{OUT}/anomaly/deforestation/{k}/" for k in ("imgs", "masks", "ignore", "pre")}
for d in dirs.values():
    os.makedirs(d, exist_ok=True)


def build_masks(polys, img_date, transform, shape):
    """
        Labels relative to the image ACQUISITION date:
            1      : alert confirmed within the last RECENT_DAYS (AFTERIMGDT in [date - RECENT_DAYS, date]) -> visible, recent clearing
            0      : alert whose BEFORE image is later than the date                                        -> still forest in this image
            ignore : alert in progress (BEFORIMGDT <= date < AFTERIMGDT) or older than RECENT_DAYS          -> ambiguous
    """
    old_limit = img_date - pd.Timedelta(days=RECENT_DAYS)
    undated = polys["after"].isna() | polys["before"].isna()
    recent = (polys["after"] <= img_date) & (polys["after"] >= old_limit) & ~undated
    ambiguous = ((polys["before"] <= img_date) & (polys["after"] > img_date)) | (polys["after"] < old_limit) | undated

    def burn(sel):
        shapes = [(g, 1) for g in polys.loc[sel, "geometry"]]
        return rasterize(shapes, out_shape=shape, transform=transform, fill=0, dtype="uint8") if shapes else np.zeros(shape, np.uint8)

    mask = burn(recent)
    ignore = burn(ambiguous) * (1 - mask)    # a recent (re-)clearing overlapping an old alert stays anomalous
    return mask, ignore


sampled = gdf_full.sample(n=N_SAMPLE, random_state=42).reset_index(drop=True)
boxes_metric = sampled.to_crs(epsg=3857)
boxes_metric["geometry"] = boxes_metric.geometry.centroid.buffer(600).envelope
boxes_metric = boxes_metric.iloc[non_overlapping(boxes_metric.geometry)].reset_index(drop=True)
boxes_wgs84 = boxes_metric.to_crs(epsg=4326)
boxes_native = boxes_metric.to_crs(native_crs)
print(f"{len(boxes_metric)} non-overlapping candidate scenes")

meta = []
for i, row in boxes_metric.iterrows():
    if len(meta) >= N_TARGET:
        break
    try:
        roi = ee.Geometry.Rectangle(list(boxes_wgs84.geometry[i].bounds))
        img, img_date, ds = fetch_patch(roi, row["after"], row["after"] + pd.Timedelta(days=WINDOW_DAYS))
        if img is None:
            print(f"Skipping ROI {i}: no clean, fully covering image")
            continue

        # All alerts intersecting the patch, labeled relative to the acquisition date
        box_native = boxes_native.geometry[i]
        cand = gdf_all.iloc[list(gdf_all.sindex.intersection(box_native.bounds))]
        cand = cand.set_geometry(cand.geometry.make_valid())       # repair invalid rings BEFORE testing intersections
        polys = cand[cand.intersects(box_native)].explode(index_parts=False).reset_index(drop=True)
        polys = polys[polys.geom_type.isin(["Polygon", "MultiPolygon"])]   # make_valid can leave stray lines / points
        polys = polys.to_crs(epsg=3857)

        mask, ignore = build_masks(polys, img_date, ds.rio.transform(), img.shape[1:])
        img, mask, ignore = img[:, :TARGET_SIZE, :TARGET_SIZE], mask[:TARGET_SIZE, :TARGET_SIZE], ignore[:TARGET_SIZE, :TARGET_SIZE]

        if mask.sum() < MIN_ANOM_PX:
            print(f"Skipping ROI {i}: recent deforestation too small in this patch")
            continue
        if ignore.mean() > MAX_IGNORE_FRAC:
            print(f"Skipping ROI {i}: ambiguous (old / in-progress) alerts cover {ignore.mean():.0%} of the patch")
            continue

        name = f"deforest_img_{i}_anomaly"
        np.save(os.path.join(dirs["imgs"], f"{name}.npy"), img)
        np.save(os.path.join(dirs["masks"], f"{name}_mask.npy"), mask)
        np.save(os.path.join(dirs["ignore"], f"{name}_ignore.npy"), ignore)

        if SAVE_PRE_EVENT:
            pre, _, _ = fetch_patch(roi, row["before"] - pd.Timedelta(days=WINDOW_DAYS), row["before"] + pd.Timedelta(days=1), ascending=False)
            if pre is not None:
                np.save(os.path.join(dirs["pre"], f"{name}_pre.npy"), pre[:, :TARGET_SIZE, :TARGET_SIZE])

        meta.append({"name": name, "roi": i, "img_date": str(img_date.date()), "alert_after": str(row["after"].date()),
                     "anom_frac": float(mask.mean()), "ignore_frac": float(ignore.mean())})
        print(f"Saved {len(meta)}/{N_TARGET}: ROI {i}, image {img_date.date()}, anomaly {mask.mean():.0%}, ignore {ignore.mean():.1%}")
    except Exception as e:
        print(f"Skipping ROI {i}, error: {e}")

meta = pd.DataFrame(meta)
if meta.empty:
    raise RuntimeError("No anomalous patch saved: check GEE access / filters before extracting normals")
meta.to_csv(f"{OUT}/metadata/deforestation_anomaly.csv", index=False)
gpd.GeoDataFrame(meta, geometry=boxes_metric.geometry.iloc[meta["roi"]].values, crs="EPSG:3857").to_file(f"{OUT}/metadata/deforestation_anomaly_rois.gpkg")


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

import os
import glob
import numpy as np
import matplotlib.pyplot as plt

def validate_deforestation(file_path, mask_dir):
    # Extract the base filename to find the corresponding mask
    basename = os.path.basename(file_path)
    # basename looks like: deforest_img_42_anomaly.npy
    mask_name = basename.replace(".npy", "_mask.npy")
    mask_path = os.path.join(mask_dir, mask_name)
    
    if not os.path.exists(mask_path):
        print(f"Mask not found for {basename}")
        return

    # Load arrays
    img_array = np.load(file_path)   # Shape: (12, 120, 120)
    mask_array = np.load(mask_path)  # Shape: (120, 120)
    
    print(f"\nLoaded: {basename}")
    print(f"Image shape: {img_array.shape} | Min: {img_array.min():.4f} | Max: {img_array.max():.4f}")
    print(f"Mask shape: {mask_array.shape} | Unique values: {np.unique(mask_array)}")

    # Extract True Color (RGB: B4, B3, B2 -> Indices 3, 2, 1)
    rgb = img_array[[3, 2, 1], :, :].transpose(1, 2, 0)
    # Brighten the image slightly for better viewing
    rgb_bright = np.clip(rgb * 2.5, 0, 1)

    # Extract False Color Infrared (NIR, Red, Green: B8, B4, B3 -> Indices 7, 3, 2)
    # Healthy forest appears bright red; deforested areas appear cyan/grey/brown
    fcc = img_array[[7, 3, 2], :, :].transpose(1, 2, 0)
    fcc_bright = np.clip(fcc * 2.5, 0, 1)

    # Setup the plot
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # 1. True Color
    axes[0].imshow(rgb_bright)
    axes[0].set_title("True Color (RGB)")
    axes[0].axis("off")

    # 2. False Color
    axes[1].imshow(fcc_bright)
    axes[1].set_title("False Color (NIR-R-G)\nForest=Red, Cleared=Cyan/Brown")
    axes[1].axis("off")

    # 3. Mask Only
    axes[2].imshow(mask_array, cmap="gray")
    axes[2].set_title("Binary Mask (Ground Truth)")
    axes[2].axis("off")

    # 4. Mask Overlay on True Color
    axes[3].imshow(rgb_bright)
    # Overlay the mask using a red/yellow colormap and some transparency (alpha)
    mask_overlay = np.ma.masked_where(mask_array == 0, mask_array)
    axes[3].imshow(mask_overlay, cmap="autumn", alpha=0.5)
    
    axes[3].set_title("RGB with Mask Overlay")
    axes[3].axis("off")

    plt.tight_layout()
    
    # Create a unique filename for the plot based on the original numpy file
    plot_path = os.path.join(OUT, "metadata", "plots", "anom_" + basename.replace(".npy", "_plot.png"))
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot as: {plot_path}")
    
    # Close the figure to free up memory
    plt.close(fig)


# Run validation
img_dir = f"{OUT}/anomaly/deforestation/imgs/"
mask_dir = f"{OUT}/anomaly/deforestation/masks/"

# Find all generated anomaly patches
test_files = glob.glob(os.path.join(img_dir, "*.npy"))

if not test_files:
    print(f"No .npy files found in {img_dir}. (Is the extraction script still running?)")
else:
    print(f"Found {len(test_files)} test patches. Visualizing a random sample of 5...")
    
    # Shuffle the list so you can see a random variety of shapes
    np.random.shuffle(test_files)
    
    # Visualize up to 5 images
    for file_path in test_files[:5]:
        validate_deforestation(file_path, mask_dir)



############################################################
######################################### NORMAL DATA
############################################################


N_SAMPLE_N, N_TARGET_N = 10000, 2000
save_dir_normal = f"{OUT}/normal/deforestation/"
save_dir_normal_pre = f"{OUT}/normal_pre/deforestation/"      # outside normal/: never loaded as training normals
os.makedirs(save_dir_normal, exist_ok=True)
os.makedirs(save_dir_normal_pre, exist_ok=True)

anom_rois = gpd.read_file(f"{OUT}/metadata/deforestation_anomaly_rois.gpkg").to_crs(epsg=3857)
anom_tree = STRtree(list(anom_rois.geometry))

rng = random.Random(101)
def apply_random_shift(pt):
    angle, distance = rng.uniform(0, 2 * np.pi), rng.uniform(3000, 6000)     # 3-6 km in a random direction
    return translate(pt, xoff=distance * np.cos(angle), yoff=distance * np.sin(angle))

src = gdf_full.sample(n=N_SAMPLE_N, random_state=101).reset_index(drop=True)
boxes_metric = src.to_crs(epsg=3857)
boxes_metric = boxes_metric.set_geometry(gpd.GeoSeries([apply_random_shift(p).buffer(600).envelope for p in boxes_metric.geometry.centroid],
                                                       index=boxes_metric.index, crs="EPSG:3857"))
boxes_metric = boxes_metric.iloc[non_overlapping(boxes_metric.geometry)].reset_index(drop=True)
boxes_wgs84 = boxes_metric.to_crs(epsg=4326)
boxes_native = boxes_metric.to_crs(native_crs)

meta_n = []
for i, row in boxes_metric.iterrows():
    if len(meta_n) >= N_TARGET_N:
        break
    try:
        # Truly normal: no alert polygon of ANY date inside, and no overlap with an anomaly patch
        box_native = boxes_native.geometry[i]
        cand = gdf_all.iloc[list(gdf_all.sindex.intersection(box_native.bounds))]
        if cand.geometry.make_valid().intersects(box_native).any():
            print(f"Skipping ROI {i}: spatial shift landed on an alert")
            continue
        if len(anom_tree.query(boxes_metric.geometry[i], predicate="intersects")):
            print(f"Skipping ROI {i}: overlaps an anomaly patch")
            continue

        # Same date distribution and filters as the anomalies (post-alert dates of the source alert) -> no seasonal shortcut
        roi = ee.Geometry.Rectangle(list(boxes_wgs84.geometry[i].bounds))
        img, img_date, _ = fetch_patch(roi, row["after"], row["after"] + pd.Timedelta(days=WINDOW_DAYS))
        if img is None:
            print(f"Skipping ROI {i}: no clean, fully covering image")
            continue

        name = f"deforest_biome_normal_{len(meta_n)}"
        np.save(os.path.join(save_dir_normal, f"{name}.npy"), img[:, :TARGET_SIZE, :TARGET_SIZE])

        if SAVE_PRE_EVENT:
            pre, _, _ = fetch_patch(roi, row["before"] - pd.Timedelta(days=WINDOW_DAYS), row["before"] + pd.Timedelta(days=1), ascending=False)
            if pre is not None:
                np.save(os.path.join(save_dir_normal_pre, f"{name}_pre.npy"), pre[:, :TARGET_SIZE, :TARGET_SIZE])

        meta_n.append({"name": name, "roi": i, "img_date": str(img_date.date())})
        print(f"Saved normal {len(meta_n)}/{N_TARGET_N}: ROI {i}, image {img_date.date()}")
    except Exception as e:
        print(f"Skipping ROI {i}, error: {e}")

pd.DataFrame(meta_n).to_csv(f"{OUT}/metadata/deforestation_normal.csv", index=False)




import os
import glob
import numpy as np
import matplotlib.pyplot as plt

def validate_normal_deforestation(file_path):
    # Load the numpy array
    img_array = np.load(file_path)   # Expected shape: (12, 120, 120)
    
    file_name = os.path.basename(file_path)
    print(f"\nLoaded: {file_name}")
    print(f"Shape: {img_array.shape} | Min: {img_array.min():.4f} | Max: {img_array.max():.4f}")

    # Extract True Color (RGB: B4, B3, B2 -> Indices 3, 2, 1)
    rgb = img_array[[3, 2, 1], :, :].transpose(1, 2, 0)
    # Brighten the image slightly for better viewing
    rgb_bright = np.clip(rgb * 2.5, 0, 1)

    # Extract False Color Infrared (NIR, Red, Green: B8, B4, B3 -> Indices 7, 3, 2)
    # Healthy forest appears bright red
    fcc = img_array[[7, 3, 2], :, :].transpose(1, 2, 0)
    fcc_bright = np.clip(fcc * 2.5, 0, 1)

    # Setup the plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # 1. True Color
    axes[0].imshow(rgb_bright)
    axes[0].set_title(f"True Color (RGB)\n{file_name}")
    axes[0].axis("off")

    # 2. False Color
    axes[1].imshow(fcc_bright)
    axes[1].set_title("False Color (NIR-R-G)\nNo recent alert (may include pasture / old clearings)")
    axes[1].axis("off")

    plt.tight_layout()
    
    # Create a unique filename based on the input file
    plot_filename = os.path.join(OUT, "metadata", "plots", "norm_" + file_name.replace(".npy", "_plot.png"))
    os.makedirs(os.path.dirname(plot_filename), exist_ok=True)
    
    # Save the plot to the current directory
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    print(f"Saved plot as: {plot_filename}")
    
    # Close the figure to prevent GUI popups and memory leaks
    plt.close(fig)

# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------
train_deforest_dir = f"{OUT}/normal/deforestation/"

# Find all generated normal patches
normal_files = glob.glob(os.path.join(train_deforest_dir, "*.npy"))

if not normal_files:
    print(f"No .npy files found in {train_deforest_dir}")
else:
    print(f"Found {len(normal_files)} normal biome patches. Visualizing a random sample of 5...")
    
    # Shuffle the list to see a varied sample
    np.random.shuffle(normal_files)
    
    # Visualize up to 5 images
    for file_path in normal_files[:5]:
        validate_normal_deforestation(file_path)