############################################################
######################### BURN SCARS (HLS Burn Scars dataset)
############################################################

import os, glob, time
import numpy as np
import pandas as pd
import geopandas as gpd
import rioxarray
from rasterio.enums import Resampling
from scipy.ndimage import uniform_filter
from shapely.geometry import box
import matplotlib.pyplot as plt
import ee
import wxee
from dotenv import load_dotenv
load_dotenv()
ee.Authenticate()
ee.Initialize(project=os.getenv("EAI_PROJECT_ID"))

BANDS_TO_USE = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
TARGET_SIZE = 120
OUT = "./new_data_v3"
HLS_DIR = "../datasets/hls_burn_scars"

SEED = 0
P_PARTIAL = 0.5                   # share of scenes whose anomaly window is placed on the scar boundary
PARTIAL_RANGE = (0.2, 0.8)        # burned fraction of a "partial" window
FULL_MIN = 0.8                    # burned fraction of a "full" window (as before: mostly burned)
SMALL_MIN = 0.02                  # fallback for scars smaller than the window
HALF_M = 600                      # half side of the 1.2 km window (m)
MASK_RES = 30                     # HLS mask resolution (m)
HALF_PX = int(np.ceil(HALF_M / MASK_RES))   # 20 mask pixels
EDGE_PX = HALF_PX + 2             # window (+ download margin) fully inside the 512x512 annotated subset
NORMAL_MARGIN_PX = 10             # normal windows: no burned pixel within 300 m around the window
MAX_POS_TRIES = 30                # window positions tested per scene (overlap check)
MAX_DOWNLOADS = 2                 # downloads attempted per scene and per window type
MAX_BAD_FRAC = 0.05               # max cloud / no-data fraction in the window (smoke is common after fires)
SCL_BAD = [0, 1, 8, 9, 10]        # no data, defective, cloud medium/high prob., cirrus
                                  # (cloud shadow = 3 NOT used: Sen2Cor often labels dark burn scars as shadow)
MIN_ANOM_PX = 20

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


dirs = {"imgs": f"{OUT}/anomaly/burn_scars/imgs/", "masks": f"{OUT}/anomaly/burn_scars/masks/",
        "normal": f"{OUT}/normal/burn_scars/", "meta": f"{OUT}/metadata/", "plots": f"{OUT}/metadata/plots/"}
for d in dirs.values():
    os.makedirs(d, exist_ok=True)
rng = np.random.default_rng(SEED)


def scene_info(path):
    """
        subsetted_512x512_HLS.S30.T10SEH.2018245.v1.4.mask.tif -> (sensor, acquisition date, scene id)
    """
    parts = os.path.basename(path).split(".")
    return parts[1], pd.to_datetime(parts[3], format="%Y%j"), f"{parts[2]}_{parts[3]}"


def load_scene(path):
    """
        HLS mask (30 m; 1 = burned, 0 = not burned, other = no data) and, for the window centered on every pixel:
        burned fraction, valid fraction, and burned / valid fraction of the window enlarged by the normal margin.
    """
    mask_da = rioxarray.open_rasterio(path).squeeze()
    vals = mask_da.values
    burned = (vals == 1).astype(np.float32)
    valid = np.isin(vals, [0, 1]).astype(np.float32)
    k, kb = 2 * HALF_PX + 1, 2 * (HALF_PX + NORMAL_MARGIN_PX) + 1
    stats = {"frac": uniform_filter(burned, size=k, mode="constant"),
             "vfrac": uniform_filter(valid, size=k, mode="constant"),
             "near": uniform_filter(burned, size=kb, mode="constant"),
             "vnear": uniform_filter(valid, size=kb, mode="constant")}
    return mask_da, burned, valid, stats


def inside(shape, e):
    r, c = np.mgrid[0:shape[0], 0:shape[1]]
    return (r >= e) & (r < shape[0] - e) & (c >= e) & (c < shape[1] - e)


def candidates(sel, ref=None):
    """
        Candidate window centers (mask pixels): closest to the scar center first (full windows, as before), else random.
    """
    pts = np.argwhere(sel)
    if len(pts) == 0:
        return pts
    if ref is not None:
        pts = pts[np.argsort(((pts - ref) ** 2).sum(axis=1))][::10]    # spaced out: next tries are not adjacent pixels
    else:
        pts = pts[rng.permutation(len(pts))]
    return pts[:MAX_POS_TRIES]


def window(mask_da, r, c):
    """
        1.2 km square centered on mask pixel (r, c): footprint in EPSG:4326 and Earth Engine rectangle.
    """
    x, y = mask_da.rio.transform() * (c + 0.5, r + 0.5)
    sq_wgs = gpd.GeoSeries([box(x - HALF_M, y - HALF_M, x + HALF_M, y + HALF_M)], crs=mask_da.rio.crs).to_crs(epsg=4326).iloc[0]
    return sq_wgs, ee.Geometry.Rectangle(list(sq_wgs.bounds))


def scene_footprint(mask_da):
    return gpd.GeoSeries([box(*mask_da.rio.bounds())], crs=mask_da.rio.crs).to_crs(epsg=4326).iloc[0]


def download(roi, date, crs):
    """
        Sentinel-2 L2A of the exact HLS acquisition day (same-day tiles mosaicked) on the window, in the mask UTM CRS.
        Returns (image (12, 120, 120), dataset, reason).
    """
    col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
           .filterDate(date.strftime('%Y-%m-%d'), (date + pd.Timedelta(days=1)).strftime('%Y-%m-%d'))
           .filterBounds(roi))
    if with_retries(lambda: col.size().getInfo()) == 0:
        return None, None, "no image on the acquisition day"
    img = (col.mosaic().select(BANDS_TO_USE + ['SCL']).toUint16()     # SCL is uint8: one dtype for the download
           .set('system:time_start', ee.Date(date.strftime('%Y-%m-%d')).millis()))   # mosaic drops properties; wxee needs a time
    ds = with_retries(lambda: img.wx.to_xarray(region=roi, scale=10, crs=crs, progress=False)).rio.write_crs(crs)
    data = np.stack([np.squeeze(ds[b].values) for b in BANDS_TO_USE], axis=0).astype(np.float32) / 10000.0
    scl = np.nan_to_num(np.squeeze(ds['SCL'].values), nan=0).astype(int)
    if data.shape[1] < TARGET_SIZE or data.shape[2] < TARGET_SIZE:
        return None, None, "too small"
    crop = data[:, :TARGET_SIZE, :TARGET_SIZE]
    nodata = ~np.isfinite(crop).all(axis=0) | (np.nan_to_num(crop).sum(axis=0) == 0)
    bad = (nodata | np.isin(scl[:TARGET_SIZE, :TARGET_SIZE], SCL_BAD)).mean()
    if bad > MAX_BAD_FRAC:
        return None, None, f"cloud / no-data {bad:.0%}"
    if np.nanmax(crop) > 2.0:
        return None, None, "invalid reflectance"
    return np.nan_to_num(crop), ds, "ok"


def window_labels(mask_da, burned, valid, ds):
    """
        Burned / valid maps of the 30 m mask reprojected (nearest) on the downloaded 10 m grid, cropped like the image.
        Pixels outside the annotated subset or with no-data mask values get valid = 0 (unknown label).
    """
    b = mask_da.copy(data=burned.astype(np.uint8)).rio.write_nodata(0)
    v = mask_da.copy(data=valid.astype(np.uint8)).rio.write_nodata(0)
    b = b.rio.reproject_match(ds, resampling=Resampling.nearest).values[:TARGET_SIZE, :TARGET_SIZE]
    v = v.rio.reproject_match(ds, resampling=Resampling.nearest).values[:TARGET_SIZE, :TARGET_SIZE]
    return b.astype(np.uint8), v.astype(np.uint8)


def extract(path, kind, kept_geoms, meta):
    """
        Save one window of the scene: kind = "anomaly" (partial / full / small, see P_PARTIAL) or "normal"
        (no burned pixel within NORMAL_MARGIN_PX). Windows never overlap previously saved ones (any scene, any date).
    """
    sensor, date, scene_id = scene_info(path)
    if sensor != "S30":
        print(f"Skipping {scene_id}: not a Sentinel-2 (S30) scene")
        return
    mask_da, burned, valid, s = load_scene(path)
    crs = mask_da.rio.crs.to_string()

    if kind == "anomaly":
        ok = inside(burned.shape, EDGE_PX) & (s["vfrac"] > 0.999)
        tiers = {"partial": ok & (s["frac"] >= PARTIAL_RANGE[0]) & (s["frac"] <= PARTIAL_RANGE[1]),
                 "full": ok & (s["frac"] > FULL_MIN),
                 "small": ok & (s["frac"] >= SMALL_MIN) & (s["frac"] < PARTIAL_RANGE[0])}
        order = ["partial", "small", "full"] if rng.random() < P_PARTIAL else ["full", "partial", "small"]
        ref = np.median(np.argwhere(burned > 0), axis=0) if burned.any() else None
        plan = [(m, candidates(tiers[m], ref if m == "full" else None)) for m in order]
    else:
        sel = (inside(burned.shape, EDGE_PX + NORMAL_MARGIN_PX) & (s["vfrac"] > 0.999)
               & (s["vnear"] > 0.999) & (s["near"] < 1e-6))
        plan = [("normal", candidates(sel))]

    footprint = scene_footprint(mask_da)
    near_kept = [g for g in kept_geoms if g.intersects(footprint)]
    downloads, reasons = 0, []
    for mode, pts in plan:
        for r, c in pts:
            if downloads >= MAX_DOWNLOADS:
                break
            sq_wgs, roi = window(mask_da, r, c)
            if any(sq_wgs.intersects(g) for g in near_kept):
                continue
            downloads += 1
            img, ds, reason = download(roi, date, crs)
            if img is None:
                reasons.append(reason)
                continue
            mask, valid_w = window_labels(mask_da, burned, valid, ds)
            if valid_w.min() < 1:
                reasons.append("unknown labels in the window")
                continue
            if kind == "anomaly" and mask.sum() < MIN_ANOM_PX:
                reasons.append("too few burned pixels")
                continue
            if kind == "normal" and mask.sum() > 0:
                reasons.append("burned pixels in a normal window")
                continue

            if kind == "anomaly":
                name = f"burn_{scene_id}_{mode}_anomaly"
                np.save(os.path.join(dirs["imgs"], f"{name}.npy"), img)
                np.save(os.path.join(dirs["masks"], f"{name}_mask.npy"), mask)
            else:
                name = f"burn_{scene_id}_normal"
                np.save(os.path.join(dirs["normal"], f"{name}.npy"), img)
            kept_geoms.append(sq_wgs)
            meta.append({"name": name, "kind": kind, "mode": mode, "scene": scene_id,
                         "folder": os.path.basename(os.path.dirname(path)), "date": str(date.date()),
                         "burned_frac": float(mask.mean()), "lon": sq_wgs.centroid.x, "lat": sq_wgs.centroid.y})
            print(f"Saved {name} (burned {mask.mean():.0%})")
            return
    print(f"Skipping {scene_id} ({kind}): {', '.join(reasons) if reasons else 'no valid window position'}")


scene_files = sorted(glob.glob(f"{HLS_DIR}/training/*.mask.tif") + glob.glob(f"{HLS_DIR}/validation/*.mask.tif"))
print(f"{len(scene_files)} HLS scenes")

kept_geoms, meta = [], []                     # every saved window (anomalies AND normals): no two patches overlap
for kind in ("anomaly", "normal"):            # all anomalies first: they get priority on the space
    for path in scene_files:
        try:
            extract(path, kind, kept_geoms, meta)
        except Exception as e:
            print(f"Skipping {os.path.basename(path)} ({kind}), error: {e}")

meta = pd.DataFrame(meta)
if meta.empty:
    raise RuntimeError("No burn-scar window saved: check GEE access / HLS_DIR")
meta.to_csv(f"{dirs['meta']}burn_scars.csv", index=False)
print(meta.groupby(["kind", "mode"]).agg(n=("name", "count"), median_burned=("burned_frac", "median")))


# ---------------------------------------------------------
# Validation plots (RGB + mask contour)
# ---------------------------------------------------------
def save_plot(name, img, mask=None):
    rgb = img[[3, 2, 1]].transpose(1, 2, 0)
    lo, hi = np.percentile(rgb, 2), np.percentile(rgb, 98)
    rgb = np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1)
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(rgb); ax[0].set_title(name, fontsize=7)
    ax[1].imshow(rgb)
    if mask is not None and 0 < mask.mean() < 1:
        ax[1].contour(mask, levels=[0.5], colors="red", linewidths=1)
    ax[1].set_title(f"burned {0 if mask is None else mask.mean():.0%} (red = mask border)", fontsize=8)
    for a in ax:
        a.axis("off")
    plt.savefig(os.path.join(dirs["plots"], f"{name}.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

for (kind, mode), grp in meta.groupby(["kind", "mode"]):
    for name in grp["name"].sample(min(3, len(grp)), random_state=SEED):
        if kind == "anomaly":
            save_plot(name, np.load(f"{dirs['imgs']}{name}.npy"), np.load(f"{dirs['masks']}{name}_mask.npy"))
        else:
            save_plot(name, np.load(f"{dirs['normal']}{name}.npy"))