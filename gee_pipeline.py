"""
Phase 1: Sentinel-1 SAR Data Ingestion and Regional Event Extraction Pipeline.

Following Section 3.1 & Table 1 of:
Ta et al. (2026), "SAR Flood Anomaly Mapping Through Statistical Time-Series
Feature Classification and Pixel-Wise TCEV Modeling", Remote Sens. 2026, 18, 3323.

This module interfaces Google Earth Engine (GEE) to:
  1. Define and load regional flood study areas (Zagora, Córdoba, Guangxi, NSW, or Custom).
  2. Filter COPERNICUS/S1_GRD by IW mode, VH polarization, consistent orbit geometry,
     and same-season calendar windows across multi-year baselines (2015-2026).
  3. Extract pixel time-series stacks and event scenes for TFAM anomaly processing.
"""

import argparse
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

# Regional Event Presets from Table 1 and Table 2 in Ta et al. (2026)
REGIONAL_PRESETS: Dict[str, dict] = {
    "zagora": {
        "name": "Zagora & Thessaly Plain, Greece",
        "country": "Greece",
        "cems_activation": "EMSR692",
        "hazard": "Medicane Daniel",
        "bbox": [22.0, 39.0, 23.2, 40.0],  # [min_lon, min_lat, max_lon, max_lat]
        "center": [22.45, 39.55],
        "pass_direction": "ASCENDING",
        "relative_orbit": 102,
        "seasonal_months": [9, 10, 11],  # September - November (Autumn, SON)
        "baseline_years": (2015, 2023),
        "event_dates": ["2023-09-07", "2023-09-19"],
        "primary_event_date": "2023-09-07",
    },
    "cordoba": {
        "name": "Córdoba Department, Colombia",
        "country": "Colombia",
        "cems_activation": "EMSR865",
        "hazard": "Severe Out-of-Season Rainstorm",
        "bbox": [-76.2, 8.2, -75.4, 9.2],
        "center": [-75.8, 8.7],
        "pass_direction": "DESCENDING",
        "relative_orbit": 142,
        "seasonal_months": [12, 1, 2],  # December - February (Dry season, DJF)
        "baseline_years": (2015, 2026),
        "event_dates": ["2026-02-02", "2026-02-08", "2026-02-14"],
        "primary_event_date": "2026-02-02",
    },
    "guangxi": {
        "name": "Guangxi Zhuang Autonomous Region, China",
        "country": "China",
        "cems_activation": None,
        "hazard": "Typhoons Ragasa, Bualoi & Matmo",
        "bbox": [108.0, 22.5, 109.5, 23.8],
        "center": [108.8, 23.1],
        "pass_direction": "DESCENDING",
        "relative_orbit": 91,
        "seasonal_months": [9, 10, 11],  # September - November (Autumn, SON)
        "baseline_years": (2015, 2025),
        "event_dates": ["2025-10-01"],
        "primary_event_date": "2025-10-01",
    },
    "nsw": {
        "name": "New South Wales, Australia",
        "country": "Australia",
        "cems_activation": "EMSR570",
        "hazard": "La Niña Relentless Rainfall",
        "bbox": [150.0, -34.5, 151.5, -33.2],
        "center": [150.8, -33.8],
        "pass_direction": "DESCENDING",
        "relative_orbit": 74,
        "seasonal_months": [3, 4, 5],  # March - May (Autumn, MAM)
        "baseline_years": (2015, 2022),
        "event_dates": ["2022-03-31"],
        "primary_event_date": "2022-03-31",
    },
}


def initialize_earth_engine(project_id: str = "sar-flood-anomaly-mapping") -> None:
    """Initialize the Earth Engine Python API using the authenticated GCP project."""
    import ee
    try:
        ee.Initialize(project=project_id)
        print(f"[GEE] Successfully initialized with project: '{project_id}'")
    except Exception as e:
        raise RuntimeError(
            f"Failed to initialize Earth Engine for project '{project_id}': {e}. "
            "Ensure you have run 'earthengine authenticate'."
        ) from e


def get_sentinel1_seasonal_collection(
    region_config: dict,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
):
    """
    Build the filtered Sentinel-1 GRD collection for the region adhering to Section 3.1:
      - Instrument mode: IW (Interferometric Wide Swath)
      - Polarization: VH (superior water specular reflection detection)
      - Consistent orbit geometry: fixed pass (ASCENDING/DESCENDING) and relative orbit number
      - Seasonal window filtering: acquisitions within same calendar months
    """
    import ee

    bbox = region_config["bbox"]
    geom = ee.Geometry.BBox(bbox[0], bbox[1], bbox[2], bbox[3])
    pass_dir = region_config["pass_direction"]
    rel_orbit = region_config["relative_orbit"]
    months = region_config["seasonal_months"]

    s_year = start_year or region_config["baseline_years"][0]
    e_year = end_year or region_config["baseline_years"][1]
    start_date = f"{s_year}-01-01"
    end_date = f"{e_year}-12-31"

    collection = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(geom)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .filter(ee.Filter.eq("orbitProperties_pass", pass_dir))
        .filter(ee.Filter.eq("relativeOrbitNumber_start", rel_orbit))
        .filterDate(start_date, end_date)
        .select("VH")
    )

    # Apply seasonal month filtering
    if len(months) == 3:
        if months == [12, 1, 2]:  # DJF crosses year boundary
            seasonal_filter = ee.Filter.Or(
                ee.Filter.calendarRange(12, 12, "month"),
                ee.Filter.calendarRange(1, 2, "month"),
            )
        else:
            seasonal_filter = ee.Filter.calendarRange(months[0], months[-1], "month")
        collection = collection.filter(seasonal_filter)

    return collection, geom


def get_collection_metadata(collection) -> dict:
    """Return summary metadata for a filtered Earth Engine collection."""
    from datetime import timezone
    size = collection.size().getInfo()
    dates = []
    if size > 0:
        first_date = collection.first().date().format("YYYY-MM-dd").getInfo()
        # Sample acquisition dates
        date_list = collection.aggregate_array("system:time_start").getInfo()
        dates = [datetime.fromtimestamp(t / 1000.0, timezone.utc).strftime("%Y-%m-%d") for t in date_list]
        unique_dates = sorted(list(set(dates)))
    else:
        first_date = None
        unique_dates = []

    return {
        "total_scenes": size,
        "first_scene_date": first_date,
        "unique_acquisition_dates_count": len(unique_dates),
        "sample_dates": unique_dates[:10],
    }


def extract_pixel_series_at_point(
    collection, lon: float, lat: float, scale: int = 30
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract calibrated VH backscatter time series (in dB) at a point coordinate.
    Returns (timestamps_ms, vh_db_array).
    """
    import ee

    pt = ee.Geometry.Point([lon, lat])
    raw = collection.getRegion(pt, scale).getInfo()
    if not raw or len(raw) <= 1:
        return np.array([]), np.array([])

    header = raw[0]
    vh_idx = header.index("VH")
    time_idx = header.index("time")

    records = [(r[time_idx], r[vh_idx]) for r in raw[1:] if r[vh_idx] is not None]
    records.sort(key=lambda x: x[0])
    times = np.array([r[0] for r in records], dtype=np.int64)
    vh_db = np.array([r[1] for r in records], dtype=float)

    return times, vh_db


def sample_grid_time_series(
    collection,
    bbox: List[float],
    grid_size: int = 10,
    scale: int = 30,
) -> Dict[str, np.ndarray]:
    """
    Sample a regular grid of N x N pixels across the bounding box and extract
    historical VH time series for each point.
    """
    import ee

    min_lon, min_lat, max_lon, max_lat = bbox
    lons = np.linspace(min_lon, max_lon, grid_size)
    lats = np.linspace(min_lat, max_lat, grid_size)

    features = []
    point_meta = []
    idx = 0
    for lat in lats:
        for lon in lons:
            feat = ee.Feature(ee.Geometry.Point([lon, lat]), {"pixel_id": idx, "lon": lon, "lat": lat})
            features.append(feat)
            point_meta.append((idx, lon, lat))
            idx += 1

    fc = ee.FeatureCollection(features)
    print(f"[GEE] Sampling {len(features)} points across bounding box...")
    raw = collection.getRegion(fc.geometry(), scale).getInfo()

    header = raw[0]
    vh_idx = header.index("VH")
    time_idx = header.index("time")
    id_idx = header.index("id")

    return {
        "raw_region": raw,
        "grid_size": grid_size,
        "lons": lons,
        "lats": lats,
        "point_meta": point_meta,
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Sentinel-1 GEE Data Ingestion Pipeline")
    parser.add_argument(
        "--region",
        type=str,
        default="zagora",
        choices=list(REGIONAL_PRESETS.keys()) + ["custom"],
        help="Target regional flood event (default: zagora).",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="sar-flood-anomaly-mapping",
        help="GCP Project ID for Google Earth Engine.",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Query and print GEE Sentinel-1 collection metadata for the selected region.",
    )
    parser.add_argument(
        "--sample-point",
        action="store_true",
        help="Extract a single-pixel historical time-series at the regional center coordinate.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Directory to save extracted datasets.",
    )

    args = parser.parse_args()

    region_cfg = REGIONAL_PRESETS.get(args.region)
    if not region_cfg:
        raise ValueError(f"Unknown region '{args.region}'.")

    print(f"\n========================================================")
    print(f"  PHASE 1: REGIONAL SAR EVENT INGESTION - {region_cfg['name'].upper()}")
    print(f"========================================================")
    print(f"  Country          : {region_cfg['country']}")
    print(f"  Hazard           : {region_cfg['hazard']}")
    print(f"  CEMS Activation  : {region_cfg.get('cems_activation', 'N/A')}")
    print(f"  Orbit Pass       : {region_cfg['pass_direction']}")
    print(f"  Relative Orbit   : {region_cfg['relative_orbit']}")
    print(f"  Seasonal Window  : Months {region_cfg['seasonal_months']}")
    print(f"  Baseline Years   : {region_cfg['baseline_years'][0]} - {region_cfg['baseline_years'][1]}")
    print(f"  Target Event Date: {region_cfg['primary_event_date']}")
    print(f"  Bounding Box     : {region_cfg['bbox']}")
    print(f"--------------------------------------------------------\n")

    initialize_earth_engine(args.project)
    collection, geom = get_sentinel1_seasonal_collection(region_cfg)

    meta = get_collection_metadata(collection)
    print(f"[GEE] Matching Seasonal Sentinel-1 Scenes: {meta['total_scenes']}")
    print(f"[GEE] Unique Seasonal Observation Dates  : {meta['unique_acquisition_dates_count']}")
    print(f"[GEE] Earliest Historical Acquisition   : {meta['first_scene_date']}")
    print(f"[GEE] Sample Acquisition Dates          : {meta['sample_dates'][:5]}...\n")

    if args.sample_point or not args.inspect:
        center_lon, center_lat = region_cfg["center"]
        print(f"[GEE] Extracting real time series at center ({center_lon}, {center_lat})...")
        times, vh_series = extract_pixel_series_at_point(collection, center_lon, center_lat)
        print(f"[GEE] Extracted {len(vh_series)} historical Sentinel-1 observations.")
        if len(vh_series) > 0:
            print(f"      Mean VH: {np.mean(vh_series):.2f} dB, Min: {np.min(vh_series):.2f} dB, Max: {np.max(vh_series):.2f} dB")
            os.makedirs(args.output_dir, exist_ok=True)
            out_file = os.path.join(args.output_dir, f"{args.region}_sample_point.npz")
            np.savez_compressed(
                out_file,
                times=times,
                vh_db=vh_series,
                lon=center_lon,
                lat=center_lat,
                region=args.region,
            )
            print(f"[OK] Saved real Sentinel-1 pixel series to: {out_file}")

    print("\nPhase 1 verification complete.")


if __name__ == "__main__":
    main()
