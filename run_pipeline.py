"""
run_pipeline.py — End-to-end TFAM flood detection pipeline.

Usage:
    # Run on a pre-downloaded .npz from gee_pipeline.py:
    python run_pipeline.py --input data/cordoba_s1.npz --output outputs/cordoba/

    # Run the full synthetic demo (no GEE data needed):
    python run_pipeline.py --demo

Phases executed:
  1. Load pixel data  (from GEE .npz or synthetic)
  2. Extract C1-C9 descriptors + fit TCEV  (batch_tcev.py)
  3. Classify flood pixels  (classify.py)
  4. Visualize outputs  (visualize.py)
"""

import argparse
import os
import sys
import time

import numpy as np


def run_demo_pipeline(out_dir: str, n_rows: int = 60, n_cols: int = 80, n_time: int = 50):
    """
    Complete end-to-end demo on synthetic data.
    Tests every module in the pipeline without needing GEE or external data.
    """
    from batch_tcev import batch_process_serial, batch_process_pixels
    from classify import FloodClassifier
    from visualize import (
        plot_anomaly_heatmap,
        plot_flood_mask,
        plot_side_by_side,
        plot_convergence_map,
        plot_metrics_summary,
        export_geotiff,
    )
    from metrics import compute_metrics

    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(42)
    n_pixels = n_rows * n_cols

    print("=" * 70)
    print("  TFAM END-TO-END PIPELINE — SYNTHETIC DEMO")
    print("=" * 70)

    # ── Step 1: Generate synthetic Sentinel-1-like data ──────────────
    print("\n▶ Step 1: Generating synthetic pixel histories...")
    histories = rng.normal(-10.0, 1.5, size=(n_pixels, n_time))

    # Create a circular flood zone
    yy, xx = np.mgrid[:n_rows, :n_cols]
    flood_zone_2d = ((yy - n_rows // 2) ** 2 + (xx - n_cols // 2) ** 2) < (min(n_rows, n_cols) * 3)
    flood_zone_flat = flood_zone_2d.flatten()
    n_flood = flood_zone_flat.sum()

    # Inject flood dips into flood-zone pixel histories
    for i in np.where(flood_zone_flat)[0]:
        n_dips = rng.integers(3, 8)
        idx = rng.choice(n_time, size=n_dips, replace=False)
        histories[i, idx] = rng.normal(-19.0, 1.5, size=n_dips)

    # Event-date observations
    events = np.where(
        flood_zone_flat,
        rng.normal(-19.0, 1.5, size=n_pixels),
        rng.normal(-10.0, 1.0, size=n_pixels),
    )

    print(f"  Grid: {n_rows}×{n_cols} = {n_pixels} pixels")
    print(f"  Temporal depth: {n_time} acquisitions")
    print(f"  Flood zone: {n_flood} pixels ({100 * n_flood / n_pixels:.1f}%)")

    # ── Step 2: Batch TCEV feature extraction ────────────────────────
    print("\n▶ Step 2: Batch TCEV feature extraction & fitting...")
    t0 = time.time()
    results = batch_process_serial(histories, events, show_progress=True)
    t_tcev = time.time() - t0

    descriptors = results["descriptors"]
    anomaly_scores = results["anomaly_scores"]
    converged = results["converged"]

    print(f"  Converged: {converged.sum()}/{n_pixels}")
    print(f"  Time: {t_tcev:.1f}s")

    # ── Step 3: Classification ───────────────────────────────────────
    print("\n▶ Step 3: Flood classification (TabPFN / fallback)...")

    # Build feature matrix: 9 descriptors + anomaly score = 10 features
    X = np.column_stack([descriptors, anomaly_scores])
    y_true = flood_zone_flat.astype(int)

    # Only use converged pixels for training/context
    mask = converged & ~np.any(np.isnan(X), axis=1)
    X_clean = X[mask]
    y_clean = y_true[mask]

    if X_clean.shape[0] < 20:
        print("  ⚠ Too few converged pixels for classification. Using anomaly threshold.")
        y_pred_flat = (anomaly_scores >= 0.5).astype(int)
    else:
        # Use a subset as context examples (paper: 83 hand-labeled points)
        from sklearn.model_selection import train_test_split
        X_ctx, X_test, y_ctx, y_test = train_test_split(
            X_clean, y_clean, test_size=0.7, random_state=42, stratify=y_clean
        )

        clf = FloodClassifier(use_fallback_if_unlicensed=True)
        clf.fit_context(X_ctx, y_ctx)

        # Metrics on held-out test set
        y_pred_test = clf.max_flood_extent_mask(X_test, threshold=0.5)
        test_metrics = compute_metrics(y_test, y_pred_test)
        print(f"  Classifier: {'TabPFN' if not clf._is_fallback else 'HistGBT (fallback)'}")
        print(f"  Test-set metrics: IoU={test_metrics['IoU']:.3f}, "
              f"F1={test_metrics['F1']:.3f}, OA={test_metrics['OA']:.3f}")

        # Predict on ALL converged pixels
        y_pred_flat = np.zeros(n_pixels, dtype=int)
        y_pred_flat[mask] = clf.max_flood_extent_mask(X_clean, threshold=0.5)

    # ── Step 4: Reshape to 2-D grids ────────────────────────────────
    print("\n▶ Step 4: Reshaping to spatial grids...")
    anomaly_grid = anomaly_scores.reshape(n_rows, n_cols)
    flood_mask = y_pred_flat.reshape(n_rows, n_cols).astype(bool)
    converged_grid = converged.reshape(n_rows, n_cols)
    gt_grid = flood_zone_2d

    # ── Step 5: Compute spatial accuracy ─────────────────────────────
    print("\n▶ Step 5: Computing spatial accuracy metrics...")
    spatial_metrics = compute_metrics(gt_grid.flatten().astype(int), flood_mask.flatten().astype(int))
    for k, v in spatial_metrics.items():
        print(f"  {k}: {v:.4f}")

    # ── Step 6: Visualization ────────────────────────────────────────
    print("\n▶ Step 6: Generating visualizations...")

    plot_anomaly_heatmap(
        anomaly_grid,
        title="TCEV Anomaly Score — Synthetic Demo",
        out_path=os.path.join(out_dir, "anomaly_heatmap.png"),
    )

    plot_flood_mask(
        flood_mask,
        title="Predicted Flood Mask",
        out_path=os.path.join(out_dir, "flood_mask_predicted.png"),
    )

    plot_flood_mask(
        gt_grid,
        title="Ground Truth Flood Mask",
        out_path=os.path.join(out_dir, "flood_mask_ground_truth.png"),
    )

    plot_side_by_side(
        anomaly_grid,
        flood_mask,
        title="TFAM Flood Detection — Synthetic Demo",
        out_path=os.path.join(out_dir, "side_by_side.png"),
    )

    plot_convergence_map(
        converged_grid,
        out_path=os.path.join(out_dir, "convergence_map.png"),
    )

    plot_metrics_summary(
        spatial_metrics,
        title="TFAM Spatial Accuracy — Synthetic Demo",
        out_path=os.path.join(out_dir, "metrics_chart.png"),
    )

    # GeoTIFF
    export_geotiff(
        anomaly_grid,
        os.path.join(out_dir, "anomaly_score.tif"),
    )
    export_geotiff(
        flood_mask.astype(np.float32),
        os.path.join(out_dir, "flood_mask.tif"),
    )

    # Save raw results as .npz
    np.savez_compressed(
        os.path.join(out_dir, "pipeline_results.npz"),
        anomaly_grid=anomaly_grid,
        flood_mask=flood_mask,
        converged=converged_grid,
        ground_truth=gt_grid,
        descriptors=descriptors,
        metrics=spatial_metrics,
    )
    print(f"\n  Raw results saved → {os.path.join(out_dir, 'pipeline_results.npz')}")

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Grid size:     {n_rows}×{n_cols} ({n_pixels} pixels)")
    print(f"  Converged:     {converged.sum()}/{n_pixels} ({100 * converged.sum() / n_pixels:.1f}%)")
    print(f"  Flood pixels:  predicted={flood_mask.sum()}, ground_truth={gt_grid.sum()}")
    print(f"  IoU:           {spatial_metrics['IoU']:.4f}")
    print(f"  F1:            {spatial_metrics['F1']:.4f}")
    print(f"  OA:            {spatial_metrics['OA']:.4f}")
    print(f"  Kappa:         {spatial_metrics['Kappa']:.4f}")
    print(f"  Outputs:       {out_dir}")
    print("=" * 70)

    return spatial_metrics


def run_real_pipeline(input_path: str, out_dir: str):
    """
    Run on real .npz data exported by gee_pipeline.py.

    Expected .npz keys:
        vh_cube: shape (rows, cols, T) — multi-temporal VH backscatter in dB
        vh_event: shape (rows, cols) — event-date VH backscatter in dB
        metadata: dict with geo_transform, epsg, etc.
    """
    from batch_tcev import batch_process_pixels
    from classify import FloodClassifier
    from visualize import (
        plot_anomaly_heatmap,
        plot_flood_mask,
        plot_side_by_side,
        plot_convergence_map,
        export_geotiff,
    )

    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("  TFAM PIPELINE — REAL SENTINEL-1 DATA")
    print("=" * 70)

    print(f"\n▶ Loading data from: {input_path}")
    data = np.load(input_path, allow_pickle=True)

    if "vh_cube" not in data:
        print("ERROR: Expected 'vh_cube' key in .npz file.")
        print(f"  Available keys: {list(data.keys())}")
        sys.exit(1)

    vh_cube = data["vh_cube"]  # (rows, cols, T)
    rows, cols, T = vh_cube.shape
    n_pixels = rows * cols
    print(f"  Grid: {rows}×{cols} = {n_pixels} pixels, {T} time steps")

    # Event observation (use last time step if not provided separately)
    if "vh_event" in data:
        vh_event = data["vh_event"]  # (rows, cols)
    else:
        vh_event = vh_cube[:, :, -1]
        vh_cube = vh_cube[:, :, :-1]
        T -= 1
        print(f"  Using last time step as event observation. History: {T} steps.")

    # Flatten to (N, T)
    histories_flat = vh_cube.reshape(n_pixels, T)
    events_flat = vh_event.flatten()

    # Batch TCEV
    print("\n▶ Running batch TCEV...")
    results = batch_process_pixels(histories_flat, events_flat, max_workers=4, show_progress=True)

    anomaly_grid = results["anomaly_scores"].reshape(rows, cols)
    converged_grid = results["converged"].reshape(rows, cols)

    # Simple threshold classification (no ground truth for supervised)
    flood_mask = anomaly_grid >= 0.5

    # Geo info
    geo_transform = None
    epsg = 4326
    if "metadata" in data:
        meta = data["metadata"].item() if hasattr(data["metadata"], "item") else {}
        geo_transform = meta.get("geo_transform")
        epsg = meta.get("epsg", 4326)

    # Visualize
    print("\n▶ Generating outputs...")
    plot_anomaly_heatmap(anomaly_grid, out_path=os.path.join(out_dir, "anomaly_heatmap.png"))
    plot_flood_mask(flood_mask, out_path=os.path.join(out_dir, "flood_mask.png"))
    plot_side_by_side(anomaly_grid, flood_mask, out_path=os.path.join(out_dir, "side_by_side.png"))
    plot_convergence_map(converged_grid, out_path=os.path.join(out_dir, "convergence_map.png"))

    export_geotiff(anomaly_grid, os.path.join(out_dir, "anomaly_score.tif"), geo_transform, epsg)
    export_geotiff(flood_mask.astype(np.float32), os.path.join(out_dir, "flood_mask.tif"), geo_transform, epsg)

    np.savez_compressed(
        os.path.join(out_dir, "pipeline_results.npz"),
        anomaly_grid=anomaly_grid,
        flood_mask=flood_mask,
        converged=converged_grid,
        descriptors=results["descriptors"].reshape(rows, cols, 9),
    )

    print(f"\n✓ Pipeline complete. Outputs → {out_dir}")
    print(f"  Converged: {converged_grid.sum()}/{n_pixels} ({100 * converged_grid.sum() / n_pixels:.1f}%)")
    print(f"  Flood pixels detected: {flood_mask.sum()}")


# ─────────────────────────── CLI ─────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="TFAM Flood Detection Pipeline — End-to-End Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --demo
  python run_pipeline.py --input data/cordoba_s1.npz --output outputs/cordoba/
        """,
    )
    parser.add_argument("--demo", action="store_true", help="Run synthetic demo (no data needed)")
    parser.add_argument("--input", type=str, help="Path to .npz file from gee_pipeline.py")
    parser.add_argument("--output", type=str, default="outputs/", help="Output directory")
    parser.add_argument("--rows", type=int, default=60, help="Demo grid rows (default: 60)")
    parser.add_argument("--cols", type=int, default=80, help="Demo grid cols (default: 80)")

    args = parser.parse_args()

    if args.demo:
        run_demo_pipeline(args.output, n_rows=args.rows, n_cols=args.cols)
    elif args.input:
        run_real_pipeline(args.input, args.output)
    else:
        parser.print_help()
        print("\nError: specify --demo or --input <path>")
        sys.exit(1)


if __name__ == "__main__":
    main()
