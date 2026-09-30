"""
Phase 5: TFAM Visualization and GeoTIFF/Map Outputs.

Produces:
- Side-by-side anomaly score heatmaps
- Binary flood mask (P_f >= 0.5) overlay
- GeoTIFF exports for GIS workflows
- Accuracy metrics visualization

Following Section 3.5 and Figures 3-5 of Ta et al. (2026).
"""

import os
from typing import Optional, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from osgeo import gdal, osr
    HAS_GDAL = True
except ImportError:
    HAS_GDAL = False


# ────────────────────────────── colour palettes ───────────────────────────────

ANOMALY_CMAP_COLORS = [
    (0.12, 0.47, 0.71),  # blue  — low anomaly
    (1.00, 1.00, 0.60),  # yellow — moderate
    (1.00, 0.50, 0.05),  # orange — high
    (0.84, 0.15, 0.16),  # red   — extreme
]

FLOOD_MASK_COLORS = [
    (0.85, 0.85, 0.85, 0.0),  # transparent = no flood
    (0.12, 0.56, 1.00, 0.7),  # semi-transparent blue = flood
]


def _anomaly_cmap():
    """Custom colormap for anomaly scores (0-1 range, blue→red)."""
    return mcolors.LinearSegmentedColormap.from_list(
        "tfam_anomaly", ANOMALY_CMAP_COLORS, N=256
    )


def _flood_cmap():
    """Binary colormap for flood mask overlay."""
    return mcolors.ListedColormap([c[:3] for c in FLOOD_MASK_COLORS])


# ──────────────────────────── plot functions ──────────────────────────────────


def plot_anomaly_heatmap(
    anomaly_grid: np.ndarray,
    title: str = "TCEV Anomaly Score",
    out_path: Optional[str] = None,
    vmin: float = 0.0,
    vmax: float = 1.0,
    figsize: Tuple[int, int] = (10, 8),
) -> Optional[str]:
    """
    Render a 2-D anomaly score grid as a heatmap.

    Args:
        anomaly_grid: 2-D array of anomaly scores (Eq 9).
        title: plot title.
        out_path: if set, save figure to this path.
        vmin, vmax: colour bar range.
        figsize: matplotlib figure size.

    Returns:
        Path to saved figure, or None.
    """
    if not HAS_MPL:
        print("[visualize] matplotlib not available. Skipping plot.")
        return None

    fig, ax = plt.subplots(figsize=figsize)
    cmap = _anomaly_cmap()
    im = ax.imshow(anomaly_grid, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Anomaly Score (TCEV CDF)", fontsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Column (pixel)")
    ax.set_ylabel("Row (pixel)")
    fig.tight_layout()

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[visualize] Saved anomaly heatmap → {out_path}")
        return out_path
    else:
        plt.close(fig)
        return None


def plot_flood_mask(
    flood_mask: np.ndarray,
    title: str = "Flood Mask (P_f ≥ 0.5)",
    out_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
) -> Optional[str]:
    """
    Render a binary flood mask.

    Args:
        flood_mask: 2-D boolean/int array (1 = flood).
        title: plot title.
        out_path: save path.
    """
    if not HAS_MPL:
        print("[visualize] matplotlib not available. Skipping plot.")
        return None

    fig, ax = plt.subplots(figsize=figsize)
    cmap = _flood_cmap()
    ax.imshow(flood_mask.astype(int), cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Column (pixel)")
    ax.set_ylabel("Row (pixel)")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=(0.85, 0.85, 0.85), edgecolor="k", label="No Flood"),
        Patch(facecolor=(0.12, 0.56, 1.00), edgecolor="k", label="Flood"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10)
    fig.tight_layout()

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[visualize] Saved flood mask → {out_path}")
        return out_path
    else:
        plt.close(fig)
        return None


def plot_side_by_side(
    anomaly_grid: np.ndarray,
    flood_mask: np.ndarray,
    title: str = "TFAM Flood Detection",
    out_path: Optional[str] = None,
    figsize: Tuple[int, int] = (18, 7),
) -> Optional[str]:
    """
    Side-by-side: anomaly heatmap + binary flood mask.
    Follows the visual layout of Figures 3-5 in the paper.
    """
    if not HAS_MPL:
        print("[visualize] matplotlib not available. Skipping plot.")
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Left: anomaly heatmap
    cmap_a = _anomaly_cmap()
    im1 = ax1.imshow(anomaly_grid, cmap=cmap_a, vmin=0, vmax=1, interpolation="nearest")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label="Anomaly Score")
    ax1.set_title("TCEV Anomaly Score Map", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Column")
    ax1.set_ylabel("Row")

    # Right: flood mask
    cmap_f = _flood_cmap()
    ax2.imshow(flood_mask.astype(int), cmap=cmap_f, vmin=0, vmax=1, interpolation="nearest")
    ax2.set_title("Flood Mask (P_f ≥ 0.5)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Column")
    ax2.set_ylabel("Row")
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=(0.85, 0.85, 0.85), edgecolor="k", label="No Flood"),
        Patch(facecolor=(0.12, 0.56, 1.00), edgecolor="k", label="Flood"),
    ]
    ax2.legend(handles=legend_elements, loc="lower right", fontsize=10)

    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[visualize] Saved side-by-side → {out_path}")
        return out_path
    else:
        plt.close(fig)
        return None


def plot_convergence_map(
    converged: np.ndarray,
    title: str = "TCEV Convergence Map",
    out_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
) -> Optional[str]:
    """Visualize which pixels converged during TCEV fitting."""
    if not HAS_MPL:
        return None

    fig, ax = plt.subplots(figsize=figsize)
    cmap = mcolors.ListedColormap(["#ff4444", "#44bb44"])
    ax.imshow(converged.astype(int), cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(title, fontsize=14, fontweight="bold")
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#ff4444", edgecolor="k", label="Failed"),
        Patch(facecolor="#44bb44", edgecolor="k", label="Converged"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10)
    fig.tight_layout()

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[visualize] Saved convergence map → {out_path}")
        return out_path
    else:
        plt.close(fig)
        return None


# ──────────────────────── GeoTIFF export ─────────────────────────────────────


def export_geotiff(
    data: np.ndarray,
    out_path: str,
    geo_transform: Optional[Tuple] = None,
    epsg: int = 4326,
    nodata: float = -9999.0,
) -> str:
    """
    Export a 2-D array as a single-band GeoTIFF.

    Args:
        data: 2-D float array.
        out_path: output file path.
        geo_transform: GDAL GeoTransform tuple (x_origin, pixel_width, 0,
                        y_origin, 0, -pixel_height). If None, uses identity.
        epsg: coordinate reference system (default WGS84).
        nodata: NoData value.

    Returns:
        Path to written file.
    """
    if not HAS_GDAL:
        # Fallback: save as .npz
        fallback = out_path.replace(".tif", ".npz").replace(".tiff", ".npz")
        np.savez_compressed(fallback, data=data)
        print(f"[visualize] GDAL not available. Saved as NumPy → {fallback}")
        return fallback

    rows, cols = data.shape
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(out_path, cols, rows, 1, gdal.GDT_Float32)

    if geo_transform:
        ds.SetGeoTransform(geo_transform)
    else:
        ds.SetGeoTransform((0, 1, 0, 0, 0, -1))

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    ds.SetProjection(srs.ExportToWkt())

    band = ds.GetRasterBand(1)
    band.SetNoDataValue(nodata)
    band.WriteArray(np.where(np.isnan(data), nodata, data))
    band.FlushCache()
    ds = None

    print(f"[visualize] GeoTIFF saved → {out_path}")
    return out_path


# ──────────────────────── metrics visualisation ──────────────────────────────


def plot_metrics_summary(
    metrics_dict: dict,
    title: str = "TFAM Accuracy Metrics",
    out_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 5),
) -> Optional[str]:
    """
    Bar chart of accuracy metrics (IoU, F1, Precision, Recall, OA, Kappa).
    metrics_dict should have keys matching the metric names.
    """
    if not HAS_MPL:
        return None

    keys = list(metrics_dict.keys())
    vals = [metrics_dict[k] for k in keys]

    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(keys)))
    bars = ax.bar(keys, vals, color=colors, edgecolor="k", linewidth=0.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    fig.tight_layout()

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[visualize] Saved metrics chart → {out_path}")
        return out_path
    else:
        plt.close(fig)
        return None


# ──────────────────────── main demo ──────────────────────────────────────────


if __name__ == "__main__":
    print("=" * 60)
    print("TFAM VISUALIZATION DEMO")
    print("=" * 60)

    rng = np.random.default_rng(42)
    rows, cols = 80, 100

    # Synthetic anomaly grid: low background, high in a circular "flood zone"
    anomaly_grid = rng.uniform(0.05, 0.30, size=(rows, cols))
    yy, xx = np.mgrid[:rows, :cols]
    flood_zone = ((yy - 40) ** 2 + (xx - 50) ** 2) < 500
    anomaly_grid[flood_zone] = rng.uniform(0.70, 0.98, size=flood_zone.sum())

    # Binary mask
    flood_mask = anomaly_grid >= 0.5

    out_dir = os.path.join(os.path.dirname(__file__), "outputs")

    # 1) Anomaly heatmap
    plot_anomaly_heatmap(
        anomaly_grid,
        title="Synthetic TCEV Anomaly Score",
        out_path=os.path.join(out_dir, "anomaly_heatmap.png"),
    )

    # 2) Flood mask
    plot_flood_mask(
        flood_mask,
        title="Synthetic Flood Mask (P_f ≥ 0.5)",
        out_path=os.path.join(out_dir, "flood_mask.png"),
    )

    # 3) Side-by-side
    plot_side_by_side(
        anomaly_grid,
        flood_mask,
        title="TFAM Flood Detection — Synthetic Demo",
        out_path=os.path.join(out_dir, "side_by_side.png"),
    )

    # 4) Convergence map (synthetic: 95% converged)
    converged = rng.random(size=(rows, cols)) > 0.05
    plot_convergence_map(
        converged,
        out_path=os.path.join(out_dir, "convergence_map.png"),
    )

    # 5) GeoTIFF export
    export_geotiff(
        anomaly_grid,
        os.path.join(out_dir, "anomaly_score.tif"),
    )

    # 6) Metrics chart
    demo_metrics = {
        "IoU": 0.823,
        "F1": 0.903,
        "Precision": 0.912,
        "Recall": 0.894,
        "OA": 0.951,
        "Kappa": 0.876,
    }
    plot_metrics_summary(
        demo_metrics,
        title="TFAM Accuracy — Synthetic Demo",
        out_path=os.path.join(out_dir, "metrics_chart.png"),
    )

    print(f"\n✓ All outputs saved to: {out_dir}")
    print("Phase 5 visualization demo complete.")
