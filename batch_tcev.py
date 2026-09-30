"""
Phase 3: Batch-parallel feature extraction and TCEV fitting over pixel grids.

Upgrades the serial per-pixel loop in test_grid.py to chunked multiprocessing,
enabling processing of real Sentinel-1 scenes (hundreds of thousands of pixels)
in minutes rather than hours.

Following Section 3.2 of Ta et al. (2026).
"""

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from features import pixel_features, empirical_cdf_sorted
from tcev import fit_tcev, event_anomaly_score, TCEVParams


def process_single_pixel(
    pixel_history: np.ndarray,
    event_obs: Optional[float] = None,
    min_segment: int = 5,
) -> dict:
    """
    Full per-pixel pipeline: features.py (Eqs 1-6) -> tcev.py (Eq 7) -> anomaly score (Eq 9).

    Args:
        pixel_history: 1-D array of historical VH backscatter in dB (NOT sign-inverted).
        event_obs: optional event-date VH observation in dB.
        min_segment: minimum segment length for piecewise fit.

    Returns:
        dict with keys: descriptors (C1-C9), tcev_params (a1,b1,a2,b2),
        anomaly_score (if event_obs given), converged (bool).
    """
    if len(pixel_history) < 2 * min_segment:
        return {
            "descriptors": np.full(9, np.nan),
            "tcev_params": (np.nan, np.nan, np.nan, np.nan),
            "anomaly_score": np.nan,
            "converged": False,
        }

    try:
        descriptors, fit = pixel_features(pixel_history, min_segment=min_segment)

        sigma_prime = -pixel_history
        x_sorted, F = empirical_cdf_sorted(sigma_prime)
        tcev = fit_tcev(x_sorted, F, fit)

        score = np.nan
        if event_obs is not None:
            score = event_anomaly_score(tcev, event_obs)

        return {
            "descriptors": descriptors,
            "tcev_params": (tcev.a1, tcev.b1, tcev.a2, tcev.b2),
            "anomaly_score": score,
            "converged": True,
        }
    except Exception:
        return {
            "descriptors": np.full(9, np.nan),
            "tcev_params": (np.nan, np.nan, np.nan, np.nan),
            "anomaly_score": np.nan,
            "converged": False,
        }


def _worker(args):
    """Unpacker for multiprocessing — calls process_single_pixel."""
    idx, history, event_obs, min_seg = args
    result = process_single_pixel(history, event_obs, min_seg)
    result["pixel_idx"] = idx
    return result


def batch_process_pixels(
    pixel_histories: np.ndarray,
    event_observations: Optional[np.ndarray] = None,
    min_segment: int = 5,
    max_workers: Optional[int] = None,
    show_progress: bool = True,
) -> dict:
    """
    Process a batch of pixels in parallel.

    Args:
        pixel_histories: shape (N, T) — N pixels, T historical observations each.
        event_observations: shape (N,) — event-date VH dB for each pixel, or None.
        min_segment: minimum segment length for piecewise fit.
        max_workers: number of parallel workers (None = auto).
        show_progress: print progress updates.

    Returns:
        dict with arrays: descriptors (N, 9), tcev_params (N, 4),
        anomaly_scores (N,), converged (N,).
    """
    n_pixels = pixel_histories.shape[0]

    if show_progress:
        print(f"[batch_tcev] Processing {n_pixels} pixels with {max_workers or 'auto'} workers...")

    t0 = time.time()

    # Build task arguments
    tasks = []
    for i in range(n_pixels):
        ev = float(event_observations[i]) if event_observations is not None else None
        tasks.append((i, pixel_histories[i], ev, min_segment))

    # Allocate output arrays
    all_descriptors = np.full((n_pixels, 9), np.nan)
    all_tcev_params = np.full((n_pixels, 4), np.nan)
    all_scores = np.full(n_pixels, np.nan)
    all_converged = np.zeros(n_pixels, dtype=bool)

    completed = 0
    log_interval = max(1, n_pixels // 10)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_worker, t): t[0] for t in tasks}
        for future in as_completed(futures):
            result = future.result()
            idx = result["pixel_idx"]
            all_descriptors[idx] = result["descriptors"]
            all_tcev_params[idx] = result["tcev_params"]
            all_scores[idx] = result["anomaly_score"]
            all_converged[idx] = result["converged"]

            completed += 1
            if show_progress and (completed % log_interval == 0 or completed == n_pixels):
                elapsed = time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 0
                print(
                    f"  [{completed}/{n_pixels}] "
                    f"{100 * completed / n_pixels:.0f}% done — "
                    f"{rate:.1f} pixels/sec — "
                    f"elapsed {elapsed:.1f}s"
                )

    elapsed = time.time() - t0
    n_ok = int(all_converged.sum())
    if show_progress:
        print(
            f"[batch_tcev] Finished: {n_ok}/{n_pixels} pixels converged "
            f"in {elapsed:.1f}s ({n_pixels / elapsed:.1f} px/s)"
        )

    return {
        "descriptors": all_descriptors,
        "tcev_params": all_tcev_params,
        "anomaly_scores": all_scores,
        "converged": all_converged,
    }


def batch_process_serial(
    pixel_histories: np.ndarray,
    event_observations: Optional[np.ndarray] = None,
    min_segment: int = 5,
    show_progress: bool = True,
) -> dict:
    """
    Serial fallback for environments where multiprocessing is problematic.
    Same interface as batch_process_pixels.
    """
    n_pixels = pixel_histories.shape[0]
    if show_progress:
        print(f"[batch_tcev] Processing {n_pixels} pixels (serial mode)...")

    t0 = time.time()
    all_descriptors = np.full((n_pixels, 9), np.nan)
    all_tcev_params = np.full((n_pixels, 4), np.nan)
    all_scores = np.full(n_pixels, np.nan)
    all_converged = np.zeros(n_pixels, dtype=bool)
    log_interval = max(1, n_pixels // 10)

    for i in range(n_pixels):
        ev = float(event_observations[i]) if event_observations is not None else None
        result = process_single_pixel(pixel_histories[i], ev, min_segment)
        all_descriptors[i] = result["descriptors"]
        all_tcev_params[i] = result["tcev_params"]
        all_scores[i] = result["anomaly_score"]
        all_converged[i] = result["converged"]

        if show_progress and ((i + 1) % log_interval == 0 or i + 1 == n_pixels):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(
                f"  [{i + 1}/{n_pixels}] "
                f"{100 * (i + 1) / n_pixels:.0f}% done — "
                f"{rate:.1f} pixels/sec — "
                f"elapsed {elapsed:.1f}s"
            )

    elapsed = time.time() - t0
    n_ok = int(all_converged.sum())
    if show_progress:
        print(
            f"[batch_tcev] Finished: {n_ok}/{n_pixels} converged "
            f"in {elapsed:.1f}s ({n_pixels / elapsed:.1f} px/s)"
        )

    return {
        "descriptors": all_descriptors,
        "tcev_params": all_tcev_params,
        "anomaly_scores": all_scores,
        "converged": all_converged,
    }


if __name__ == "__main__":
    # Quick benchmark: 500 synthetic pixels, serial vs parallel
    rng = np.random.default_rng(42)
    N, T = 500, 60

    histories = rng.normal(-10.0, 1.5, size=(N, T))
    # Inject flood dips in first 100 pixels
    for i in range(100):
        n_dips = rng.integers(3, 8)
        idx = rng.choice(T, size=n_dips, replace=False)
        histories[i, idx] = rng.normal(-19.0, 1.5, size=n_dips)

    events = np.where(
        np.arange(N) < 100,
        rng.normal(-19.0, 1.5, size=N),
        rng.normal(-10.0, 1.0, size=N),
    )

    print("=" * 60)
    print("SERIAL MODE BENCHMARK")
    print("=" * 60)
    res_serial = batch_process_serial(histories, events)

    print(f"\nConverged: {res_serial['converged'].sum()}/{N}")
    flood_scores = res_serial["anomaly_scores"][:100]
    land_scores = res_serial["anomaly_scores"][100:]
    print(f"Mean anomaly score (flood pixels):  {np.nanmean(flood_scores):.4f}")
    print(f"Mean anomaly score (land pixels):   {np.nanmean(land_scores):.4f}")
    assert np.nanmean(flood_scores) > np.nanmean(land_scores), "Flood pixels should score higher"
    print("Serial sanity check PASSED.\n")

    print("=" * 60)
    print("PARALLEL MODE BENCHMARK (max_workers=4)")
    print("=" * 60)
    res_parallel = batch_process_pixels(histories, events, max_workers=4)

    print(f"\nConverged: {res_parallel['converged'].sum()}/{N}")
    flood_scores_p = res_parallel["anomaly_scores"][:100]
    land_scores_p = res_parallel["anomaly_scores"][100:]
    print(f"Mean anomaly score (flood pixels):  {np.nanmean(flood_scores_p):.4f}")
    print(f"Mean anomaly score (land pixels):   {np.nanmean(land_scores_p):.4f}")
    assert np.nanmean(flood_scores_p) > np.nanmean(land_scores_p), "Flood pixels should score higher"

    # Verify consistency
    assert np.allclose(
        res_serial["anomaly_scores"], res_parallel["anomaly_scores"], equal_nan=True
    ), "Serial and parallel results should match"
    print("Parallel sanity check PASSED — results match serial.\n")
    print("All batch_tcev benchmarks passed.")
