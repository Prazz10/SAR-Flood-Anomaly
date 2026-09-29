# TFAM reproduction — SAR Flood Anomaly Mapping (TCEV + TabPFN)

Reproduction of the core method from:

> Ta, L.; Liu, Q.; Yu, C.; Valdes-Abellan, J. "SAR Flood Anomaly Mapping
> Through Statistical Time-Series Feature Classification and Pixel-Wise
> TCEV Modeling." *Remote Sens.* 2026, 18, 3323.
> https://doi.org/10.3390/rs18193323
> Authors' own repo (too new to be indexed yet):
> https://github.com/Liangyu2021/Temporal-Flood-Anomaly-Maps-TFAMs

## What's implemented and validated (this sandbox, synthetic data only)

| Module | Paper section | Status |
|---|---|---|
| `features.py` | 3.2, eqs. 1-6 | Sign inversion, empirical CDF, Gumbel transform, piecewise-linear breakpoint fit, C1-C9 descriptors |
| `tcev.py` | 3.2, eq. 7; 3.3, eq. 9 | Nonlinear least-squares TCEV fit (bounded), event anomaly score |
| `classify.py` | 3.3, eq. 8 | TabPFN wrapper (**not runnable here** — see Setup) |
| `tfam.py` | 3.3 (final paragraphs), eq. 14 | TFAM combination, binary threshold segmentation |
| `metrics.py` | 3.4, eqs. 10-17 | ROC-AUC, AP, threshold sweep, F1/IoU/Kappa, optimal-threshold report |

`test_synthetic.py` (single-pixel) and `test_grid.py` (500-pixel synthetic
scene) both pass: flood events score meaningfully higher than non-flood
events/pixels, and the full metrics report recovers perfect discrimination
on an easy synthetic case.

## Honest caveats — read before trusting numbers

1. **C8 and C9 are reconstructed, not quoted.** The paper describes these
   in prose ("temporal variation amplitude," "mean of the upper 10% SAR
   responses") without a closed-form equation. I implemented the most
   literal reading (range of the transformed series; mean of its top 10%).
   If you can get the authors' actual code, diff this against it.
2. **C6/C7 (breakpoint descriptors)** are the (x, y) coordinates of the
   fitted transition point. The paper doesn't specify whether "breakpoint"
   means an index, a value, or both — same caveat as above.
3. **TCEV fitting is fragile with few extreme samples** — in the synthetic
   test, the second Gumbel component's slope collapsed toward the lower
   bound when only ~5 of 60 historical points were "flood-like." This
   mirrors a limitation the paper itself names in Section 5 ("the limited
   number of observations may affect the stability of the four-parameter
   TCEV fitting") — it's a property of the method, not a bug in this code,
   but it means per-pixel TCEV fits deserve a sanity check on real data
   (e.g. flag pixels where the optimizer hits a bound).
4. **TabPFN could not be executed in this sandbox.** It installed fine, but
   its pretrained weights are gated on HuggingFace
   (`Prior-Labs/tabpfn_3_5`), and `huggingface.co` isn't reachable from
   here. `classify.py` is written to the paper's exact configuration
   (`n_estimators=8, random_state=0, device='cpu'`, threshold 0.5) but is
   untested end-to-end — you'll need to run it yourself (see Setup).
5. **No real Sentinel-1 data has touched this code yet.** Everything above
   is validated on synthetic per-pixel series I constructed to have the
   right qualitative shape (mostly-normal background + occasional low-dB
   "flood" dips). Real VH backscatter time series will have autocorrelation,
   speckle, and seasonal structure that synthetic Gaussian noise doesn't
   capture — the piecewise-fit breakpoint search in particular should be
   re-checked against real data (e.g. does it ever pick a degenerate 5- or
   6-point segment at the edge?).

## Setup needed on your machine (not doable in this sandbox)

### 1. TabPFN — accept the model license
```bash
pip install tabpfn
# Visit https://huggingface.co/Prior-Labs/tabpfn_3_5, accept terms, then:
huggingface-cli login   # or: export HF_TOKEN=...
```

### 2. Google Earth Engine — get access + authenticate
- Sign up at https://earthengine.google.com/ (free for research/education;
  needs a Google Cloud project with the Earth Engine API enabled)
- `pip install earthengine-api`
- `earthengine authenticate` (opens a browser OAuth flow) — or use a
  service account for headless/server use
- Verify with:
  ```python
  import ee
  ee.Initialize(project='your-gcp-project-id')
  print(ee.ImageCollection('COPERNICUS/S1_GRD').first().getInfo())
  ```

## Suggested next steps (in order)

1. **Pick a flood event to reproduce or a new one of your own.** The
   paper's own reference data (Córdoba/EMSR865, Zagora/EMSR692, New South
   Wales/EMSR570) come from Copernicus EMS
   (https://mapping.emergency.copernicus.eu/) — free to download. A new
   region needs you to source your own reference flood extent (EMS
   activations cover most large recent floods) or fall back to the
   Guangxi-style indirect hydro-meteorological check (GPM + ERA5-Land).
2. **Write the GEE ingestion script**: pull `COPERNICUS/S1_GRD`, filter to
   VH polarization + one orbit direction + the seasonal window, mosaic
   same-date scenes, clip to the study area, export to a stack (GeoTIFF
   time series or an HDF5 cube like the paper uses).
3. **Vectorize `features.py`/`tcev.py` over a real pixel grid** — right now
   they operate on one 1-D time series at a time; for a real scene you'll
   want to batch this (e.g. `numpy` vectorized breakpoint search, or
   multiprocessing across pixel chunks) since a 30 m grid over even a
   modest study area is hundreds of thousands of pixels × dozens of
   historical dates.
4. **Hand-label ~80-100 context points** on a high-res optical image over
   your chosen region (the paper used Planet imagery; Sentinel-2 or
   Google/Bing basemap imagery works as a free substitute) for the TabPFN
   context set.
5. **Run the full pipeline, evaluate against Table-3-style metrics**, and
   compare against the paper's ballpark numbers (ROC-AUC 0.76-0.87, F1
   0.55-0.81) as a sanity check — not to match exactly, since your region/
   event and reference data will differ.

## File layout
```
tfam/
  features.py       # eqs 1-6: CDF, Gumbel transform, piecewise fit, C1-C9
  tcev.py           # eq 7, 9: TCEV fit + event anomaly score
  classify.py       # eq 8: TabPFN wrapper (needs HF access to run)
  tfam.py           # TFAM combination + binary thresholding (eq 14)
  metrics.py        # eqs 10-17: AUC, AP, F1, IoU, Kappa, threshold sweep
  test_synthetic.py # single-pixel validation (land vs flood-prone)
  test_grid.py      # 500-pixel synthetic scene, full metrics report
```
