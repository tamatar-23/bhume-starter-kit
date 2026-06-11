# BhuMe Land-Boundary Correction Pipeline

This repository contains a robust, unsupervised, and regularized boundary correction pipeline that aligns cadastral plot geometries to satellite-derived field boundaries. It generalizes across different resolution landscapes and calibrates confidence to flag uncertain plots, protecting calibration and restraint.

---

## Technical Approach

### 1. Unsupervised Global Shift Estimation (Coarse-to-Fine)
Since ground-truth files (`example_truths.geojson`) are not available for hidden evaluation villages, the pipeline estimates the global translation vector $(dx_{global}, dy_{global})$ using the village plots themselves:
- **Sampling**: Randomly samples 100 plots from the cadastre.
- **Coarse Search**: Searches a wide window ($\pm 15$ meters with 3m steps) by rasterizing plot boundaries and cross-correlating them with the `boundaries.tif` raster.
- **Global prior**: A Gaussian prior centered at $(0, 0)$ with $\sigma_{global} = 15.0\,\text{m}$ is applied to avoid snapping to distant parallel crop-field boundary aliases.
- **Fine Search**: Searches a tight window ($\pm 2.5$ meters with 0.5m steps) around the coarse peak to refine the shift.

### 2. Local Cross-Correlation Alignment
For each plot, we start from the globally-shifted position and perform a local grid-search translation ($\pm 20$ meters):
- **Local Prior**: An L2 penalty (Gaussian weight) centered at the global shift with $\sigma_{local} = 8.0$ meters is applied. This prevents smaller plots (which suffer from the aperture problem and edge sparsity) from snapping to wrong parallel boundaries.

### 3. Confidence Modeling & Flagging
Confidence is calibrated in $[0, 1]$ using four factors:
1. **Edge Overlap (`norm_corr`)**: The fraction of the plot boundary that overlaps with the raster boundary lines.
2. **Ambiguity Penalty (`distinctness`)**: Peak-to-second-peak ratio outside an 8-meter exclusion zone.
3. **Displacement Penalty**: Scales down confidence if the local translation deviates significantly from the global shift.
4. **Area Discrepancy Penalty**: Penalizes plots where the map area differs from the recorded 7/12 area (cultivable + pot-kharaba) by >15%, which flags stale geometries.

Plots with confidence below the threshold (`0.35`) are `"flagged"` and retain their original official geometries. Plots above the threshold are `"corrected"`.

---

## Scorecard Results (vs Example Truths)

### Vadnerbhairav (Nashik)
- **Coverage**: 6 corrected, 0 flagged of 6 truths
- **Accuracy**: Median IoU = **0.868** (vs Official = 0.612, **Improvement = +0.236**, 100% of plots improved)
- **Accurate @ IoU $\ge$ 0.5**: 100%, median centroid err 5.2m
- **Calibration**: Spearman correlation = **0.83** (rank corr, -1 to 1)

### Malatavadi (Kolhapur)
- **Coverage**: 1 corrected, 2 flagged of 3 truths
- **Accuracy**: Median IoU = **0.783** (vs Official = 0.510, **Improvement = +0.273**, 100% of plots improved)
- **Accurate @ IoU $\ge$ 0.5**: 100%, median centroid err 3.4m
- **Calibration**: — (flat confidence $\rightarrow$ no signal)

---

## Execution Instructions

Initialize the virtual environment and install dependencies:
```bash
python -m venv .venv
.venv\Scripts\pip.exe install geopandas rasterio shapely numpy scipy pillow
```

Run the boundary correction pipeline on a village directory:
```bash
# Set PYTHONPATH to the root directory
$env:PYTHONPATH="."
.venv\Scripts\python.exe main.py data/Vadnerbhairav
.venv\Scripts\python.exe main.py data/Malatavadi
```

---

## Assumptions & Limitations
- **Coherent Shift Assumption**: We assume the cadastre georeferencing error is globally coherent with small local deviations. If the cadastre has non-linear shear or localized folding, a simple global translation prior may be insufficient.
- **Raster Sparsity**: If `boundaries.tif` has high false negatives (e.g. under dense tree canopy), the overlap is low and plots are flagged.
- **Small-Plot Sensitivity**: Small plots have low perimeter pixel counts and are highly susceptible to snapping to parallel agricultural rows. We mitigate this using a strict local regularization prior.

## Future Improvements
- **Multi-Modal Signal Integration**: Combine `boundaries.tif` with Canny edge detection on the RGB bands of `imagery.tif` to populate boundaries in canopy areas.
- **Adaptive Regularization**: Scale the local prior standard deviation $\sigma_{local}$ based on the plot area (tighter priors for smaller plots).
