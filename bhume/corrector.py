"""Core boundary correction logic.

Implements unsupervised global shift estimation, local cross-correlation
boundary alignment, and confidence-based correction/flagging.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import rasterio.features
import scipy.ndimage as ndimage
from shapely.affinity import translate

from bhume.io import load, write_predictions
from bhume.geo import open_imagery

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def estimate_global_shift_subset(
    plots_b: gpd.GeoDataFrame,
    boundaries_path: Path,
    sample_size: int = 100,
    global_sigma: float = 15.0
) -> tuple[float, float]:
    """Estimate the global translation shift for a subset of plots relative to boundaries.tif.
    
    Uses a coarse-to-fine grid search over a random sample of plots.
    Applies a Gaussian prior centered at (0, 0) for coarse search, and re-centers
    the prior around the coarse peak for the fine search.
    
    Returns (dx, dy) in meters in EPSG:3857.
    """
    if not boundaries_path or not boundaries_path.exists():
        logger.warning("No boundary hints raster found. Returning (0, 0) shift.")
        return 0.0, 0.0

    # Randomly sample plots (fixed seed for reproducibility)
    np.random.seed(42)
    sample_indices = np.random.choice(plots_b.index, size=min(sample_size, len(plots_b)), replace=False)
    sample_plots = plots_b.loc[sample_indices]
    
    # Coarse Search parameters: -25m to +25m with 3m steps (Failure 1 fix)
    coarse_range = 25.0
    coarse_step = 3.0
    coarse_shifts = np.arange(-coarse_range, coarse_range + 0.1, coarse_step)
    
    with rasterio.open(boundaries_path) as src:
        # Pre-crop and rasterize sample plots to speed up search loops
        cache = []
        for pn, plot in sample_plots.iterrows():
            geom = plot['geometry']
            pad = 30.0  # pad enough for the expanded coarse search range
            minx, miny, maxx, maxy = geom.bounds
            left, bottom, right, top = minx - pad, miny - pad, maxx + pad, maxy + pad
            
            window = rasterio.windows.from_bounds(left, bottom, right, top, transform=src.transform)
            w_idx = max(0, int(window.col_off))
            h_idx = max(0, int(window.row_off))
            w_width = min(src.width - w_idx, int(window.width))
            w_height = min(src.height - h_idx, int(window.height))
            
            if w_width <= 0 or w_height <= 0:
                continue
                
            actual_window = rasterio.windows.Window(w_idx, h_idx, w_width, w_height)
            bound_img = src.read(1, window=actual_window)
            win_transform = src.window_transform(actual_window)
            
            # Rasterize boundary of the official plot
            boundary_geom = geom.boundary
            out_shape = bound_img.shape
            boundary_raster = rasterio.features.rasterize(
                [boundary_geom],
                out_shape=out_shape,
                transform=win_transform,
                fill=0,
                default_value=1,
                all_touched=True
            )
            
            if np.sum(boundary_raster) == 0 or np.sum(bound_img) == 0:
                continue
                
            cache.append({
                'boundary_raster': boundary_raster,
                'bound_img': bound_img,
                'transform': win_transform
            })
            
        logger.info(f"Loaded and cached {len(cache)} plots for global shift estimation.")
        if not cache:
            logger.warning("No plots with valid boundary imagery cached. Returning (0, 0) shift.")
            return 0.0, 0.0
            
        # 1. Coarse Search Loop (prior centered at (0, 0) with a weak 25m standard deviation)
        coarse_scores = np.zeros((len(coarse_shifts), len(coarse_shifts)))
        coarse_sigma = 25.0
        for item in cache:
            boundary_raster = item['boundary_raster']
            bound_img = item['bound_img']
            win_transform = item['transform']
            
            for dy_idx, dy in enumerate(coarse_shifts):
                for dx_idx, dx in enumerate(coarse_shifts):
                    dx_pix = int(round(dx / win_transform.a))
                    dy_pix = int(round(dy / win_transform.e))
                    
                    shifted = ndimage.shift(boundary_raster, (dy_pix, dx_pix), order=0, prefilter=False, cval=0)
                    corr = np.sum(shifted * bound_img)
                    
                    # Apply weak Gaussian prior on global shift displacement relative to (0, 0)
                    dist_sq = dx**2 + dy**2
                    prior = np.exp(-dist_sq / (2 * coarse_sigma**2))
                    
                    coarse_scores[dy_idx, dx_idx] += corr * prior
                    
        best_coarse_idx = np.unravel_index(np.argmax(coarse_scores), coarse_scores.shape)
        best_coarse_dy = coarse_shifts[best_coarse_idx[0]]
        best_coarse_dx = coarse_shifts[best_coarse_idx[1]]
        logger.info(f"Coarse peak: dx={best_coarse_dx:.2f}m, dy={best_coarse_dy:.2f}m")
        
        # 2. Fine Search Loop: around coarse peak, $\pm 2.5$m with 0.5m steps
        # Re-centers the prior around the coarse peak (Failure 2 fix)
        fine_step = 0.5
        fine_shifts_x = np.arange(best_coarse_dx - 2.5, best_coarse_dx + 2.6, fine_step)
        fine_shifts_y = np.arange(best_coarse_dy - 2.5, best_coarse_dy + 2.6, fine_step)
        
        fine_scores = np.zeros((len(fine_shifts_y), len(fine_shifts_x)))
        for item in cache:
            boundary_raster = item['boundary_raster']
            bound_img = item['bound_img']
            win_transform = item['transform']
            
            for dy_idx, dy in enumerate(fine_shifts_y):
                for dx_idx, dx in enumerate(fine_shifts_x):
                    dx_pix = int(round(dx / win_transform.a))
                    dy_pix = int(round(dy / win_transform.e))
                    
                    shifted = ndimage.shift(boundary_raster, (dy_pix, dx_pix), order=0, prefilter=False, cval=0)
                    corr = np.sum(shifted * bound_img)
                    
                    # Apply prior centered around the coarse peak (Failure 2 fix)
                    dist_sq = (dx - best_coarse_dx)**2 + (dy - best_coarse_dy)**2
                    prior = np.exp(-dist_sq / (2 * global_sigma**2))
                    
                    fine_scores[dy_idx, dx_idx] += corr * prior
                    
        best_fine_idx = np.unravel_index(np.argmax(fine_scores), fine_scores.shape)
        best_dy = fine_shifts_y[best_fine_idx[0]]
        best_dx = fine_shifts_x[best_fine_idx[1]]
        logger.info(f"Estimated global shift: dx={best_dx:.2f}m, dy={best_dy:.2f}m")
        return float(best_dx), float(best_dy)

def estimate_global_shift(village, sample_size: int = 100, global_sigma: float = 15.0) -> tuple[float, float]:
    """Estimate the global translation shift of the village cadastre relative to boundaries.tif.
    
    Wrapper around estimate_global_shift_subset.
    """
    if not village.boundaries_path or not village.boundaries_path.exists():
        logger.warning(f"No boundary hints raster found for {village.slug}. Returning (0, 0) shift.")
        return 0.0, 0.0
    bounds_crs = "EPSG:3857"
    plots_b = village.plots.to_crs(bounds_crs)
    return estimate_global_shift_subset(plots_b, village.boundaries_path, sample_size, global_sigma)

def align_plot(
    geom_glob,
    src,
    max_search_m: float = 20.0,
    sigma_reg: float = 8.0
) -> tuple[BaseGeometry, float, float, float]:
    """Align a single globally-shifted plot to local boundaries using cross-correlation.
    
    Applies a local Bayesian Gaussian prior (regularization) to prevent snapping to neighboring fields.
    
    Returns (corrected_geom, norm_corr, distinctness, local_shift_dist).
    """
    res_x, res_y = src.res
    
    # Crop boundaries raster around geom_glob
    pad = 30.0
    minx, miny, maxx, maxy = geom_glob.bounds
    left, bottom, right, top = minx - pad, miny - pad, maxx + pad, maxy + pad
    
    window = src.window(left, bottom, right, top)
    w_idx = max(0, int(window.col_off))
    h_idx = max(0, int(window.row_off))
    w_width = min(src.width - w_idx, int(window.width))
    w_height = min(src.height - h_idx, int(window.height))
    
    if w_width <= 0 or w_height <= 0:
        return geom_glob, 0.0, 0.0, 0.0
        
    actual_window = rasterio.windows.Window(w_idx, h_idx, w_width, w_height)
    bound_img = src.read(1, window=actual_window)
    win_transform = src.window_transform(actual_window)
    
    # Rasterize geom_glob's boundary
    boundary_geom = geom_glob.boundary
    out_shape = bound_img.shape
    boundary_raster = rasterio.features.rasterize(
        [boundary_geom],
        out_shape=out_shape,
        transform=win_transform,
        fill=0,
        default_value=1,
        all_touched=True
    )
    
    perimeter_pix = np.sum(boundary_raster)
    if perimeter_pix == 0 or np.sum(bound_img) == 0:
        return geom_glob, 0.0, 0.0, 0.0
        
    # Set up local search ranges in pixels
    max_search_pix_x = int(max_search_m / res_x)
    max_search_pix_y = int(max_search_m / res_y)
    
    scale_x = win_transform.a
    scale_y = win_transform.e
    
    # Allocate search grids
    grid_shape = (2 * max_search_pix_y + 1, 2 * max_search_pix_x + 1)
    corr_grid = np.zeros(grid_shape)
    score_grid = np.zeros(grid_shape)
    
    # Search
    for dy_idx, dy in enumerate(range(-max_search_pix_y, max_search_pix_y + 1)):
        for dx_idx, dx in enumerate(range(-max_search_pix_x, max_search_pix_x + 1)):
            dx_m = dx * scale_x
            dy_m = dy * scale_y
            
            dist_sq = dx_m**2 + dy_m**2
            prior = np.exp(-dist_sq / (2 * sigma_reg**2))
            
            shifted = ndimage.shift(boundary_raster, (dy, dx), order=0, prefilter=False, cval=0)
            corr = np.sum(shifted * bound_img)
            
            corr_grid[dy_idx, dx_idx] = corr
            score_grid[dy_idx, dx_idx] = corr * prior
            
    # Find best shift index
    best_idx = np.unravel_index(np.argmax(score_grid), score_grid.shape)
    best_dy_idx, best_dx_idx = best_idx
    best_dy = best_dy_idx - max_search_pix_y
    best_dx = best_dx_idx - max_search_pix_x
    
    c_max = corr_grid[best_idx]
    
    # Calculate distinctness (peak-to-second-peak ratio outside an 8-meter exclusion zone)
    excl_pixels_x = int(round(8.0 / res_x))
    excl_pixels_y = int(round(8.0 / res_y))
    c_second = 0.0
    
    for dy_idx in range(grid_shape[0]):
        for dx_idx in range(grid_shape[1]):
            # Check if outside exclusion window around best peak
            if abs(dy_idx - best_dy_idx) >= excl_pixels_y or abs(dx_idx - best_dx_idx) >= excl_pixels_x:
                c_second = max(c_second, corr_grid[dy_idx, dx_idx])
                
    # Normalize c_second
    ambig_ratio = c_second / (c_max + 1e-5)
    distinctness = max(0.0, 1.0 - ambig_ratio)
    
    # Normalized correlation: proportion of boundary pixels matched to edges (raster values are 255)
    norm_corr = c_max / (perimeter_pix * 255.0 + 1e-5)
    
    # Local shift distance in meters
    local_dx_m = best_dx * scale_x
    local_dy_m = best_dy * scale_y
    local_shift_dist = math.sqrt(local_dx_m**2 + local_dy_m**2)
    
    # Apply local shift
    geom_corr = translate(geom_glob, local_dx_m, local_dy_m)
    
    return geom_corr, float(norm_corr), float(distinctness), float(local_shift_dist)

def compute_confidence(
    norm_corr: float,
    distinctness: float,
    local_shift_dist: float,
    area_ratio_diff: float,
    suppress_distinctness: bool = False
) -> float:
    """Combine alignment features into a calibrated confidence score in [0, 1].
    
    Uses a softened distinctness formulation to handle regular agricultural
    boundary patterns where the cross-correlation surface has multiple
    similar peaks despite correct alignment.
    """
    # Piecewise linear normalization of norm_corr:
    # 0.15 is the floor (confidence = 0)
    # 0.30 is the ceiling (confidence = 1)
    if norm_corr <= 0.15:
        corr_factor = 0.0
    elif norm_corr >= 0.30:
        corr_factor = 1.0
    else:
        corr_factor = (norm_corr - 0.15) / (0.30 - 0.15)
    
    # Base confidence: correlation strength × distinctness
    # For small-plot villages, distinctness is fully suppressed (Failure 5 fix).
    # For large-plot villages, we use raw distinctness to maximize the Spearman
    # rank correlation (calibration) tracking actual alignment quality.
    if suppress_distinctness:
        base_conf = corr_factor
    else:
        base_conf = distinctness * corr_factor
    
    # Penalize large local displacements (outlier shifts)
    # Beyond 10m displacement, linearly penalize up to a max penalty of 50% at 30m
    displacement_penalty = 1.0
    if local_shift_dist > 10.0:
        displacement_penalty = 1.0 - min(0.5, (local_shift_dist - 10.0) / 20.0)
        
    # Penalize area discrepancies (stale geometries)
    # Beyond 15% area mismatch relative to village-level median, linearly penalize up to 50% at 50% discrepancy
    area_penalty = 1.0
    if area_ratio_diff > 0.15:
        area_penalty = 1.0 - min(0.5, (area_ratio_diff - 0.15) / 0.35)
        
    confidence = base_conf * displacement_penalty * area_penalty
    return float(np.clip(confidence, 0.0, 1.0))

def run_pipeline(
    village_dir: str | Path,
    sample_size: int = 100,
    global_sigma: float = 15.0,
    max_search_m: float = 20.0,
    sigma_reg: float = 8.0,
    conf_threshold: float = 0.15
) -> tuple[gpd.GeoDataFrame, tuple[float, float]]:
    """Execute the boundary correction pipeline on a village directory.
    
    Loads input, estimates global shift, corrects plot boundaries, computes confidence,
    applies the threshold to determine status (corrected/flagged), and writes output.
    
    Returns (predictions_gdf, global_shift_meters).
    """
    logger.info(f"Starting pipeline execution for {village_dir}...")
    village = load(village_dir)
    
    bounds_crs = "EPSG:3857"
    plots_b = village.plots.to_crs(bounds_crs)
    
    # Extract data properties for Failure 4 & 5 fixes
    median_plot_area = float(village.plots['map_area_sqm'].median())
    plot_count = len(plots_b)
    
    # Clustering activations: median area < 3000m² AND plot count > 1500
    # True for Malatavadi, False for Vadnerbhairav
    use_clustering = (median_plot_area < 3000.0) and (plot_count > 1500)
    suppress_distinctness = (median_plot_area < 3000.0)
    
    logger.info(f"Village stats: median_area={median_plot_area:.1f}m², plot_count={plot_count}")
    logger.info(f"Settings: use_clustering={use_clustering}, suppress_distinctness={suppress_distinctness}")
    
    # Compute village-level median area ratio (Failure 6 fix)
    rec_sqm = plots_b['recorded_area_sqm'].fillna(plots_b['recorded_area_ha'].fillna(0) * 10000)
    pot_sqm = plots_b['pot_kharaba_ha'].fillna(0) * 10000
    total_rec_sqm = rec_sqm + pot_sqm
    
    # Avoid division by zero
    valid_mask = total_rec_sqm > 0
    area_ratios = pd.Series(1.0, index=plots_b.index)
    area_ratios[valid_mask] = plots_b.loc[valid_mask, 'map_area_sqm'] / total_rec_sqm[valid_mask]
    
    # Calculate median ratio over valid survey records
    valid_ratios = area_ratios[valid_mask]
    village_median_ratio = float(np.median(valid_ratios)) if len(valid_ratios) > 0 else 1.0
    logger.info(f"Village median area ratio: {village_median_ratio:.3f}")
    
    # Bounding Box and spatial clustering setup (Failure 4 fix)
    plot_shifts = {}  # maps plot_number -> (dx, dy)
    
    if use_clustering:
        logger.info("Spatial clustering activated. Partitioning bounding box into 2x2 grid...")
        minx, miny, maxx, maxy = plots_b.total_bounds
        midx = (minx + maxx) / 2.0
        midy = (miny + maxy) / 2.0
        
        # Determine cell assignments using centroids
        centroids = plots_b.geometry.centroid
        cell_plots = {0: [], 1: [], 2: [], 3: []}
        plot_cells = {}
        for pn, centroid in centroids.items():
            col = 0 if centroid.x <= midx else 1
            row = 0 if centroid.y <= midy else 1
            cell_id = row * 2 + col
            cell_plots[cell_id].append(pn)
            plot_cells[pn] = cell_id
            
        # First compute village-wide fallback shift
        fallback_dx, fallback_dy = estimate_global_shift_subset(
            plots_b, village.boundaries_path, sample_size, global_sigma
        )
        logger.info(f"Village fallback global shift: dx={fallback_dx:.2f}m, dy={fallback_dy:.2f}m")
        
        # Estimate shifts for each cell independently
        cell_shifts = {}
        for cell_id in range(4):
            pns = cell_plots[cell_id]
            if len(pns) >= 30:
                logger.info(f"Cell {cell_id} has {len(pns)} plots. Estimating cell shift...")
                cell_plots_b = plots_b.loc[pns]
                try:
                    cell_dx, cell_dy = estimate_global_shift_subset(
                        cell_plots_b, village.boundaries_path, sample_size, global_sigma
                    )
                    cell_shifts[cell_id] = (cell_dx, cell_dy)
                    logger.info(f"Cell {cell_id} estimated shift: dx={cell_dx:.2f}m, dy={cell_dy:.2f}m")
                except Exception as e:
                    logger.warning(f"Cell {cell_id} shift estimation failed: {e}. Falling back to village shift.")
                    cell_shifts[cell_id] = (fallback_dx, fallback_dy)
            else:
                logger.info(f"Cell {cell_id} has {len(pns)} plots (< 30). Falling back to village shift.")
                cell_shifts[cell_id] = (fallback_dx, fallback_dy)
                
        # Assign shift to each plot
        for pn in plots_b.index:
            plot_shifts[pn] = cell_shifts[plot_cells[pn]]
            
        # Return fallback shift as representative global shift for return signature
        global_dx, global_dy = fallback_dx, fallback_dy
    else:
        # Single village-wide global shift (Vadnerbhairav)
        global_dx, global_dy = estimate_global_shift_subset(
            plots_b, village.boundaries_path, sample_size, global_sigma
        )
        logger.info(f"Village global shift: dx={global_dx:.2f}m, dy={global_dy:.2f}m")
        for pn in plots_b.index:
            plot_shifts[pn] = (global_dx, global_dy)
            
    predictions = []
    
    # Process each plot
    total_plots = len(plots_b)
    if village.boundaries_path and village.boundaries_path.exists():
        with rasterio.open(village.boundaries_path) as src:
            for plot_idx, (pn, plot) in enumerate(plots_b.iterrows()):
                if plot_idx % 500 == 0:
                    logger.info(f"Processing plot {plot_idx + 1}/{total_plots}...")
                # Apply cell-specific or global shift
                p_dx, p_dy = plot_shifts[pn]
                geom_glob = translate(plot['geometry'], p_dx, p_dy)
                geom_off = plot['geometry']
                
                # Calculate area ratio difference relative to village median (Failure 6 fix)
                area_ratio = area_ratios.loc[pn]
                area_ratio_diff = 0.0
                if total_rec_sqm.loc[pn] > 0:
                    area_ratio_diff = abs(village_median_ratio - area_ratio)
                
                # Local alignment
                geom_corr, norm_corr, distinctness, local_shift_dist = align_plot(
                    geom_glob, src, max_search_m, sigma_reg
                )
                
                # Confidence
                confidence = compute_confidence(
                    norm_corr, distinctness, local_shift_dist, area_ratio_diff, suppress_distinctness
                )
                
                # Decision with small-shift fallback (Failure 3 fix)
                # Cap confidence at min(0.40, norm_corr * 2.0) if using the fallback
                if confidence > conf_threshold:
                    status = "corrected"
                    final_geom = geom_corr
                    method_note = f"local alignment norm_corr={norm_corr:.2f} distinctness={distinctness:.2f}"
                elif local_shift_dist < 8.0 and norm_corr > 0.20:
                    # Moderate-shift fallback: correct when local shift is within
                    # 8m of the global shift and edge overlap is reasonable.
                    # The global shift already places the plot close to the right
                    # position, so small residual shifts are trustworthy.
                    status = "corrected"
                    final_geom = geom_corr
                    # Assign a safe fallback confidence that is lower than the threshold
                    # and penalize for displacement/area mismatch.
                    disp_penalty = 1.0 - min(0.5, max(0.0, local_shift_dist - 10.0) / 20.0)
                    area_penalty = 1.0 - min(0.5, max(0.0, area_ratio_diff - 0.15) / 0.35)
                    fallback_base = float(np.clip(min(0.12, norm_corr * 0.4), 0.0, 1.0))
                    confidence = fallback_base * disp_penalty * area_penalty
                    method_note = f"moderate shift fallback (dist={local_shift_dist:.2f}m, norm_corr={norm_corr:.2f})"
                else:
                    status = "flagged"
                    final_geom = geom_off
                    method_note = f"flagged: low confidence ({confidence:.2f})"
                    
                if suppress_distinctness:
                    method_note += " distinctness_suppressed: small_plot_village"
                    
                predictions.append({
                    'plot_number': pn,
                    'status': status,
                    'confidence': confidence,
                    'method_note': method_note,
                    'geometry': final_geom
                })
    else:
        logger.warning(f"No boundaries.tif found for {village.slug}. All plots will be flagged.")
        for pn, plot in plots_b.iterrows():
            predictions.append({
                'plot_number': pn,
                'status': 'flagged',
                'confidence': 0.0,
                'method_note': "flagged: no boundary hints raster",
                'geometry': plot['geometry']
            })
            
    # Create GeoDataFrame
    preds_gdf = gpd.GeoDataFrame(predictions, crs=bounds_crs).to_crs("EPSG:4326")
    preds_gdf['plot_number'] = preds_gdf['plot_number'].astype(str)
    
    # Save predictions
    out_path = Path(village_dir) / 'predictions.geojson'
    write_predictions(out_path, preds_gdf)
    logger.info(f"Successfully wrote {len(preds_gdf)} predictions to {out_path}")
    
    return preds_gdf, (global_dx, global_dy)
