# Spatial Cross-Validation Reveals a Predictive Ceiling for Environmental Correction of GPS Error in Virtual-Fencing Collars — Analysis Code

Companion repository for [AUTHORS] (submitted to GPS Solutions). Every statistic, table, and figure in the paper is traceable to a script in this repository.

## Data

Input data are the public NoFence collar datasets of Versluijs et al. (2024), DataverseNO: https://doi.org/10.18710/TUCMOJ (CC0 1.0). The preparation scripts in `pipeline/` reconstruct the analysis file `ml_ready_stat.txt` (one row per GPS fix: station labels, error components delta_easting / delta_northing, and the predictor columns of Section 2.2.1) from the repository files.

## Repository layout

- `pipeline/` — data preparation (raw repository files → ml_ready_stat.txt)
- `scripts/` — all analysis and figure-generation scripts
- `results/` — text outputs backing every reported number
- `figures/` — final PNG figures as they appear in the paper

## Script → paper mapping

| Script | Produces | Paper location |
|---|---|---|
| pipeline/preprocess_v3.py | ml_ready_stat.txt (static dataset) | Sect. 2.1-2.2 |
| pipeline/preprocess_nofence_py.py | Mobile fusion; telemetry variables | Sect. 2.2.1, 3.1 |
| station_variance_explained.py | eta2 = 48.0% / 37.0% | Sect. 3.1 |
| analyse_circulaire_gps.py | Rayleigh & Watson-Williams tests; Fig. 8 | Sect. 3.5, Fig. 8 |
| make_figure4.py | Naive vs LOSO with bootstrap CIs; Fig. 4 | Sect. 3.2, Fig. 4 |
| wp1_wp2_uncertainty.py | Station-bootstrap CIs; 50-partition replication | Sect. 3.2, 3.4, Fig. 6b |
| wp1b_verify.py | Pooled vs fold-mean R2; RMSE-difference CI [+0.46,+1.39] | Sect. 3.2 |
| loao_validation.py | Leave-one-aspect-out (4 folds) | Sect. 3.3 |
| reconcile_permutation_features.py | Enrichment A-D under station-block null | Table 2, Fig. 6a |
| permutation_loo.py | Environment-only precision test (p = 0.005) | Table 2, Sect. 3.5 |
| permutations_10k.py | Row-level null (median -0.035): comparison evidence | Table 2 note |
| wp5_spatial_importance.py | LOSO donor-station feature reliance; Fig. 5b | Sect. 3.4, Fig. 5b |
| make_figure7.py | Six alternative target formulations; Fig. 7 | Sect. 3.5, Fig. 7 |
| ml_direction.py | Circular-direction model (87.9 vs 68.0 deg) | Sect. 3.5 (vi) |
| extended_features_loso.py | Extended 19-feature model (6.44 vs 6.50 m) | Sect. 3.6 |
| ml_improvement.py | RBF-SVR under LOSO (subsample N = 2000) | Sect. 3.7 |
| kriging_fixed.py | Linear Ridge (5.97 m) and Ordinary Kriging (6.54 m) under LOSO | Sect. 3.7 |
| tuning_3configs.py | Hyperparameter sensitivity (5.73-5.82 m) | Sect. 3.7 |
| feature_engineering_aggressive.py | Polynomial/interaction RF (-16.6%) | Sect. 3.7, H8 |
| satellite_geometry_proxy.py | HDOP/sqrt(nsat) proxy (3.8-8.0%) | Sect. 3.7, H5 |
| wp4_spatial_diagnostics.py | Distance-band Moran's I; directional variograms | Sect. 3.7 |
| variogram_range_morans_i_v3.py | Variogram ranges 1087 / 41 m; bootstrap CI [45, 2377] | Sect. 3.7 |
| quantile_regression.py | Quantile-forest coverage 78% vs 90% | Sect. 3.7, H9 |
| quiver_plot_mobile.py | Correction-vector plausibility map | Sect. 3.7 |
| make_figure11.py | Geofencing violations (72%, 2.8-fold); Fig. 11 | Sect. 3.8, Fig. 11 |
| wp6_adaptive_buffer.py | Adaptive vs uniform buffers | Table 3, Sect. 3.8, 4.5 |
| static_to_mobile_transfer_9features.py | Static-to-mobile transfer (13.67 vs 12.92 m) | Sect. 3.9 |
| mobile_to_mobile.py | Mobile-only model (11.24 vs 10.64 m) | Sect. 3.9 |
| static_imputation_sensitivity.py | Imputation sensitivity check | Sect. 2.4.3 |
| make_figure5b.py / make_figure6.py / make_figure_wp2.py / make_figure9.py / make_figure10.py | Remaining figure panels | Figs. 5b, 6, 9, 10 |

Figures 1-3 are schematic diagrams produced in a vector editor; their content is fully specified by their captions.

## Requirements

Python 3.8+; numpy, pandas, scikit-learn, scipy, matplotlib, scikit-gstat, pykrige. GLMM reproduction and circular statistics additionally use R (glmmTMB, circular), following the original pipeline of Versluijs et al. (2024).

## Runtimes

permutation_loo.py performs a complete 30-fold LOSO per permutation (~4 h at P = 1000). All other scripts complete within minutes on a standard desktop.
