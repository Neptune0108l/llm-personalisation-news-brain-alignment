# Reproduction scope and commands

Run commands from the repository root. They write under `outputs/`, which is excluded from Git. The package preparation did not run model inference, scientific reanalysis or redraw figures.

## 1. Inspect the released evidence

```bash
python code/verify_release.py
```

Frozen source values and current PDF/PNG figures cover every display in the manuscript. Figure 1 is a schematic with a PPTX source. Group maps are supplied in NIfTI; individual beta images are external.

## 2. Replot saved results

```bash
python code/plot_saved_summaries.py 02
python code/plot_saved_summaries.py 03
python code/plot_saved_summaries.py 04b
python code/plot_saved_summaries.py S03
python code/plot_saved_summaries.py S05
python code/plot_figures_05_06.py
python code/plot_figure_S04.py
```

The summary plotter draws compact layouts from existing estimates; it does not recreate exact manuscript typography. The Figures 5/6 and S4 scripts adapt the current source plotting logic. Figure 6a uses the current frozen cortical artwork. Rendering may differ with installed fonts; Arial is not redistributed. No figure command recalculates significance or interval endpoints.

For a volumetric display from saved significant group maps, obtain a matching MNI background image separately:

```bash
python code/plot_spatial_maps.py --contrast T_minus_S --setting raw --background /path/to/MNI_T1.nii.gz
```

This is a new view of saved maps, not an exact reconstruction of the manuscript’s surface/axial layout. Saved publication exports remain authoritative for presentation.

## 3. Recompute adopted group tests from derived metrics

```bash
python code/analyse_saved_metrics.py platform
python code/analyse_saved_metrics.py laboratory
python code/analyse_saved_metrics.py roi
python code/analyse_saved_metrics.py layers
python code/analyse_saved_metrics.py readout
python code/analyse_saved_metrics.py semantic
```

| Entry point | Calculation and family | Interval handling |
|---|---|---|
| `platform` | Paired t tests, 18 comparisons | Reads saved paired bootstrap limits; this command does not regenerate them |
| `laboratory` | Paired t tests, 32 comparisons; finite A/S/T observations per metric | 10,000 paired participant bootstrap draws, seed 20261015 |
| `roi` | Two models × three focal ROIs × two contrasts; Holm12 | Frozen Raw RSA bootstrap indices; 99,999 saved sign flips |
| `layers` | Two readouts × tested layers × three ROIs × two contrasts; Holm342 | Preserves saved pointwise limits; sign-test tie tolerance 1e−12 |
| `readout` | Layer-mean T−S last_token minus target_mean over layers 1–28; Holm3 | Frozen Raw RSA bootstrap indices/signs |
| `semantic` | Within-person adjusted-minus-Raw T−S; Holm3 | 10,000 paired bootstrap draws, seed 2026102304; saved semantic sign matrix |

These compact entry points retain the archived calculation definitions. They were syntax-checked, not scientifically rerun during packaging. Compare generated outputs with the full-precision reference tables before using a different environment for inference. Floating-point tie handling and library versions can matter at discrete permutation thresholds.

## 4. Reuse core operations with external inputs

`code/core/rsa.py` contains the archived rank correlation, SVD residualisation and layer-aggregation functions. `decoding.py` contains the original train-fold standardisation and leave-one-trial-out ridge procedures. `platform_metrics.py` implements nDCG@5/MAP/AUC with expected treatment of exact ties. `mind_history.py` contains history-label permutation and formatting helpers. `inference.py` contains local categorical scoring and hidden-state extraction; it uses local checkpoint files and offline model loading.

These are callable kernels, not a one-command recreation of the original data acquisition. The prompt files provide the current English methods material. API calls and credentials are intentionally absent.

## 5. Original T−S searchlight analysis with external arrays

`code/upstream/searchlight_analysis.py` retains the original computational body for the adopted Raw/Semantic control spatial analysis. Paths are configured by environment variables:

- `NEWS_BRAIN_ARRAYS`: `analysis_mask.nii.gz`, `<sub>_betas.npy` and `<sub>_predictors.npz`.
- `NEWS_MODEL_GRAMS`: `participants.json` and `qwen25_7b/<sub>_grams.npy`.
- `NEWS_RSA_PREPARED`: `data/prepared/qwen25_7b/<sub>.npz`, including semantic RDMs and trial identifiers.

Obtain or prepare these inputs from the original authorised data and model pipeline. They are not bundled in the compact public package. Run only after configuring those directories and installing optional dependencies:

```bash
python code/upstream/searchlight_analysis.py
```

The script performs a full spatial analysis and can be computationally expensive. It does not write to the external input directories. T−A group spatial outputs and null maxima are supplied, but a separate end-to-end T−A rerun entry point is not included.

## Known coverage limits

| Item | Included | External or unavailable |
|---|---|---|
| All current figures/tables | Source estimates, saved statistics and final figures | Exact typography may depend on fonts/earlier composition tools |
| Eight-model laboratory evaluation | Derived participant metrics and scoring instructions | Complete API request/output archives |
| Model–brain RSA | Three focal ROIs, both local models; adopted processing/readout data | Full hidden states and per-trial brain inputs |
| Nine-ROI brain–behaviour RSA | Current Figure 5a/Table S2 summary | Code constructing the updated intervals was not recovered |
| Spatial maps | Group maps, corrected voxel tables, null maxima and T−S source code | Large participant spatial arrays and individual beta maps |
| Raw-data-to-result replay | Core algorithms and known input schemas | Not claimed to be complete in this release |
