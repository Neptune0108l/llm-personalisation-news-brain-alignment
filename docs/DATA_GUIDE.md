# Source-data guide

CSV files use UTF-8 (some copied originals include a UTF-8 BOM). Read them with `utf-8-sig` or `pandas.read_csv`. Empty cells/`NA` in source metrics retain the source missing-value representation. Do not replace undefined correlations with zero. All reported coefficients, confidence limits and P values retain the source precision.

## Conditions and direction

| Label | Meaning |
|---|---|
| `article_only`, A | Article-only input |
| `shuffled_history` or `shuffled`, S | Shuffled history; the task determines the shuffling procedure |
| `true_history`, T | True history |
| `raw` | Unadjusted RSA |
| `semantic_only` | Semantic control, partial correlation of ranked RDM entries |
| `gap1` | Exclude RDM pairs whose original trial indices differ by at most 1 |
| `voxel_centered` | Centre each voxel across trials before forming the brain RDM |

Ranking, correlation and RSA benefits use T minus the comparator. MAE benefits use the comparator minus T. Other explicit contrasts such as S−A and `(T−S)_Semantic control − (T−S)_Raw` retain their own signed definitions.

## Analysis units and files

`user_id` identifies a coded platform user; `sub` or `participant` identifies a coded laboratory participant. Shuffle repetitions and layers are repeated calculations within a person, not independent participants. Source tables retain denominators needed for reconstruction; the code applies the documented finite-value rules.

- `figure_02`: two local models, user-level ranking metrics, shuffle repeats and Holm18 tests.
- `figure_03`: eight-model participant metrics, common-finite condition summaries, paired benefits and Holm32 tests. `Figure3.csv` is the wide input table.
- `figure_04`: whole-brain decoding summaries, nine-ROI activity correlations, BH9 tests and group weight maps.
- `figure_05`: nine-ROI brain–behaviour summary, Qwen model–behaviour means/layers, participant endpoints and exploratory tests.
- `figure_06`: three-ROI model–brain absolute estimates, paired T−A/T−S tests, participant RSA and semantic-adjustment comparisons.
- `figure_S04`: layer/readout arrays, paired differences, Holm342 tests and direct readout Holm3 tests.
- `figure_S05`: the four adopted processing settings only, including saved means/intervals and the matching selected participant RSA records.
- `searchlight`: saved spatial tables, projections, group NIfTI maps and max-|t| null distributions for Figures 6a/S1/S2. Coordinates are in MNI millimetres. `significant_components.csv` describes connected components of a voxelwise-FWER mask; it is not cluster-level inference.
- `resampling`: losslessly retained bootstrap indices and sign vectors. `roi_rsa_indices.npz` contains `subjects`, `signs`, `bootstrap_000` and associated index fields. `semantic_control_signflips.npz` contains `signs`.
- `tables`: display exports of the current Supplementary Information, including the revised Table S3b analysis-level labels. These are rounded manuscript displays, not substitutes for full-precision analysis files.

The [column inventory](data_columns.csv) lists every CSV field, row count and missing-value count. The [figure map](figure_table_map.csv) assigns files to current manuscript panels. Some retained source filenames contain former panel letters; current placement is defined by that map, not by the old filename.

## Important field conventions

| Fields | Interpretation |
|---|---|
| `mean`, `rho_mean`, `mean_delta`, `mean_benefit` | Mean of the named independent-unit metric or paired contrast; consult the table’s condition/contrast fields |
| `ci_low`, `ci_high`, `ci95_low`, `ci95_high`, `bootstrap_ci95_low/high` | Saved interval endpoints; the figure map/reproduction guide distinguishes bootstrap, paired-t and unresolved-provenance intervals |
| `activation_behavior_r_sem` | SEM of participant activity correlations, not a confidence interval |
| `p_raw` | Uncorrected test P value |
| `p_holm`, `p_holm342` | Holm-adjusted P value within the stated family |
| `activation_behavior_r_p_bh9` | BH-FDR adjusted P for nine mean-activity ROIs |
| `p_FWER_joint` | Joint max-|t| FWER-adjusted voxel/centre P value across Raw and Semantic control |
| `layer`, `layer_set` | Model-layer index or averaging range; `last_token` layer 0 is undefined in the layer comparison |
| `pooling` | `last_token` or `target_mean` representation readout |
| `n_pairs`, `n_trials`, `n_voxels` | Within-participant data descriptors, not the group sampling unit |
| `cohen_dz` | A value already present in the archived statistical output; it is not added to the manuscript by this release |

The updated Figure 5a/Table S2 interval endpoints are distributed as supplied. Their construction code is not included because it was not recovered. Paired-t intervals in Table S3b and bootstrap intervals in Figure 5b/c must not be interchanged.
