# News-history personalisation: code and source data

**Personalisation improves large language model alignment with human news preferences and brain representations**

This repository contains a compact source-data and code package for the manuscript. It covers platform click prediction, laboratory willingness-to-read prediction, brain–behaviour associations, and model–behaviour and model–brain representational similarity analyses (RSA).

The release corresponds to manuscript v8 and the final reorganised Figures 5 and 6. A, S and T denote article-only input, shuffled history and true history. The code and source data are intended for public access, not reviewer-restricted access.

Project repository: [llm-personalisation-news-brain-alignment](https://github.com/Neptune0108l/llm-personalisation-news-brain-alignment).

## Start here

- [Figure and table map](docs/figure_table_map.csv): each manuscript display linked to released data and code.
- [Data guide](docs/DATA_GUIDE.md) and [column inventory](docs/data_columns.csv).
- [Reproduction guide](docs/REPRODUCTION.md): commands and the exact scope of each entry point.
- [Prompt materials](prompts/README.md): eight English text blocks from the current Supplementary Methods.
- [Provenance](docs/source_manifest.csv): source checksums and recorded packaging transformations.
- [Rights and third-party sources](RIGHTS.md).

```text
code/          Saved-data analysis, plotting, reusable kernels and one upstream spatial script
config/        Recorded model order and inferential settings
data/          User/participant metrics, saved tests, source tables and group spatial maps
figures/       Current main and supplementary PDF/PNG figures; editable Figure 1
prompts/       English prompt templates, instruction excerpts and field structures
docs/          Data dictionary, figure mapping and reproduction scope
```

## Installation and validation

Use Python 3.10 or later. From the repository root:

```bash
python -m pip install -r requirements.txt
python code/verify_release.py
```

The verifier checks file hashes, Python syntax, table readability and required links. It does not run scientific analyses. Additional dependencies for fMRI arrays, decoding or LLM inference are listed in `requirements-optional.txt`; they are not needed to inspect the CSV source data.

## Reusing the saved results

```bash
# A compact visualisation from saved means and intervals (no statistics recalculated)
python code/plot_saved_summaries.py 03

# Re-render current Figures 5/6 from saved tables and fixed cortical artwork
python code/plot_figures_05_06.py

# Optional: recompute the adopted laboratory group tests from participant metrics
python code/analyse_saved_metrics.py laboratory
```

All generated files go to `outputs/`. The frozen files in `data/` and `figures/` remain unchanged. Source figures preserve the manuscript version; compact redraws are not claimed to match every typographic detail. See the reproduction guide before running upstream analyses.

## What is included

The package includes saved estimates and tests for all current main and supplementary figures and Tables S1–S3, user/participant-level derived metrics for the principal comparisons, frozen ROI resampling arrays, and group searchlight/decoding maps. Model–brain participant tables retain the three focal ROIs. The adopted nine-ROI brain–behaviour and mean-activity summaries remain available.

## External inputs and scope

This is a source-data release, not a complete raw-data and model-inference archive. Original MIND news text and click logs, individual fMRI images, model weights, full hidden-state arrays, API credentials and historical working files are not bundled. Original sources:

- MIND: [dataset site](https://msnews.github.io/) and [Wu et al. (2020)](https://doi.org/10.18653/v1/2020.acl-main.331).
- News-reading study: [Chan et al. (2023)](https://doi.org/10.1073/pnas.2313175120) and the original study’s [OSF record](https://doi.org/10.17605/OSF.IO/CAXFQ).

For the six API-accessed laboratory models, this compact release contains derived participant metrics, not a complete request-by-request replay archive. Figure 5a/Table S2 use the author-supplied updated summary: its reported interval endpoints are included, but the script constructing those updated intervals was not recovered. Do not substitute older participant files to claim exact reproduction of those endpoints.

The package was checked for file integrity and code syntax when prepared. Scientific analyses, full model inference and all plotting pipelines were not rerun as part of packaging. Preparation-time checks and dependency availability are documented in [validation status](docs/VALIDATION.md).

## Citation and licence

Please use [CITATION.cff](CITATION.cff) for this code package and cite the original datasets when using them. The manuscript publication DOI can be added after publication; none is invented here.

The original study-specific software in `code/` is released under the [MIT licence](LICENSE). MIT does not relicense the data, figures, original datasets or model weights; their status is described in [RIGHTS.md](RIGHTS.md).
