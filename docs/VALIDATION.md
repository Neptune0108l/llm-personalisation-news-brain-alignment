# Validation status

Preparation date: 2026-10-28.

- 13 Python files parsed successfully without executing the analysis bodies.
- Copied reference data and figures match their source SHA-256 values. Selected CSV rows preserve original field strings. Semantic sign arrays are losslessly compressed.
- Eight English prompt blocks match the latest Supplementary Methods text.
- Table exports use current SI content, including the revised Table S3b analysis-level labels.
- No selected text file matched the configured private-key/token or local-machine-path patterns. This is a bounded packaging scan, not a formal security certification.
- Original project manuscripts, code, data, figures and configuration files remained unchanged (846 file hashes).
- `code/verify_release.py` checks the final release manifest, local links, Python syntax and CSV shapes.
- Command-line `--help` checks passed for the metric-analysis, compact-plotting and spatial-view entry points. Imports of the RSA, platform-metric, MIND-history and local-inference core modules passed without running their scientific functions.

Scientific analysis, LLM/API inference, optional neuroimaging imports and figure redraws were not run during packaging. No claim of complete raw-data-to-result reproduction is made. The package includes source evidence plus runnable/adapted entry points with the limits stated in REPRODUCTION.md. The packaging interpreter's installed packages are recorded in `packaging_environment.json`; optional dependencies are not all installed in that interpreter.
