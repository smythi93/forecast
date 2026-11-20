# Evaluation results

The results of the paper's evaluation of FORECAST on SWE-bench Lite. See
[Evaluation](../../README.md#evaluation) in the top-level README for what was run,
how to reproduce it, and what each field means.

| File | Content |
|---|---|
| `results.json` | per instance the diagnosis, and per model, agent, arm and instance the repair; `evaluation/scripts/analyze.py` answers the research questions from it |
| `manifest.json` | per part of the Zenodo archives: files, raw bytes, forbidden-string hits and files with run dates (all none) |
| `SHA256SUMS` | checksums of the Zenodo archives `forecast-results.tar.zst` and `forecast-trees.tar.zst` (DOI: [10.5281/zenodo.23769792](https://doi.org/10.5281/zenodo.23769792)) |

All three are written by `evaluation/scripts/pack.py`.
