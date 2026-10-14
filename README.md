# Calibration Transfer and Updating in Educational Risk Prediction: A Cross-Cohort Reliability Audit

This repository accompanies the manuscript and provides analysis code, frozen aggregate evidence, and reproduction instructions. The paper compares historical calibration transfer, retrospective target out-of-fold recalibration, and updating evaluated on later learners.

## Start here

The aggregate evidence is distributed as **`Frozen_Results.zip`** to keep every repository file below 25 MB. Keep this archive compressed when uploading the repository to GitHub. After cloning or downloading the repository, extract it locally before any reproduction command:

```bash
python -m zipfile -e Frozen_Results.zip .
```

This restores `frozen/aggregate_results.json` (170 checksummed CSV/JSON aggregate payloads) and `frozen/evidence_registry.json`. It does not download or distribute raw learner records. The archive is byte-identical to the frozen-results archive identified in the supplementary material.

Use Python 3.12 in an isolated environment:

```bash
python -m venv .venv
# Windows PowerShell: .venv/Scripts/Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce.py quick --run-dir ../calibration-quick
python -B tests/test_runtime_integrity.py -v
```

Run commands from the repository root. Choose a new `--run-dir` outside the repository; quote paths containing spaces. The default limit is four threads and 4 GB for DuckDB, configurable through `--threads` and `--memory`. Full preparation and fitting require considerably more disk space than this download. All individual-level runtime outputs stay in the selected private run directory.

## What each workflow verifies

| Workflow | Inputs | Scope |
|---|---|---|
| `quick` | Bundled aggregate evidence | Recomputes arithmetic, summaries, course-cluster intervals, numeric tables and figures; no model fitting |
| `prepare` + `run --smoke` | Original dataset archives | Rebuilds features and fits one real OULAD and one real XuetangX job; not the full experiment |
| `prepare` + `run` | Original dataset archives | Runs the OULAD and initial XuetangX primary, control and sensitivity workflows |
| `prepare` + `review3` | Original XuetangX archive | Runs the baseline, recent-active and joint-assignment two-/three-presentation workflows used in the final manuscript |

The complete manuscript evidence requires both the `run` and `review3` workflows. `review3` can also be used directly after XuetangX preparation without first executing the other analyses. Command and directory names such as `review3` are retained implementation identifiers for traceability; they do not denote external peer-review status.

Quick verification exports all 170 payloads, repeats the three populations' 100,000-draw uncertainty summaries and 288 population-specific update comparisons, and regenerates current numeric tables and figures. Its report is `quick_verification.json`, with the population-specific checks in `r3_verification.json`. The current common-support Figure 5 is in `outputs/manuscript_r3/figures` under the private run directory; earlier plot variants in other output folders are not the final manuscript panel.

## Obtain the original datasets

- **OULAD:** <https://research.stem.open.ac.uk/ouanalyse/dataset/>. Use the original ZIP containing `OULAD.names` and the seven CSV files at its root, subject to the publisher's terms.
- **XuetangX MOOC Dataset:** <https://www.kaggle.com/datasets/samyakjhaveri/mooc-final>. The expected archive contains `train/train.csv` and `test/test.csv`. Acquire it through the provider and comply with its terms.

The entry point reads locally supplied archives and records their hashes. It does not download private data or bypass access requirements. Full XuetangX preparation checks the expected 247 presentations, 225,642 enrollments and 42,110,402 events. The supplied archival label's generation window is unverified; the explicitly timed seven-day logged-inactivity endpoint is a separate analysis, not authenticated dropout or completion.

## Full reproduction

```bash
python reproduce.py prepare --run-dir ../calibration-full --dataset both --oulad-zip /path/OULAD.zip --xuetangx-zip /path/mooc-final.zip
python reproduce.py run --run-dir ../calibration-full --dataset both
python reproduce.py review3 --run-dir ../calibration-full --dataset xuetangx
```

For only the time-defined population and later-presentation analyses:

```bash
python reproduce.py prepare --run-dir ../calibration-populations --dataset xuetangx --xuetangx-zip /path/mooc-final.zip
python reproduce.py review3 --run-dir ../calibration-populations --dataset xuetangx
```

For real-data smoke checks, use another fresh directory:

```bash
python reproduce.py prepare --run-dir ../calibration-smoke --dataset both --oulad-zip /path/OULAD.zip --xuetangx-zip /path/mooc-final.zip --smoke
python reproduce.py run --run-dir ../calibration-smoke --dataset both --smoke
```

The XuetangX smoke uses two real MITx `6_00_1x` offerings. Its preparation still scans the original archive. Quick, smoke and full runtimes cannot be mixed. Changed source/configuration, changed raw archives (including changes at the same path), or copied runtime source changes are rejected before continuation. Documentation and paper updates do not change the scientific cache key.

After quick-mode export and an OULAD rerun, compare the final estimates:

```bash
python work/oulad_experiment/verify_rebuild.py --regenerated ../calibration-full/outputs/oulad_experiment3 --reference ../calibration-quick/outputs/oulad_experiment3 --output ../calibration-full/oulad_rebuild_comparison.json
```

OULAD reproduction rebuilds the final 80 base-model jobs and final decomposition, not discarded pilot searches. The population workflow includes the separate million-draw stability check for four tail signals; those draws do not replace the primary 100,000-draw intervals.

## Interpretation and validation boundaries

- `G_S` is source calibration gain, `G_T` is unchanged historical transfer gain, `G_T_star` is retrospective target OOF gain, and `A = G_T_star - G_T`. Target OOF and fitted target-event-rate baselines require target outcomes.
- Positive retrospective adaptation gaps do not guarantee positive later updating utility `U`. All six full-budget full-Platt intervals cross zero after within-population six-comparison adjustment in each population.
- The three populations overlap. Recent-active Day 7 duplicates baseline on supported jobs. Joint user assignment covers the eligible 81-pair structural pool, not every platform course. Three favorable recent-active shrinkage signals and one joint small-budget harm signal remain exploratory; each population has its own 96-comparison family.
- Course-cluster bootstrap inference conditions on already fitted models. It does not incorporate full-pipeline refitting uncertainty or resolve unverified label-generation and live-log-arrival timing.
- Author-run records include fresh raw preparation and 7,610 base-model fits for the time-defined XuetangX two-/three-presentation workflows. This is another execution of the same algorithms, not an independent implementation or team validation; it does not newly reproduce every OULAD and historical extension. Quick mode does not repeat those fits.

See `docs/VALIDATION.json` for dated records and their individual scopes. Frozen scientific records retain their original identifiers and numerical contents. The present distribution updates documentation and paper files; no new raw-data experiment is claimed.

## Files and provenance

- `paper/main.pdf`, `paper/supplement.pdf`: the current manuscript and supplement, with two school-level affiliations and Li Yu as corresponding author.
- `paper/LaTeX_Source.zip`: main-manuscript source; select `main.tex` and compile with pdfLaTeX. It includes the matching compiled main PDF and excludes supplementary material. The local source/PDF build uses PDF 1.4. This is not a claim of acceptance by any submission platform.
- `paper/Figure1_Editable.pptx`, `paper/figure1.pdf`: editable native PowerPoint diagram and vector export; see `docs/figure1_provenance.txt`.
- `docs/artifact_map.csv`, `docs/feature_schema.csv`, `docs/protocol_and_seeds.md`: result-to-code mapping, input definitions and protocols.
- `docs/author_information.csv`, `CITATION.cff`: the five authors and the two school-level affiliations.
- `Frozen_Results.zip`: unchanged aggregate evidence; extract locally before reproduction, but keep compressed for GitHub browser uploads.
- `MANIFEST.json`: checksums and sizes for the distributed files, plus hashes of the frozen contents.

The code and datasets have separate rights. Third-party notices, including the DFMC adapter's MIT notice, are retained. No new repository-wide license grant is added; see `docs/PORTABILITY_AND_LICENSES.md`. Raw logs, learner identifiers, individual predictions, environments, credentials and author-submitted forms are excluded.

This package does not create a remote commit or release. Cite the actual GitHub commit after uploading rather than treating a previous commit as the identity of this distribution. When updating through GitHub's website, extract the outer delivery ZIP and upload its files and folders at the repository root. Keep the nested `Frozen_Results.zip` and `paper/LaTeX_Source.zip` compressed.
