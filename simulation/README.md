# Run and review the hedging study

Execution status: the pilot, full E1–E3 grid, numerical-refinement runs, and saved-output audit are complete. All 324 cells passed the final audit; no paths were excluded. The author asked to finish the full procedure. The experiment uses newly simulated data and source-based reference parameters, with implementation choices frozen in [protocol.json](protocol.json) before pilot hedge outcomes.

## Start here

- [02_run_and_review.ipynb](02_run_and_review.ipynb): the shared review sequence, from one path and its cash ledger to the full comparison.
- [PROTOCOL.md](PROTOCOL.md): inputs, parameter provenance, contracts, accounting, uncertainty, numerical checks, and scope.
- [RESULTS_REPORT.md](RESULTS_REPORT.md): findings, validation evidence, and implications for the next writing stage.
- [results/](results/): compact tables, figures, ledgers, validation reports, and the main run manifest.
- [DEVELOPMENT_LOG.md](DEVELOPMENT_LOG.md): the initial numerical failure and the corrections made before the main study.

## What is being tested

| Experiment | Change | Contribution to the paper |
|---|---|---|
| E1 | Three hedge rules at reference inputs, daily trading, zero fees | Tests the reference comparison in the introduction and framework. |
| E2a–d | One-at-a-time changes in pricing kappa, theta, xi, and rho | Tests structural-input sensitivity while actual variance and BS reference-price histories remain fixed. |
| E3a–c | Fees and trading schedules crossed with all input scenarios | Tests the interaction of error, costs, and discrete rebalancing discussed in the literature review. |
| Contract comparisons | Three moneyness levels and two maturities | Shows where the same hedging comparison changes with contract conditions. |

The main design has 4,096 common paths, six contracts, nine input scenarios, three fee levels, and two schedules: 324 cells with three strategies each. The two primary paired contrasts produce 648 comparisons. The 128-path pilot and 512-path coupled refinement sample use separate seeds. There is no cross-contract pooling or assumption that Heston must win.

## Software and execution

The computation platform is the project-local Python environment. Jupyter provides the review interface; GitHub stores the source and compact artifacts. Software versions and assistance are documented in [SOFTWARE_AND_ASSISTANCE.md](SOFTWARE_AND_ASSISTANCE.md) and [requirements-lock.txt](requirements-lock.txt).

From the project root:

```bash
.venv/bin/python simulation/validate_accounting.py
.venv/bin/python simulation/validate_paths.py
.venv/bin/python simulation/validate_pricing.py
.venv/bin/python simulation/validate_pricing_stress.py
.venv/bin/python simulation/run_study.py pilot
.venv/bin/python simulation/run_study.py main
.venv/bin/python simulation/run_study.py convergence --substeps 8
.venv/bin/python simulation/run_study.py convergence --substeps 16
.venv/bin/python simulation/run_study.py summarize-convergence
.venv/bin/python simulation/audit_results.py
.venv/bin/python simulation/analyze_results.py
.venv/bin/python simulation/build_review_notebook.py
.venv/bin/python simulation/execute_review_notebook.py
.venv/bin/python -m jupyterlab simulation/02_run_and_review.ipynb
```

Completed runs are protected against accidental overwriting. Archive their corresponding `outputs/` directory before an intentional repeat. Raw paths, reference-price/IV histories, and pathwise errors remain in `outputs/`; those larger files and `.venv/` are excluded from Git. Every production run records its protocol, source hashes, seed, completion status, exclusions, and diagnostics. The compact `results/` artifacts are intended for version control. The initial [01_setup.ipynb](01_setup.ipynb) checks software only and is retained for environment reproduction.
