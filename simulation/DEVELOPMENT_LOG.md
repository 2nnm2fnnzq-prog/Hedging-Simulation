# Implementation and execution record

The author requested completion of the staged procedure. The numerical design in protocol version 2.0 was recorded before pilot hedge outcomes or main-grid results. The pilot, main study, and numerical-refinement sample use separate fixed seeds; this log records implementation corrections without changing the research grid in response to results.

1. Path and accounting modules were implemented independently. Eight path checks and 22 deterministic accounting checks passed. A separate review checked common inputs, array dimensions, the 648-contrast family, and the treatment of costs.
2. The first 128-path pilot attempt stopped before hedge outcomes because QuantLib's adaptive fallback reached its integration limit while computing a variance derivative at low variance. No path was excluded or substituted. The partial manifest, paths, and error log are retained locally at `outputs/development/pilot_failed_01/`. This was a numerical implementation failure, not an experiment result.

Subsequent validated execution is recorded in the completed-run manifests and the results report. Changes to the estimator, parameter grid, or sample count would require an explicit protocol amendment, not silent replacement of this record.

3. A second adaptive QuantLib control-variate calculation with an explicit absolute tolerance was added to recover integration failures when necessary. The original 128 pilot paths then completed without exclusions. Pricing checks covered 900 states across all nine structural scenarios; prices, spot derivatives, variance derivatives, IV inversion, and the constant-variance limit passed their declared tolerances.
4. Before the main run, a checked 180-node Fourier calculation was added between the initial quadrature disagreement and the slower scalar fallback. Acceptance uses the same price/Greek tolerances. The same pilot was rerun: the largest change in any terminal error was 1.254e-7 currency units. The earlier successful pilot remains at `outputs/development/pilot_before_pricing_refinement/`. Contract computations run in three separate processes; paths and scientific settings are unchanged.

5. The main grid completed with 4,096 paths and no exclusions. Both 512-path coupled refinement runs completed. The saved-output audit verified 972 strategy summaries and 648 primary contrasts for each production grid. The review notebook executed its seven code cells in order through IPython, retaining their actual outputs; no research outcomes were rerun for notebook display.
