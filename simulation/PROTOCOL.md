# Frozen simulation protocol

Version 2.0 was recorded before the pilot's hedge outcomes and the main runs. The author asked to complete the simulation procedure. The numerical grid is an implementation choice under that request; it is not a previously approved empirical calibration. `protocol.json` is the machine-readable authority.

## Data and contracts

The data are newly simulated stock and variance paths. The source paper supplies reference parameters, not observations to reuse as this project's dataset. [Poulsen, Schenk-Hoppé and Ewald (2009), Table 2](https://d-nb.info/1191914100/34) reports the parameter set used here and attributes the original estimates to Eraker (2004). This project checked the former source, not the original estimation independently.

The market law uses annual drift 0.10, variance mean reversion 4.75, long-run variance 0.0483, volatility of variance 0.550, and correlation −0.569. The option pricing law uses mean reversion 2.75 and long-run variance 0.0834, with the same volatility of variance and correlation. The stock starts at 100 and current variance at 0.0483. Interest is 0.04 and dividends are zero. A measure difference is not an imposed parameter error.

Six European calls combine 63 or 252 trading days with strike/initial-forward ratios 0.9, 1.0, and 1.1. The forward is 100 exp(0.04 T), and a year contains 252 trading days. These contract choices provide shorter/longer maturity and ITM/ATM/OTM comparisons without adding another research question.

The pilot uses 128 paths and a 63-day forward-ATM call. The main study uses 4,096 independent paths, shared across every contract and strategy. The 63-day contracts use the prefix of the one-year paths. Main and pilot seeds differ. Eight log-stock/full-truncation-variance substeps per trading day provide the initial production resolution. A separate 512-path sample compares eight and sixteen coupled substeps for every cell. The full-truncation choice follows the scheme discussed by [Lord, Koekkoek and van Dijk (2010)](https://repub.eur.nl/pub/18571); its suitability for these hedge outcomes is assessed separately. Negative latent Euler variance is retained internally for full truncation; the supplied variance is its nonnegative part. Boundary counts are reported.

## What changes and what stays fixed

E1 compares Black–Scholes IV delta, plain Heston delta, and correlation-adjusted Heston at reference inputs, daily trading, and zero fees. E2 adds eight one-at-a-time structural-error settings: ±20% in each of pricing kappa, theta, and xi, and ±0.10 in rho. Each erroneous vector persists over all dates and paths. The errors are selected sensitivity magnitudes, not a sample from an estimated error distribution.

E3 applies fees of 0, 5, or 10 basis points of traded stock value and rebalancing every one or five trading days to all nine input scenarios. These choices create 324 contract/scenario/trading cells and 648 primary paired contrasts. E1 and E2 are subsets of this grid. The two Heston rules receive the same structural vector and variance. The adjusted rule adds rho times xi times the derivative with respect to variance divided by spot to plain delta; a derivative with respect to volatility cannot be substituted.

At each daily date before maturity, reference-Q option prices are calculated from the current simulated spot and variance. Their contract-specific implied volatilities supply the Black–Scholes benchmark, using current information only. The entire realized reference-price and IV history is calculated once per contract and reused across structural-error settings. The Heston variance history is also fixed. Weekly trading samples these daily input histories, so changing the trading interval does not change information-update dates.

## Portfolio accounting

Each contract has one initial endowment: its reference-Q time-zero option price. Every rule pays opening stock costs from that endowment. Cash earns or pays continuously compounded interest, including negative balances. A trade costs fee_bps/10,000 times absolute shares traded times current spot. At maturity, stock is liquidated with the same proportional charge and the call payoff is deducted once. The terminal error is remaining cash; positive values are surplus and negative values are shortfall. There are no additional funding, dividend, lot-size, or position constraints.

MSE is measured in currency squared per call on one share. No contracts are pooled or normalized. The difference plain-minus-BS measures their combined specification/input comparison. Adjusted-minus-plain isolates implementation of the correlation correction at identical inputs, including its trading costs.

## Uncertainty and validation

For each contrast, form the difference of squared errors on each common path, then estimate its mean and standard error. Approximate Student-t 95% pointwise intervals aid estimation. Bonferroni intervals cover the prespecified family of 648 contrasts at approximate 95% familywise confidence under the IID-path and finite-variance assumptions for squared-loss differences. Finite fourth moments of hedge errors are sufficient for the latter condition; finite MSE alone is not. Their validity is asymptotic; Monte Carlo intervals do not cover model misspecification or numerical bias. An interval containing zero is unresolved, not evidence of equivalence.

The first 2,048 and all 4,096 paths and eight consecutive 512-path batches are reported as prespecified precision diagnostics. Squared-loss concentration helps reveal dependence on extreme paths. These checks do not set a stopping rule or establish tail-moment existence.

Pricing is checked against QuantLib and Greeks against finite differences. Implied volatilities are checked by repricing; numerical boundary conventions and fallback counts are recorded. Deterministic ledger examples check funding, opening charges, rebalance charges, liquidation, and payoff accounting. The path generator is checked against analytic moments and coupled time refinements. The production loss differences are compared with their 16-substep counterparts on a separate coupled sample. A main sign is labeled stable under this check only if its simultaneous interval still excludes zero after expansion by the absolute refinement shift plus 1.96 refinement standard errors. This is a diagnostic, not a proof that discretization bias is bounded.

Nonfinite results stop the run. There are no planned path exclusions. Code/configuration hashes, package versions, seeds, paths, reference histories, pathwise errors, summaries, checks, and logs are retained. Any implementation correction after the protocol freeze is recorded; results from an incorrect version are superseded rather than merged.

## Scope of the eventual writing

The merged manuscript Section 4 reports inputs, data generation, experimental controls, accounting, estimation, and validation. Section 5 reports the actual run summaries and distinguishes Monte Carlo uncertainty from numerical checks. Appendix A records detailed numerical conventions and reproducibility artifacts. This editorial organization leaves the frozen numerical protocol unchanged. The discussion can interpret differences within this single Heston regime with observed variance. It cannot establish empirical performance, a calibration-error distribution, individual identification of each Heston feature, or total implementation value.
