"""Build a review notebook with saved, executed evidence (does not rerun the study)."""
from pathlib import Path
import nbformat as nbf
ROOT=Path(__file__).resolve().parents[1]
nb=nbf.v4.new_notebook()
md=nbf.v4.new_markdown_cell
code=nbf.v4.new_code_cell
nb.cells=[
md('''# Run and review the Heston hedging study

This notebook walks through the completed procedure using its saved outputs. The numerical design was fixed before hedge outcomes. Values are simulated research data, not observations taken from another paper.

The parameter source is [Poulsen, Schenk-Hoppé and Ewald (2009), Table 2](https://d-nb.info/1191914100/34). Contract choices, perturbations, fees, schedules, and path counts are this project's design choices. See `PROTOCOL.md` for the full rationale and scope.'''),
code('''from pathlib import Path
import json
import pandas as pd
from IPython.display import display, Image, Markdown
ROOT = Path.cwd()
if not (ROOT / 'simulation/protocol.json').exists():
    ROOT = ROOT.parent
P = json.loads((ROOT / 'simulation/protocol.json').read_text())
R = ROOT / 'simulation/results'
assert (R / 'complete.json').exists()
display(pd.DataFrame({'Market P': P['market'], 'Pricing Q': P['pricing']}))
display(pd.DataFrame([P['pilot'], P['main']], index=['pilot','main']))'''),
md('''## 1. Inspect one path and its holdings

The pilot uses 128 paths and a 63-day forward-ATM call. Path 0 was selected before inspecting its outcome. The cash plot and ledger below use 10 bps to expose every cost entry; the initial diagnostic comparison uses zero fees.'''),
code("display(Image(filename=str(R/'pilot_path_ledger.png')))"),
md('''## 2. Reconcile cash through settlement

`cash_before + interest - trade * spot - fee = cash_after`. At expiry, the stock holding is zero after liquidation. The terminal error is `cash_after - payoff`. Nominal cumulative fees are a diagnostic and are not subtracted again.'''),
code('''for strategy in ['BS','Heston','Adjusted']:
    ledger = pd.read_csv(R/f'ledger_{strategy}.csv')
    display(Markdown(f'**{strategy}: opening rows and final settlement**'))
    display(pd.concat([ledger.head(2),ledger.tail(2)]))'''),
md('''## 3. Check the small comparison and validation

The pilot estimates are diagnostic and use only 128 paths. They do not select the main grid, sample count, or seed.'''),
code('''pilot = pd.read_csv(R/'pilot_mse_summary.csv')
display(pilot.query('fee_bps == 0 and rebalance_days == 1')[['strategy','mse','mean_error','mean_fees','n_paths']])
for path in sorted((R/'validation').glob('*.json')):
    report = json.loads(path.read_text())
    print(path.name, report.get('status', report.get('passed', 'see report')))
print(json.dumps(json.loads((R/'complete.json').read_text()), indent=2))'''),
md('''## 4. Read E1: the reference-input comparison

These MSE estimates use 4,096 shared paths, daily trading, and zero fees. Units are currency squared per one call on one share. Compare strategies within a contract; contracts are not pooled.'''),
code("display(pd.read_csv(R/'e1_reference_mse.csv').round(4))"),
md('''## 5. Read E2 and E3 together

Each matrix covers parameter errors, fees, trading frequency, moneyness, and maturity. A negative difference favors the more complex rule in the named comparison. Gray means unresolved under the prespecified statistical and numerical-sensitivity checks; it does not mean equivalence.'''),
code("display(Image(filename=str(R/'Heston_minus_BS_grid.png')))\ndisplay(Image(filename=str(R/'Adjusted_minus_Heston_grid.png')))"),
md('''## 6. Inspect uncertainty, convergence, and failures

Pointwise and Bonferroni intervals estimate Monte Carlo uncertainty using paired squared-loss differences. They are approximate and require finite variance of those differences. The refinement expansion is a numerical-sensitivity diagnostic, not a simultaneous confidence bound on numerical bias. The 2,048-path check is nested inside the 4,096-path run.'''),
code('''contrasts = pd.read_csv(R/'contrasts_with_convergence.csv')
display(contrasts.query("scenario == 'reference' and fee_bps == 0 and rebalance_days == 1").round(5))
display(pd.read_csv(R/'sign_counts.csv'))
print('Pricing diagnostics:', json.dumps(json.loads((R/'pricing_diagnostics.json').read_text()), indent=2))
print('Path diagnostics:', json.dumps(json.loads((R/'path_diagnostics.json').read_text()), indent=2))'''),
md('''## 7. Reproduce deliberately

Run from the project root. Scripts refuse to overwrite a completed study; archive the matching output directory before an intentional rerun. `requirements-lock.txt` records package versions, and each manifest records seeds plus code and protocol hashes.

```bash
.venv/bin/python simulation/validate_paths.py
.venv/bin/python simulation/validate_accounting.py
.venv/bin/python simulation/validate_pricing.py
.venv/bin/python simulation/run_study.py pilot
.venv/bin/python simulation/run_study.py main
.venv/bin/python simulation/run_study.py convergence --substeps 8
.venv/bin/python simulation/run_study.py convergence --substeps 16
.venv/bin/python simulation/run_study.py summarize-convergence
.venv/bin/python simulation/analyze_results.py
```

`RESULTS_REPORT.md` explains the findings and writing implications. All conclusions remain conditional on the tested Heston regime, observed variance, selected errors, and accounting assumptions.''')]
nb.metadata={'kernelspec':{'display_name':'Python 3 (project .venv)','language':'python','name':'python3'},
             'language_info':{'name':'python','version':'3.12.14'}}
nbf.write(nb,ROOT/'simulation/02_run_and_review.ipynb')
print('Created simulation/02_run_and_review.ipynb')
