"""Create auditable tables and publication-style figures from completed runs."""
from pathlib import Path
import json
import os
import shutil
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'outputs/matplotlib-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

OUT=ROOT/'simulation/results'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                     'axes.spines.right':False,'savefig.dpi':180})
COLORS={'BS':'#0072B2','Heston':'#D55E00','Adjusted':'#009E73'}


def save(fig,name):
    fig.savefig(OUT/f'{name}.png',bbox_inches='tight')
    fig.savefig(OUT/f'{name}.pdf',bbox_inches='tight')
    plt.close(fig)


def pilot():
    folder=ROOT/'outputs/pilot/d63_m1.0'
    p=pd.read_csv(folder/'first_path_and_targets.csv')
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    axes[0,0].plot(p.day,p.stock,color='#0072B2'); axes[0,0].set(ylabel='Stock price',xlabel='Trading day')
    axes[0,1].plot(p.day,p.variance,color='#CC79A7'); axes[0,1].set(ylabel='Annual variance',xlabel='Trading day')
    for name,color in COLORS.items():
        axes[1,0].plot(p.day[:-1],p[name][:-1],label=name,color=color)
        ledger=pd.read_csv(folder/f'ledger_{name}.csv')
        axes[1,1].plot(ledger.day,ledger.cash_after,label=name,color=color)
    axes[1,0].set(ylabel='Target shares before expiry',xlabel='Trading day')
    axes[1,1].set(ylabel='Cash after interest and trades',xlabel='Trading day')
    axes[1,0].legend(frameon=False)
    fig.suptitle('Pilot path 0: 63-day forward-ATM call\nCash ledger uses daily trading and 10 bps, including opening and liquidation')
    save(fig,'pilot_path_ledger')
    for f in folder.glob('ledger_*.csv'):
        shutil.copy2(f,OUT/f.name)
    shutil.copy2(folder/'first_path_and_targets.csv',OUT/'pilot_first_path.csv')
    shutil.copy2(ROOT/'outputs/pilot/mse_summary.csv',OUT/'pilot_mse_summary.csv')


def main_plots():
    mse=pd.read_csv(ROOT/'outputs/main/mse_summary.csv')
    contrasts=pd.read_csv(ROOT/'outputs/main/contrasts_with_convergence.csv')
    config=json.loads((ROOT/'simulation/protocol.json').read_text())
    reference=mse.query("scenario == 'reference'")
    for freq,label in [(1,'daily'),(5,'weekly')]:
        fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained',sharex=True)
        for ri,days in enumerate(config['maturity_days']):
            for ci,ratio in enumerate(config['strike_over_forward']):
                ax=axes[ri,ci]
                subset=reference.query('maturity_days == @days and strike_over_forward == @ratio and rebalance_days == @freq')
                for strategy,color in COLORS.items():
                    d=subset[subset.strategy==strategy].sort_values('fee_bps')
                    ax.plot(d.fee_bps,d.mse,'o-',color=color,label=strategy)
                ax.set(title=f'{days} days; strike/forward {ratio:.1f}',xticks=[0,5,10],ylabel='Terminal MSE (currency²)',xlabel='Fee (bps of stock traded)')
                ax.grid(alpha=.15)
        axes[0,0].legend(frameon=False)
        fig.suptitle(f'Reference inputs, {label} rebalancing; 4,096 common paths\nPoint estimates; paired uncertainty is reported separately')
        save(fig,f'reference_{label}_mse')
    labels={'Heston_minus_BS':'Plain Heston minus Black–Scholes','Adjusted_minus_Heston':'Adjusted Heston minus plain Heston'}
    scenarios=[x['id'] for x in config['scenarios']]
    row_labels=['Reference','κ −20%','κ +20%','θ −20%','θ +20%','ξ −20%','ξ +20%','ρ −0.10','ρ +0.10']
    columns=[(1,0),(1,5),(1,10),(5,0),(5,5),(5,10)]
    for contrast,title in labels.items():
        fig,axes=plt.subplots(2,3,figsize=(14,9),layout='constrained')
        for ri,days in enumerate(config['maturity_days']):
            for ci,ratio in enumerate(config['strike_over_forward']):
                ax=axes[ri,ci]
                sub=contrasts.query('maturity_days == @days and strike_over_forward == @ratio and contrast == @contrast')
                data=np.zeros((9,6)); vals=np.zeros_like(data)
                for i,scenario in enumerate(scenarios):
                    for j,(freq,cost) in enumerate(columns):
                        row=sub.query('scenario == @scenario and fee_bps == @cost and rebalance_days == @freq').iloc[0]
                        data[i,j]={'negative':-1,'unresolved':0,'positive':1}[row.supported_sign]
                        vals[i,j]=row.difference
                ax.imshow(data,vmin=-1,vmax=1,cmap=ListedColormap(['#B8DED5','#F5F5F5','#F4C3A8']),aspect='auto')
                for (i,j),value in np.ndenumerate(vals):
                    ax.text(j,i,f'{value:.2f}',ha='center',va='center',fontsize=8)
                ax.set(yticks=range(9),yticklabels=row_labels,xticks=range(6),
                       xticklabels=['D 0','D 5','D 10','W 0','W 5','W 10'],title=f'{days} days; strike/forward {ratio:.1f}')
                ax.tick_params(length=0)
        fig.suptitle(title+' — paired MSE differences (currency²)\nGreen: negative; orange: positive; gray: unresolved after simultaneous intervals and refinement check\nD/W: daily/weekly; number: fee in bps. Color is conditional on the numerical diagnostic.')
        save(fig,contrast+'_grid')
    for name in ['mse_summary.csv','paired_contrasts.csv','contrasts_with_convergence.csv','mse_convergence.csv','batch_contrasts.csv','contracts.csv',
                 'manifest.json','complete.json','path_diagnostics.json','pricing_diagnostics.json']:
        shutil.copy2(ROOT/'outputs/main'/name,OUT/name)
    validation=OUT/'validation'; validation.mkdir(exist_ok=True)
    for f in (ROOT/'outputs/validation').glob('*.json'):
        shutil.copy2(f,validation/f.name)
    # Compact reference table suitable for results prose.
    reference.query('fee_bps == 0 and rebalance_days == 1').pivot(
        index=['maturity_days','strike_over_forward'],columns='strategy',values='mse').to_csv(OUT/'e1_reference_mse.csv')
    counts=contrasts.groupby(['contrast','supported_sign']).size().unstack(fill_value=0)
    counts.to_csv(OUT/'sign_counts.csv')
    summary={'cells':int(len(mse)//3),'comparisons':len(contrasts),
             'sign_counts':counts.to_dict(),
             'excluded_paths':0,
             'max_top1pct_loss_share':float(mse.top1pct_loss_share.max()),
             'median_top1pct_loss_share':float(mse.top1pct_loss_share.median()),
             'max_absolute_refinement_shift':float(contrasts.refinement_shift.abs().max())}
    (OUT/'analysis_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    pilot()
    main_plots()
