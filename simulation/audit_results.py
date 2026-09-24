"""Independently reconcile persisted outcomes, summaries and run provenance."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
C=json.loads((ROOT/'simulation/protocol.json').read_text())
STRATEGIES=['BS','Heston','Adjusted']

def main():
    checks=[]
    h=hashlib.sha256((ROOT/'simulation/protocol.json').read_bytes()).hexdigest()
    manifests=[]
    maximum_mse_error=0.0
    maximum_contrast_error=0.0
    for folder in ['main','convergence_8','convergence_16']:
        out=ROOT/'outputs'/folder
        done=json.loads((out/'complete.json').read_text())
        manifest=json.loads((out/'manifest.json').read_text())
        assert done['protocol_sha256']==h==manifest['protocol_sha256']
        assert done['excluded_paths']==0
        assert done['strategy_rows']==972 and done['contrast_rows']==648
        manifests.append(manifest)
        losses=pd.read_csv(out/'mse_summary.csv')
        differences=pd.read_csv(out/'paired_contrasts.csv')
        assert len(losses)==972 and len(differences)==648
        market=np.load(out/'market_paths.npz')
        assert market['S'].shape==(manifest['n_paths'],253)
        assert np.isfinite(market['S']).all() and (market['S']>0).all()
        assert np.isfinite(market['V']).all() and (market['V']>=0).all()
        for days in C['maturity_days']:
            for ratio in C['strike_over_forward']:
                name=f'd{days}_m{ratio:.1f}'
                arrays=np.load(out/name/'pathwise_outcomes.npz')
                ee=arrays['errors']; ff=arrays['fees']
                assert ee.shape==(9,3,2,3,manifest['n_paths'])
                assert np.isfinite(ee).all() and np.isfinite(ff).all()
                assert (ff>=0).all() and (ff[:,0]==0).all()
                assert np.array_equal(ee[:,:,:,0],np.broadcast_to(ee[0:1,:,:,0],ee[:,:,:,0].shape))
                inp=np.load(out/name/'reference_inputs.npz')
                assert inp['prices'].shape==(manifest['n_paths'],days)
                assert inp['ivs'].shape==inp['prices'].shape
                assert np.isfinite(inp['prices']).all() and np.isfinite(inp['ivs']).all()
                for i,scenario in enumerate(C['scenarios']):
                    for j,fee in enumerate(C['fee_bps']):
                        for k,freq in enumerate(C['rebalance_days']):
                            meta=(losses.maturity_days==days)&(losses.strike_over_forward==ratio)&(losses.scenario==scenario['id'])&(losses.fee_bps==fee)&(losses.rebalance_days==freq)
                            sel=losses[meta].set_index('strategy')
                            mse=np.mean(ee[i,j,k]**2,axis=1)
                            residual=np.max(np.abs(sel.loc[STRATEGIES,'mse'].to_numpy()-mse))
                            maximum_mse_error=max(maximum_mse_error,float(residual))
                            assert residual<1e-11
                            assert np.allclose(sel.loc[STRATEGIES,'mean_fees'],ff[i,j,k].mean(axis=1),rtol=1e-12,atol=1e-12)
                            mask=(differences.maturity_days==days)&(differences.strike_over_forward==ratio)&(differences.scenario==scenario['id'])&(differences.fee_bps==fee)&(differences.rebalance_days==freq)
                            ds=differences[mask].set_index('contrast')
                            for a,b,label in [(1,0,'Heston_minus_BS'),(2,1,'Adjusted_minus_Heston')]:
                                x=ee[i,j,k,a]**2-ee[i,j,k,b]**2
                                residual=abs(float(x.mean())-ds.loc[label,'difference'])
                                maximum_contrast_error=max(maximum_contrast_error,float(residual))
                                assert residual<1e-11
                                assert np.isclose(x.std(ddof=1)/np.sqrt(len(x)),ds.loc[label,'se'],rtol=1e-11,atol=1e-12)
        checks.append({'run':folder,'status':'PASS','cells':324,'strategy_rows':972,'contrasts':648})
    assert manifests[1]['seed']==manifests[2]['seed']
    assert manifests[1]['n_paths']==manifests[2]['n_paths']==512
    for key in ['paths.py','pricing.py','hedging.py']:
        assert len({m['code_sha256'][key] for m in manifests})==1
    report={'status':'PASS','checks':checks,'max_summary_mse_residual':maximum_mse_error,
            'max_summary_contrast_residual':maximum_contrast_error,
            'common_benchmark_outcomes_bitwise_identical':True,'excluded_paths':0,
            'protocol_sha256':h,'notes':'Recomputed summaries and paired SEs from all persisted outcomes; core hashes and completed-run markers agree.'}
    path=ROOT/'outputs/validation/results_audit.json'
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
