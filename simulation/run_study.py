"""Execute the frozen study; all reported losses use a paired common-path design."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from paths import generate_paths
from pricing import heston_price_greeks, bs_iv_delta, get_diagnostics
from hedging import hedge_paths

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / 'simulation/protocol.json'
STRATEGIES = ['BS', 'Heston', 'Adjusted']
FAMILY_SIZE = 648


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def log(message):
    print(f'{datetime.now(timezone.utc).isoformat()} {message}', flush=True)


def scenario_params(config, scenario):
    result = dict(config['pricing'])
    key = scenario['parameter']
    if key:
        if scenario['operation'] == 'multiply':
            result[key] *= scenario['amount']
        else:
            result[key] += scenario['amount']
    return result


def paired_row(x, label, nfamily=FAMILY_SIZE):
    n = len(x)
    mean = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n))
    critical = float(student_t.ppf(0.975, n - 1))
    simultaneous = float(student_t.ppf(1 - 0.05 / (2 * nfamily), n - 1))
    return {'contrast': label, 'difference': mean, 'se': se,
            'ci95_low': mean-critical*se, 'ci95_high': mean+critical*se,
            'sim95_low': mean-simultaneous*se, 'sim95_high': mean+simultaneous*se,
            'n_paths': n}


def summarize(errors, fees, config, days, ratio, scenarios):
    if not np.isfinite(errors).all() or not np.isfinite(fees).all():
        raise FloatingPointError('nonfinite accounting output: run stopped, no exclusions')
    rows, contrasts, batches = [], [], []
    for i, scenario in enumerate(scenarios):
        for j, cost in enumerate(config['fee_bps']):
            for k, freq in enumerate(config['rebalance_days']):
                meta = {'maturity_days': days, 'strike_over_forward': ratio,
                        'scenario': scenario['id'], 'fee_bps': cost, 'rebalance_days': freq}
                ee = errors[i,j,k]
                for s, name in enumerate(STRATEGIES):
                    e = ee[s]
                    loss = e*e
                    if not np.isfinite(loss).all():
                        raise FloatingPointError('nonfinite squared losses: run stopped')
                    mse = float(loss.mean())
                    mean = float(e.mean())
                    variance = float(e.var(ddof=0))
                    if not np.isclose(mse, variance+mean*mean, rtol=1e-12, atol=1e-12):
                        raise AssertionError('MSE decomposition failed')
                    top = max(1, int(np.ceil(len(e)*.01)))
                    rows.append({**meta, 'strategy':name, 'mse':mse, 'mean_error':mean,
                        'error_variance':variance, 'mean_fees':float(fees[i,j,k,s].mean()),
                        'top1pct_loss_share':float(np.sort(loss)[-top:].sum()/loss.sum()) if mse else 0,
                        'max_abs_error':float(np.max(np.abs(e))), 'n_paths':len(e)})
                for a,b,label in [(1,0,'Heston_minus_BS'),(2,1,'Adjusted_minus_Heston')]:
                    x = ee[a]**2-ee[b]**2
                    row = paired_row(x,label)
                    row['first_half_difference'] = float(x[:len(x)//2].mean())
                    contrasts.append({**meta, **row})
                    if len(x) >= 4096:
                        for batch, xx in enumerate(np.array_split(x,8)):
                            batches.append({**meta,'contrast':label,'batch':batch,
                                            'n_paths':len(xx),'difference':float(xx.mean())})
    return rows, contrasts, batches


def price_contract(S,V,K,days,config,scenarios,outdir,pilot=False):
    n = len(S)
    targets = np.zeros((len(scenarios),2,n,days+1))
    bs = np.zeros((n,days+1))
    refs = np.zeros((n,days))
    ivs = np.zeros_like(refs)
    r = config['rate']
    tpy = config['trading_days_per_year']
    endowment = float(np.asarray(heston_price_greeks(config['spot0'],config['variance0'],K,days/tpy,r,config['pricing'])[0]))
    for day in range(days):
        tau = (days-day)/tpy
        p, delta, dv = heston_price_greeks(S[:,day],V[:,day],K,tau,r,config['pricing'])
        refs[:,day] = p
        ivs[:,day], bs[:,day] = bs_iv_delta(p,S[:,day],K,tau,r)
        for i, scenario in enumerate(scenarios):
            params = scenario_params(config,scenario)
            if scenario['id'] != 'reference':
                _, delta_i, dv_i = heston_price_greeks(S[:,day],V[:,day],K,tau,r,params)
            else:
                delta_i, dv_i = delta, dv
            targets[i,0,:,day] = delta_i
            targets[i,1,:,day] = delta_i+params['rho']*params['xi']*dv_i/S[:,day]
        if day in [0,days//2,days-1]:
            log(f'  priced date {day}/{days-1}')
    if not all(np.all(np.isfinite(x)) for x in [targets,bs,refs,ivs]):
        raise FloatingPointError('nonfinite strategy inputs: run stopped, no exclusions')
    # These saved histories are constructed once and shared by all structural scenarios.
    np.savez_compressed(outdir/'reference_inputs.npz',prices=refs,ivs=ivs,bs_target=bs,
                        initial_endowment=endowment,strike=K)
    shape = (len(scenarios),len(config['fee_bps']),len(config['rebalance_days']),3,n)
    errors, fees = np.empty(shape), np.empty(shape)
    bs_cache = {}
    for i,scenario in enumerate(scenarios):
        for j,cost in enumerate(config['fee_bps']):
            for k,freq in enumerate(config['rebalance_days']):
                for s,name in enumerate(STRATEGIES):
                    if s == 0 and (j,k) in bs_cache:
                        err, fee = bs_cache[(j,k)]
                    else:
                        target = bs if s == 0 else targets[i,s-1]
                        err,fee,ledger = hedge_paths(S[:,:days+1],target,K,r,cost,freq,endowment,
                            trading_days=tpy,ledger_path=0 if pilot and cost==10 and freq==1 else None)
                        if ledger is not None:
                            ledger.to_csv(outdir/f'ledger_{name}.csv',index=False)
                        if s == 0:
                            bs_cache[(j,k)] = (err,fee)
                    errors[i,j,k,s] = err
                    fees[i,j,k,s] = fee
    if not np.array_equal(errors[:,:,:,0],np.broadcast_to(errors[0:1,:,:,0],errors[:,:,:,0].shape)):
        raise AssertionError('BS outcomes changed with Heston structural inputs')
    np.savez_compressed(outdir/'pathwise_outcomes.npz',errors=errors,fees=fees,
                        scenarios=np.array([s['id'] for s in scenarios]),strategies=np.array(STRATEGIES))
    if pilot:
        frame = pd.DataFrame({'day':np.arange(days+1),'stock':S[0,:days+1],
                              'variance':V[0,:days+1],'BS':bs[0],
                              'Heston':targets[0,0,0],'Adjusted':targets[0,1,0]})
        frame.to_csv(outdir/'first_path_and_targets.csv',index=False)
    return errors,fees,endowment


def contract_job(job):
    """A contract owns its output directory and its QuantLib process state."""
    outdir, config, scenarios, days, ratio, pilot = job
    outdir = Path(outdir)
    paths = np.load(outdir/'market_paths.npz')
    S,V = paths['S'],paths['V']
    name = f'd{days}_m{ratio:.1f}'
    contractdir = outdir/name
    contractdir.mkdir(exist_ok=True)
    K = ratio*config['spot0']*np.exp(config['rate']*days/config['trading_days_per_year'])
    get_diagnostics(reset=True)
    log(f'{outdir.name}: {name}, strike {K:.8f}')
    ee,ff,endowment = price_contract(S,V,K,days,config,scenarios,contractdir,pilot=pilot)
    rr,cc,bb = summarize(ee,ff,config,days,ratio,scenarios)
    contract = {'contract':name,'maturity_days':days,'strike_over_forward':ratio,
                'strike':K,'initial_endowment':endowment}
    log(f'{outdir.name}: {name} complete')
    return rr,cc,bb,contract,get_diagnostics()


def run_suite(mode, config, substeps=None):
    run_config = config['pilot'] if mode=='pilot' else config['main'] if mode=='main' else config['convergence']
    n,seed = run_config['n_paths'],run_config['seed']
    steps = substeps or run_config['substeps']
    outdir = ROOT/'outputs'/('convergence_'+str(steps) if mode=='convergence' else mode)
    outdir.mkdir(parents=True,exist_ok=True)
    if (outdir/'complete.json').exists():
        raise FileExistsError(f'Completed run exists at {outdir}; archive it explicitly before rerunning.')
    max_days = run_config['maturity_days'] if mode=='pilot' else max(config['maturity_days'])
    days_list = [max_days] if mode=='pilot' else config['maturity_days']
    ratios = [run_config['strike_over_forward']] if mode=='pilot' else config['strike_over_forward']
    scenarios = config['scenarios'][:1] if mode=='pilot' else config['scenarios']
    manifest = {'started_utc':datetime.now(timezone.utc).isoformat(),'mode':mode,'n_paths':n,
        'seed':seed,'substeps':steps,'protocol_sha256':digest(PROTOCOL),
        'code_sha256':{p.name:digest(p) for p in sorted((ROOT/'simulation').glob('*.py'))},
        'package_lock_sha256':digest(ROOT/'simulation/requirements-lock.txt'),
        'excluded_paths':0, 'configuration':config}
    (outdir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    log(f'{mode}: generate {n} common paths, {steps} substeps/day, seed {seed}')
    start = time.perf_counter()
    kwargs = {'coupling_substeps':run_config['coupling_substeps']} if mode=='convergence' else {}
    S,V,path_diag = generate_paths(n,max_days,steps,seed,config['market'],
        s0=config['spot0'],v0=config['variance0'],trading_days=config['trading_days_per_year'],**kwargs)
    np.savez_compressed(outdir/'market_paths.npz',S=S,V=V)
    (outdir/'path_diagnostics.json').write_text(json.dumps(path_diag,indent=2)+'\n')
    get_diagnostics(reset=True)
    rows, contrasts, batches, contracts = [],[],[],[]
    jobs = [(str(outdir),config,scenarios,days,ratio,mode=='pilot') for days in days_list for ratio in ratios]
    diagnostics = Counter()
    with ProcessPoolExecutor(max_workers=1 if mode=='pilot' else 3) as pool:
        for rr,cc,bb,contract,diag in pool.map(contract_job,jobs):
            rows.extend(rr); contrasts.extend(cc); batches.extend(bb); contracts.append(contract)
            diagnostics.update(diag)
            # Checkpoint summaries support inspection of interrupted runs, never imply completion.
            pd.DataFrame(rows).to_csv(outdir/'mse_summary.csv',index=False)
            pd.DataFrame(contrasts).to_csv(outdir/'paired_contrasts.csv',index=False)
            pd.DataFrame(contracts).to_csv(outdir/'contracts.csv',index=False)
    if batches:
        pd.DataFrame(batches).to_csv(outdir/'batch_contrasts.csv',index=False)
    (outdir/'pricing_diagnostics.json').write_text(json.dumps(diagnostics,indent=2,default=str)+'\n')
    complete = {'completed_utc':datetime.now(timezone.utc).isoformat(),
                'elapsed_seconds':time.perf_counter()-start,'contracts':len(contracts),
                'strategy_rows':len(rows),'contrast_rows':len(contrasts),
                'excluded_paths':0,'protocol_sha256':digest(PROTOCOL)}
    (outdir/'complete.json').write_text(json.dumps(complete,indent=2)+'\n')
    log(f'{mode} complete: {complete}')


def summarize_convergence(config):
    manifests = []
    for folder in ['main', 'convergence_8', 'convergence_16']:
        location = ROOT/'outputs'/folder
        completion = json.loads((location/'complete.json').read_text())
        manifest = json.loads((location/'manifest.json').read_text())
        if completion['protocol_sha256'] != digest(PROTOCOL):
            raise AssertionError('protocol mismatch across completed runs')
        manifests.append(manifest)
    if manifests[1]['seed'] != manifests[2]['seed'] or manifests[1]['n_paths'] != manifests[2]['n_paths']:
        raise AssertionError('refinement samples do not share seed and path count')
    for filename in ['paths.py', 'pricing.py', 'hedging.py']:
        if len({m['code_sha256'][filename] for m in manifests}) != 1:
            raise AssertionError('core computation code changed across production runs')
    main = pd.read_csv(ROOT/'outputs/main/paired_contrasts.csv')
    keys = ['maturity_days','strike_over_forward','scenario','fee_bps','rebalance_days','contrast']
    records=[]
    loss_records=[]
    for days in config['maturity_days']:
        for ratio in config['strike_over_forward']:
            name = f'd{days}_m{ratio:.1f}'
            coarse = np.load(ROOT/f'outputs/convergence_8/{name}/pathwise_outcomes.npz')['errors']
            fine = np.load(ROOT/f'outputs/convergence_16/{name}/pathwise_outcomes.npz')['errors']
            for i,sc in enumerate(config['scenarios']):
                for j,fee in enumerate(config['fee_bps']):
                    for k,freq in enumerate(config['rebalance_days']):
                        for s,strategy in enumerate(STRATEGIES):
                            shift = fine[i,j,k,s]**2 - coarse[i,j,k,s]**2
                            loss_records.append({'maturity_days':days,'strike_over_forward':ratio,
                                'scenario':sc['id'],'fee_bps':fee,'rebalance_days':freq,
                                'strategy':strategy,'coarse_mse':float(np.mean(coarse[i,j,k,s]**2)),
                                'fine_mse':float(np.mean(fine[i,j,k,s]**2)),
                                'mse_refinement_shift':float(shift.mean()),
                                'mse_refinement_se':float(shift.std(ddof=1)/np.sqrt(len(shift)))})
                        for a,b,label in [(1,0,'Heston_minus_BS'),(2,1,'Adjusted_minus_Heston')]:
                            x = (fine[i,j,k,a]**2-fine[i,j,k,b]**2)-(coarse[i,j,k,a]**2-coarse[i,j,k,b]**2)
                            mean=float(x.mean()); se=float(x.std(ddof=1)/np.sqrt(len(x)))
                            records.append(dict(zip(keys,[days,ratio,sc['id'],fee,freq,label]),
                                refinement_shift=mean,refinement_se=se,
                                refinement_margin=abs(mean)+1.96*se))
    frame=main.merge(pd.DataFrame(records),on=keys,validate='one_to_one')
    if len(frame) != FAMILY_SIZE or frame.duplicated(keys).any():
        raise AssertionError('expected exactly 648 unique prespecified comparisons')
    frame['expanded_low']=frame.sim95_low-frame.refinement_margin
    frame['expanded_high']=frame.sim95_high+frame.refinement_margin
    frame['supported_sign']=np.where(frame.expanded_high<0,'negative',np.where(frame.expanded_low>0,'positive','unresolved'))
    frame.to_csv(ROOT/'outputs/main/contrasts_with_convergence.csv',index=False)
    pd.DataFrame(loss_records).to_csv(ROOT/'outputs/main/mse_convergence.csv',index=False)
    log('Paired refinement check complete: '+str(frame.supported_sign.value_counts().to_dict()))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['pilot','main','convergence','summarize-convergence'])
    parser.add_argument('--substeps',type=int,choices=[8,16])
    args=parser.parse_args()
    if args.substeps and args.mode != 'convergence':
        parser.error('--substeps is reserved for the convergence runs')
    config=json.loads(PROTOCOL.read_text())
    if args.mode=='summarize-convergence':
        summarize_convergence(config)
    else:
        if args.mode=='convergence' and not args.substeps:
            parser.error('--substeps required for convergence')
        run_suite(args.mode,config,args.substeps)

if __name__=='__main__':
    main()
