"""Fail closed when any supported model job is absent, duplicated or failed."""
import numpy as np

def validate_job_coverage(results, support, failures=None, label='analysis'):
    key=['pair_id','day','seed','base_model']
    supported=support[support.admissible]
    expected={(str(r.pair_id),int(r.day),int(r.seed),model)
              for r in supported.itertuples() for model in ('logistic','lightgbm')}
    if not expected:
        raise RuntimeError(f'{label}: no supported jobs')
    if failures is not None and len(failures):
        raise RuntimeError(f'{label}: {len(failures)} failed supported jobs; see failure diagnostics')
    if results[key].duplicated().any():
        raise RuntimeError(f'{label}: duplicated result keys')
    actual={(str(r.pair_id),int(r.day),int(r.seed),str(r.base_model)) for r in results.itertuples()}
    if actual!=expected:
        raise RuntimeError(f'{label}: incomplete or unexpected supported jobs: {len(expected-actual)} missing, {len(actual-expected)} unexpected')
    for column in ('G_S','G_T','G_T_star','A'):
        if not np.isfinite(results[column].to_numpy(dtype=float)).all():
            raise RuntimeError(f'{label}: nonfinite {column}')
    return len(expected)
