"""P2-4 regression at the module level: eigvals is an explicit argument."""
import inspect

import numpy as np
import pandas as pd

from eigenrmt import build_eigenportfolios, compute_eigen, classify_eigenvalues


def test_signature_takes_eigvals_explicitly():
    params = list(inspect.signature(build_eigenportfolios).parameters)
    assert params[0] == 'eigvals'  # the P2-4 fix, structurally


def test_weights_normalized_and_explained_var_from_argument():
    rng = np.random.default_rng(3)
    X = pd.DataFrame(0.5 * rng.standard_normal(400)[:, None] * np.ones(6)
                     + rng.normal(0, 1, (400, 6)))
    ev, evec, _, _ = compute_eigen(X)
    is_sig, _, n_sig = classify_eigenvalues(ev, 6 / 400)
    assert n_sig >= 1
    eps = build_eigenportfolios(ev, evec, is_sig, [f'A{i}' for i in range(6)])
    for k, ep in enumerate(eps):
        assert np.isclose(np.abs(ep['weights']).sum(), 1.0)
        assert np.isclose(ep['explained_var'], ev[k] / ev.sum())
