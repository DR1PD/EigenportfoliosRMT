"""P2-3 regression tests (ROADMAP 1.4): the Marchenko-Pastur noise level
sigma^2 must be re-estimated iteratively after excluding signal eigenvalues.

The naive one-pass recipe (mean of eigenvalues below the sigma^2=1 edge)
lets moderate factor eigenvalues contaminate the bulk average, inflating
sigma^2 and the edge lambda_+ — so real factors get classified as noise.
The iterative recipe removes detected signals from the trace budget and
re-draws the edge until the classification stops changing.
"""
import numpy as np


# ── same recipes the notebook implements (test-side reference copies) ──

def mp_edges(q, sigma2=1.0):
    return sigma2 * (1 - np.sqrt(q)) ** 2, sigma2 * (1 + np.sqrt(q)) ** 2


def naive_sigma2(eigvals, q):
    """Pre-fix recipe: mean of the bulk below the preliminary sigma^2=1 edge."""
    _, lp = mp_edges(q, 1.0)
    noise = eigvals[eigvals <= lp]
    return noise.mean() if len(noise) > 5 else 1.0


def iterative_sigma2(eigvals, q, max_iter=100, tol=1e-8):
    """P2-3 fix: sigma^2_(k) = (1 - sum_signal(lam)/n) / (1 - k/n), iterated."""
    n = len(eigvals)
    sigma2, k_prev = 1.0, -1
    for _ in range(max_iter):
        _, lam_plus = mp_edges(q, sigma2)
        is_signal = eigvals > lam_plus
        k = int(is_signal.sum())
        if k >= n:
            break
        sigma2_new = (1.0 - eigvals[is_signal].sum() / n) / (1.0 - k / n)
        if sigma2_new <= 0:
            break
        if k == k_prev and abs(sigma2_new - sigma2) < tol:
            sigma2 = sigma2_new
            break
        k_prev, sigma2 = k, sigma2_new
    return sigma2


def n_signals(eigvals, q, sigma2):
    return int((eigvals > mp_edges(q, sigma2)[1]).sum())


def corr_eigvals(X):
    Xs = (X - X.mean(0)) / X.std(0)
    return np.sort(np.linalg.eigvalsh(Xs.T @ Xs / len(Xs)))[::-1]


N, T = 100, 250
Q = N / T


def test_pure_noise_recovers_zero_signals():
    """ROADMAP 1.4 acceptance: no factors planted → no signals found."""
    rng = np.random.default_rng(11)
    ev = corr_eigvals(rng.standard_normal((T, N)))
    assert n_signals(ev, Q, iterative_sigma2(ev, Q)) == 0
    assert n_signals(ev, Q, naive_sigma2(ev, Q)) == 0


def test_single_strong_factor_recovers_exactly_one():
    """ROADMAP 1.4 acceptance: one planted factor → exactly one signal."""
    rng = np.random.default_rng(11)
    f = rng.standard_normal(T)[:, None]
    X = 0.6 * f * np.ones(N) + rng.standard_normal((T, N))
    ev = corr_eigvals(X)
    assert n_signals(ev, Q, iterative_sigma2(ev, Q)) == 1


def test_iterative_edge_beats_naive_on_moderate_factors():
    """11 planted factors (1 market + 10 sector blocks) with the blocks tuned
    to land below the naive sigma^2=1 preliminary edge: they contaminate the
    naive bulk average. Fixed seed: iterative recovers 10 (the 11th sits at
    the BBP detectability limit — no spectral method can see it); naive
    recovers only 7. The iterative edge must sit strictly lower."""
    B = np.zeros((11, N))
    B[0] = 0.7
    for j in range(10):
        B[j + 1, 10 * j:10 * (j + 1)] = 0.45
    rng = np.random.default_rng(1)
    X = rng.standard_normal((T, 11)) @ B + rng.standard_normal((T, N))
    ev = corr_eigvals(X)

    k_naive = n_signals(ev, Q, naive_sigma2(ev, Q))
    k_iter = n_signals(ev, Q, iterative_sigma2(ev, Q))

    assert k_iter == 10 and k_naive == 7
    assert k_iter > k_naive
    assert iterative_sigma2(ev, Q) < naive_sigma2(ev, Q)  # lower noise floor → lower fence
