import numpy as np
from scipy.stats import multivariate_normal as mvn

def rmvnorm(n, mean, sigma, random_state=None, allow_psd=True):
    return mvn.rvs(
        mean=mean,
        cov=sigma,
        size=n,
        random_state=random_state,
    )


def sample_norm(n, m=0.0, s2=1.0, random_state=None):
    """Draw n samples from N(m, s2)."""
    rng = np.random.default_rng(random_state)
    return rng.normal(loc=m, scale=np.sqrt(s2), size=n)


def scale_fast2(X, *, center=True, scale=False):
    """
    Centre (and optionally scale) the columns of X quickly.
    Returns a dict with centred/scaled matrix under key 'M'.
    """
    X = np.asarray(X, dtype=float)
    if center:
        col_means = X.mean(axis=0, keepdims=True)
        X = X - col_means
    if scale:  # not used in the R code
        col_sds = X.std(axis=0, ddof=1, keepdims=True)
        col_sds[col_sds == 0] = 1
        X = X / col_sds
    return {"M": X}


def col_vars(X, ddof=0):
    """Column-wise variance (like colVars in R matrixStats)."""
    return X.var(axis=0, ddof=ddof)


def matrix_normal_indep_rows(M, V, random_state=None):
    """
    Draw Y ~ MN(M, I_n, V)  (row-independent matrix normal).
    Each row i ~ N(M[i, :], V).
    """
    n, r = M.shape
    rng = np.random.default_rng(random_state)
    return rng.multivariate_normal(mean=np.zeros(r), cov=V, size=n) + M


# -------------------------------------------------------------------
def simulate_mr_mash_data(
    n,
    p,
    p_causal,
    r,
    r_causal=None,  # list of lists, as in the R code
    intercepts=None,  # length-r vector
    pve=0.20,
    B_cor=1.0,
    B_scale=1.0,
    w=1.0,
    X_cor=0.0,
    X_scale=1.0,
    V_cor=0.0,
    seed=None,
):
    """
    Simulate multivariate regression data under a mixture-of-shrinkage (mr-mash) model.

    Parameters
    ----------
    n : int
        Number of samples.
    p : int
        Number of variables (predictors).
    p_causal : int
        Number of causal variables.
    r : int
        Number of response variables.
    r_causal : list of list of int, optional
        For each mixture component, the indices of responses influenced by causal variables.
        Default is all responses affected in a single component.
    intercepts : array-like of float, optional
        Intercept for each response. Default is ones of length r.
    pve : float or array-like of float, default 0.20
        Per-response proportion of variance explained by causal variables.
    B_cor : float or array-like of float, default 1.0
        Correlation(s) between causal effects in each mixture component.
    B_scale : float or array-like of float, default 1.0
        Scale(s) (diagonal entries) for each component covariance Sigma_k.
    w : float or array-like of float, default 1.0
        Mixture proportions for the components; must sum to 1.
    X_cor : float, default 0.0
        Correlation between predictor variables.
    X_scale : float, default 1.0
        Diagonal scale for predictor covariance Gamma.
    V_cor : float, default 0.0
        Correlation among residuals across responses.
    random_state : int or RandomState, optional
        Seed or random state for reproducibility.

    Returns
    -------
    dict
        Dictionary containing:
        - 'X': array of shape (n, p), simulated predictor matrix.
        - 'Y': array of shape (n, r), simulated responses.
        - 'B': array of shape (p, r), true effect sizes.
        - 'V': array of shape (r, r), residual covariance matrix.
        - 'Sigma': dict of component covariance matrices.
        - 'Gamma': predictor covariance (or label).
        - 'intercepts': intercept used for each response.
        - 'causal_responses': mapping of component to causal response indices.
        - 'causal_variables': indices of causal predictors.
        - 'causal_vars_to_mixture_comps': assignment of causal variables to mixture components.
    """
    rng = np.random.default_rng(seed)

    # ---------------- defaults & checks -----------------------------
    if intercepts is None:
        intercepts = np.ones(r)

    if r_causal is None:
        r_causal = [list(range(r))]  # same default as R

    intercepts = np.asarray(intercepts, dtype=float)
    if intercepts.size != r:
        raise ValueError("intercepts must have length r")

    if any(len(rc) > r for rc in r_causal):
        raise ValueError("some r_causal entry is longer than r")

    # - convert scalars to length-K vectors so arithmetic is uniform
    def as_vector(x):
        if isinstance(x, (list, tuple)):
            x = np.asarray(x)
        return x if np.ndim(x) > 0 else np.asarray([x])

    B_cor = as_vector(B_cor).astype(float)
    B_scale = as_vector(B_scale).astype(float)
    w = as_vector(w).astype(float)

    if not (B_cor.size == B_scale.size == w.size):
        raise ValueError("B_cor, B_scale, w must have identical length")
    if not np.isclose(w.sum(), 1.0):
        raise ValueError("elements of w must sum to 1")

    pve = np.asarray(pve, dtype=float)
    if pve.size not in (1, r):
        raise ValueError("pve must be scalar or length r")

    K = w.size
    # ---------------------------------------------------------------

    # 1) build component-specific covariance matrices Sigma_k --------
    Sigma = []
    for k in range(K):
        rc = r_causal[k]
        d = len(rc)
        offdiag = B_scale[k] * B_cor[k]
        S = np.full((d, d), offdiag)
        np.fill_diagonal(S, B_scale[k])
        Sigma.append(S)

    # 2) sample true causal effects B_causal -------------------------
    B_causal = np.zeros((p_causal, r))

    if K > 1:
        mixcomps = rng.choice(K, size=p_causal, p=w, replace=True)
        for j, comp in enumerate(mixcomps):
            rc = r_causal[comp]
            B_causal[j, rc] = rmvnorm(
                1,
                mean=np.zeros(len(rc)),
                sigma=Sigma[comp],
                random_state=rng,
            ).ravel()
    else:
        rc = r_causal[0]
        B_causal[:, rc] = rmvnorm(
            p_causal,
            mean=np.zeros(len(rc)),
            sigma=Sigma[0],
            random_state=rng,
        )

    # 3) embed causal rows into full B -------------------------------
    B = np.zeros((p, r))
    causal_variables = rng.choice(p, size=p_causal, replace=False)
    B[causal_variables, :] = B_causal

    # 4) simulate X ---------------------------------------------------
    if X_cor != 0:
        Gamma = np.full((p, p), X_scale * X_cor)
        np.fill_diagonal(Gamma, X_scale)
        X = rmvnorm(n, mean=np.zeros(p), sigma=Gamma, random_state=rng)
    else:
        X = np.column_stack(
            [sample_norm(n, m=0.0, s2=X_scale, random_state=rng) for _ in range(p)]
        )
        Gamma = f"I_{p}"  # mimic R's character label

    X = scale_fast2(X, center=True, scale=False)["M"]

    # 5) genetic component of the mean & its variance ----------------
    G = X @ B
    Var_G = col_vars(G)  # length-r

    # 6) residual covariance V ---------------------------------------
    #    Var_E_s = ((1/pve)-1)*Var_G_s   (element-wise)
    Var_E = ((1.0 / pve) - 1.0) * Var_G
    Var_E[Var_E <= np.finfo(float).eps] = 1.0
    D = np.diag(np.sqrt(Var_E))

    V_cor_mat = np.full((r, r), V_cor)
    np.fill_diagonal(V_cor_mat, 1.0)
    V = D @ V_cor_mat @ D

    # 7) simulate Y ---------------------------------------------------
    M = G + intercepts.reshape(1, r)  # broadcast intercept
    Y = matrix_normal_indep_rows(M, V, random_state=rng)

    # 8) compile outputs ---------------------------------------------
    output = {
        "X": X,
        "Y": Y,
        "B": B,
        "V": V,
        "Sigma": {f"Component{k+1}": S for k, S in enumerate(Sigma)},
        "Gamma": Gamma,
        "intercepts": intercepts,
        "causal_responses": {f"Component{k+1}": rc for k, rc in enumerate(r_causal)},
    }

    if K > 1:
        if p_causal > 1:
            ordering = np.argsort(causal_variables)
            output["causal_variables"] = causal_variables[ordering]
            output["causal_vars_to_mixture_comps"] = mixcomps[ordering]
        else:
            output["causal_variables"] = causal_variables
            output["causal_vars_to_mixture_comps"] = mixcomps
    else:
        output["causal_variables"] = np.sort(causal_variables)
        output["causal_vars_to_mixture_comps"] = np.ones(p_causal, dtype=int)

    return output
