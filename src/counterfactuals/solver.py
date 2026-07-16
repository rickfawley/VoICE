import numpy as np

from scipy.optimize import minimize

def directional_rho(
    a,
    u,
    halfspaces,
    x_lo=None,
    x_hi=None,
    eps=1e-12,
    return_diagnostics=False,
    respect_box_bounds=True,
):
    """
    Compute rho for the ray r(lambda) = a + lambda u.

    The minimum counterfactual is assumed to occur at lambda = 1.
    rho is the largest lambda before the ray leaves the feasible region.
    """

    a = np.asarray(a, dtype=float)
    u = np.asarray(u, dtype=float)

    upper_bounds = []
    diagnostics = []

    for w, b, _ in halfspaces:
        w = np.asarray(w, dtype=float)
        denom = float(np.dot(w, u))
        numer = float(b - np.dot(w, a))

        if denom > eps:
            bound = numer / denom
            upper_bounds.append(bound)
            diagnostics.append({
                "source": "halfspace",
                "bound": float(bound),
                "numer": float(numer),
                "denom": float(denom),
            })
        #end-if
    #end-for

    if (
        respect_box_bounds
        and x_lo is not None
        and x_hi is not None
    ):

        x_lo = np.asarray(x_lo, dtype=float)
        x_hi = np.asarray(x_hi, dtype=float)

        for j, uj in enumerate(u):

            if uj > eps:
                bound = (x_hi[j] - a[j]) / uj
                upper_bounds.append(bound)
                diagnostics.append({
                    "source": "upper_bound",
                    "feature_index": int(j),
                    "bound": float(bound),
                    "numer": float(x_hi[j] - a[j]),
                    "denom": float(uj),
                })
            elif uj < -eps:
                bound = (x_lo[j] - a[j]) / uj
                upper_bounds.append(bound)
                diagnostics.append({
                    "source": "lower_bound",
                    "feature_index": int(j),
                    "bound": float(bound),
                    "numer": float(x_lo[j] - a[j]),
                    "denom": float(uj),
                })
            #end-if

        #end-for

    #end-if

    if len(upper_bounds) == 0:
        if return_diagnostics:
            return np.inf, {
                "raw_rho": np.inf,
                "clipped_rho": np.inf,
                "active_bound": None,
                "n_upper_bounds": 0,
                "diagnostics": diagnostics,
            }
        return np.inf

    raw_rho = float(min(upper_bounds))
    clipped_rho = max(1.0, raw_rho)

    active_idx = int(np.argmin(upper_bounds))
    active_bound = diagnostics[active_idx] if diagnostics else None

    if return_diagnostics:
        return clipped_rho, {
            "raw_rho": raw_rho,
            "clipped_rho": clipped_rho,
            "active_bound": active_bound,
            "n_upper_bounds": len(upper_bounds),
            "was_clipped": raw_rho < 1.0,
            "tau_raw": raw_rho - 1.0,
            "tau_clipped": clipped_rho - 1.0,
        }

    return clipped_rho

def project_to_target_with_fixed_features(
    a,
    free_idx,
    halfspaces,
    x_lo=None,
    x_hi=None,
    weights=None,
):
    """
    Solve:
        min ||z - a||^2
        s.t. z lies in target halfspaces
             z_k = a_k for fixed features
             optional box bounds x_lo <= z <= x_hi

    Returns z, success, message.
    """
    a = np.asarray(a, dtype=float)
    d = a.shape[0]

    if weights is None:
        weights = np.ones(d, dtype=float) / d
    #end-if

    weights = np.asarray(weights, dtype=float)

    if weights.shape[0] != d:
        raise ValueError("weights must have the same length as a")
    #end-if

    free_idx = np.array(sorted(free_idx), dtype=int)
    fixed_idx = np.array([i for i in range(d) if i not in free_idx], dtype=int)

    if free_idx.size == 0:
        z = a.copy()
        feasible = True
        for w, b, _ in halfspaces:
            if np.dot(w, z) > b + 1e-9:
                feasible = False
                break
        return z, feasible, "No free features"

    x0 = a[free_idx].copy()

    def unpack(u):
        z = a.copy()
        z[free_idx] = u
        return z

    def obj(u):
        z = unpack(u)
        dlt = z - a
        return float(np.sum(weights * dlt ** 2))

    constraints = []

    for w, b, _ in halfspaces:
        constraints.append({
            "type": "ineq",
            "fun": lambda u, w=w, b=b: float(b - np.dot(w, unpack(u)))
        })

    bounds = None
    if x_lo is not None and x_hi is not None:
        bounds = [(float(x_lo[i]), float(x_hi[i])) for i in free_idx]

    res = minimize(
        obj,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-9},
    )

    z = unpack(res.x if res.success else x0)

    feasible = True
    for w, b, _ in halfspaces:
        if np.dot(w, z) > b + 1e-6:
            feasible = False
            break

    return z, bool(res.success and feasible), res.message