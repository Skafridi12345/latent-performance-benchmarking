from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

from sfa.sfa_halfnormal import (
    LOG_SIGMA_BOUNDS,
    MIN_SIGMA,
    _mills_ratio,
    _truncated_normal_exp_moment,
)


class TruncatedNormalSFA:
    """
    Truncated-normal stochastic frontier model.

    The composed-error convention is

        y = X beta + v - u

    with u drawn from N(mu, sigma_u^2) truncated below at zero. The model is
    available for comparison, while the half-normal model remains the default
    because it is more parsimonious and usually more stable in rolling windows.
    """

    model_type = "truncated_normal"

    def __init__(
        self,
        y: np.ndarray,
        X: np.ndarray,
        *,
        feature_names: list[str] | None = None,
    ):
        self.y = np.asarray(y, dtype=float)
        self.X = np.asarray(X, dtype=float)
        if self.X.ndim != 2:
            raise ValueError("X must be a two-dimensional design matrix.")
        if self.y.ndim != 1:
            raise ValueError("y must be a one-dimensional response vector.")
        if len(self.y) != self.X.shape[0]:
            raise ValueError("y and X have incompatible lengths.")
        if not np.isfinite(self.y).all() or not np.isfinite(self.X).all():
            raise ValueError("y and X must contain only finite values.")

        self.n, self.k = self.X.shape
        if self.n <= self.k + 1:
            raise ValueError("SFA requires more observations than parameters.")

        self.feature_names = feature_names or [f"x{i}" for i in range(self.k)]
        self.res = None
        self.theta = None

    def _unpack(self, theta: np.ndarray) -> tuple[np.ndarray, float, float, float]:
        beta = np.asarray(theta[: self.k], dtype=float)
        mu = float(theta[self.k])
        sigma_v = float(np.exp(theta[self.k + 1]))
        sigma_u = float(np.exp(theta[self.k + 2]))
        return beta, mu, max(sigma_v, MIN_SIGMA), max(sigma_u, MIN_SIGMA)

    def _loglike_obs(self, theta: np.ndarray) -> np.ndarray:
        beta, mu, sigma_v, sigma_u = self._unpack(theta)
        eps = self.y - self.X @ beta
        sigma = np.sqrt(sigma_v**2 + sigma_u**2)
        lam = sigma_u / sigma_v
        trunc = norm.logcdf(mu / sigma_u)
        z = (eps + mu) / sigma
        arg = mu / (sigma * lam) - (lam * eps) / sigma
        return -np.log(sigma) + norm.logpdf(z) + norm.logcdf(arg) - trunc

    def _neg_loglik(self, theta: np.ndarray) -> float:
        if not np.isfinite(theta).all():
            return np.inf
        ll = self._loglike_obs(theta)
        if not np.isfinite(ll).all():
            return np.inf
        return float(-np.sum(ll))

    def _ols_start(self) -> np.ndarray:
        beta_ols = np.linalg.lstsq(self.X, self.y, rcond=None)[0]
        resid = self.y - self.X @ beta_ols
        sigma = float(np.std(resid, ddof=max(1, self.k)) + 1e-6)
        return np.concatenate(
            [beta_ols, [0.0, np.log(sigma * 0.8), np.log(sigma * 0.5)]]
        )

    def _bounds(self) -> list[tuple[float | None, float | None]]:
        beta_bounds = [(None, None)] * self.k
        return beta_bounds + [(None, None), LOG_SIGMA_BOUNDS, LOG_SIGMA_BOUNDS]

    def fit(
        self,
        *,
        theta0: np.ndarray | None = None,
        maxiter: int = 500,
        raise_on_failure: bool = False,
    ) -> TruncatedNormalSFA:
        starts = []
        used_warm_start = False
        if theta0 is not None and len(theta0) == self.k + 3:
            starts.append(np.asarray(theta0, dtype=float))
            used_warm_start = True

        base = self._ols_start()
        starts.append(base)
        if not used_warm_start:
            starts.append(np.r_[base[: self.k], 0.01, base[self.k + 1 :]])

        best = None
        for start in starts:
            res = minimize(
                self._neg_loglik,
                start,
                method="L-BFGS-B",
                bounds=self._bounds(),
                options={"maxiter": maxiter, "ftol": 1e-9},
            )
            if best is None or (bool(res.success), -float(res.fun)) > (
                bool(best.success),
                -float(best.fun),
            ):
                best = res

        self.res = best
        self.theta = np.asarray(best.x, dtype=float)
        self._postprocess()

        if raise_on_failure and not self.converged:
            raise RuntimeError(f"TruncatedNormalSFA did not converge: {self.message}")

        return self

    def _postprocess(self) -> None:
        beta, mu, sigma_v, sigma_u = self._unpack(self.theta)
        self.beta = beta
        self.alpha = float(beta[0])
        self.mu = mu
        self.sigma_v = sigma_v
        self.sigma_u = sigma_u
        self.lambda_ = sigma_u / sigma_v
        self.log_likelihood = float(-self.res.fun)
        self.n_params = self.k + 3
        self.aic = float(2 * self.n_params - 2 * self.log_likelihood)
        self.bic = float(np.log(self.n) * self.n_params - 2 * self.log_likelihood)
        self.converged = bool(self.res.success)
        self.message = str(self.res.message)
        self.n_iter = int(getattr(self.res, "nit", 0))

        self.frontier = self.X @ beta
        self.composed_error = self.y - self.frontier
        sigma2 = sigma_v**2 + sigma_u**2
        self.sigma_total = float(np.sqrt(sigma2))
        mu_star = (mu * sigma_v**2 - self.composed_error * sigma_u**2) / sigma2
        sigma_star = (sigma_v * sigma_u) / np.sqrt(sigma2)
        z = mu_star / max(sigma_star, MIN_SIGMA)
        self.u_hat = np.maximum(mu_star + sigma_star * _mills_ratio(z), 0.0)
        self.u_hat_standardized = self.u_hat / self.sigma_total
        self.AE_raw_plugin = np.clip(
            np.exp(-self.u_hat), np.finfo(float).tiny, 1.0
        )
        self.AE = np.clip(
            _truncated_normal_exp_moment(
                mu_star,
                sigma_star,
                rate=1.0 / self.sigma_total,
            ),
            np.finfo(float).tiny,
            1.0,
        )
        self.TE = self.AE
        self.fitted_values = self.frontier - self.u_hat
        self.residuals = self.y - self.fitted_values

    def summary(self) -> dict:
        return {
            "model_type": self.model_type,
            "beta": self.beta,
            "alpha": self.alpha,
            "mu": self.mu,
            "sigma_v": self.sigma_v,
            "sigma_u": self.sigma_u,
            "lambda": self.lambda_,
            "AE_mean": float(np.mean(self.AE)),
            "AE_median": float(np.median(self.AE)),
            "TE_mean": float(np.mean(self.AE)),
            "TE_median": float(np.median(self.AE)),
            "u_hat_mean": float(np.mean(self.u_hat)),
            "u_hat_standardized_mean": float(np.mean(self.u_hat_standardized)),
            "log_likelihood": self.log_likelihood,
            "AIC": self.aic,
            "BIC": self.bic,
            "converged": self.converged,
            "message": self.message,
            "n_iter": self.n_iter,
        }
