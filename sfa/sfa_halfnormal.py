from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2, norm

LOG_SIGMA_BOUNDS = (-20.0, 2.0)
MIN_SIGMA = 1e-10


@dataclass
class SFAResult:
    beta: np.ndarray
    sigma_v: float
    sigma_u: float
    log_likelihood: float
    aic: float
    bic: float
    converged: bool
    message: str
    n_iter: int


def _mills_ratio(z: np.ndarray) -> np.ndarray:
    """Compute phi(z) / Phi(z) with log-scale protection."""

    log_ratio = norm.logpdf(z) - norm.logcdf(z)
    return np.exp(np.clip(log_ratio, -745.0, 50.0))


def _truncated_normal_exp_moment(
    mu: np.ndarray,
    sigma: float,
    *,
    rate: float,
) -> np.ndarray:
    """Return E[exp(-rate * U)] for U ~ N(mu, sigma^2), U >= 0."""

    sigma = max(float(sigma), MIN_SIGMA)
    rate = max(float(rate), 0.0)
    z = np.asarray(mu, dtype=float) / sigma
    log_moment = (
        -rate * np.asarray(mu, dtype=float)
        + 0.5 * (rate * sigma) ** 2
        + norm.logcdf(z - rate * sigma)
        - norm.logcdf(z)
    )
    return np.exp(np.clip(log_moment, -745.0, 0.0))


class HalfNormalSFA:
    """
    Half-normal stochastic frontier model for a production-style frontier.

    The composed-error convention is

        y = X beta + v - u

    where v is symmetric Gaussian noise and u is non-negative latent
    performance shortfall. The class keeps the historical TE attribute for
    compatibility, but the project reports it as adjusted efficiency (AE).
    """

    model_type = "half_normal"

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
        if self.n <= self.k:
            raise ValueError("SFA requires more observations than parameters.")

        self.feature_names = feature_names or [f"x{i}" for i in range(self.k)]
        self.res = None
        self.theta = None

    def _unpack(self, theta: np.ndarray) -> tuple[np.ndarray, float, float]:
        beta = np.asarray(theta[: self.k], dtype=float)
        sigma_v = float(np.exp(theta[self.k]))
        sigma_u = float(np.exp(theta[self.k + 1]))
        return beta, max(sigma_v, MIN_SIGMA), max(sigma_u, MIN_SIGMA)

    def _loglike_obs(self, theta: np.ndarray) -> np.ndarray:
        beta, sigma_v, sigma_u = self._unpack(theta)
        eps = self.y - self.X @ beta
        sigma = np.sqrt(sigma_v**2 + sigma_u**2)
        lam = sigma_u / sigma_v
        z = eps / sigma
        arg = -(lam * eps) / sigma
        return np.log(2.0) - np.log(sigma) + norm.logpdf(z) + norm.logcdf(arg)

    def _neg_loglik(self, theta: np.ndarray) -> float:
        if not np.isfinite(theta).all():
            return np.inf
        ll = self._loglike_obs(theta)
        if not np.isfinite(ll).all():
            return np.inf
        return float(-np.sum(ll))

    def _neg_loglik_grad(self, theta: np.ndarray) -> np.ndarray:
        beta, sigma_v, sigma_u = self._unpack(theta)
        eps = self.y - self.X @ beta
        sigma = np.sqrt(sigma_v**2 + sigma_u**2)
        lam = sigma_u / sigma_v
        arg = -(lam * eps) / sigma
        ratio = _mills_ratio(arg)

        d_ll_d_eps = -eps / sigma**2 - (lam / sigma) * ratio
        grad_beta = -np.sum(d_ll_d_eps[:, None] * (-self.X), axis=0)

        d_sigma_d_sv = sigma_v / sigma
        d_sigma_d_su = sigma_u / sigma
        d_lam_d_sv = -sigma_u / sigma_v**2
        d_lam_d_su = 1.0 / sigma_v

        common_sigma = -1.0 / sigma + eps**2 / sigma**3
        d_arg_d_sv = -eps * (d_lam_d_sv / sigma - lam * d_sigma_d_sv / sigma**2)
        d_arg_d_su = -eps * (d_lam_d_su / sigma - lam * d_sigma_d_su / sigma**2)

        d_ll_d_sv = common_sigma * d_sigma_d_sv + ratio * d_arg_d_sv
        d_ll_d_su = common_sigma * d_sigma_d_su + ratio * d_arg_d_su

        grad_log_sv = -np.sum(d_ll_d_sv) * sigma_v
        grad_log_su = -np.sum(d_ll_d_su) * sigma_u
        return np.concatenate([grad_beta, [grad_log_sv, grad_log_su]])

    def _ols_start(self) -> np.ndarray:
        beta_ols = np.linalg.lstsq(self.X, self.y, rcond=None)[0]
        resid = self.y - self.X @ beta_ols
        sigma = float(np.std(resid, ddof=max(1, self.k)) + 1e-6)
        return np.concatenate([beta_ols, [np.log(sigma * 0.8), np.log(sigma * 0.5)]])

    def _bounds(self) -> list[tuple[float | None, float | None]]:
        beta_bounds = [(None, None)] * self.k
        return beta_bounds + [LOG_SIGMA_BOUNDS, LOG_SIGMA_BOUNDS]

    def fit(
        self,
        *,
        theta0: np.ndarray | None = None,
        maxiter: int = 500,
        raise_on_failure: bool = False,
    ) -> HalfNormalSFA:
        starts = []
        used_warm_start = False
        if theta0 is not None and len(theta0) == self.k + 2:
            starts.append(np.asarray(theta0, dtype=float))
            used_warm_start = True

        base = self._ols_start()
        starts.append(base)
        if not used_warm_start:
            starts.extend(
                [
                    np.r_[
                        base[: self.k],
                        np.log(np.exp(base[self.k]) * 0.75),
                        np.log(np.exp(base[self.k]) * 1.50),
                    ],
                    np.r_[
                        base[: self.k],
                        np.log(np.exp(base[self.k]) * 1.50),
                        np.log(np.exp(base[self.k]) * 0.75),
                    ],
                ]
            )

        best = None
        for start in starts:
            res = minimize(
                self._neg_loglik,
                start,
                jac=self._neg_loglik_grad,
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
            raise RuntimeError(f"HalfNormalSFA did not converge: {self.message}")

        return self

    def _postprocess(self) -> None:
        beta, sigma_v, sigma_u = self._unpack(self.theta)
        self.beta = beta
        self.alpha = float(beta[0])
        self.sigma_v = sigma_v
        self.sigma_u = sigma_u
        self.lambda_ = sigma_u / sigma_v
        self.sigma_v_unconstrained = sigma_v
        self.sigma_u_unconstrained = sigma_u
        self.lambda_unconstrained = self.lambda_
        self.log_likelihood = float(-self.res.fun)
        self.n_params = self.k + 2
        self.aic = float(2 * self.n_params - 2 * self.log_likelihood)
        self.bic = float(np.log(self.n) * self.n_params - 2 * self.log_likelihood)
        self.converged = bool(self.res.success)
        self.message = str(self.res.message)
        self.n_iter = int(getattr(self.res, "nit", 0))

        self.frontier = self.X @ beta
        self.composed_error = self.y - self.frontier

        beta_ols = np.linalg.lstsq(self.X, self.y, rcond=None)[0]
        ols_residuals = self.y - self.X @ beta_ols
        ols_variance = max(float(ols_residuals @ ols_residuals) / self.n, MIN_SIGMA**2)
        self.normal_log_likelihood = float(
            -0.5
            * self.n
            * (np.log(2.0 * np.pi) + 1.0 + np.log(ols_variance))
        )
        self.boundary_lr_stat = max(
            0.0, 2.0 * (self.log_likelihood - self.normal_log_likelihood)
        )
        self.boundary_mixture_p_value = float(
            0.5 * chi2.sf(self.boundary_lr_stat, df=1)
        )
        self.one_sided_component_supported = bool(
            self.boundary_mixture_p_value < 0.05
        )

        sigma2 = sigma_v**2 + sigma_u**2
        self.sigma_total = float(np.sqrt(sigma2))
        mu_star = -(sigma_u**2 * self.composed_error) / sigma2
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
        if not self.one_sided_component_supported:
            # Near the sigma_u=0 boundary the likelihood is flat and arbitrary
            # tiny sigma_u estimates create unstable rankings. Report the null
            # diagnostic instead of turning numerical noise into inefficiency.
            self.u_hat = np.zeros_like(self.composed_error)
            self.u_hat_standardized = np.zeros_like(self.composed_error)
            self.AE_raw_plugin = np.ones_like(self.composed_error)
            self.AE = np.ones_like(self.composed_error)
            self.beta = beta_ols
            self.alpha = float(beta_ols[0])
            self.sigma_v = float(np.sqrt(ols_variance))
            self.sigma_u = 0.0
            self.lambda_ = 0.0
            self.sigma_total = self.sigma_v
            self.frontier = self.X @ beta_ols
            self.composed_error = self.y - self.frontier
        self.TE = self.AE
        self.fitted_values = self.frontier - self.u_hat
        self.residuals = self.y - self.fitted_values

    def result(self) -> SFAResult:
        return SFAResult(
            beta=self.beta,
            sigma_v=self.sigma_v,
            sigma_u=self.sigma_u,
            log_likelihood=self.log_likelihood,
            aic=self.aic,
            bic=self.bic,
            converged=self.converged,
            message=self.message,
            n_iter=self.n_iter,
        )

    def summary(self) -> dict:
        return {
            "model_type": self.model_type,
            "beta": self.beta,
            "alpha": self.alpha,
            "sigma_v": self.sigma_v,
            "sigma_u": self.sigma_u,
            "lambda": self.lambda_,
            "AE_mean": float(np.mean(self.AE)),
            "AE_median": float(np.median(self.AE)),
            "TE_mean": float(np.mean(self.AE)),
            "TE_median": float(np.median(self.AE)),
            "u_hat_mean": float(np.mean(self.u_hat)),
            "u_hat_standardized_mean": float(np.mean(self.u_hat_standardized)),
            "normal_log_likelihood": self.normal_log_likelihood,
            "boundary_lr_stat": self.boundary_lr_stat,
            "boundary_mixture_p_value": self.boundary_mixture_p_value,
            "one_sided_component_supported": self.one_sided_component_supported,
            "log_likelihood": self.log_likelihood,
            "AIC": self.aic,
            "BIC": self.bic,
            "converged": self.converged,
            "message": self.message,
            "n_iter": self.n_iter,
        }
