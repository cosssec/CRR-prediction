import pandas as pd
from numpy import exp, log
from scipy.special import gammaln
from scipy.optimize import minimize
from math import isclose
import random


MODEL_DEF = {
    "params": ["alpha", "beta"],
    "model_name": "sbg",
    "transform_func": "sbg"
}


def fit(df_data: pd.DataFrame):

    def sum_ll(params):
        alpha = params[0]
        beta = params[1]

        df_data['ll'] = df_data.apply(
            lambda x: ll(x['t'], x['delta'], alpha, beta) * x['amount'],
            axis=1
        )

        return - df_data['ll'].sum()

    params_init = [random.random(), random.random()]
    bnds = ((0.0001, 10000), (0.0001, 10000))

    res = minimize(
        sum_ll,
        params_init,
        bounds=bnds,
        method='Nelder-Mead'
    )

    if any(
            isclose(param, 0.0001, rel_tol=0.05, abs_tol=0.001) or isclose(param, 100000, rel_tol=0.05, abs_tol=0.001)
            for param in res.x):
        raise "Inappropriate funnel"

    return res


def pdf(t, alpha_, beta_):
    return alpha_ * exp(
        gammaln(alpha_ + beta_) + gammaln(beta_ + t - 1) -
        gammaln(beta_) - gammaln(alpha_ + beta_ + t)
    )


def survival_function(t, alpha_, beta_):
    return exp(
        gammaln(alpha_ + beta_) + gammaln(beta_ + t) -
        gammaln(beta_) - gammaln(alpha_ + beta_ + t)
    )


def coeff(t, alpha_, beta_):
    return (beta_ - exp(
        gammaln(alpha_ + beta_) + gammaln(beta_ + t + 1) - gammaln(beta_) - gammaln(alpha_ + beta_ + t)
    )
            ) / (alpha_ - 1) + 1 if t > 0 else 1


def ll(t, delta, alpha_, beta_):
    return delta * log(pdf(t, alpha_, beta_)) + (1 - delta) * log(survival_function(t, alpha_, beta_))
