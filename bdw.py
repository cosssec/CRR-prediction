import pandas as pd
from numpy import exp, log, inf
from scipy.special import gammaln
from scipy import optimize
from math import isclose
import random


MODEL_DEF = {
    "params": ["gamma", "delta", "c"],
    "model_name": "bdw",
    "transform_func": "bdw"
}


def fit(df_data: pd.DataFrame):

    def sum_ll(params):
        gamma = params[0]
        delta = params[1]
        c = params[2]

        df_data['ll'] = df_data.apply(
            lambda x: ll(x['t'], x['delta'], gamma, delta, c) * x['amount'],
            axis=1
        )
        return - df_data['ll'].sum()

    params_init = [random.random(), random.random(), random.random()]
    bnds = ((0.0001, 10000), (0.0001, 10000), (0.0001, 3))

    res = optimize.minimize(
        sum_ll,
        params_init,
        bounds=bnds,
        method='Powell'    # ?
    )

    assert any(
        isclose(param, 0.0001, rel_tol=0.05, abs_tol=0.001) or
        isclose(param, 10000, rel_tol=0.05, abs_tol=0.001)
        for param in res.x
    ) is False, "Inappropriate funnel"

    return res


def pdf(t, gamma_, delta_, c):
    return survival_function(t - 1, gamma_, delta_, c) - survival_function(t, gamma_, delta_, c)


def survival_function(t, gamma_, delta_, c):
    return exp(
        gammaln(gamma_ + delta_) + gammaln(delta_ + t**c) -
        gammaln(delta_) - gammaln(gamma_ + delta_ + t**c)
    )


def coeff(t, gamma_, delta_, c):
    # no explicit form
    return sum(
        [survival_function(i, gamma_, delta_, c) for i in range(0, t + 1)]
    )


def ll(t, delta, gamma_, delta_, c):
    return delta * log(pdf(t, gamma_, delta_, c)) + \
           (1 - delta) * log(survival_function(t, gamma_, delta_, c))
