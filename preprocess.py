import logging
import json

import pandas as pd
import numpy as np

from datetime import date
from ltv.constants import (
    DIMENSIONS,
    SLICES,
    MAX_TENURE,
    MIN_TENURE,
    STATIC_LIFETIMES,
    LEVELS_TO_IGNORE,
    MIN_SUBS,
    TRUE_DURATION,
    WINDOW_SIZE
)

def moving_calculation(df, window_size):
    # Pivot Table
    pivot = df.pivot_table(values='renewals', index='tenure', columns=['iteration'], aggfunc='sum').sort_index(ascending=False)
    # Add missing tenures (filled with NaNs)
    pivot = pivot.reindex(range(max(pivot.index), min(pivot.index)-1, -1))
    # Calculate moving sum
    moving_calc = pivot.rolling(window_size).sum()
    # Calculate moving sum with smaller window where can't with full
    for i in reversed(range(1, window_size)):
        moving_calc.loc[moving_calc[0].isna()] = pivot.rolling(i).sum().loc[moving_calc[0].isna()]
    # Unpivot table back
    unpivoted = pd.melt(moving_calc.reset_index(), id_vars=['tenure'], var_name='iteration', value_name='renewals')
    unpivoted = unpivoted.loc[(unpivoted['iteration'] <= unpivoted['tenure']) & unpivoted['renewals'].notna()]
    unpivoted['renewals'] = unpivoted['renewals'].astype(int)
    return unpivoted


def preprocess_level(df_level, level_info):
    tenures = df_level['tenure'].unique()

    seq = [(i, j) for i in tenures for j in range(i + 1)]
    df_tenure = pd.DataFrame(seq, columns=['tenure', 'iteration'])
    if len(df_tenure) != len(df_level):
        df_level = pd.merge(df_level, df_tenure, on=['tenure', 'iteration'], how='outer')
        df_level['renewals'].fillna(0, inplace=True)

    good_tenures = df_level.loc[
        (df_level['iteration'] == 0) & (df_level['renewals'] >= MIN_SUBS), 'tenure'
    ].unique()

    level_min_tenure = MIN_TENURE[level_info['duration_interval']]
    
    if np.count_nonzero(good_tenures >= level_min_tenure) < 2:
        print('Used moving calculation, not enough data')
        df_level = moving_calculation(df_level, WINDOW_SIZE[level_info['duration_interval']])
        good_tenures = df_level.loc[
            (df_level['iteration'] == 0) & (df_level['renewals'] >= MIN_SUBS), 'tenure'
        ].unique()

    if not any(tenure >= level_min_tenure for tenure in good_tenures):
        raise ValueError("Short funnel")
    
    df_level = df_level.loc[df_level['tenure'].isin(good_tenures)]

    # empty data
    if df_level.empty:
        raise ValueError("Not enough subs")

    # sort
    df_level.sort_values(['tenure', 'iteration'], inplace=True)

    # amount
    df_level.reset_index(inplace=True)
    df_level.loc[:, 'amount'] = df_level.groupby('tenure')['renewals'].shift(-1)

    # last iteration in tenure is renewals in amount
    df_level['amount'].fillna(0, inplace=True)

    # drops and renewals
    df_level.loc[:, 'amount'] = df_level['renewals'] - df_level['amount']

    # poor values processing
    df_level.loc[:, 'amount'] = df_level['amount'].apply(lambda x: 0 if x < 0 else x)

    # mark renewals/drops
    df_level.loc[:, 'delta'] = df_level.apply(lambda x: 0 if x['tenure'] == x['iteration'] else 1, axis=1)

    # t; remark drop iters
    df_level.loc[:, 't'] = df_level.apply(lambda x: x['iteration'] + 1 if x['delta'] == 1 else x['iteration'], axis=1)

    return df_level