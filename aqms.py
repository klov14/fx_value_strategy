import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt


def calc_log_returns(df, num_periods=1):
    """
    Calculate log returns of a given price column in a DataFrame.

    Parameters:
    df (pd.DataFrame):     DataFrame containing price data.
    num_periods (integer): Numbers of periods to calculate log return over (default = 1)
    Returns:
    pd.Series: A pandas Series containing log returns.
    """
    log_returns = np.log(df / df.shift(num_periods))
    return log_returns


def calculate_portfolio_performance(holdings, returns, days_lag=0):
    """
    Calculates portfolio performance using holdings and returns DataFrames.

    Parameters:
    holdings (pd.DataFrame): DataFrame of asset weights (rows = time, columns = assets)
    returns (pd.DataFrame): DataFrame of asset returns (rows = time, columns = assets)
    benchmark_returns (pd.Series, optional): Series of benchmark returns for information ratio calculation

    Returns:
    dict: A dictionary containing DataFrames with portfolio returns, asset-wise contributions, and statistics.
    """

    time_d = (holdings.index[1:] - holdings.index[:-1])

    periods_per_year = 365 / (time_d.days).to_series().mean()

    # Ensure data alignment
    holdings, returns = holdings.align(returns, join='inner', axis=0)
    holdings, returns = holdings.align(returns, join='inner', axis=1)

    # Portfolio returns calculation
    portfolio_returns = (holdings.shift(days_lag) * returns).sum(axis=1)

    # Asset-wise contributions
    asset_returns = holdings.shift(days_lag) * returns

    portfolio_cum_returns = portfolio_returns.cumsum()

    asset_cum_returns = asset_returns.cumsum()

    portfolio_ann_return = portfolio_returns.mean() * periods_per_year

    portfolio_ann_std = portfolio_returns.std() * np.sqrt(periods_per_year)

    portfolio_info_ratio = portfolio_ann_return / portfolio_ann_std if portfolio_ann_std != 0 else np.nan

    transactions = holdings.diff(1)

    turnover = transactions.abs().sum().sum() * periods_per_year / len(holdings.index)

    holding_period = 2 * (holdings.abs().sum(axis=1) / (turnover / periods_per_year)).mean()

    stats = {
        'portfolio_ann_return Return': portfolio_ann_return,
        'portfolio_ann_std': portfolio_ann_std,
        'portfolio_info_ratio': portfolio_info_ratio,
        'holding_period': holding_period,
        'turnover': turnover
    }

    return {
        'portfolio_returns': portfolio_returns.to_frame('portfolio_returns'),
        'portfolio_cum_returns': portfolio_cum_returns.to_frame('portfolio_cum_returns'),
        'asset_cum_returns': asset_cum_returns,
        'asset_returns': asset_returns,
        'perf_stats': pd.DataFrame([stats])
    }


def calc_stacked_cov(df, half_life_in_years=1, seed_length_years=1, return_period_days=5):
    """
    Calculates a time series of exponentially weighted moving average (EWMA) covariance matrices.

    For each date starting after a seed period, the function computes the EWMA covariance matrix
    based on rolling returns. The covariances are annualized and stacked across dates into a
    multi-indexed DataFrame.

    Args:
        df (pd.DataFrame): DataFrame of asset prices or returns, indexed by date.
        half_life_in_years (float, optional): Half-life for EWMA decay in years. Default is 1 year.
        seed_length_years (float, optional): Initial period (in years) to skip before starting the covariance calculations. Default is 1 year.
        return_period_days (int, optional): Number of days over which returns are aggregated before calculating covariance. Default is 5 days.

    Returns:
        pd.DataFrame: A multi-indexed DataFrame where:
                      - First level of columns is the date.
                      - Second level is the asset.
                      Each column contains the annualized covariance values for that date and asset pair.

    Notes:
        - Assumes approximately 260 trading days in a year.
        - Uses EWMA with a span calculated based on the specified half-life.
        - Annualizes the covariance matrices by scaling by (260 / return_period_days).
        - Requires `tqdm` for progress tracking.
    """
    days_in_year = 260
    half_life_days = days_in_year * half_life_in_years
    seed_days = int(days_in_year * seed_length_years)
    span_for_ewma = half_life_days / np.log(2)

    cov_dict = {}

    # Precompute rolling returns once outside the loop for efficiency
    rolling_returns = df.rolling(return_period_days).sum()

    for curr_date in tqdm(df.index[seed_days:], desc="Calculating stacked covariances"):
        curr_data = rolling_returns.loc[:curr_date]

        if len(curr_data) > 1:
            ewma_cov = curr_data.ewm(span=span_for_ewma).cov(pairwise=True).loc[curr_date]
            cov_dict[curr_date] = ewma_cov * (days_in_year / return_period_days)

    concat_cov = pd.concat(cov_dict, axis=1)
    concat_cov.columns.names = ['Date', 'Asset']

    return concat_cov


def calc_portfolio_vol(df, cov_matrix):
    """
    Calculates the portfolio volatility over time based on the provided holdings and covariance matrices.

    For each date in the holdings DataFrame, the function matches the appropriate covariance matrix,
    aligns the assets between holdings and covariance, and computes the portfolio volatility using
    the standard quadratic form.

    Args:
        df (pd.DataFrame): DataFrame of portfolio holdings (weights) indexed by date.
        cov_matrix (pd.DataFrame): Multi-indexed DataFrame of covariance matrices, where the first
                                   column level is the date and the second is the asset.

    Returns:
        pd.DataFrame: A DataFrame indexed by date with a single column 'portfolio_vol' containing
                      the portfolio's volatility for each date.

    Notes:
        - Requires `match_cov` function to find the correct covariance matrix for a given date.
        - Requires `intersect_cov` function to align holdings and covariance matrix to common assets.
        - Holdings with missing values are treated as zero.
        - Volatility is calculated as the square root of the quadratic form: √(wᵀ Σ w).
    """

    df_out = pd.DataFrame(index=df.index, columns=['portfolio_vol'])

    for i, x in enumerate(df.index):
        curr_date = datetime.strftime(x, '%Y-%m-%d')

        curr_cov = match_cov(cov_matrix, curr_date)

        curr_data = df.loc[curr_date]
        curr_data = curr_data.to_frame().T
        curr_data, curr_cov = intersect_cov(curr_data, curr_cov)
        curr_data = curr_data.fillna(0)

        curr_risk = np.sqrt(np.dot(curr_data, np.dot(curr_cov, curr_data.T)))

        df_out.iloc[i, :] = curr_risk

    return df_out


def match_cov(cov_matrix, date_required):
    """
    Matches and retrieves the appropriate covariance matrix for a given required date.

    The function searches the covariance matrix for the earliest available date
    that is greater than or equal to the `date_required`. If no such date exists
    (i.e., `date_required` is beyond the latest available date), it defaults
    to using the last available covariance matrix.

    Args:
        cov_matrix (pd.DataFrame): Multi-indexed DataFrame of covariance matrices,
                                   where the first level of columns is 'Date' and
                                   the second level is 'Asset'.
        date_required (str): Date string in 'YYYY-MM-DD' format specifying the desired date.

    Returns:
        pd.DataFrame: Covariance matrix (single date slice) with assets as columns and index.

    Notes:
        - Assumes that `cov_matrix` columns have a multi-index with 'Date' as the first level.
        - If multiple matrices match, the function selects the first valid one.
        - If no future dates are available, it falls back to the most recent past matrix.
    """

    cov_dates = cov_matrix.columns.get_level_values('Date')

    # Find indices where date is greater than or equal to the required date
    valid_indices = np.where(cov_dates >= date_required)[0]

    if valid_indices.size > 0:
        indx = valid_indices.min()
    else:
        # Use the last available date if date_required is beyond the latest cov_date
        indx = -1

    selected_date = cov_dates[indx]
    cov_out = cov_matrix.xs(selected_date, level='Date', axis=1)

    return cov_out

    import pandas as pd


def intersect_cov(df1: pd.DataFrame, df_cov: pd.DataFrame):
    """
    Takes an asset returns DataFrame (df1) and a covariance matrix (df_cov),
    returning both with only the intersecting assets.

    Parameters:
    df1 (pd.DataFrame): DataFrame with assets as columns and datetime as index
    df_cov (pd.DataFrame): Covariance matrix with assets as both index and columns

    Returns:
    pd.DataFrame, pd.DataFrame: Filtered asset returns DataFrame and covariance matrix
    """
    # Find common assets and maintain order from df_cov
    common_assets = [asset for asset in df_cov.index if asset in df1.columns]
    # Subset DataFrames
    df1_common = df1[common_assets].copy()
    df_cov_common = df_cov.loc[common_assets, common_assets].copy()

    return df1_common, df_cov_common


def target_vol(df, cov_matrix, volatility=0.01):
    """
    Scales portfolio holdings to target a specified volatility level over time.

    For each date, the function matches the corresponding covariance matrix,
    aligns the assets between the holdings and the covariance matrix, calculates
    the current portfolio volatility, and scales the holdings proportionally
    to achieve the desired target volatility.

    Args:
        df (pd.DataFrame): DataFrame of portfolio holdings (weights), indexed by date.
        cov_matrix (pd.DataFrame): Multi-indexed DataFrame of covariance matrices,
                                   with the first level as dates and the second as assets.
        volatility (float, optional): Target portfolio volatility level. Default is 0.01 (1%).

    Returns:
        tuple:
            - pd.DataFrame: Scaled holdings DataFrame, same shape as input `df`.
            - pd.DataFrame: DataFrame of scale factors applied at each date (`scale_fac`).

    Notes:
        - Requires `match_cov` function to select the appropriate covariance matrix for each date.
        - Requires `intersect_cov` function to align holdings and covariance to common assets.
        - Holdings with missing values are treated as zero before scaling.
        - Scaling factor is computed as: target_volatility / current_portfolio_volatility.
    """

    df_out = pd.DataFrame(index=df.index, columns=df.columns)
    df_scale_factor = pd.DataFrame(index=df.index, columns=['scale_fac'])

    for i, x in enumerate(df.index):
        curr_date = datetime.strftime(x, '%Y-%m-%d')

        curr_cov = match_cov(cov_matrix, curr_date)

        curr_data = df.loc[curr_date]
        curr_data = curr_data.to_frame().T

        curr_data, curr_cov = intersect_cov(curr_data, curr_cov)
        curr_data = curr_data.fillna(0)

        curr_risk = np.sqrt(np.dot(curr_data, np.dot(curr_cov, curr_data.T)))

        curr_sf = volatility / curr_risk

        data_scaled = curr_data * curr_sf

        df_out.loc[curr_date, data_scaled.columns] = data_scaled.loc[curr_date].values

        df_scale_factor.loc[curr_date] = curr_sf
    return df_out, df_scale_factor


def scale_by_vol(df, cov_matrix):
    """
    Scales portfolio holdings by their corresponding asset volatilities on each date.

    For each date, the function matches the relevant covariance matrix, aligns it with
    the current holdings, and divides each holding weight by the asset's volatility
    (i.e., the square root of the variance) to standardize exposures.

    Args:
        df (pd.DataFrame): DataFrame of portfolio holdings (weights), indexed by date.
        cov_matrix (pd.DataFrame): Multi-indexed DataFrame of covariance matrices,
                                   with the first level as dates and the second as assets.

    Returns:
        pd.DataFrame: A DataFrame of the same shape as `df`, where each holding has been
                      scaled by the corresponding asset's volatility for that date.

    Notes:
        - Requires `match_cov` function to retrieve the covariance matrix for a given date.
        - Requires `intersect_cov` function to align holdings and covariance matrices to common assets.
        - Holdings with missing values are treated as zero before scaling.
        - Scaling is performed by dividing holdings by the square root of the diagonal entries of the covariance matrix (individual asset volatilities).
    """

    df_out = pd.DataFrame(index=df.index, columns=df.columns)

    for i, x in enumerate(df.index):
        curr_date = datetime.strftime(x, '%Y-%m-%d')

        curr_cov = match_cov(cov_matrix, curr_date)

        curr_data = df.loc[curr_date]
        curr_data = curr_data.to_frame().T

        curr_data, curr_cov = intersect_cov(curr_data, curr_cov)
        curr_data = curr_data.fillna(0)

        curr_vol = np.sqrt(np.diagonal(curr_cov))

        data_scaled = curr_data.div(curr_vol)

        df_out.loc[curr_date, data_scaled.columns] = data_scaled.loc[curr_date].values
    return df_out


def evaluate_strategy(strategy_returns):
    portfolio_cum_returns = (1 + strategy_returns).cumprod()
    portfolio_cum_returns.plot(title="Cumulative Return")
    plt.ylabel("Cumulative Return")
    plt.xlabel("Date")
    plt.grid(True)
    plt.show()
    plt.show()

    running_max = portfolio_cum_returns.cummax()

    # Calculate drawdown
    drawdown = portfolio_cum_returns / running_max - 1

    # Calculate maximum drawdown
    max_drawdown = drawdown.min()

    annual_return = (1 + strategy_returns.mean()) ** 12 - 1
    annual_volatility = strategy_returns.std() * (12 ** 0.5)
    sharpe_ratio = (annual_return-0.01) / annual_volatility


    stats = {
        #'portfolio_returns': strategy_returns,
        'portfolio_cum_returns': portfolio_cum_returns.iloc[-1],
        'ann_return': annual_return,
        'ann_vol': annual_volatility,
        'sharpe': sharpe_ratio,
        'max_drawdown(%)': max_drawdown*100
    }

    return stats


def calc_lead_lag_IR(df, returns, lead_lag_range):
    """
    Calculates the portfolio's Information Ratio (IR) across different lead-lag shifts of the holdings.

    Shifts the holdings DataFrame by different lead-lag values and evaluates how the
    Information Ratio changes for each shift. This can help assess the timing sensitivity
    of the portfolio strategy.

    Args:
        df (pd.DataFrame): DataFrame of portfolio holdings (weights), where columns represent assets.
        returns (pd.DataFrame): DataFrame of asset returns.
        lead_lag_range (iterable): Range or list of integers representing the number of periods to shift holdings.
                                   Positive values imply leading (future) holdings, negative values imply lagging.

    Returns:
        pd.DataFrame: A DataFrame indexed by the lead-lag values, with a single column
                      'portfolio_info_ratio' showing the IR for each shift.

    Notes:
        - The function assumes the existence of `calculate_portfolio_performance`.
        - Shifting introduces NaNs which are dropped (`dropna`) to maintain alignment.
        - `days_lag` parameter in performance calculation is set to 0 to match shifted holdings directly.
    """

    df_out = pd.DataFrame(index=lead_lag_range, columns=['portfolio_info_ratio'])
    for i, x in enumerate(lead_lag_range):
        holds_shifted = df.shift(x).dropna()
        results = calculate_portfolio_performance(holds_shifted, returns, days_lag=0)
        df_out.iloc[i, :] = results['perf_stats']['portfolio_info_ratio']

    return df_out


def plot_bootstrapped_rtns(df_insample, df_all, num_bootstraps):
    """
    Plots bootstrapped cumulative return paths against actual out-of-sample returns.

    This function performs a bootstrap simulation on in-sample returns to generate
    multiple hypothetical out-of-sample return paths. It compares these simulated paths
    with the actual out-of-sample returns from the full return dataset.

    Parameters:
    ----------
    df_insample : pandas.DataFrame
        DataFrame of in-sample returns. Assumed to be a single-column DataFrame with a datetime index.

    df_all : pandas.DataFrame
        DataFrame of full returns including both in-sample and out-of-sample periods.
        Must share the same structure and index format as df_insample.

    num_bootstraps : integer
        Number of bootstrap samples to use


    Notes:
    ------
    - Generates 'num_bootstraps' bootstrapped return paths by randomly sampling with replacement
      from in-sample returns.
    - Computes the cumulative return for each path and plots:
        - All bootstrapped paths
        - The mean of the bootstrapped paths
        - A 5–95% confidence band
        - The actual cumulative return in the out-of-sample period
    - Plots are aligned by date using the out-of-sample date index.

    Returns:
    -------
    None
        Displays a matplotlib plot.
    """
    # -------------------
    # INPUTS
    # -------------------
    # Assume you have:
    # df_insample (your in-sample returns)
    # df_all (full returns, including in-sample and out-of-sample)

    n_paths = num_bootstraps
    n_days = len(df_all.index.difference(df_insample.index))  # OOS period length

    # -------------------
    # BOOTSTRAPPING
    # -------------------

    bootstrap_paths = []

    for i in range(n_paths):
        sampled_returns = df_insample.sample(n=n_days, replace=True).values.flatten()
        # cumulative_returns = (1 + sampled_returns).cumprod() - 1
        cumulative_returns = sampled_returns.cumsum()
        bootstrap_paths.append(cumulative_returns)

    bootstrap_paths = np.array(bootstrap_paths)  # Shape (n_paths, n_days)

    # -------------------
    # SUMMARY STATS
    # -------------------
    mean_path = np.mean(bootstrap_paths, axis=0)
    lower_band = np.percentile(bootstrap_paths, 5, axis=0)
    upper_band = np.percentile(bootstrap_paths, 95, axis=0)

    # Force them to pure floats
    mean_path = np.array(mean_path, dtype=float)
    lower_band = np.array(lower_band, dtype=float)
    upper_band = np.array(upper_band, dtype=float)

    # -------------------
    # OUT-OF-SAMPLE RETURNS
    # -------------------
    out_of_sample_dates = df_all.index.difference(df_insample.index)
    out_of_sample_returns = df_all.loc[out_of_sample_dates].sort_index()
    # out_of_sample_cum_returns = (1 + out_of_sample_returns).cumprod() - 1
    out_of_sample_cum_returns = out_of_sample_returns.cumsum()

    # -------------------
    # PLOTTING
    # -------------------

    x_dates = out_of_sample_returns.index
    x_numeric = np.arange(len(x_dates))

    plt.figure(figsize=(12, 7))

    # Plot all bootstrap paths
    for path in bootstrap_paths:
        plt.plot(x_numeric, path, color='lightblue', alpha=0.3)

    # Plot mean path
    plt.plot(x_numeric, mean_path, color='blue', lw=2, label='Mean Bootstrap Path')

    # Plot confidence interval
    plt.fill_between(
        x_numeric,
        lower_band,
        upper_band,
        color='blue',
        alpha=0.2,
        label='5-95% Confidence Band'
    )

    # Plot actual out-of-sample returns
    plt.plot(x_numeric, out_of_sample_cum_returns.values, color='black', lw=2.5, label='Actual Out-of-Sample')

    # Format x-axis
    ax = plt.gca()
    ax.set_xticks(x_numeric[::max(len(x_numeric) // 10, 1)])
    ax.set_xticklabels([date.strftime('%Y-%m-%d') for date in x_dates[::max(len(x_numeric) // 10, 1)]], rotation=45)

    plt.title('Bootstrapped Cumulative Returns vs Actual (Out-of-Sample)', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Cumulative Return', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
