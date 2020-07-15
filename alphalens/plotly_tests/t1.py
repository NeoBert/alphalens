# %%
from numpy import nan
from pandas import (DataFrame, date_range, Timedelta, concat)
import alphalens.performance as perf
from alphalens import plotting
from alphalens import utils

# %%
# 数据
tickers = ['A', 'B', 'C', 'D', 'E', 'F']

factor_groups = {'A': 1, 'B': 2, 'C': 1, 'D': 2, 'E': 1, 'F': 2}

price_data = [[1.25**i, 1.50**i, 1.00**i, 0.50**i, 1.50**i, 1.00**i]
              for i in range(1, 51)]

factor_data = [[3, 4, 2, 1, nan, nan], [3, 4, 2, 1, nan, nan],
               [3, 4, 2, 1, nan, nan], [3, 4, 2, 1, nan, nan],
               [3, 4, 2, 1, nan, nan], [3, 4, 2, 1, nan, nan],
               [3, nan, nan, 1, 4, 2], [3, nan, nan, 1, 4, 2],
               [3, 4, 2, 1, nan, nan], [3, 4, 2, 1, nan, nan],
               [3, nan, nan, 1, 4, 2], [3, nan, nan, 1, 4, 2],
               [3, nan, nan, 1, 4, 2], [3, nan, nan, 1, 4, 2],
               [3, nan, nan, 1, 4, 2], [3, nan, nan, 1, 4, 2],
               [3, nan, nan, 1, 4, 2], [3, nan, nan, 1, 4, 2],
               [3, nan, nan, 1, 4, 2], [3, nan, nan, 1, 4, 2],
               [3, 4, 2, 1, nan, nan], [3, 4, 2, 1, nan, nan],
               [3, 4, 2, 1, nan, nan], [3, 4, 2, 1, nan, nan],
               [3, 4, 2, 1, nan, nan], [3, 4, 2, 1, nan, nan],
               [3, 4, 2, 1, nan, nan], [3, 4, 2, 1, nan, nan],
               [3, nan, nan, 1, 4, 2], [3, nan, nan, 1, 4, 2]]

event_data = [[1, nan, nan, nan, nan, nan],
              [4, nan, nan, 7, nan, nan],
              [nan, nan, nan, nan, nan, nan],
              [nan, 3, nan, 2, nan, nan],
              [1, nan, nan, nan, nan, nan],
              [nan, nan, 2, nan, nan, nan],
              [nan, nan, nan, 2, nan, nan],
              [nan, nan, nan, 1, nan, nan],
              [2, nan, nan, nan, nan, nan],
              [nan, nan, nan, nan, 5, nan],
              [nan, nan, nan, 2, nan, nan],
              [nan, nan, nan, nan, nan, nan],
              [2, nan, nan, nan, nan, nan],
              [nan, nan, nan, nan, nan, 5],
              [nan, nan, nan, 1, nan, nan],
              [nan, nan, nan, nan, 4, nan],
              [5, nan, nan, 4, nan, nan],
              [nan, nan, nan, 3, nan, nan],
              [nan, nan, nan, 4, nan, nan],
              [nan, nan, 2, nan, nan, nan],
              [5, nan, nan, nan, nan, nan],
              [nan, 1, nan, nan, nan, nan],
              [nan, nan, nan, nan, 4, nan],
              [0, nan, nan, nan, nan, nan],
              [nan, 5, nan, nan, nan, 4],
              [nan, nan, nan, nan, nan, nan],
              [nan, nan, 5, nan, nan, 3],
              [nan, nan, 1, 2, 3, nan],
              [nan, nan, nan, 5, nan, nan],
              [nan, nan, 1, nan, 3, nan]]

#
# business days calendar
#
bprice_index = date_range(start='2015-1-10', end='2015-3-22', freq='B')
bprice_index.name = 'date'
bprices = DataFrame(index=bprice_index, columns=tickers, data=price_data)

bfactor_index = date_range(start='2015-1-15', end='2015-2-25', freq='B')
bfactor_index.name = 'date'
bfactor = DataFrame(index=bfactor_index, columns=tickers,
                    data=factor_data).stack()

#
# full calendar
#
price_index = date_range(start='2015-1-10', end='2015-2-28')
price_index.name = 'date'
prices = DataFrame(index=price_index, columns=tickers, data=price_data)

factor_index = date_range(start='2015-1-15', end='2015-2-13')
factor_index.name = 'date'
factor = DataFrame(index=factor_index, columns=tickers,
                   data=factor_data).stack()

#
# intraday factor
#
today_open = DataFrame(index=price_index+Timedelta('9h30m'),
                       columns=tickers, data=price_data)
today_open_1h = DataFrame(index=price_index+Timedelta('10h30m'),
                          columns=tickers, data=price_data)
today_open_1h += today_open_1h*0.001
today_open_3h = DataFrame(index=price_index+Timedelta('12h30m'),
                          columns=tickers, data=price_data)
today_open_3h -= today_open_3h*0.002
intraday_prices = concat([today_open, today_open_1h, today_open_3h]) \
    .sort_index()

intraday_factor = DataFrame(index=factor_index+Timedelta('9h30m'),
                            columns=tickers, data=factor_data).stack()

#
# event factor
#
bevent_factor = DataFrame(index=bfactor_index, columns=tickers,
                          data=event_data).stack()

event_factor = DataFrame(index=factor_index, columns=tickers,
                         data=event_data).stack()

all_prices = [prices, bprices]
all_factors = [factor, bfactor]
all_events = [event_factor, bevent_factor]

# %%
# 参数
long_short = True
group_neutral = False
by_group = False

# %%
# 计算
factor_returns = perf.factor_returns(
    factor, long_short, group_neutral
)

mean_quant_ret, std_quantile = perf.mean_return_by_quantile(
    factor,
    by_group=False,
    demeaned=long_short,
    group_adjust=group_neutral,
)

mean_quant_rateret = mean_quant_ret.apply(
    utils.rate_of_return, axis=0, base_period=mean_quant_ret.columns[0]
)

mean_quant_ret_bydate, std_quant_daily = perf.mean_return_by_quantile(
    factor,
    by_date=True,
    by_group=False,
    demeaned=long_short,
    group_adjust=group_neutral,
)

mean_quant_rateret_bydate = mean_quant_ret_bydate.apply(
    utils.rate_of_return,
    axis=0,
    base_period=mean_quant_ret_bydate.columns[0],
)

compstd_quant_daily = std_quant_daily.apply(
    utils.std_conversion, axis=0, base_period=std_quant_daily.columns[0]
)

alpha_beta = perf.factor_alpha_beta(
    factor, factor_returns, long_short, group_neutral
)

mean_ret_spread_quant, std_spread_quant = perf.compute_mean_returns_spread(
    mean_quant_rateret_bydate,
    factor["factor_quantile"].max(),
    factor["factor_quantile"].min(),
    std_err=compstd_quant_daily,
)

# %%
# 图表
plotting.plot_returns_table(
    alpha_beta, mean_quant_rateret, mean_ret_spread_quant
)

# %%
