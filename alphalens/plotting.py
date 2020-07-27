import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import statsmodels.api as sm
from plotly.subplots import make_subplots
from scipy import stats

from . import performance as perf
from . import utils

plotly.offline.init_notebook_mode(connected=True)

DECIMAL_TO_BPS = 10000
F_DIGIT = 4


def _date_tickformat(fig, fmt=r'%m月%d日<br>%Y年'):
    fig.update_layout(xaxis_tickformat=fmt)


def _round(df):
    for col in df.columns:
        if pd.api.types.is_float_dtype(df[col]):
            df[col] = df[col].round(decimals=F_DIGIT)
    return df


def _print_table(table, title):
    if isinstance(table, pd.Series):
        table = pd.DataFrame(table)

    if isinstance(table, pd.DataFrame):
        table.columns.name = None

    table = _round(table)

    header_values = [''] + [f"{k} <br>"for k in table.columns]
    cell_values = [table.index.to_list()] + [table[k].tolist()
                                             for k in table.columns]
    fig = go.Figure(data=[
        go.Table(header=dict(
            values=header_values,
            align=['left', 'right'],
            fill_color='paleturquoise',
        ), cells=dict(
            values=cell_values,
            fill_color='lavender',
            align=['left', 'right'])
        )],
        layout={"title": {"text": title},
                'height': 350})
    fig.show()


def plot_returns_table(alpha_beta,
                       mean_ret_quantile,
                       mean_ret_spread_quantile):
    returns_table = pd.DataFrame()
    returns_table = returns_table.append(alpha_beta)
    returns_table.loc["顶部分位数期间平均收益率(基点)"] = mean_ret_quantile.iloc[-1] * \
        DECIMAL_TO_BPS
    returns_table.loc["底部分位数期间平均收益率(基点)"] = mean_ret_quantile.iloc[0] * \
        DECIMAL_TO_BPS
    returns_table.loc["期间平均展布(基点)"] = mean_ret_spread_quantile.mean(
    ) * DECIMAL_TO_BPS

    title = "收益率分析"
    # table = returns_table.apply(lambda x: x.round(3))
    _print_table(returns_table, title)


def plot_turnover_table(autocorrelation_data, quantile_turnover):
    turnover_table = pd.DataFrame()
    for period in sorted(quantile_turnover.keys()):
        for quantile, p_data in quantile_turnover[period].iteritems():
            turnover_table.loc["分位数 {} 平均换手率".format(quantile),
                               "{}D".format(period)] = p_data.mean()
    auto_corr = pd.DataFrame()
    for period, p_data in autocorrelation_data.iteritems():
        auto_corr.loc["平均因子秩自相关",
                      "{}D".format(period)] = p_data.mean()

    title = "换手率分析"
    # utils.print_table(turnover_table.apply(lambda x: x.round(3)))
    tabel_1 = turnover_table.apply(lambda x: x.round(3))
    _print_table(tabel_1, title)
    # utils.print_table(auto_corr.apply(lambda x: x.round(3)))
    table_2 = auto_corr.apply(lambda x: x.round(3))
    _print_table(table_2, '')


def plot_quantile_statistics_table(factor_data):
    quantile_stats = factor_data.groupby('factor_quantile') \
        .agg(['min', 'max', 'mean', 'std', 'count'])['factor']
    quantile_stats['count %'] = quantile_stats['count'] \
        / quantile_stats['count'].sum() * 100.

    title = "分位数统计"
    _print_table(quantile_stats, title)


def plot_quantile_returns_bar(mean_ret_by_q,
                              by_group=False,
                              ylim_percentiles=None):
    """
    Plots mean period wise returns for factor quantiles.

    Parameters
    ----------
    mean_ret_by_q : pd.DataFrame
        DataFrame with quantile, (group) and mean period wise return values.
    by_group : bool
        Disaggregated figures by group.
    ylim_percentiles : tuple of integers
        Percentiles of observed data to use as y limits for plot.
    """

    mean_ret_by_q = mean_ret_by_q.copy()
    colors = px.colors.qualitative.Plotly

    if ylim_percentiles is not None:
        ymin = (np.nanpercentile(mean_ret_by_q.values,
                                 ylim_percentiles[0]) * DECIMAL_TO_BPS)
        ymax = (np.nanpercentile(mean_ret_by_q.values,
                                 ylim_percentiles[1]) * DECIMAL_TO_BPS)
    else:
        ymin = None
        ymax = None
    if by_group:
        group_keys = sorted(
            mean_ret_by_q.index.get_level_values('group').unique())
        num_group = len(group_keys)
        grouped = mean_ret_by_q.groupby(level='group')
        v_spaces = ((num_group - 1) // 2) + 1
        # 每次显示二列
        for i in range(v_spaces):
            subplot_titles = group_keys[i*2:(i+1)*2]
            # 如实际只有一列，也设定为2
            gf = make_subplots(rows=1, cols=2,
                               y_title='收益率(基点)',
                               subplot_titles=subplot_titles,
                               shared_yaxes=True)
            for j, sc in enumerate(subplot_titles, 1):
                cor = grouped.get_group(sc)
                bar_data = cor.xs(sc, level='group').multiply(DECIMAL_TO_BPS)
                columns = bar_data.columns
                for k, name in enumerate(columns):
                    gf.add_trace(
                        go.Bar(name=name,
                               legendgroup=sc,
                               marker_color=colors[k],
                               showlegend=True if j == 1 else False,
                               x=bar_data.index,
                               y=bar_data[name].values),
                        row=1, col=j)
            gf.update_layout(barmode='group')
            gf.update_yaxes(range=[ymin, ymax])
            gf.show()
    else:
        # gf = make_subplots(x_title='周期频率', y_title='收益率(基点)')
        bar_data = mean_ret_by_q.multiply(DECIMAL_TO_BPS)
        columns = bar_data.columns
        bar_data.reset_index(inplace=True)

        fig = px.bar(bar_data, x='factor_quantile', y=columns, barmode='group')

        fig.update_layout(
            title_text="因子分位数分组期间平均收益率")
        fig.update_xaxes(title_text='分位数')
        fig.update_yaxes(range=[ymin, ymax], title_text='收益率(基点)')
        fig.update_layout(
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        fig.show()


def plot_information_table(ic_data):
    ic_summary_table = pd.DataFrame()
    ic_summary_table["IC Mean"] = ic_data.mean()
    ic_summary_table["IC Std."] = ic_data.std()
    ic_summary_table["Risk-Adjusted IC"] = \
        ic_data.mean() / ic_data.std()
    t_stat, p_value = stats.ttest_1samp(ic_data, 0)
    ic_summary_table["t-stat(IC)"] = t_stat
    ic_summary_table["p-value(IC)"] = p_value
    ic_summary_table["IC Skew"] = stats.skew(ic_data)
    ic_summary_table["IC Kurtosis"] = stats.kurtosis(ic_data)

    title = "信息分析"
    # utils.print_table(ic_summary_table.apply(lambda x: x.round(3)).T)
    table = ic_summary_table.apply(lambda x: x.round(3)).T
    _print_table(table, title)


def plot_quantile_returns_violin(return_by_q,
                                 ylim_percentiles=None):
    """
    Plots a violin box plot of period wise returns for factor quantiles.

    Parameters
    ----------
    return_by_q : pd.DataFrame - MultiIndex
        DataFrame with date and quantile as rows MultiIndex,
        forward return windows as columns, returns as values.
    ylim_percentiles : tuple of integers
        Percentiles of observed data to use as y limits for plot.
    """

    return_by_q = return_by_q.copy()

    if ylim_percentiles is not None:
        ymin = (np.nanpercentile(return_by_q.values,
                                 ylim_percentiles[0]) * DECIMAL_TO_BPS)
        ymax = (np.nanpercentile(return_by_q.values,
                                 ylim_percentiles[1]) * DECIMAL_TO_BPS)
    else:
        ymin = None
        ymax = None

    gf = make_subplots(rows=1, cols=1, x_title='分位数',
                       y_title='收益率(基点)', shared_xaxes=True)

    unstacked_dr = return_by_q.multiply(DECIMAL_TO_BPS)
    unstacked_dr.columns = unstacked_dr.columns.set_names('forward_periods')
    unstacked_dr = unstacked_dr.stack()
    unstacked_dr.name = 'return'
    unstacked_dr = unstacked_dr.reset_index()

    groups = unstacked_dr['forward_periods'].unique()
    for name in groups:
        gf.add_trace(go.Violin(x=unstacked_dr['factor_quantile'][unstacked_dr['forward_periods'] == name],
                               y=unstacked_dr['return'][unstacked_dr['forward_periods'] == name],
                               box_visible=False,
                               legendgroup=name, scalegroup=name, name=name))

    gf.update_traces(meanline_visible=True)
    gf.update_layout(violinmode='group')
    gf.update_layout(title_text="因子分位数期间收益率")
    gf.update_yaxes(range=[ymin, ymax])
    gf.update_layout(
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    gf.show()


def plot_cumulative_returns(factor_returns,
                            period,
                            freq=None,
                            title=None):
    """
    Plots the cumulative returns of the returns series passed in.

    Parameters
    ----------
    factor_returns : pd.Series
        Period wise returns of dollar neutral portfolio weighted by factor
        value.
    period : pandas.Timedelta or string
        Length of period for which the returns are computed (e.g. 1 day)
        if 'period' is a string it must follow pandas.Timedelta constructor
        format (e.g. '1 days', '1D', '30m', '3h', '1D1h', etc)
    freq : pandas DateOffset
        Used to specify a particular trading calendar e.g. BusinessDay or Day
        Usually this is inferred from utils.infer_trading_calendar, which is
        called by either get_clean_factor_and_forward_returns or
        compute_forward_returns
    title: string, optional
        Custom title
    """
    title = "投资组合累积收益率({} 预测周期)".format(period) if title is None else title
    gf = make_subplots(rows=1, cols=1, y_title='累积收益率')
    factor_returns = perf.cumulative_returns(factor_returns)
    gf.add_trace(
        go.Scatter(x=factor_returns.index, y=factor_returns.values,
                   name='累积',
                   opacity=0.6,
                   line=dict(color='forestgreen', width=3))
    )
    gf.add_trace(
        go.Scatter(x=[factor_returns.index[0], factor_returns.index[-1]], y=[1.0]*2, name='基准',
                   line=dict(color='black', width=1, dash='dash'))
    )
    gf.update_layout(title_text=title, yaxis_tickformat='.3f')
    _date_tickformat(gf)
    gf.show()


def plot_cumulative_returns_by_quantile(quantile_returns,
                                        period,
                                        freq=None):
    """
    Plots the cumulative returns of various factor quantiles.

    Parameters
    ----------
    quantile_returns : pd.DataFrame
        Returns by factor quantile
    period : pandas.Timedelta or string
        Length of period for which the returns are computed (e.g. 1 day)
        if 'period' is a string it must follow pandas.Timedelta constructor
        format (e.g. '1 days', '1D', '30m', '3h', '1D1h', etc)
    freq : pandas DateOffset
        Used to specify a particular trading calendar e.g. BusinessDay or Day
        Usually this is inferred from utils.infer_trading_calendar, which is
        called by either get_clean_factor_and_forward_returns or
        compute_forward_returns
    """
    title = f'分位数分组累积收益率({period} 周期预测收益率)'
    # gf = make_subplots(y_title='累积收益率(log)', subplot_titles=[title])
    ret_wide = quantile_returns.unstack('factor_quantile')

    cum_ret = ret_wide.apply(perf.cumulative_returns)

    cum_ret = cum_ret.loc[:, ::-1]

    ymin, ymax = cum_ret.min(skipna=True).min(), cum_ret.max(skipna=True).max()
    fig = go.Figure()
    for col in cum_ret:
        fig.add_trace(go.Scatter(x=cum_ret.index,
                                 y=cum_ret[col].values,
                                 legendgroup="分位数",
                                 mode='lines',
                                 name=col))
    fig.add_trace(
        go.Scatter(x=cum_ret.index, y=[1.0]*len(cum_ret), name='基准',
                   line=dict(color='black', width=1, dash='dash'))
    )
    tickvals = np.logspace(np.log10(ymin), np.log10(ymax), 5, endpoint=True)
    fig.update_layout(
        yaxis_type="log",
        yaxis_tickformat='.2f',
        yaxis=dict(
            tickmode='array',
            tickvals=tickvals,
            ticktext=[f"{x:.2f}" for x in tickvals]
        ),
    )
    fig.update_xaxes(title_text='日期')
    fig.update_yaxes(title_text='累积收益率(log)')
    _date_tickformat(fig)
    fig.update_layout(
        title_text=title,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    fig.show()


def plot_mean_quantile_returns_spread_time_series(mean_returns_spread,
                                                  std_err=None,
                                                  bandwidth=1):
    """
    Plots mean period wise returns for factor quantiles.

    Parameters
    ----------
    mean_returns_spread : pd.Series
        Series with difference between quantile mean returns by period.
    std_err : pd.Series
        Series with standard error of difference between quantile
        mean returns each period.
    bandwidth : float
        Width of displayed error bands in standard deviations.
    """
    def _gen_figure(s1, s2):
        if s1.isnull().all():
            return
        periods = s1.name
        title = ('顶、底分位数之差的平均收益率({} 周期预测收益率)'
                 .format(periods if periods is not None else ""))

        gf = make_subplots(y_title='分位数平均收益率之差(基点)')

        mean_returns_spread_bps = s1 * DECIMAL_TO_BPS

        gf.add_trace(
            go.Scatter(x=mean_returns_spread_bps.index,
                       y=mean_returns_spread_bps.values,
                       name='平均收益率展布',
                       opacity=0.4,
                       line=dict(color='forestgreen', width=0.7))
        )

        m = mean_returns_spread_bps.rolling(window=22).mean()
        gf.add_trace(
            go.Scatter(x=m.index,
                       y=m.values,
                       name='一个月移动平均',
                       opacity=0.7,
                       line=dict(color='orangered', width=0.7))
        )

        if s2 is not None:
            std_err_bps = s2 * DECIMAL_TO_BPS
            upper = mean_returns_spread_bps.values + (std_err_bps * bandwidth)
            lower = mean_returns_spread_bps.values - (std_err_bps * bandwidth)
            x = s2.index.to_list()
            x += x[::-1]
            y = lower.tolist() + upper.tolist()

            gf.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    name='spread',
                    fill='toself',
                    fillcolor="steelblue",
                    opacity=0.3,
                    line_width=0,
                ))

        ylim = np.nanpercentile(abs(mean_returns_spread_bps.values), 95)
        gf.add_shape(
            type="line",
            opacity=0.8,
            x0=s1.index[0],
            y0=0,
            x1=s1.index[-1],
            y1=0,
            line=dict(
                color="black",
                width=1,
                dash="dash",
            )
        )
        gf.update_yaxes(range=[-ylim, ylim])
        gf.update_layout(title_text=title)
        return gf

    if isinstance(mean_returns_spread, pd.DataFrame):
        fig_list = []

        ymin, ymax = (None, None)
        for name, fr_column in mean_returns_spread.iteritems():
            stdn = None if std_err is None else std_err[name]
            fig = _gen_figure(fr_column, stdn)
            fig_list.append(fig)
            range_ = fig.layout.yaxis.range
            if range_:
                curr_ymin, curr_ymax = range_[0], range_[1]
            else:
                curr_ymin, curr_ymax = None, None
            ymin = curr_ymin if ymin is None else min(ymin, curr_ymin)
            ymax = curr_ymax if ymax is None else max(ymax, curr_ymax)

        fig_list = [fig for fig in fig_list if fig]
        # 对图列表统一限定区间，使其图直观可比。为关键一步，不可省略。
        for fig in fig_list:
            fig.update_yaxes(range=[ymin, ymax])
            _date_tickformat(fig)
            fig.show()
    elif isinstance(mean_returns_spread, pd.Series):
        fig = _gen_figure(mean_returns_spread, std_err)
        if fig:
            _date_tickformat(fig)
            fig.show()
    else:
        raise ValueError(
            f'传入错误参数`mean_returns_spread`:\n {mean_returns_spread} std_err:\n{std_err}')


def plot_ic_ts(ic):
    """
    Plots Spearman Rank Information Coefficient and IC moving
    average for a given factor.

    Parameters
    ----------
    ic : pd.DataFrame
        DataFrame indexed by date, with IC for each forward return.
     """
    ic = ic.copy()
    ic.sort_index(inplace=True)
    num_plots = len(ic.columns)
    fig_list = [make_subplots(y_title='IC', shared_xaxes=True)
                for _ in range(num_plots)]
    ymin, ymax = (None, None)
    for fig, (period_num, ic_) in zip(fig_list, ic.iteritems()):
        title = "{} 期间预测收益率信息系数(IC)".format(
            period_num)
        fig.add_trace(
            go.Scatter(x=ic_.index,
                       y=ic_.values,
                       name='IC',
                       opacity=0.7,
                       line=dict(color='steelblue', width=0.7))
        )

        m = ic_.rolling(window=22).mean().dropna()
        fig.add_trace(
            go.Scatter(x=m.index,
                       y=m.values,
                       name='一个月移动平均',
                       opacity=0.8,
                       line=dict(color='forestgreen', width=2))
        )

        fig.add_trace(
            go.Scatter(x=[ic_.index[0], ic_.index[-1]],
                       y=[0.0]*2, name='基准', opacity=0.8,
                       line=dict(color='black', width=1, dash='dash'))
        )

        text = "均值 %.3f \n 标准差 %.3f" % (ic_.mean(), ic_.std())

        range_ = fig.layout.yaxis.range
        if range_:
            curr_ymin, curr_ymax = range_[0], range_[1]
        else:
            curr_ymin, curr_ymax = None, None
        ymin = curr_ymin if ymin is None else min(ymin, curr_ymin)
        ymax = curr_ymax if ymax is None else max(ymax, curr_ymax)

        fig.update_layout(title_text=title)
        fig.add_annotation(
            x=0.05,
            y=0.95,
            xref="paper",
            yref="paper",
            text=text,
            showarrow=False,
            align="center",
            opacity=0.8
        )

    for fig in fig_list:
        fig.update_yaxes(range=[ymin, ymax])
        _date_tickformat(fig)
        fig.show()


def _ic_hist_fig(ic, period_num):
    ic_col = ic[period_num]
    title = "%s 周期信息系数" % period_num
    text = "均值 %.3f \n 标准差 %.3f" % (ic_col.mean(), ic_col.std())
    ic_col = ic_col.copy()
    hist_data = ic_col.replace(np.nan, 0.)

    fig = ff.create_distplot([hist_data.values],
                             [period_num],
                             show_rug=False,
                             curve_type='normal',
                             bin_size=0.25)

    fig.add_shape(
        # Line Vertical
        dict(
            type="line",
            yref='paper',
            x0=ic_col.mean(),
            y0=0,
            x1=ic_col.mean(),
            y1=1,
            line=dict(
                color="black",
                width=2,
                dash='dash'
            )
        ))
    fig.update_layout(
        title_text=title,
        showlegend=False,
        annotations=[
            dict(
                x=0.05,
                y=0.95,
                showarrow=False,
                xref="paper",
                yref="paper",
                text=text,
            )
        ]
    )
    fig.update_xaxes(title_text="IC", range=[-1, 1])
    return fig


def plot_ic_hist(ic):
    """
    Plots Spearman Rank Information Coefficient histogram for a given factor.

    Parameters
    ----------
    ic : pd.DataFrame
        DataFrame indexed by date, with IC for each forward return.
    """
    ic = ic.copy()
    # 分列绘制不同周期的IC直方图
    for col in ic.columns:
        fig = _ic_hist_fig(ic, col)
        fig.show()


def _ic_qq_fig(ic, period_num, theoretical_dist=stats.norm):
    data = ic[period_num]
    if isinstance(theoretical_dist, stats.norm.__class__):
        dist_name = 'Normal'
    elif isinstance(theoretical_dist, stats.t.__class__):
        dist_name = 'T'
    else:
        dist_name = 'Theoretical'
    title = "{} 周期信息系数 {} Q-Q分布".format(period_num, dist_name)
    probplot = sm.ProbPlot(data.replace(np.nan, 0.).values,
                           theoretical_dist, fit=True)
    x = probplot.theoretical_quantiles
    y = probplot.sample_quantiles
    fig = go.Figure()
    m_min, m_max = min(min(x), min(y)), max(max(x), max(y))
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode='markers',
        ))
    # 红色45°对角线
    range_ = [m_min-0.01*m_min, m_max+0.01*m_max]
    fig.add_shape(
        dict(
            type="line",
            xref='paper',
            yref='paper',
            x0=0,
            y0=0,
            x1=1,
            y1=1,
            line=dict(
                color="red",
            )
        ))
    fig.update_xaxes(title_text="IC", range=range_)
    fig.update_yaxes(range=range_)
    fig.update_layout(
        title_text=title,
        showlegend=False,
    )
    fig.update_xaxes(title_text='{}分布分位数'.format(dist_name))
    fig.update_yaxes(title_text='观测分位数')
    return fig


def plot_ic_qq(ic, theoretical_dist=stats.norm):
    """
    Plots Spearman Rank Information Coefficient "Q-Q" plot relative to
    a theoretical distribution.

    Parameters
    ----------
    ic : pd.DataFrame
        DataFrame indexed by date, with IC for each forward return.
    theoretical_dist : scipy.stats._continuous_distns
        Continuous distribution generator. scipy.stats.norm and
        scipy.stats.t are popular options.
    """

    ic = ic.copy()
    # 分列绘制不同周期的IC-QQ图
    for col in ic.columns:
        fig = _ic_qq_fig(ic, col)
        fig.show()


def plot_ic_hist_qq(ic, period_num, theoretical_dist=stats.norm):
    """绘制直方图、QQ图"""
    ic = ic.copy()
    hist_fig = _ic_hist_fig(ic, period_num)
    qq_fig = _ic_qq_fig(ic, period_num)
    hist_layout = hist_fig.layout
    qq_layout = qq_fig.layout

    # 子图 标题
    subplot_titles = (hist_layout.title.text, qq_layout.title.text)
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.15)

    # 添加柱状图
    fig.add_traces(hist_fig.data, rows=1, cols=1)
    # 添加图例(不得指定row、col)
    for an in hist_layout.annotations:
        fig.add_annotation(
            showarrow=False,
            text=an.text,
            x=0.05,
            y=0.95,
            xref='paper',
            yref='paper')

    # 添加柱状图 shape
    x1_factor = fig.layout.xaxis.domain[1]
    for s in hist_layout.shapes:
        x0 = s.x0 * x1_factor
        x1 = s.x1 * x1_factor
        s.update(dict(x0=x0, x1=x1))
        fig.add_shape(s)

    # 添加QQ图
    fig.add_traces(qq_fig.data, rows=1, cols=2)
    # QQ 对角线
    m_min, m_max = qq_layout.xaxis.range[0], qq_layout.xaxis.range[1]
    fig.add_shape(
        dict(
            type="line",
            xref='paper',
            yref='paper',
            x0=m_min,
            y0=m_min,
            x1=m_max,
            y1=m_max,
            line=dict(
                color="red",
            )
        ),
        row=1,
        col=2
    )
    # 更新坐标轴title
    fig.update_layout(
        # 不显示图例
        showlegend=False,
        xaxis=dict(
            title_text=hist_layout.xaxis.title.text
        ),
        xaxis2=dict(
            title_text=qq_layout.xaxis.title.text,
        ),
        yaxis2=dict(
            title_text=qq_layout.yaxis.title.text,
        ),
    )
    fig.show()


def plot_monthly_ic_heatmap(mean_monthly_ic):
    """
    Plots a heatmap of the information coefficient or returns by month.

    Parameters
    ----------
    mean_monthly_ic : pd.DataFrame
        The mean monthly IC for N periods forward.
    """

    mean_monthly_ic = mean_monthly_ic.copy()

    new_index_year = []
    new_index_month = []
    for date in mean_monthly_ic.index:
        new_index_year.append(date.year)
        new_index_month.append(date.month)

    mean_monthly_ic.index = pd.MultiIndex.from_arrays(
        [new_index_year, new_index_month],
        names=["年", "月"])
    # 原设计模式每行分2列绘图。此处简化为分月逐个打印。
    # 主要是简化代码，充分利用`plotly.figure_factory`模块
    for periods_num, ic in mean_monthly_ic.iteritems():
        title = "月度平均 {} 周期信息系数".format(periods_num)
        ic_data = ic.unstack(level=-1).apply(lambda x: x.round(2))
        fig = ff.create_annotated_heatmap(
            x=[f"{str(x).zfill(2)}月" for x in ic_data.columns],
            y=[f"{str(x)[:4]}年" for x in ic_data.index.values],
            z=ic_data.values,
            colorscale='Bluered',
        )
        # Make text size smaller
        for i in range(len(fig.layout.annotations)):
            fig.layout.annotations[i].font.size = 12
        fig.update_layout(title_text=title)
        fig.update_yaxes(autorange="reversed")
        fig.show()


def plot_ic_by_group(ic_group):
    """
    Plots Spearman Rank Information Coefficient for a given factor over
    provided forward returns. Separates by group.

    Parameters
    ----------
    ic_group : pd.DataFrame
        group-wise mean period wise returns.
    """
    ic_group_data = ic_group.stack().reset_index()
    ic_group_data.columns = ['行业', '周期', 'IC']
    fig = px.bar(ic_group_data, x="行业", y="IC",
                 color='周期', barmode='group')
    fig.update_layout(title_text='分组信息系数')
    fig.update_xaxes(tickangle=45)
    fig.show()


def plot_top_bottom_quantile_turnover(quantile_turnover, period=1):
    """
    Plots period wise top and bottom quantile factor turnover.

    Parameters
    ----------
    quantile_turnover: pd.Dataframe
        Quantile turnover (each DataFrame column a quantile).
    period: int, optional
        Period over which to calculate the turnover.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """
    max_quantile = quantile_turnover.columns.max()
    min_quantile = quantile_turnover.columns.min()
    turnover = pd.DataFrame()
    turnover['top quantile turnover'] = quantile_turnover[max_quantile]
    turnover['bottom quantile turnover'] = quantile_turnover[min_quantile]
    title = '{}D 周期最高与最低分位数换手率'.format(period)
    turnover = turnover.stack().reset_index()
    turnover.columns = ['日期', '分位数', '换手率']
    fig = px.line(turnover, x='日期', y='换手率', color='分位数')

    fig.update_yaxes(title_text='分位数新名称的比例')
    fig.update_traces(marker_line_width=2, opacity=0.6)
    fig.update_layout(title=title)
    _date_tickformat(fig)
    fig.show()


def plot_factor_rank_auto_correlation(factor_autocorrelation,
                                      period=1):
    """
    Plots factor rank autocorrelation over time.
    See factor_rank_autocorrelation for more details.

    Parameters
    ----------
    factor_autocorrelation : pd.Series
        Rolling 1 period (defined by time_rule) autocorrelation
        of factor values.
    period: int, optional
        Period over which the autocorrelation is calculated
    """
    title = '{}D 周期因子秩自相关系数'.format(period)
    df = factor_autocorrelation.sort_index().reset_index()
    df.columns = ['日期', '系数']
    fig = px.line(df, x='日期', y='系数')

    text = "均值 %.3f" % factor_autocorrelation.mean()
    fig.add_annotation(
        x=0.05,
        y=0.95,
        xref="paper",
        yref="paper",
        text=text,
        showarrow=False,
        align="center",
        opacity=0.8
    )
    fig.add_trace(
        go.Scatter(x=[df['日期'].min(), df['日期'].max()], y=[0.0]*2,
                   line=dict(color='black', width=1, dash='dash'))
    )
    fig.update_yaxes(title_text='自相关系数')
    fig.update_layout(title=title, showlegend=False)
    _date_tickformat(fig)
    fig.show()


def plot_quantile_average_cumulative_return(avg_cumulative_returns,
                                            by_quantile=False,
                                            std_bar=False,
                                            title=None):
    """
    Plots sector-wise mean daily returns for factor quantiles
    across provided forward price movement columns.

    Parameters
    ----------
    avg_cumulative_returns: pd.Dataframe
        The format is the one returned by
        performance.average_cumulative_return_by_quantile
    by_quantile : boolean, optional
        Disaggregated figures by quantile (useful to clearly see std dev bars)
    std_bar : boolean, optional
        Plot standard deviation plot
    title: string, optional
        Custom title
    """
    avg_cumulative_returns = avg_cumulative_returns.multiply(DECIMAL_TO_BPS)

    # 分位数
    qs = sorted(avg_cumulative_returns.index.levels[0].unique())
    quantiles = len(qs)
    colors = plotly.colors.sequential.Rainbow
    palette = [colors[i % len(colors)] for i in range(quantiles)]

    if by_quantile:
        for i, (quantile, q_ret) in enumerate(avg_cumulative_returns.groupby(level='factor_quantile')):
            mean = q_ret.loc[(quantile, 'mean')]
            if i % 2 == 0:
                j = i // 2
                subplot_titles = [
                    f"分位数 {int(t)} 平均累积收益率" for t in qs[j*2:(j+1)*2]]
                fig = make_subplots(
                    rows=1,
                    cols=2,
                    # shared_yaxes=True,
                    subplot_titles=subplot_titles)
            col = i % 2 + 1
            name = '分位数 ' + str(quantile)
            fig.add_trace(
                go.Scatter(x=mean.index,
                           y=mean.values,
                           name=name,
                           line_color=palette[i],
                           #    showlegend=True if col == 1 else False,
                           mode='lines'),
                row=1,
                col=col,
            )

            if std_bar:
                std = q_ret.loc[(quantile, 'std')]
                fig.add_trace(
                    go.Scatter(x=std.index,
                               y=mean,
                               name='误差',
                               showlegend=True if col == 1 else False,
                               error_y=dict(
                                   type='data',
                                   color=palette[i],
                                   array=std)),
                    row=1,
                    col=col
                )

            fig.add_shape(
                type="line",
                yref='paper',
                name='基准',
                x0=0,
                y0=0,
                x1=0,
                y1=1,
                line=dict(
                    color="black",
                    width=1,
                    dash="dash",
                ),
                row=1, col=col,
            )
            if i > 0 and i % 2 == 1 or (i+1) == quantiles:
                fig.update_layout(legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                ))
                fig.update_xaxes(title_text='周期')
                fig.update_yaxes(title_text='平均收益率(基点)',
                                 tickformat='.2f',
                                 #  range=[m_min, m_max],
                                 )
                fig.show()
    else:
        fig = make_subplots(rows=1, cols=1)
        for i, (quantile, q_ret) in enumerate(avg_cumulative_returns
                                              .groupby(level='factor_quantile')
                                              ):
            mean = q_ret.loc[(quantile, 'mean')]
            name = '分位数 ' + str(quantile)

            fig.add_trace(
                go.Scatter(x=mean.index,
                           y=mean.values,
                           name=name,
                           line=dict(color=palette[i]),
                           mode='lines')
            )
            if std_bar:
                std = q_ret.loc[(quantile, 'std')]
                fig.add_trace(
                    go.Scatter(x=std.index,
                               y=mean,
                               name='误差',
                               error_y=dict(
                                   type='data',
                                   color=palette[i],
                                   array=std))
                )

        fig.add_shape(
            type="line",
            yref='paper',
            name='基准',
            x0=0,
            y0=0,
            x1=0,
            y1=1,
            line=dict(
                color="black",
                width=1,
                dash="dash",
            )
        )
        title = "分位数的平均累积收益率" if title is None else title
        fig.update_layout(title=title,
                          legend=dict(
                              yanchor="top",
                              y=0.99,
                              xanchor="left",
                              x=0.01
                          ))
        fig.update_xaxes(title_text='周期')
        fig.update_yaxes(title_text='平均收益率(基点)', tickformat='.2f')
        fig.show()


def plot_events_distribution(events, num_bars=50):
    """
    Plots the distribution of events in time.

    Parameters
    ----------
    events : pd.Series
        A pd.Series whose index contains at least 'date' level.
    num_bars : integer, optional
        Number of bars to plot
    ax : matplotlib.Axes, optional
        Axes upon which to plot.

    Returns
    -------
    ax : matplotlib.Axes
    """
    start = events.index.get_level_values('date').min()
    end = events.index.get_level_values('date').max()
    group_interval = (end - start) / num_bars
    grouper = pd.Grouper(level='date', freq=group_interval)
    grouped = events.groupby(grouper).count()
    df = grouped.reset_index()
    df.columns = ['日期', '事件次数']
    fig = px.bar(df, x='日期', y='事件次数')
    fig.update_xaxes(title_text='日期')
    fig.update_yaxes(title_text='事件次数')
    fig.update_layout(title='事件时间分布', showlegend=False)
    _date_tickformat(fig)
    fig.show()
