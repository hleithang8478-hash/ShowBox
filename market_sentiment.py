# -*- coding: utf-8 -*-
"""
每日市场情绪打分 v2 — 四大硬核改造
  改造一：成交额加权情绪（Amount-Weighted Breadth）
  改造二：风格因子敞口与崩溃监控（Size + Vol Factor Spreads）
  改造三：聪明钱与日内承接力（Smart Money & Intraday Premium）
  改造四：股票池锚定（Universe Isolation — 剔除流动性后10%）

八维度：
  M1 资金多空比（成交额加权）
  M2 流动性缩量
  M3 加权横截面离散度 (WCSV)
  M4 宽基指数趋势
  M5 风格因子监控（Size + Vol Factor Spreads）
  M6 短线赚钱效应
  M7 聪明钱与日内承接力（Smart Money）
  M8 强势股补跌
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

SENTIMENT_VERSION = 'v2'

# ─── 常量 ────────────────────────────────────────────
LIMIT_THRESHOLD = 0.098
UNIVERSE_LIQUIDITY_CUTOFF = 0.10  # 每日剔除成交额最低10%

# Module 1: 资金多空比
BREADTH_LOOKBACK_DAYS = 252
BREADTH_PERCENTILE_PANIC = 90
BREADTH_PERCENTILE_CRASH = 98
MONEY_FLOW_ICE = 0.3   # 资金多空比 < 0.3 视为冰点（空头控 77%+）
PENALTY_PANIC = 20
PENALTY_CRASH = 20
PENALTY_ICE = 10

# Module 2: 流动性
LIQUIDITY_SHRINK_RATIO = 0.7
LIQUIDITY_LOOKBACK_DAYS = 252
LIQUIDITY_TURNOVER_P10 = 10
PENALTY_SHRINK = 15
PENALTY_TURNOVER_P10 = 15

# Module 3: 加权横截面离散度
CSV_LOOKBACK_DAYS = 61
CSV_PERCENTILE_LOW = 10
PENALTY_CSV_LOW = 20

# Module 4: 宽基指数趋势
BROAD_INDEX_MA_DAYS = 20
BROAD_INDEX_MA60_DAYS = 60
INNERCODE_CSI1000 = 3145
INNERCODE_SH50 = 11089
CORE_INDEX_MA60 = (46, 30, 3145)
PENALTY_CSI1000_BELOW_MA20 = 15
PENALTY_ALL_BELOW_MA60 = 30

# Module 5: 风格因子
STYLE_DIFF_THRESHOLD = -0.02
PENALTY_STYLE_DIFF = 20
FACTOR_QUINTILE = 0.2        # 分组用 20% 分位
FACTOR_SPREAD_THRESH = -0.015  # 连续3日 spread < -1.5%
FACTOR_CONSEC_DAYS = 3
PENALTY_SIZE_FACTOR = 20
PENALTY_VOL_FACTOR = 15
FACTOR_VOL_WINDOW = 20       # 20日滚动波动率

# Module 6: 短线赚钱效应
HOT_LOOKBACK_DAYS = 3
HOT_AVG_THRESHOLD = -0.02
HOT_NUKE_RATIO = 0.2
HOT_MIN_COUNT = 10
PENALTY_HOT_AVG = 20
PENALTY_HOT_NUKE = 20

# Module 7: 聪明钱
SMART_DISTRIB_THRESHOLD = 0.60   # 高开低走成交额占比 > 60%
PENALTY_SMART_DISTRIB = 15
TURNOVER_TOP_PCT = 0.10          # 换手率前10%
TURNOVER_PREMIUM_THRESH = -0.02  # 活跃池收益 < -2%
PENALTY_TURNOVER_PREMIUM = 15

# Module 8: 强势股补跌
STRONG_TOP_PERCENT = 10
STRONG_CRASH_THRESHOLD = -0.03
PENALTY_STRONG_CRASH = 20


# ─── 股票池锚定（改造四）────────────────────────────

def _filter_universe(df: pd.DataFrame) -> pd.DataFrame:
    """剔除每日成交额最低10%的股票（流动性过差，实盘无法交易）。
    使用 merge 代替 groupby.transform(lambda) 以大幅提速。"""
    if df is None or df.empty or 'Amount' not in df.columns:
        return df
    q_df = df.groupby('Date')['Amount'].quantile(UNIVERSE_LIQUIDITY_CUTOFF).rename('_amt_cutoff')
    df2 = df.merge(q_df, left_on='Date', right_index=True, how='left')
    result = df2[df2['Amount'] >= df2['_amt_cutoff']].drop(columns=['_amt_cutoff'])
    return result


# ─── 向量化辅助（改造一：成交额加权）──────────────────

def _vectorized_daily_breadth(df: pd.DataFrame) -> pd.DataFrame:
    """一次性计算所有日期的涨跌停统计 + 成交额/市值加权多空比。
    输入 df 需包含列：Date, R, Open, High, Low, Close, Amount, NegotiableMV"""
    if df is None or df.empty or 'R' not in df.columns:
        return pd.DataFrame()
    d = df.dropna(subset=['R']).copy()
    if d.empty:
        return pd.DataFrame()

    for col in ['Open', 'High', 'Low', 'Close', 'Amount', 'NegotiableMV']:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors='coerce')

    has_amount = 'Amount' in d.columns
    has_mv = 'NegotiableMV' in d.columns
    if has_amount:
        d['Amount'] = d['Amount'].fillna(0)
    if has_mv:
        d['NegotiableMV'] = d['NegotiableMV'].fillna(0)

    d['_up'] = d['R'] > 0
    d['_down'] = d['R'] < 0
    d['_limit_up'] = d['R'] > LIMIT_THRESHOLD
    d['_limit_down'] = d['R'] < -LIMIT_THRESHOLD

    if has_amount:
        d['_amt_up'] = d['Amount'] * d['_up'].astype(float)
        d['_amt_down'] = d['Amount'] * d['_down'].astype(float)
    if has_mv:
        d['_mv_up'] = d['NegotiableMV'] * d['_up'].astype(float)
        d['_mv_down'] = d['NegotiableMV'] * d['_down'].astype(float)

    agg_dict = {
        '_up': 'sum', '_down': 'sum',
        '_limit_up': 'sum', '_limit_down': 'sum',
        'R': 'size',
    }
    if has_amount:
        agg_dict['_amt_up'] = 'sum'
        agg_dict['_amt_down'] = 'sum'
    if has_mv:
        agg_dict['_mv_up'] = 'sum'
        agg_dict['_mv_down'] = 'sum'

    stats = d.groupby('Date').agg(agg_dict)
    stats = stats.rename(columns={
        'R': 'total_count', '_up': 'up_count', '_down': 'down_count',
        '_limit_up': 'limit_up', '_limit_down': 'limit_down',
    })

    if has_amount:
        stats.rename(columns={'_amt_up': 'amount_up', '_amt_down': 'amount_down'}, inplace=True)
        denom = stats['amount_down'].replace(0, np.nan)
        stats['money_flow_ratio'] = stats['amount_up'] / denom
    if has_mv:
        stats.rename(columns={'_mv_up': 'mv_up', '_mv_down': 'mv_down'}, inplace=True)
        denom = stats['mv_down'].replace(0, np.nan)
        stats['mv_flow_ratio'] = stats['mv_up'] / denom

    has_ohlc = all(c in d.columns for c in ['Open', 'High', 'Low', 'Close'])
    if has_ohlc:
        d2 = d.dropna(subset=['Open', 'High', 'Low', 'Close'])
        d2 = d2.assign(
            _one_word=(d2['Open'] == d2['High']) & (d2['High'] == d2['Low']) & (d2['Low'] == d2['Close']),
        )
        d2 = d2.assign(
            _owld=d2['_one_word'] & (d2['R'] < -LIMIT_THRESHOLD),
            _owlu=d2['_one_word'] & (d2['R'] > LIMIT_THRESHOLD),
        )
        ow = d2.groupby('Date').agg(
            one_word_limit_down=('_owld', 'sum'),
            one_word_limit_up=('_owlu', 'sum'),
        )
        stats = stats.join(ow, how='left')
    else:
        stats['one_word_limit_down'] = 0
        stats['one_word_limit_up'] = 0

    int_cols = ['total_count', 'up_count', 'down_count',
                'limit_up', 'limit_down', 'one_word_limit_down', 'one_word_limit_up']
    stats = stats.fillna(0)
    for c in int_cols:
        if c in stats.columns:
            stats[c] = stats[c].astype(int)
    stats['up_ratio'] = stats['up_count'] / stats['total_count'].replace(0, np.nan)
    stats = stats.reset_index()
    return stats


# ─── 模块 1（改造一：成交额加权情绪）────────────────

def module1_market_breadth(daily_df: pd.DataFrame, trading_day: str) -> Dict[str, Any]:
    """资金多空比（成交额加权情绪）。daily_df 来自 _vectorized_daily_breadth。"""
    empty = {
        'money_flow_ratio': None, 'mv_flow_ratio': None,
        'divergence': None,
        'up_count': 0, 'down_count': 0,
        'limit_up': 0, 'limit_down': 0,
        'one_word_limit_down': 0, 'total_count': 0, 'up_ratio': None,
        'p90_limit_down': None, 'p98_limit_down': None,
        'penalty': 0, 'penalty_detail': [],
    }
    if daily_df is None or daily_df.empty:
        empty['error'] = '无有效数据'
        return empty

    daily_df = daily_df.copy()
    daily_df['Date'] = daily_df['Date'].astype(str)
    today_row = daily_df[daily_df['Date'] == trading_day]
    if today_row.empty:
        today_row = daily_df.iloc[[-1]]
        trading_day = str(today_row['Date'].iloc[0])
    today = today_row.iloc[0]

    hist = daily_df[daily_df['Date'] != trading_day]
    if hist.empty:
        hist = daily_df

    ld_series = hist['limit_down'].values
    ow_series = hist['one_word_limit_down'].values
    p90_limit = float(np.nanpercentile(ld_series, BREADTH_PERCENTILE_PANIC))
    p98_limit = float(np.nanpercentile(ld_series, BREADTH_PERCENTILE_CRASH))
    p90_one = float(np.nanpercentile(ow_series, BREADTH_PERCENTILE_PANIC))
    p98_one = float(np.nanpercentile(ow_series, BREADTH_PERCENTILE_CRASH))

    limit_down_today = int(today.get('limit_down', 0))
    one_word_today = int(today.get('one_word_limit_down', 0))
    up_ratio_today = float(today.get('up_ratio', 0) or 0)
    up_count = int(today.get('up_count', 0))
    down_count = int(today.get('down_count', 0))
    total_count = int(today.get('total_count', 0))
    limit_up_today = int(today.get('limit_up', 0))

    mfr = today.get('money_flow_ratio', None)
    mfr = round(float(mfr), 4) if mfr is not None and np.isfinite(mfr) else None
    mvfr = today.get('mv_flow_ratio', None)
    mvfr = round(float(mvfr), 4) if mvfr is not None and np.isfinite(mvfr) else None
    divergence = round(mfr - mvfr, 4) if mfr is not None and mvfr is not None else None

    penalty = 0
    penalty_detail = []
    if limit_down_today >= p90_limit or one_word_today >= p90_one:
        penalty += PENALTY_PANIC
        penalty_detail.append(
            f'跌停/一字跌停数达252日{BREADTH_PERCENTILE_PANIC}%分位以上（恐慌蔓延），扣{PENALTY_PANIC}分')
    if limit_down_today >= p98_limit or one_word_today >= p98_one:
        penalty += PENALTY_CRASH
        penalty_detail.append(
            f'跌停数达252日{BREADTH_PERCENTILE_CRASH}%分位（极端股灾），追加扣{PENALTY_CRASH}分')
    if mfr is not None and mfr < MONEY_FLOW_ICE and total_count > 0:
        penalty += PENALTY_ICE
        penalty_detail.append(
            f'资金多空比{mfr:.2f}<{MONEY_FLOW_ICE}（空头控制77%+活跃资金），扣{PENALTY_ICE}分')

    return {
        'money_flow_ratio': mfr, 'mv_flow_ratio': mvfr,
        'divergence': divergence,
        'up_count': up_count, 'down_count': down_count,
        'limit_up': limit_up_today, 'limit_down': limit_down_today,
        'one_word_limit_down': one_word_today,
        'total_count': total_count, 'up_ratio': round(up_ratio_today, 4),
        'p90_limit_down': round(p90_limit, 0), 'p98_limit_down': round(p98_limit, 0),
        'penalty': penalty, 'penalty_detail': penalty_detail,
    }


# ─── 模块 2 ─────────────────────────────────────────

def module2_liquidity(
    turnover_today: float,
    turnover_series_5: List[float],
    turnover_series_20: List[float],
    turnover_series_252: Optional[List[float]] = None,
) -> Dict[str, Any]:
    vol_5 = np.mean(turnover_series_5) if turnover_series_5 else 0
    vol_20 = np.mean(turnover_series_20) if turnover_series_20 else 0
    penalty = 0
    penalty_detail = []
    if vol_20 > 0 and vol_5 < LIQUIDITY_SHRINK_RATIO * vol_20:
        penalty += PENALTY_SHRINK
        penalty_detail.append(f'短期均量/20日均量={vol_5/vol_20:.2%}<70%（短期断崖缩量），扣{PENALTY_SHRINK}分')
    p10_turnover = None
    if turnover_series_252 and len(turnover_series_252) > 0:
        p10_turnover = float(np.nanpercentile(turnover_series_252, LIQUIDITY_TURNOVER_P10))
        if turnover_today < p10_turnover:
            penalty += PENALTY_TURNOVER_P10
            penalty_detail.append(f'当日成交额{turnover_today:.0f}亿低于252日10%分位({p10_turnover:.0f}亿)，历史绝对地量，扣{PENALTY_TURNOVER_P10}分')
    return {
        'turnover_today': round(turnover_today, 2),
        'vol_5': round(vol_5, 2),
        'vol_20': round(vol_20, 2),
        'p10_turnover_252': round(p10_turnover, 2) if p10_turnover is not None else None,
        'penalty': penalty,
        'penalty_detail': penalty_detail,
    }


# ─── 模块 3（改造一：加权横截面离散度 WCSV）─────────

def _weighted_cross_sectional_std(df_day: pd.DataFrame) -> float:
    """成交额加权的横截面收益率标准差。"""
    r = df_day['R'].values
    w = df_day['Amount'].values if 'Amount' in df_day.columns else np.ones(len(r))
    mask = np.isfinite(r) & np.isfinite(w) & (w > 0)
    r, w = r[mask], w[mask]
    if len(r) < 10:
        return np.nan
    w_norm = w / w.sum()
    mean_r = np.average(r, weights=w_norm)
    return float(np.sqrt(np.average((r - mean_r) ** 2, weights=w_norm)))


def module3_cross_sectional_volatility(
    sigma_weighted_today: float,
    sigma_weighted_history: List[float],
    sigma_equal_today: Optional[float] = None,
) -> Dict[str, Any]:
    """加权横截面离散度 (WCSV)：扣分基于成交额加权 sigma。"""
    if sigma_weighted_today is None or not sigma_weighted_history:
        return {'sigma_weighted': None, 'sigma_equal': None,
                'percentile': None, 'penalty': 0, 'penalty_detail': []}
    arr = np.array(sigma_weighted_history, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {'sigma_weighted': round(sigma_weighted_today, 6),
                'sigma_equal': round(sigma_equal_today, 6) if sigma_equal_today else None,
                'percentile': None, 'penalty': 0, 'penalty_detail': []}
    percentile = float(np.sum(arr <= sigma_weighted_today) / len(arr) * 100)
    penalty = 0
    penalty_detail = []
    if percentile <= CSV_PERCENTILE_LOW:
        penalty = PENALTY_CSV_LOW
        penalty_detail.append(
            f'加权σ处于过去60日后{CSV_PERCENTILE_LOW}%分位（同涨同跌/因子失效），扣{PENALTY_CSV_LOW}分')
    return {
        'sigma_weighted': round(sigma_weighted_today, 6),
        'sigma_equal': round(sigma_equal_today, 6) if sigma_equal_today is not None else None,
        'percentile': round(percentile, 2),
        'penalty': penalty,
        'penalty_detail': penalty_detail,
    }


# ─── 模块 4 ─────────────────────────────────────────

def module4_broad_index_trend(
    df_today: pd.DataFrame,
    df_series_20: pd.DataFrame,
    df_series_60: pd.DataFrame,
    trading_day: str,
) -> Dict[str, Any]:
    if df_today is None or df_today.empty:
        return {
            'indices': [], 'summary': '无指数数据',
            'penalty': 0, 'penalty_detail': [],
            'below_ma_count': 0, 'core_below_ma': [],
            'csi1000_below_ma20': False, 'all_three_below_ma60': False,
        }
    df_today = df_today.copy()
    df_today['ChangePCT'] = pd.to_numeric(df_today['ChangePCT'], errors='coerce')
    df_today['ClosePrice'] = pd.to_numeric(df_today['ClosePrice'], errors='coerce')

    ma20_by_index = {}
    if df_series_20 is not None and not df_series_20.empty and 'ClosePrice' in df_series_20.columns:
        d = df_series_20.copy()
        d['ClosePrice'] = pd.to_numeric(d['ClosePrice'], errors='coerce')
        for ic, g in d.groupby('InnerCode'):
            g = g.dropna(subset=['ClosePrice']).sort_values('TradingDay').tail(BROAD_INDEX_MA_DAYS)
            if len(g) >= int(BROAD_INDEX_MA_DAYS * 0.8):
                ma20_by_index[ic] = float(g['ClosePrice'].mean())

    ma60_by_index = {}
    if df_series_60 is not None and not df_series_60.empty and 'ClosePrice' in df_series_60.columns:
        d = df_series_60.copy()
        d['ClosePrice'] = pd.to_numeric(d['ClosePrice'], errors='coerce')
        for ic, g in d.groupby('InnerCode'):
            g = g.dropna(subset=['ClosePrice']).sort_values('TradingDay').tail(BROAD_INDEX_MA60_DAYS)
            if len(g) >= int(BROAD_INDEX_MA60_DAYS * 0.8):
                ma60_by_index[ic] = float(g['ClosePrice'].mean())

    rows = []
    csi1000_below_ma20 = False
    for _, r in df_today.iterrows():
        ic = r.get('InnerCode')
        name = r.get('ChiName') or r.get('SecuAbbr', '')
        close = r.get('ClosePrice')
        change_pct = r.get('ChangePCT')
        ma20 = ma20_by_index.get(ic)
        below_ma20 = None
        if ma20 is not None and close is not None and np.isfinite(close) and np.isfinite(ma20):
            below_ma20 = close < ma20
            if ic == INNERCODE_CSI1000:
                csi1000_below_ma20 = below_ma20
        rows.append({
            'name': name, 'abbr': r.get('SecuAbbr', ''),
            'change_pct': round(float(change_pct), 2) if change_pct is not None and np.isfinite(change_pct) else None,
            'category': r.get('Category', ''),
            'close': round(float(close), 2) if close is not None and np.isfinite(close) else None,
            'ma20': round(ma20, 2) if ma20 is not None else None,
            'below_ma': below_ma20,
        })

    all_three_below_ma60 = True
    for ic in CORE_INDEX_MA60:
        sub = df_today[df_today['InnerCode'] == ic]
        close = float(sub['ClosePrice'].iloc[0]) if len(sub) and pd.notna(sub['ClosePrice'].iloc[0]) else None
        ma60 = ma60_by_index.get(ic)
        if close is None or ma60 is None or not np.isfinite(close) or close >= ma60:
            all_three_below_ma60 = False
            break

    penalty = 0
    penalty_detail = []
    if csi1000_below_ma20:
        penalty += PENALTY_CSI1000_BELOW_MA20
        penalty_detail.append(f'中证1000收盘跌破20日均线（小盘趋势破位），扣{PENALTY_CSI1000_BELOW_MA20}分')
    if all_three_below_ma60:
        penalty += PENALTY_ALL_BELOW_MA60
        penalty_detail.append(f'沪深300、中证500、中证1000同时跌破60日均线（系统性空头），扣{PENALTY_ALL_BELOW_MA60}分')
    return {
        'indices': rows,
        'summary': f'共 {len(rows)} 只指数；中证1000破MA20={csi1000_below_ma20}，三指数同破MA60={all_three_below_ma60}',
        'penalty': penalty, 'penalty_detail': penalty_detail,
        'csi1000_below_ma20': csi1000_below_ma20,
        'all_three_below_ma60': all_three_below_ma60,
    }


# ─── 模块 5 ─────────────────────────────────────────

def module5_style_factors(
    df_today_slice: pd.DataFrame,
    size_spread_history: Optional[List[float]] = None,
    vol_spread_history: Optional[List[float]] = None,
    df_idx_pair: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """风格因子监控：Size Factor + Volatility Factor + 中证1000/上证50 spread。
    df_today_slice: 当日全市场（需含 R, NegotiableMV, _vol20）。
    size/vol_spread_history: 最近 N 日的 spread 序列（用于判断连续3日）。
    df_idx_pair: 当日指数数据（含 ChangePCT, InnerCode）。
    """
    empty = {
        'spread_size': None, 'spread_vol': None,
        'r_csi1000_vs_sh50': None,
        'size_consec_days': 0, 'vol_consec_days': 0,
        'penalty': 0, 'penalty_detail': [],
    }
    if df_today_slice is None or df_today_slice.empty or 'R' not in df_today_slice.columns:
        empty['error'] = '无当日股票数据'
        return empty

    d = df_today_slice.dropna(subset=['R']).copy()
    penalty = 0
    detail = []

    # --- Size Factor ---
    spread_size = None
    if 'NegotiableMV' in d.columns and d['NegotiableMV'].notna().sum() > 50:
        mv_q_lo = d['NegotiableMV'].quantile(FACTOR_QUINTILE)
        mv_q_hi = d['NegotiableMV'].quantile(1 - FACTOR_QUINTILE)
        r_small = d.loc[d['NegotiableMV'] <= mv_q_lo, 'R'].mean()
        r_large = d.loc[d['NegotiableMV'] >= mv_q_hi, 'R'].mean()
        if np.isfinite(r_small) and np.isfinite(r_large):
            spread_size = round(float(r_small - r_large), 6)

    size_consec = 0
    if spread_size is not None and size_spread_history:
        recent = list(size_spread_history) + [spread_size]
        for v in reversed(recent):
            if v is not None and v < FACTOR_SPREAD_THRESH:
                size_consec += 1
            else:
                break
    if size_consec >= FACTOR_CONSEC_DAYS:
        penalty += PENALTY_SIZE_FACTOR
        detail.append(
            f'Size Factor连续{size_consec}日<{FACTOR_SPREAD_THRESH:.1%}（弃小买大抽血），扣{PENALTY_SIZE_FACTOR}分')

    # --- Volatility Factor ---
    spread_vol = None
    if '_vol20' in d.columns and d['_vol20'].notna().sum() > 50:
        vol_q_lo = d['_vol20'].quantile(FACTOR_QUINTILE)
        vol_q_hi = d['_vol20'].quantile(1 - FACTOR_QUINTILE)
        r_low_vol = d.loc[d['_vol20'] <= vol_q_lo, 'R'].mean()
        r_high_vol = d.loc[d['_vol20'] >= vol_q_hi, 'R'].mean()
        if np.isfinite(r_low_vol) and np.isfinite(r_high_vol):
            spread_vol = round(float(r_high_vol - r_low_vol), 6)

    vol_consec = 0
    if spread_vol is not None and vol_spread_history:
        recent = list(vol_spread_history) + [spread_vol]
        for v in reversed(recent):
            if v is not None and v < FACTOR_SPREAD_THRESH:
                vol_consec += 1
            else:
                break
    if vol_consec >= FACTOR_CONSEC_DAYS:
        penalty += PENALTY_VOL_FACTOR
        detail.append(
            f'Vol Factor连续{vol_consec}日<{FACTOR_SPREAD_THRESH:.1%}（高波被闷杀/风险偏好收缩），扣{PENALTY_VOL_FACTOR}分')

    # --- CSI1000 vs SH50 (保留) ---
    r_csi1000_vs_sh50 = None
    if df_idx_pair is not None and not df_idx_pair.empty and 'ChangePCT' in df_idx_pair.columns:
        ip = df_idx_pair.copy()
        ip['ChangePCT'] = pd.to_numeric(ip['ChangePCT'], errors='coerce')
        r50s = ip.loc[ip['InnerCode'] == INNERCODE_SH50, 'ChangePCT']
        r1000s = ip.loc[ip['InnerCode'] == INNERCODE_CSI1000, 'ChangePCT']
        if not r50s.empty and not r1000s.empty:
            r50 = float(r50s.iloc[0]) / 100.0
            r1000 = float(r1000s.iloc[0]) / 100.0
            diff = r1000 - r50
            r_csi1000_vs_sh50 = round(diff, 4)
            if diff < STYLE_DIFF_THRESHOLD:
                penalty += PENALTY_STYLE_DIFF
                detail.append(
                    f'中证1000 vs 上证50 日收益差{diff:.2%}<{STYLE_DIFF_THRESHOLD:.0%}（中小盘被抽血），扣{PENALTY_STYLE_DIFF}分')

    return {
        'spread_size': spread_size,
        'spread_vol': spread_vol,
        'r_csi1000_vs_sh50': r_csi1000_vs_sh50,
        'size_consec_days': size_consec,
        'vol_consec_days': vol_consec,
        'penalty': penalty,
        'penalty_detail': detail,
    }


# ─── 模块 6 ─────────────────────────────────────────

def module6_hot_money(df_2: pd.DataFrame, yesterday: str, today: str) -> Dict[str, Any]:
    if df_2 is None or df_2.empty or 'R' not in df_2.columns:
        return {
            'hot_avg_return': None, 'hot_count': 0,
            'nuke_count': 0, 'nuke_ratio': None,
            'penalty': 0, 'penalty_detail': [], 'error': '无最近两日收益率数据'
        }
    df = df_2
    dates = df['Date'].astype(str)
    df_y = df[dates == yesterday]
    df_t = df[dates == today]
    if df_y.empty or df_t.empty:
        return {
            'hot_avg_return': None, 'hot_count': 0,
            'nuke_count': 0, 'nuke_ratio': None,
            'penalty': 0, 'penalty_detail': [], 'error': '缺少昨日或今日数据'
        }
    hot_codes = set(df_y.loc[df_y['R'] > LIMIT_THRESHOLD, 'SecuCode'])
    if not hot_codes:
        return {
            'hot_avg_return': None, 'hot_count': 0,
            'nuke_count': 0, 'nuke_ratio': None,
            'penalty': 0, 'penalty_detail': [], 'error': '昨日无涨停股'
        }
    df_t_hot = df_t[df_t['SecuCode'].isin(hot_codes)]
    hot_count = len(df_t_hot)
    if hot_count == 0:
        return {
            'hot_avg_return': None, 'hot_count': 0,
            'nuke_count': 0, 'nuke_ratio': None,
            'penalty': 0, 'penalty_detail': [], 'error': '昨日涨停股今日无行情'
        }
    hot_avg = float(df_t_hot['R'].mean())
    nuke_count = int((df_t_hot['R'] < -LIMIT_THRESHOLD).sum())
    nuke_ratio = nuke_count / hot_count
    penalty = 0
    detail: List[str] = []
    if hot_avg < HOT_AVG_THRESHOLD:
        penalty += PENALTY_HOT_AVG
        detail.append(f'昨日涨停股今平均收益{hot_avg:.2%}<{HOT_AVG_THRESHOLD:.0%}，扣{PENALTY_HOT_AVG}分（接力意愿极差）')
    if hot_count >= HOT_MIN_COUNT and nuke_ratio >= HOT_NUKE_RATIO:
        penalty += PENALTY_HOT_NUKE
        detail.append(f'昨日涨停股中有{nuke_ratio:.1%} 近似跌停（{nuke_count}/{hot_count}），扣{PENALTY_HOT_NUKE}分（热钱被核按钮）')
    return {
        'hot_avg_return': round(hot_avg, 4), 'hot_count': hot_count,
        'nuke_count': nuke_count,
        'nuke_ratio': round(nuke_ratio, 4) if nuke_ratio is not None else None,
        'penalty': penalty, 'penalty_detail': detail,
    }


# ─── 模块 7（改造三：聪明钱与日内承接力）────────────

def module7_smart_money(
    df_today_slice: pd.DataFrame,
    market_return_today: Optional[float],
) -> Dict[str, Any]:
    """Smart Money: 成交额加权假阳真阴比例 + 高换手率溢价。
    df_today_slice: 当日全市场，需含 Open, Close, R, Amount, TurnoverRate。"""
    empty = {
        'smart_distrib_ratio': None,
        'count_false_green_ratio': None,
        'top_turnover_avg_return': None,
        'top_turnover_count': 0,
        'market_return_today': market_return_today,
        'penalty': 0, 'penalty_detail': [],
    }
    if df_today_slice is None or df_today_slice.empty:
        empty['error'] = '当日无股票数据'
        return empty

    d = df_today_slice.copy()
    for col in ['Open', 'Close', 'Amount', 'TurnoverRate', 'R']:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors='coerce')

    penalty = 0
    detail: List[str] = []

    # --- Smart Distribution: 高开低走的成交额加权占比 ---
    smart_ratio = None
    count_ratio = None
    has_oc = 'Open' in d.columns and 'Close' in d.columns
    has_amt = 'Amount' in d.columns
    if has_oc:
        mask = d['Open'].notna() & d['Close'].notna()
        dm = d[mask]
        total = len(dm)
        is_fg = dm['Close'] < dm['Open']
        count_fg = int(is_fg.sum())
        count_ratio = round(count_fg / total, 4) if total > 0 else None

        if has_amt and total > 0:
            total_amount = dm['Amount'].sum()
            fg_amount = dm.loc[is_fg, 'Amount'].sum()
            smart_ratio = round(float(fg_amount / total_amount), 4) if total_amount > 0 else None

        if smart_ratio is not None and smart_ratio > SMART_DISTRIB_THRESHOLD:
            penalty += PENALTY_SMART_DISTRIB
            detail.append(
                f'高开低走股票吸收了{smart_ratio:.1%}的全市场成交额（>{SMART_DISTRIB_THRESHOLD:.0%}），'
                f'活跃资金被套/主力派发，扣{PENALTY_SMART_DISTRIB}分')

    # --- Turnover Premium: 高换手率前10%股票的平均收益 ---
    top_ret = None
    top_count = 0
    if 'TurnoverRate' in d.columns and 'R' in d.columns:
        dr = d.dropna(subset=['TurnoverRate', 'R'])
        if len(dr) > 20:
            q_hi = dr['TurnoverRate'].quantile(1 - TURNOVER_TOP_PCT)
            top_pool = dr[dr['TurnoverRate'] >= q_hi]
            top_count = len(top_pool)
            if top_count > 0:
                top_ret = round(float(top_pool['R'].mean()), 4)
                if top_ret < TURNOVER_PREMIUM_THRESH:
                    penalty += PENALTY_TURNOVER_PREMIUM
                    detail.append(
                        f'换手率前{TURNOVER_TOP_PCT:.0%}活跃股（{top_count}只）平均收益{top_ret:.2%}'
                        f'<{TURNOVER_PREMIUM_THRESH:.0%}（热钱被收割），扣{PENALTY_TURNOVER_PREMIUM}分')

    return {
        'smart_distrib_ratio': smart_ratio,
        'count_false_green_ratio': count_ratio,
        'top_turnover_avg_return': top_ret,
        'top_turnover_count': top_count,
        'market_return_today': market_return_today,
        'penalty': penalty,
        'penalty_detail': detail,
    }


# ─── 模块 8（向量化） ────────────────────────────────

def module8_strong_stock_crash(
    df_window20: pd.DataFrame,
    df_today_slice: pd.DataFrame,
    market_return_today: Optional[float],
) -> Dict[str, Any]:
    """向量化版：df_window20 = 过去20日（不含今日）全市场，df_today_slice = 今日。"""
    empty = {
        'strong_count': 0, 'strong_avg_return': None,
        'threshold_cum20': None, 'limitdown_count': 0,
        'market_return_today': market_return_today,
        'penalty': 0, 'penalty_detail': [],
    }
    if df_window20 is None or df_window20.empty or df_today_slice is None or df_today_slice.empty:
        empty['error'] = '窗口期或今日数据不足'
        return empty
    if 'R' not in df_window20.columns or 'R' not in df_today_slice.columns:
        empty['error'] = '缺少R列'
        return empty

    cum20 = df_window20.groupby('SecuCode').agg(
        cnt=('R', 'size'),
        cum=('R', lambda x: float((1 + x).prod() - 1)),
    )
    cum20 = cum20[cum20['cnt'] >= 10]
    if cum20.empty:
        empty['error'] = '无可用于计算的强势股历史数据'
        return empty

    threshold = float(np.nanpercentile(cum20['cum'].values, 100 - STRONG_TOP_PERCENT))
    strong_codes = cum20.index[cum20['cum'] >= threshold]

    today_map = df_today_slice.set_index('SecuCode')['R']
    strong_today = today_map.reindex(strong_codes).dropna()
    strong_count = len(strong_today)
    if strong_count == 0:
        empty['threshold_cum20'] = round(threshold, 4)
        empty['error'] = '强势股池为空'
        return empty

    strong_avg = float(strong_today.mean())
    limitdown_count = int((strong_today < -LIMIT_THRESHOLD).sum())

    penalty = 0
    detail: List[str] = []
    if market_return_today is not None and market_return_today < 0 and strong_avg <= STRONG_CRASH_THRESHOLD:
        penalty += PENALTY_STRONG_CRASH
        detail.append(
            f'大盘下跌且强势股池（前{STRONG_TOP_PERCENT}%）今日平均跌幅{strong_avg:.2%}≤{STRONG_CRASH_THRESHOLD:.0%}，'
            f'出现强势股补跌，扣{PENALTY_STRONG_CRASH}分')
    return {
        'strong_count': strong_count,
        'strong_avg_return': round(strong_avg, 4),
        'threshold_cum20': round(threshold, 4),
        'limitdown_count': limitdown_count,
        'market_return_today': market_return_today,
        'penalty': penalty, 'penalty_detail': detail,
    }


# ══════════════════════════════════════════════════════
#  单日入口（优化：仅 1 次全市场查询 + 1 次指数查询）
# ══════════════════════════════════════════════════════

def compute_daily_sentiment(trading_day: str, fetcher) -> Dict[str, Any]:
    trading_day = pd.to_datetime(trading_day).strftime('%Y-%m-%d')

    dates_252 = fetcher.get_trading_days_inclusive(trading_day, count=BREADTH_LOOKBACK_DAYS)
    if not dates_252:
        return {
            'trading_day': trading_day, 'version': SENTIMENT_VERSION,
            'module1': {}, 'module2': {}, 'module3': {}, 'module4': {},
            'module5': {}, 'module6': {}, 'module7': {}, 'module8': {},
            'total_score': 100, 'total_penalty': 0, 'penalty_details': [],
            'error': '无交易日',
        }
    trading_day = dates_252[-1]

    result = {
        'trading_day': trading_day, 'version': SENTIMENT_VERSION,
        'module1': {}, 'module2': {}, 'module3': {}, 'module4': {},
        'module5': {}, 'module6': {}, 'module7': {}, 'module8': {},
        'total_score': 100, 'total_penalty': 0, 'penalty_details': [],
    }

    # ── 数据拉取 ──
    df_252 = fetcher.get_all_stocks_returns_for_dates(dates_252)
    df_252_ok = df_252 is not None and not df_252.empty and 'R' in df_252.columns
    if df_252_ok:
        df_252['Date'] = df_252['Date'].astype(str)
        df_252 = _filter_universe(df_252)

    index_inner_codes = [1, 1059]
    all_index_codes = list(set(
        index_inner_codes +
        [1, 1055, 11089, 3145, 4978, 46, 30, 4074, 39144, 36324,
         6973, 4078, 3036, 7544, 39376, 31398, 48542, 19475, 217313,
         3469, 3471, 4089, 225892, 303968] +
        [INNERCODE_SH50, INNERCODE_CSI1000] + list(CORE_INDEX_MA60)
    ))
    start_date_idx = dates_252[0]
    df_idx_all = fetcher.get_index_quote_for_dates(all_index_codes, start_date_idx, trading_day)
    market_return_today: Optional[float] = None

    # ── MODULE 1: 资金多空比 ──
    if not df_252_ok:
        result['module1'] = {'penalty': 0, 'penalty_detail': [], 'error': '无252日收益率数据'}
    else:
        breadth_df = _vectorized_daily_breadth(df_252)
        result['module1'] = module1_market_breadth(breadth_df, trading_day)

    # ── MODULE 2: 流动性 ──
    dates_20 = dates_252[-20:] if len(dates_252) >= 20 else dates_252
    if df_idx_all is None or df_idx_all.empty or 'TurnoverValue' not in df_idx_all.columns:
        result['module2'] = {'penalty': 0, 'penalty_detail': [], 'error': '无指数成交额'}
    else:
        df_idx_all['TurnoverValue'] = pd.to_numeric(df_idx_all['TurnoverValue'], errors='coerce').fillna(0)
        if 'ChangePCT' in df_idx_all.columns:
            df_idx_all['ChangePCT'] = pd.to_numeric(df_idx_all['ChangePCT'], errors='coerce')
        df_liq = df_idx_all[df_idx_all['InnerCode'].isin(index_inner_codes)]
        base = df_liq[(df_liq['InnerCode'] == index_inner_codes[0]) & (df_liq['TradingDay'] == trading_day)]
        if not base.empty and 'ChangePCT' in base.columns and not pd.isna(base['ChangePCT'].iloc[0]):
            market_return_today = float(base['ChangePCT'].iloc[0]) / 100.0
        daily_total = df_liq.groupby('TradingDay')['TurnoverValue'].sum()
        daily_billion = (daily_total / 1e8).to_dict()
        turnover_today = float(daily_billion.get(trading_day, 0))
        series_5 = [daily_billion.get(d, 0) for d in dates_20[-5:]]
        series_20 = [daily_billion.get(d, 0) for d in dates_20]
        series_252 = [daily_billion.get(d, 0) for d in dates_252]
        result['module2'] = module2_liquidity(turnover_today, series_5, series_20, turnover_series_252=series_252)

    # ── MODULE 3: 加权横截面离散度 ──
    if not df_252_ok:
        result['module3'] = {'sigma_weighted': None, 'sigma_equal': None,
                             'percentile': None, 'penalty': 0, 'penalty_detail': []}
    else:
        dates_61 = dates_252[-CSV_LOOKBACK_DAYS:] if len(dates_252) >= CSV_LOOKBACK_DAYS else dates_252
        df_61 = df_252[df_252['Date'].isin(set(dates_61))]
        sigma_w_series = df_61.groupby('Date').apply(_weighted_cross_sectional_std)
        sigma_e_series = df_61.groupby('Date')['R'].std()
        sw_today = float(sigma_w_series.get(trading_day, np.nan)) if trading_day in sigma_w_series.index else np.nan
        se_today = float(sigma_e_series.get(trading_day, np.nan)) if trading_day in sigma_e_series.index else np.nan
        result['module3'] = module3_cross_sectional_volatility(
            sw_today, sigma_w_series.tolist(), sigma_equal_today=se_today)

    # ── MODULE 4: 宽基指数趋势 ──
    CATEGORY_MAP = {
        1: '市场综合基准', 1055: '市场综合基准', 11089: '市场综合基准', 3145: '市场综合基准', 4978: '市场综合基准',
        46: '规模与风格', 30: '规模与风格', 4074: '规模与风格', 39144: '规模与风格', 36324: '规模与风格',
        6973: '主题与策略', 4078: '主题与策略', 3036: '主题与策略', 7544: '主题与策略',
        39376: '主题与策略', 31398: '主题与策略', 48542: '主题与策略', 19475: '主题与策略', 217313: '主题与策略',
        3469: '其他常用', 3471: '其他常用', 4089: '其他常用', 225892: '其他常用', 303968: '其他常用',
    }
    broad_codes = list(CATEGORY_MAP.keys())
    if df_idx_all is not None and not df_idx_all.empty:
        df_broad = df_idx_all[(df_idx_all['InnerCode'].isin(broad_codes)) & (df_idx_all['TradingDay'] == trading_day)].copy()
        df_broad['Category'] = df_broad['InnerCode'].map(CATEGORY_MAP).fillna('')
        dates_20_idx = dates_252[-BROAD_INDEX_MA_DAYS:] if len(dates_252) >= BROAD_INDEX_MA_DAYS else dates_252
        df_series_20 = df_idx_all[
            (df_idx_all['InnerCode'].isin(broad_codes)) & (df_idx_all['TradingDay'].isin(set(dates_20_idx)))]
        dates_60_idx = dates_252[-BROAD_INDEX_MA60_DAYS:] if len(dates_252) >= BROAD_INDEX_MA60_DAYS else dates_252
        df_series_60 = df_idx_all[
            (df_idx_all['InnerCode'].isin(list(CORE_INDEX_MA60))) & (df_idx_all['TradingDay'].isin(set(dates_60_idx)))]
        result['module4'] = module4_broad_index_trend(df_broad, df_series_20, df_series_60, trading_day)
    else:
        result['module4'] = {'indices': [], 'summary': '无指数数据', 'penalty': 0, 'penalty_detail': []}

    # ── MODULE 5: 风格因子监控 ──
    try:
        if df_252_ok:
            df_today_stocks = df_252[df_252['Date'] == trading_day].copy()
            # 20日滚动波动率
            vol20 = df_252.groupby('SecuCode')['R'].rolling(FACTOR_VOL_WINDOW, min_periods=10).std()
            vol20 = vol20.reset_index(level=0, drop=True)
            df_252_tmp = df_252.copy()
            df_252_tmp['_vol20'] = vol20
            df_today_with_vol = df_252_tmp[df_252_tmp['Date'] == trading_day]

            # 最近几天的 spread history
            recent_dates = dates_252[-(FACTOR_CONSEC_DAYS + 1):-1] if len(dates_252) > FACTOR_CONSEC_DAYS else []
            size_hist, vol_hist = [], []
            for dd in recent_dates:
                dd_slice = df_252_tmp[df_252_tmp['Date'] == dd]
                if dd_slice.empty:
                    size_hist.append(None); vol_hist.append(None); continue
                dd_d = dd_slice.dropna(subset=['R'])
                if 'NegotiableMV' in dd_d.columns and dd_d['NegotiableMV'].notna().sum() > 50:
                    mql = dd_d['NegotiableMV'].quantile(FACTOR_QUINTILE)
                    mqh = dd_d['NegotiableMV'].quantile(1 - FACTOR_QUINTILE)
                    rs = dd_d.loc[dd_d['NegotiableMV'] <= mql, 'R'].mean()
                    rl = dd_d.loc[dd_d['NegotiableMV'] >= mqh, 'R'].mean()
                    size_hist.append(float(rs - rl) if np.isfinite(rs) and np.isfinite(rl) else None)
                else:
                    size_hist.append(None)
                if '_vol20' in dd_d.columns and dd_d['_vol20'].notna().sum() > 50:
                    vql = dd_d['_vol20'].quantile(FACTOR_QUINTILE)
                    vqh = dd_d['_vol20'].quantile(1 - FACTOR_QUINTILE)
                    rlv = dd_d.loc[dd_d['_vol20'] <= vql, 'R'].mean()
                    rhv = dd_d.loc[dd_d['_vol20'] >= vqh, 'R'].mean()
                    vol_hist.append(float(rhv - rlv) if np.isfinite(rlv) and np.isfinite(rhv) else None)
                else:
                    vol_hist.append(None)

            df_idx_pair = None
            if df_idx_all is not None and not df_idx_all.empty:
                df_idx_pair = df_idx_all[
                    (df_idx_all['InnerCode'].isin([INNERCODE_SH50, INNERCODE_CSI1000])) &
                    (df_idx_all['TradingDay'] == trading_day)]
            result['module5'] = module5_style_factors(
                df_today_with_vol, size_hist, vol_hist, df_idx_pair)
        else:
            result['module5'] = {'penalty': 0, 'penalty_detail': [], 'error': '无行情数据'}
    except Exception as e:
        logger.error(f'计算风格因子失败: {e}', exc_info=True)
        result['module5'] = {'penalty': 0, 'penalty_detail': [], 'error': str(e)}

    # ── MODULE 6: 短线赚钱效应 ──
    if not df_252_ok or len(dates_252) < 2:
        result['module6'] = {'penalty': 0, 'penalty_detail': [], 'error': '交易日不足两天'}
    else:
        hot_dates = dates_252[-HOT_LOOKBACK_DAYS:]
        df_hot = df_252[df_252['Date'].isin(set(hot_dates))]
        yesterday, today2 = hot_dates[-2], hot_dates[-1]
        try:
            result['module6'] = module6_hot_money(df_hot, yesterday, today2)
        except Exception as e:
            logger.error(f'计算短线赚钱效应失败: {e}')
            result['module6'] = {'penalty': 0, 'penalty_detail': [], 'error': str(e)}

    # ── MODULE 7: 聪明钱与日内承接力 ──
    if df_252_ok:
        df_today_stocks = df_252[df_252['Date'] == trading_day]
        result['module7'] = module7_smart_money(df_today_stocks, market_return_today)
    else:
        result['module7'] = {'penalty': 0, 'penalty_detail': [], 'error': '无行情数据'}

    # ── MODULE 8: 强势股补跌 ──
    if df_252_ok and len(dates_252) >= 21:
        dates_21 = dates_252[-21:]
        window20 = dates_21[:-1]
        df_window20 = df_252[df_252['Date'].isin(set(window20))]
        df_today_stocks = df_252[df_252['Date'] == trading_day]
        result['module8'] = module8_strong_stock_crash(df_window20, df_today_stocks, market_return_today)
    else:
        result['module8'] = {'penalty': 0, 'penalty_detail': [], 'error': '历史交易日不足21天'}

    # ── 汇总 ──
    total_penalty = 0
    penalty_details = []
    for key in ['module1', 'module2', 'module3', 'module4', 'module5', 'module6', 'module7', 'module8']:
        m = result.get(key, {})
        p = m.get('penalty', 0) or 0
        if p > 0:
            total_penalty += p
            penalty_details.extend(m.get('penalty_detail', []))
    result['total_penalty'] = total_penalty
    result['penalty_details'] = penalty_details
    result['total_score'] = 100 - total_penalty
    return result


# ══════════════════════════════════════════════════════
#  批量入口：一次拉数据 + 滚动窗口
# ══════════════════════════════════════════════════════

def compute_batch_sentiment(
    start_date: str, end_date: str, fetcher
) -> List[Dict[str, Any]]:
    """同步版批量计算（供导出 CSV 等非流式场景使用）。"""
    results = []
    for msg in compute_batch_sentiment_streaming(start_date, end_date, fetcher):
        if msg.get('type') == 'result':
            results.append(msg['data'])
    return results


def compute_batch_sentiment_streaming(
    start_date: str, end_date: str, fetcher,
    skip_existing: bool = True,
):
    """
    批量计算生成器，yield 进度消息字典。
    skip_existing=True 时，先从 SQLite 读取已有结果，跳过不重算。
    """
    import time as _time

    start_date = pd.to_datetime(start_date).strftime('%Y-%m-%d')
    end_date = pd.to_datetime(end_date).strftime('%Y-%m-%d')

    yield {'type': 'stage', 'stage': 'trading_days', 'msg': '正在获取交易日列表…'}

    big_count = BREADTH_LOOKBACK_DAYS + 1500
    all_dates_raw = fetcher.get_trading_days_inclusive(end_date, count=big_count)
    if not all_dates_raw:
        yield {'type': 'error', 'error': '无法获取交易日列表'}
        return

    all_dates = [str(d) for d in all_dates_raw]

    try:
        idx_start = next(i for i, d in enumerate(all_dates) if d >= start_date)
    except StopIteration:
        yield {'type': 'error', 'error': f'{start_date} 超出可用交易日范围'}
        return
    try:
        idx_end = next(i for i in range(len(all_dates) - 1, -1, -1) if all_dates[i] <= end_date)
    except StopIteration:
        yield {'type': 'error', 'error': f'{end_date} 超出可用交易日范围'}
        return

    earliest_idx = max(0, idx_start - BREADTH_LOOKBACK_DAYS)
    needed_dates = all_dates[earliest_idx:idx_end + 1]
    target_dates = all_dates[idx_start:idx_end + 1]

    if not needed_dates or not target_dates:
        yield {'type': 'error', 'error': '计算区间为空'}
        return

    total = len(target_dates)

    # 查询已有缓存，跳过已计算的日期
    cached_results = {}
    if skip_existing:
        try:
            import sqlite3, json as _jl
            conn = sqlite3.connect('database.db')
            c = conn.cursor()
            c.execute(
                "SELECT trading_day, result_json FROM market_sentiment_results "
                "WHERE trading_day >= ? AND trading_day <= ?",
                (target_dates[0], target_dates[-1])
            )
            for row in c.fetchall():
                try:
                    cached_results[row[0]] = _jl.loads(row[1])
                except Exception:
                    pass
            conn.close()
        except Exception:
            pass

    to_compute = [d for d in target_dates if d not in cached_results]
    cached_count = total - len(to_compute)
    if cached_count > 0:
        yield {'type': 'stage', 'stage': 'cache_check',
               'msg': f'共 {total} 个交易日，其中 {cached_count} 天已有缓存可直接使用，需新计算 {len(to_compute)} 天',
               'total': total, 'cached_count': cached_count}

    if not to_compute:
        # 全部命中缓存，直接逐条返回
        for i, td in enumerate(target_dates):
            done = i + 1
            yield {
                'type': 'result', 'data': cached_results[td],
                'current': done, 'total': total,
                'pct': round(done / total * 100, 1),
                'date': td, 'msg': f'{done}/{total}（缓存）{td}',
                'from_cache': True,
            }
        logger.info(f'批量计算：{total} 天全部命中缓存')
        return

    import threading as _threading
    import queue as _queue

    # ── 后台线程分批加载，通过队列实时推送每批进度 ──
    progress_q = _queue.Queue()
    _result_box = {}

    def _on_batch_progress(batch_idx, total_batches, rows_so_far, label):
        progress_q.put((batch_idx, total_batches, rows_so_far, label))

    def _load_data():
        try:
            _result_box['df_all'] = fetcher.get_all_stocks_returns_for_dates(
                needed_dates, on_progress=_on_batch_progress)
        except Exception as e:
            _result_box['error'] = str(e)
        finally:
            progress_q.put(None)  # sentinel

    yield {'type': 'stage', 'stage': 'load_stocks',
           'msg': f'正在分批加载全市场行情（{len(needed_dates)} 天）…',
           'total': total}

    t_load = _time.time()
    loader_thread = _threading.Thread(target=_load_data, daemon=True)
    loader_thread.start()

    while True:
        try:
            item = progress_q.get(timeout=2.0)
        except _queue.Empty:
            elapsed = round(_time.time() - t_load, 1)
            yield {'type': 'heartbeat', 'stage': 'load_stocks',
                   'msg': f'正在查询数据库… 已耗时 {elapsed} 秒',
                   'elapsed': elapsed}
            continue
        if item is None:
            break
        batch_idx, total_batches, rows_so_far, label = item
        elapsed = round(_time.time() - t_load, 1)
        yield {'type': 'heartbeat', 'stage': 'load_stocks',
               'msg': f'数据加载 第{batch_idx}/{total_batches}批 [{label}]，累计 {rows_so_far:,} 行，耗时 {elapsed}s',
               'elapsed': elapsed,
               'batch': batch_idx, 'total_batches': total_batches, 'rows': rows_so_far}

    loader_thread.join()

    if 'error' in _result_box:
        yield {'type': 'error', 'error': f'数据加载失败: {_result_box["error"]}'}
        return

    df_all = _result_box.get('df_all')
    if df_all is None or df_all.empty:
        yield {'type': 'error', 'error': '全市场行情数据为空'}
        return
    df_all['Date'] = df_all['Date'].astype(str)
    load_stock_sec = round(_time.time() - t_load, 1)
    logger.info(f'全市场行情加载完成: {load_stock_sec}秒, {len(df_all)}条')

    yield {'type': 'stage', 'stage': 'load_index',
           'msg': f'行情加载完成（{load_stock_sec}秒，{len(df_all):,}条），正在加载指数…',
           'total': total}

    all_index_codes = list(set(
        [1, 1059, 1055, 11089, 3145, 4978, 46, 30, 4074, 39144, 36324,
         6973, 4078, 3036, 7544, 39376, 31398, 48542, 19475, 217313,
         3469, 3471, 4089, 225892, 303968] +
        [INNERCODE_SH50, INNERCODE_CSI1000] + list(CORE_INDEX_MA60)
    ))
    df_idx_all = fetcher.get_index_quote_for_dates(all_index_codes, needed_dates[0], needed_dates[-1])
    if df_idx_all is not None and not df_idx_all.empty:
        df_idx_all['TurnoverValue'] = pd.to_numeric(df_idx_all['TurnoverValue'], errors='coerce').fillna(0)
        if 'ChangePCT' in df_idx_all.columns:
            df_idx_all['ChangePCT'] = pd.to_numeric(df_idx_all['ChangePCT'], errors='coerce')
        df_idx_all['TradingDay'] = df_idx_all['TradingDay'].astype(str)

    t_pre = _time.time()
    yield {'type': 'stage', 'stage': 'precompute',
           'msg': '正在预计算：股票池过滤、涨跌停、加权离散度、风格因子…',
           'total': total}

    # 改造四：股票池锚定 — 剔除流动性后10%
    df_all = _filter_universe(df_all)

    breadth_df = _vectorized_daily_breadth(df_all)
    if breadth_df is not None and not breadth_df.empty:
        breadth_df['Date'] = breadth_df['Date'].astype(str)

    # 加权横截面离散度 (WCSV)
    sigma_w_by_date = df_all.groupby('Date').apply(_weighted_cross_sectional_std)
    sigma_e_by_date = df_all.groupby('Date')['R'].std()

    # 风格因子 spread 预计算（向量化：merge 分位数替代逐日 for 循环）
    vol20 = df_all.groupby('SecuCode')['R'].rolling(FACTOR_VOL_WINDOW, min_periods=10).std()
    vol20 = vol20.reset_index(level=0, drop=True)
    df_all['_vol20'] = vol20

    size_spread_by_date = {}
    vol_spread_by_date = {}
    _df_r = df_all.dropna(subset=['R'])
    if 'NegotiableMV' in _df_r.columns:
        mv_lo = _df_r.groupby('Date')['NegotiableMV'].quantile(FACTOR_QUINTILE).rename('_mv_lo')
        mv_hi = _df_r.groupby('Date')['NegotiableMV'].quantile(1 - FACTOR_QUINTILE).rename('_mv_hi')
        _tmp = _df_r[['Date', 'R', 'NegotiableMV']].merge(mv_lo, left_on='Date', right_index=True)
        _tmp = _tmp.merge(mv_hi, left_on='Date', right_index=True)
        r_small = _tmp[_tmp['NegotiableMV'] <= _tmp['_mv_lo']].groupby('Date')['R'].mean()
        r_large = _tmp[_tmp['NegotiableMV'] >= _tmp['_mv_hi']].groupby('Date')['R'].mean()
        spread_size = (r_small - r_large).dropna()
        size_spread_by_date = spread_size.to_dict()

    if '_vol20' in _df_r.columns:
        _vr = _df_r.dropna(subset=['_vol20'])
        vol_lo = _vr.groupby('Date')['_vol20'].quantile(FACTOR_QUINTILE).rename('_vl')
        vol_hi = _vr.groupby('Date')['_vol20'].quantile(1 - FACTOR_QUINTILE).rename('_vh')
        _vtmp = _vr[['Date', 'R', '_vol20']].merge(vol_lo, left_on='Date', right_index=True)
        _vtmp = _vtmp.merge(vol_hi, left_on='Date', right_index=True)
        r_low_v = _vtmp[_vtmp['_vol20'] <= _vtmp['_vl']].groupby('Date')['R'].mean()
        r_high_v = _vtmp[_vtmp['_vol20'] >= _vtmp['_vh']].groupby('Date')['R'].mean()
        spread_vol = (r_high_v - r_low_v).dropna()
        vol_spread_by_date = spread_vol.to_dict()

    index_liq_codes = [1, 1059]
    daily_turnover_billion = {}
    if df_idx_all is not None and not df_idx_all.empty:
        df_liq = df_idx_all[df_idx_all['InnerCode'].isin(index_liq_codes)]
        daily_total = df_liq.groupby('TradingDay')['TurnoverValue'].sum()
        daily_turnover_billion = (daily_total / 1e8).to_dict()

    CATEGORY_MAP = {
        1: '市场综合基准', 1055: '市场综合基准', 11089: '市场综合基准', 3145: '市场综合基准', 4978: '市场综合基准',
        46: '规模与风格', 30: '规模与风格', 4074: '规模与风格', 39144: '规模与风格', 36324: '规模与风格',
        6973: '主题与策略', 4078: '主题与策略', 3036: '主题与策略', 7544: '主题与策略',
        39376: '主题与策略', 31398: '主题与策略', 48542: '主题与策略', 19475: '主题与策略', 217313: '主题与策略',
        3469: '其他常用', 3471: '其他常用', 4089: '其他常用', 225892: '其他常用', 303968: '其他常用',
    }
    broad_codes = list(CATEGORY_MAP.keys())

    pre_sec = round(_time.time() - t_pre, 1)
    logger.info(f'预计算完成: {pre_sec}秒')

    compute_count = len(to_compute)
    yield {'type': 'stage', 'stage': 'computing',
           'msg': f'预计算完成（{pre_sec}秒），开始逐日评分（共 {total} 天，新算 {compute_count}，缓存 {cached_count}）…',
           'total': total, 'current': 0, 'pct': 0}

    t_calc = _time.time()
    import json as _json
    from concurrent.futures import ThreadPoolExecutor, as_completed
    try:
        import psutil as _psutil
    except ImportError:
        _psutil = None
    import os as _os

    # ── 自适应线程数：根据 CPU / 内存实时决定 ──
    def _auto_workers():
        cpu_count = _os.cpu_count() or 4
        if _psutil is None:
            return min(cpu_count, 16)
        try:
            mem = _psutil.virtual_memory()
            mem_avail_gb = mem.available / (1024 ** 3)
            cpu_pct = _psutil.cpu_percent(interval=0.1)
        except Exception:
            return min(cpu_count, 16)
        if mem_avail_gb < 1.0 or cpu_pct > 90:
            return max(4, cpu_count // 2)
        if mem_avail_gb < 2.0 or cpu_pct > 75:
            return cpu_count
        return min(cpu_count * 2, 32)

    def _calc_one(td):
        try:
            return _compute_single_day_from_cache(
                td, all_dates, df_all, df_idx_all,
                breadth_df, sigma_w_by_date, sigma_e_by_date,
                daily_turnover_billion, CATEGORY_MAP, broad_codes,
                size_spread_by_date=size_spread_by_date,
                vol_spread_by_date=vol_spread_by_date,
            )
        except Exception as e:
            logger.error(f'批量计算 {td} 失败: {e}', exc_info=True)
            return {
                'trading_day': td, 'version': SENTIMENT_VERSION,
                'module1': {}, 'module2': {}, 'module3': {}, 'module4': {},
                'module5': {}, 'module6': {}, 'module7': {}, 'module8': {},
                'total_score': 100, 'total_penalty': 0, 'penalty_details': [],
                'error': str(e),
            }

    computed_results = {}
    WORKERS = min(_auto_workers(), max(1, len(to_compute)))

    if to_compute:
        _done_atomic = [0]
        _lock = _threading.Lock()

        try:
            mem = _psutil.virtual_memory()
            cpu_pct = _psutil.cpu_percent(interval=0.1)
            sys_info = f'CPU {cpu_pct:.0f}% / 内存 {mem.percent:.0f}%（可用 {mem.available/1024**3:.1f}GB）'
        except Exception:
            sys_info = ''

        yield {'type': 'heartbeat', 'stage': 'computing',
               'msg': f'启动 {WORKERS} 线程并行评分（{compute_count} 天） {sys_info}'}
        logger.info(f'并行评分: {WORKERS} workers, {compute_count} 天, {sys_info}')

        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            future_map = {executor.submit(_calc_one, td): td for td in to_compute}
            for future in as_completed(future_map):
                td = future_map[future]
                r = future.result()
                computed_results[td] = r
                with _lock:
                    _done_atomic[0] += 1
                    done_new = _done_atomic[0]
                elapsed_calc = _time.time() - t_calc
                speed = done_new / elapsed_calc if elapsed_calc > 0 else 0
                remaining = compute_count - done_new
                eta = round(remaining / speed, 1) if speed > 0 else 0
                if done_new % 10 == 0 or done_new == compute_count:
                    try:
                        mem = _psutil.virtual_memory()
                        cpu_pct = _psutil.cpu_percent(interval=0)
                        res_tag = f'CPU {cpu_pct:.0f}% 内存 {mem.percent:.0f}%'
                    except Exception:
                        res_tag = ''
                    yield {'type': 'heartbeat', 'stage': 'computing',
                           'msg': f'并行评分 {done_new}/{compute_count}，{speed:.1f} 天/s，ETA {eta}s | {res_tag}',
                           'batch': done_new, 'total_batches': compute_count}

    calc_sec = round(_time.time() - t_calc, 1)
    logger.info(f'并行评分完成: {calc_sec}秒, {len(computed_results)}天')

    # ── 批量写入 SQLite（串行，避免锁冲突）──
    if computed_results:
        try:
            import sqlite3
            conn = sqlite3.connect('database.db')
            c = conn.cursor()
            rows_to_write = [(td, _json.dumps(r, ensure_ascii=False))
                             for td, r in computed_results.items()]
            c.executemany(
                "INSERT OR REPLACE INTO market_sentiment_results (trading_day, result_json) VALUES (?, ?)",
                rows_to_write
            )
            conn.commit()
            conn.close()
        except Exception as ex:
            logger.error(f'批量写入缓存失败: {ex}')

    # ── 按日期顺序 yield 结果（缓存 + 新算合并）──
    for i, td in enumerate(target_dates):
        from_cache = td in cached_results
        r = cached_results[td] if from_cache else computed_results.get(td, {})
        done = i + 1
        pct = round(done / total * 100, 1)
        tag = '缓存' if from_cache else '新算'

        yield {
            'type': 'result',
            'data': r,
            'current': done,
            'total': total,
            'pct': pct,
            'date': td,
            'msg': f'{done}/{total}（{pct}%）{td}（{tag}）',
            'from_cache': from_cache,
        }

    total_sec = round(_time.time() - t_calc, 1)
    logger.info(f'批量计算完成：{total} 天（新算 {len(computed_results)}，缓存 {cached_count}），评分+推送 {total_sec}s')


def _compute_single_day_from_cache(
    trading_day: str,
    all_dates: List[str],
    df_all: pd.DataFrame,
    df_idx_all: pd.DataFrame,
    breadth_df: pd.DataFrame,
    sigma_w_by_date: pd.Series,
    sigma_e_by_date: pd.Series,
    daily_turnover_billion: dict,
    category_map: dict,
    broad_codes: list,
    size_spread_by_date: Optional[dict] = None,
    vol_spread_by_date: Optional[dict] = None,
) -> Dict[str, Any]:
    """从预加载的全量数据中，切片计算单日情绪（v2 升级版）。"""
    result = {
        'trading_day': trading_day, 'version': SENTIMENT_VERSION,
        'module1': {}, 'module2': {}, 'module3': {}, 'module4': {},
        'module5': {}, 'module6': {}, 'module7': {}, 'module8': {},
        'total_score': 100, 'total_penalty': 0, 'penalty_details': [],
    }

    idx_td = None
    for i, d in enumerate(all_dates):
        if d == trading_day:
            idx_td = i
            break
    if idx_td is None:
        result['error'] = f'{trading_day} 不在交易日列表中'
        return result

    w252_start = max(0, idx_td - BREADTH_LOOKBACK_DAYS + 1)
    dates_252 = all_dates[w252_start:idx_td + 1]
    set_252 = set(dates_252)

    market_return_today: Optional[float] = None

    # MODULE 1
    if breadth_df is not None and not breadth_df.empty:
        breadth_252 = breadth_df[breadth_df['Date'].isin(set_252)]
        result['module1'] = module1_market_breadth(breadth_252, trading_day)
    else:
        result['module1'] = {'penalty': 0, 'penalty_detail': [], 'error': '无breadth数据'}

    # MODULE 2
    dates_20 = dates_252[-20:] if len(dates_252) >= 20 else dates_252
    if daily_turnover_billion:
        turnover_today = daily_turnover_billion.get(trading_day, 0)
        series_5 = [daily_turnover_billion.get(d, 0) for d in dates_20[-5:]]
        series_20 = [daily_turnover_billion.get(d, 0) for d in dates_20]
        series_252 = [daily_turnover_billion.get(d, 0) for d in dates_252]
        if df_idx_all is not None and not df_idx_all.empty:
            base = df_idx_all[(df_idx_all['InnerCode'] == 1) & (df_idx_all['TradingDay'] == trading_day)]
            if not base.empty and 'ChangePCT' in base.columns and not pd.isna(base['ChangePCT'].iloc[0]):
                market_return_today = float(base['ChangePCT'].iloc[0]) / 100.0
        result['module2'] = module2_liquidity(turnover_today, series_5, series_20, turnover_series_252=series_252)
    else:
        result['module2'] = {'penalty': 0, 'penalty_detail': [], 'error': '无成交额数据'}

    # MODULE 3 (WCSV)
    dates_61 = dates_252[-CSV_LOOKBACK_DAYS:] if len(dates_252) >= CSV_LOOKBACK_DAYS else dates_252
    sw_slice = sigma_w_by_date.reindex(dates_61).dropna() if sigma_w_by_date is not None else pd.Series(dtype=float)
    sw_today = float(sw_slice.get(trading_day, np.nan)) if trading_day in sw_slice.index else np.nan
    se_today = float(sigma_e_by_date.get(trading_day, np.nan)) if sigma_e_by_date is not None and trading_day in sigma_e_by_date.index else np.nan
    result['module3'] = module3_cross_sectional_volatility(sw_today, sw_slice.tolist(), sigma_equal_today=se_today)

    # MODULE 4
    if df_idx_all is not None and not df_idx_all.empty:
        df_broad_today = df_idx_all[
            (df_idx_all['InnerCode'].isin(broad_codes)) & (df_idx_all['TradingDay'] == trading_day)].copy()
        df_broad_today['Category'] = df_broad_today['InnerCode'].map(category_map).fillna('')
        dates_20_idx = dates_252[-BROAD_INDEX_MA_DAYS:] if len(dates_252) >= BROAD_INDEX_MA_DAYS else dates_252
        df_series_20 = df_idx_all[
            (df_idx_all['InnerCode'].isin(broad_codes)) & (df_idx_all['TradingDay'].isin(set(dates_20_idx)))]
        dates_60_idx = dates_252[-BROAD_INDEX_MA60_DAYS:] if len(dates_252) >= BROAD_INDEX_MA60_DAYS else dates_252
        df_series_60 = df_idx_all[
            (df_idx_all['InnerCode'].isin(list(CORE_INDEX_MA60))) & (df_idx_all['TradingDay'].isin(set(dates_60_idx)))]
        result['module4'] = module4_broad_index_trend(df_broad_today, df_series_20, df_series_60, trading_day)
    else:
        result['module4'] = {'indices': [], 'summary': '无指数数据', 'penalty': 0, 'penalty_detail': []}

    # MODULE 5 (Style Factors)
    try:
        df_today_stocks = df_all[df_all['Date'] == trading_day]
        recent_idx = max(0, idx_td - FACTOR_CONSEC_DAYS)
        recent_dates = all_dates[recent_idx:idx_td]
        size_hist = [size_spread_by_date.get(d) for d in recent_dates] if size_spread_by_date else []
        vol_hist = [vol_spread_by_date.get(d) for d in recent_dates] if vol_spread_by_date else []
        df_idx_pair = None
        if df_idx_all is not None and not df_idx_all.empty:
            df_idx_pair = df_idx_all[
                (df_idx_all['InnerCode'].isin([INNERCODE_SH50, INNERCODE_CSI1000])) &
                (df_idx_all['TradingDay'] == trading_day)]
        result['module5'] = module5_style_factors(df_today_stocks, size_hist, vol_hist, df_idx_pair)
    except Exception as e:
        result['module5'] = {'penalty': 0, 'penalty_detail': [], 'error': str(e)}

    # MODULE 6
    if len(dates_252) >= HOT_LOOKBACK_DAYS:
        hot_dates = dates_252[-HOT_LOOKBACK_DAYS:]
        df_hot = df_all[df_all['Date'].isin(set(hot_dates))]
        yesterday, today2 = hot_dates[-2], hot_dates[-1]
        try:
            result['module6'] = module6_hot_money(df_hot, yesterday, today2)
        except Exception as e:
            result['module6'] = {'penalty': 0, 'penalty_detail': [], 'error': str(e)}
    else:
        result['module6'] = {'penalty': 0, 'penalty_detail': [], 'error': '交易日不足'}

    # MODULE 7 (Smart Money)
    df_today_stocks = df_all[df_all['Date'] == trading_day]
    result['module7'] = module7_smart_money(df_today_stocks, market_return_today)

    # MODULE 8
    if len(dates_252) >= 21:
        dates_21 = dates_252[-21:]
        window20_set = set(dates_21[:-1])
        df_window20 = df_all[df_all['Date'].isin(window20_set)]
        result['module8'] = module8_strong_stock_crash(df_window20, df_today_stocks, market_return_today)
    else:
        result['module8'] = {'penalty': 0, 'penalty_detail': [], 'error': '历史交易日不足21天'}

    # 汇总
    total_penalty = 0
    penalty_details = []
    for key in ['module1', 'module2', 'module3', 'module4', 'module5', 'module6', 'module7', 'module8']:
        m = result.get(key, {})
        p = m.get('penalty', 0) or 0
        if p > 0:
            total_penalty += p
            penalty_details.extend(m.get('penalty_detail', []))
    result['total_penalty'] = total_penalty
    result['penalty_details'] = penalty_details
    result['total_score'] = 100 - total_penalty
    return result


# ═══════════════════════════════════════════════════════════════════
#  报告生成器 —— 纯文字（Markdown），与计算逻辑完全解耦
# ═══════════════════════════════════════════════════════════════════

def generate_sentiment_report(data: dict) -> str:
    """
    根据 compute_daily_sentiment / 缓存返回的 result dict，
    生成一份专业的市场情绪诊断报告（Markdown 格式）。
    只读取 data 中已有字段，不调用任何计算函数。
    """
    day = data.get('trading_day', '未知')
    score = data.get('total_score', 100)
    penalty = data.get('total_penalty', 0)

    if score >= 85:
        regime, pos, risk = '主升浪 / 活跃资金主导', '建议满仓进攻，趋势共振强烈', '绿灯'
    elif score >= 70:
        regime, pos, risk = '温和多头 / 结构性行情', '建议7成仓持有核心标的', '绿灯'
    elif score >= 50:
        regime, pos, risk = '震荡分化 / 选股为王', '建议5成仓精选个股，回避弱势板块', '黄灯'
    elif score >= 30:
        regime, pos, risk = '弱势防御 / 风控优先', '建议3成以下仓位，严格止损', '黄灯'
    else:
        regime, pos, risk = '极端恐慌 / 系统性风险', '建议空仓回避，等待右侧信号', '红灯'

    def _pct(v):
        return f'{v * 100:+.2f}%' if v is not None else '-'
    def _f(v):
        return f'{v:.2f}' if v is not None else '-'

    L = []
    L.append(f'# 市场情绪诊断报告 · {day}\n')
    L.append(f'**综合得分：{score} / 100**（扣分 {penalty}） · 风控信号：{risk}\n')
    L.append(f'**环境研判**：{regime}')
    L.append(f'**仓位指令**：{pos}\n')

    # ── M1 ──
    m1 = data.get('module1') or {}
    L.append('---\n## M1 资金多空比')
    if m1.get('error'):
        L.append(f'> 异常：{m1["error"]}')
    else:
        L.append(f'- 成交额多空比 = {_f(m1.get("money_flow_ratio"))}（>1 多头占优，<0.5 极端恐慌）')
        L.append(f'- 市值加权多空比 = {_f(m1.get("mv_flow_ratio"))}')
        dv = m1.get('divergence')
        if dv is not None:
            L.append(f'- 背离度 = {_f(dv)}（{"资金面强于市值面" if dv > 0 else "资金面弱于市值面"}）')
        L.append(f'- 涨 {m1.get("up_count", "-")} / 跌 {m1.get("down_count", "-")}，'
                 f'涨停 {m1.get("limit_up", "-")} / 跌停 {m1.get("limit_down", "-")}')
        p1 = m1.get('penalty', 0)
        if p1:
            L.append(f'- **扣分 {p1}**：{"；".join(m1.get("penalty_detail", []))}')
        else:
            L.append('- 本模块无扣分，多空力量均衡')
    L.append('')

    # ── M2 ──
    m2 = data.get('module2') or {}
    L.append('## M2 流动性缩量')
    if m2.get('error'):
        L.append(f'> 异常：{m2["error"]}')
    else:
        L.append(f'- 当日 {m2.get("turnover_today", "-")} 亿，'
                 f'5日均量 {m2.get("vol_5", "-")}，20日均量 {m2.get("vol_20", "-")}')
        if m2.get('p10_turnover_252') is not None:
            L.append(f'- 252日 10% 分位 = {_f(m2["p10_turnover_252"])} 亿')
        p2 = m2.get('penalty', 0)
        if p2:
            L.append(f'- **扣分 {p2}**：{"；".join(m2.get("penalty_detail", []))}')
        else:
            L.append('- 流动性充裕，未触发缩量警报')
    L.append('')

    # ── M3 ──
    m3 = data.get('module3') or {}
    L.append('## M3 横截面离散度 (WCSV)')
    if m3.get('error'):
        L.append(f'> 异常：{m3["error"]}')
    else:
        L.append(f'- 加权σ = {m3.get("sigma_weighted", "-")}，'
                 f'等权σ = {m3.get("sigma_equal", "-")}，'
                 f'60日分位 = {m3.get("percentile", "-")}%')
        p3 = m3.get('penalty', 0)
        if p3:
            L.append(f'- **扣分 {p3}**：{"；".join(m3.get("penalty_detail", []))}')
        else:
            L.append('- 截面波动处于正常区间，选股因子有效')
    L.append('')

    # ── M4 ──
    m4 = data.get('module4') or {}
    L.append('## M4 宽基指数趋势')
    if m4.get('error'):
        L.append(f'> 异常：{m4["error"]}')
    else:
        for nm in ['上证50', '沪深300', '中证500', '中证1000']:
            for r in (m4.get('indices') or []):
                if nm in (r.get('name', '') + r.get('abbr', '')):
                    st = '**跌破 MA20**' if r.get('below_ma') else '站上 MA20'
                    L.append(f'- {nm}：{r.get("close","-")}，MA20={r.get("ma20","-")}，{st}')
                    break
        p4 = m4.get('penalty', 0)
        if p4:
            L.append(f'- **扣分 {p4}**：{"；".join(m4.get("penalty_detail", []))}')
        else:
            L.append('- 核心宽基均在均线之上，趋势健康')
    L.append('')

    # ── M5 ──
    m5 = data.get('module5') or {}
    L.append('## M5 风格因子监控')
    if m5.get('error'):
        L.append(f'> 异常：{m5["error"]}')
    else:
        L.append(f'- Size Spread = {_pct(m5.get("spread_size"))}，连续 {m5.get("size_consec_days",0)} 天')
        L.append(f'- Vol Spread = {_pct(m5.get("spread_vol"))}，连续 {m5.get("vol_consec_days",0)} 天')
        if m5.get('r_csi1000_vs_sh50') is not None:
            L.append(f'- 中证1000 vs 上证50 = {_pct(m5["r_csi1000_vs_sh50"])}')
        p5 = m5.get('penalty', 0)
        if p5:
            L.append(f'- **扣分 {p5}**：{"；".join(m5.get("penalty_detail", []))}')
        else:
            L.append('- 风格因子均衡，未出现极端割裂')
    L.append('')

    # ── M6 ──
    m6 = data.get('module6') or {}
    L.append('## M6 短线赚钱效应')
    if m6.get('error'):
        L.append(f'> 异常：{m6["error"]}')
    else:
        L.append(f'- 昨日涨停 {m6.get("hot_count","-")} 只，今日均收益 {_pct(m6.get("hot_avg_return"))}')
        L.append(f'- 核按钮 {m6.get("nuke_count","-")} 只（{_pct(m6.get("nuke_ratio"))}）')
        p6 = m6.get('penalty', 0)
        if p6:
            L.append(f'- **扣分 {p6}**：{"；".join(m6.get("penalty_detail", []))}')
        else:
            L.append('- 打板资金有正收益，接力情绪存续')
    L.append('')

    # ── M7 ──
    m7 = data.get('module7') or {}
    L.append('## M7 聪明钱与日内承接力')
    if m7.get('error'):
        L.append(f'> 异常：{m7["error"]}')
    else:
        sdr = m7.get('smart_distrib_ratio')
        ttr = m7.get('top_turnover_avg_return')
        if sdr is not None:
            L.append(f'- 派发资金占比 = {_pct(sdr)}（{"主力派发明显" if sdr > 0.6 else "派发压力可控"}）')
        if ttr is not None:
            L.append(f'- 活跃池收益 = {_pct(ttr)}（{"热钱被闷杀" if ttr < -0.02 else "活跃资金可盈利"}）')
        p7 = m7.get('penalty', 0)
        if p7:
            L.append(f'- **扣分 {p7}**：{"；".join(m7.get("penalty_detail", []))}')
        else:
            L.append('- 聪明钱指标正常，日内承接有力')
    L.append('')

    # ── M8 ──
    m8 = data.get('module8') or {}
    L.append('## M8 强势股补跌预警')
    if m8.get('error'):
        L.append(f'> 异常：{m8["error"]}')
    else:
        L.append(f'- 强势股池 {m8.get("strong_count","-")} 只，'
                 f'今日均值 {_pct(m8.get("strong_avg_return"))}')
        if m8.get('market_return_today') is not None:
            L.append(f'- 大盘日收益 {_pct(m8["market_return_today"])}')
        p8 = m8.get('penalty', 0)
        if p8:
            L.append(f'- **扣分 {p8}**：{"；".join(m8.get("penalty_detail", []))}')
        else:
            L.append('- 动量因子稳定，强势股未集体补跌')
    L.append('')

    # ── 综合诊断 ──
    L.append('---\n## 综合诊断')
    modules = [data.get(f'module{i}') or {} for i in range(1, 9)]
    triggered = [f'M{i+1}' for i, mx in enumerate(modules) if (mx.get('penalty') or 0) > 0]
    if not triggered:
        L.append('八大维度均未触发扣分，市场全面健康，可积极参与主线方向。')
    else:
        L.append(f'触发扣分模块：{"、".join(triggered)}。')
        if score >= 70:
            L.append('整体风险可控，局部扣分不影响大局，保持多头思维，关注边际变化。')
        elif score >= 50:
            L.append('结构性分化信号出现，精选个股、缩小战线，避免追高弱势板块。')
        elif score >= 30:
            L.append('多项风控指标亮灯，进入防御区间，降低仓位、严格止损、保全本金。')
        else:
            L.append('系统性风险密集释放，极端恐慌环境，建议清仓等待右侧反转确认。')

    return '\n'.join(L)
