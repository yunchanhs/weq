# -*- coding: utf-8 -*-
import time
import os
import sys
import math
import pickle
import logging
import threading
from logging.handlers import RotatingFileHandler
from datetime import datetime, timedelta, date
from collections import defaultdict, deque

import pyupbit
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# =============== 실행/환경 기본 ===============
DRY_RUN = False
BEAR_ALLOW_BUYS = True                 # 하락장 예외 진입 허용
torch.set_num_threads(1)               # t3.medium CPU 스레드 고정
torch.set_num_interop_threads(1)

# Upbit 키 (환경변수 권장)
ACCESS_KEY = os.getenv("UPBIT_ACCESS_KEY", "")
SECRET_KEY = os.getenv("UPBIT_SECRET_KEY", "")

# =============== 전략 파라미터 ===============
last_trained_time = {}
TRAINING_INTERVAL = timedelta(hours=6)

ML_BASE_THRESHOLD = 0.50
STOP_LOSS_THRESHOLD = -0.05
COOLDOWN_TIME = timedelta(minutes=30)
SURGE_COOLDOWN_TIME = timedelta(minutes=60)

MAX_ACTIVE_POSITIONS_BASE = 3
FOCUS_SLOT_ENABLE = True               # 고신뢰 1개 추가(최대 4개)
USE_CASH_RATIO_BASE = 0.95
MIN_ORDER_KRW = 6000

TOP_POOL_MULTIPLIER = 12
TOP_POOL_BASE = 4

# 적응형 문턱(유연형)
TBUY_LAX_FACTOR = 0.95                 # 완화 진입: T_buy * 0.95
TSELL_GAP = 0.04                       # T_sell = T_buy - 0.04

# 자본/리저브/드로우다운
DAILY_MAX_LOSS = 0.02
MAX_CONSECUTIVE_LOSSES = 3
PROFIT_SKIM_TRIGGER = 0.03
PROFIT_SKIM_RATIO = 0.25
RESERVE_RELEASE_DD = 0.02
POS_RISK_CAP = 0.0075                  # 포지션당 계좌위험 상한 0.75%

# 부분 익절/트레일링
PARTIAL_TP1, TP1_RATIO = 0.08, 0.40
PARTIAL_TP2, TP2_RATIO = 0.15, 0.30
TRAIL_DROP_BULL, TRAIL_DROP_BEAR = 0.04, 0.025

# =============== 상태 ===============
entry_prices = {}
highest_prices = {}
recent_trades = {}
recent_surge_tickers = {}
ml_hist = defaultdict(lambda: deque(maxlen=300))
pos_plan = {}
last_top_update = datetime.min

reserved_profit = 0.0
equity_hwm = 0.0
pnl_today = 0.0
consecutive_losses = 0
try:
    pnl_day = datetime.now().date()
except Exception:
    pnl_day = datetime.now().date()

state_lock = threading.Lock()

# =============== 로깅 ===============
log = logging.getLogger("bot")
log.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
fh = RotatingFileHandler("bot.log", maxBytes=5_000_000, backupCount=3, encoding="utf-8")
fh.setFormatter(fmt)
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(fmt)
log.addHandler(fh)
log.addHandler(ch)

def atomic_save(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

def load_pickle(filename, default_value):
    if os.path.exists(filename):
        try:
            with open(filename, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            log.warning(f"[로드 실패] {filename}: {e}")
    return default_value

# 상태 복원
entry_prices = load_pickle("entry_prices.pkl", {})
recent_trades = load_pickle("recent_trades.pkl", {})
highest_prices = load_pickle("highest_prices.pkl", {})
reserved_profit = load_pickle("reserved_profit.pkl", 0.0)
equity_hwm = load_pickle("equity_hwm.pkl", 0.0)
pnl_today = load_pickle("pnl_today.pkl", 0.0)
try:
    _pday = load_pickle("pnl_day.pkl", datetime.now().date().isoformat())
    pnl_day = datetime.fromisoformat(_pday).date() if isinstance(_pday, str) else datetime.now().date()
except Exception:
    pnl_day = datetime.now().date()
consecutive_losses = load_pickle("consecutive_losses.pkl", 0)

def auto_save_state(interval=300):
    while True:
        try:
            with state_lock:
                atomic_save(entry_prices, "entry_prices.pkl")
                atomic_save(recent_trades, "recent_trades.pkl")
                atomic_save(highest_prices, "highest_prices.pkl")
                atomic_save(reserved_profit, "reserved_profit.pkl")
                atomic_save(equity_hwm, "equity_hwm.pkl")
                atomic_save(pnl_today, "pnl_today.pkl")
                atomic_save(pnl_day.isoformat(), "pnl_day.pkl")
                atomic_save(consecutive_losses, "consecutive_losses.pkl")
            log.info("[백업] 상태 자동 저장 완료")
        except Exception as e:
            log.exception(f"[백업 오류] 상태 저장 실패: {e}")
        time.sleep(interval)

# =============== 안전 OHLCV & 검증 ===============
def safe_get_ohlcv(ticker, interval="minute5", count=200, max_retries=5, base_sleep=0.7):
    for attempt in range(1, max_retries+1):
        try:
            df = pyupbit.get_ohlcv(ticker, interval=interval, count=count)
            if df is not None and not df.empty and all(c in df.columns for c in ["open","high","low","close","volume"]):
                return df
            else:
                log.info(f"[safe_get_ohlcv] 빈 DF/컬럼 부족: {ticker} {interval} (시도 {attempt}/{max_retries})")
        except Exception as e:
            log.info(f"[safe_get_ohlcv] 예외: {ticker} {interval} (시도 {attempt}/{max_retries}) → {e}")
        time.sleep(base_sleep * attempt)
    return None

def is_valid_df(df, min_len=5):
    return df is not None and not df.empty and len(df) >= min_len and all(
        c in df.columns for c in ["open","high","low","close","volume"]
    )

# =============== 지표/피처 ===============
def get_macd_from_df(df):
    df = df.copy()
    df['short_ema'] = df['close'].ewm(span=12, adjust=False).mean()
    df['long_ema'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['short_ema'] - df['long_ema']
    df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    return df

def get_rsi_from_df(df, period=14):
    df = df.copy()
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    return df

def get_adx_from_df(df, period=14):
    df = df.copy()
    df['H-L'] = df['high'] - df['low']
    df['H-C'] = (df['high'] - df['close'].shift(1)).abs()
    df['L-C'] = (df['low'] - df['close'].shift(1)).abs()
    df['TR'] = df[['H-L','H-C','L-C']].max(axis=1)
    df['+DM'] = (df['high'] - df['high'].shift(1)).clip(lower=0)
    df['-DM'] = (df['low'].shift(1) - df['low']).clip(lower=0)
    df['+DM'] = df['+DM'].where(df['+DM'] > df['-DM'], 0)
    df['-DM'] = df['-DM'].where(df['-DM'] > df['+DM'], 0)
    df['TR_smooth'] = df['TR'].rolling(window=period).sum()
    df['+DM_smooth'] = df['+DM'].rolling(window=period).sum()
    df['-DM_smooth'] = df['-DM'].rolling(window=period).sum()
    df['+DI'] = 100 * (df['+DM_smooth'] / df['TR_smooth'])
    df['-DI'] = 100 * (df['-DM_smooth'] / df['TR_smooth'])
    df['DX'] = 100 * (df['+DI'] - df['-DI']).abs() / ((df['+DI'] + df['-DI']).replace(0,np.nan))
    df['adx'] = df['DX'].rolling(window=period).mean()
    return df

def get_atr_from_df(df, period=14):
    df = df.copy()
    df['H-L'] = df['high'] - df['low']
    df['H-C'] = (df['high'] - df['close'].shift(1)).abs()
    df['L-C'] = (df['low'] - df['close'].shift(1)).abs()
    df['TR'] = df[['H-L','H-C','L-C']].max(axis=1)
    df['atr'] = df['TR'].rolling(window=period).mean()
    return df

def get_features(ticker, normalize=True):
    df = safe_get_ohlcv(ticker, interval="minute5", count=1000)
    if not is_valid_df(df, min_len=100):
        return pd.DataFrame()
    df = get_macd_from_df(df)
    df = get_rsi_from_df(df)
    df = get_adx_from_df(df)
    df = get_atr_from_df(df)
    df['return'] = df['close'].pct_change()
    df['future_return'] = df['close'].shift(-1) / df['close'] - 1
    df.dropna(inplace=True)
    if normalize and not df.empty:
        scaler = MinMaxScaler()
        cols = ['macd','signal','rsi','adx','atr','return','future_return']
        df[cols] = scaler.fit_transform(df[cols])
    return df

# =============== 자본/리스크/유틸 ===============
def reset_daily_if_needed():
    global pnl_today, pnl_day, consecutive_losses
    today = datetime.now().date()
    if pnl_day != today:
        pnl_day = today
        pnl_today = 0.0
        consecutive_losses = 0

def record_trade_pnl(pnl_ratio):
    global pnl_today, consecutive_losses
    pnl_today += pnl_ratio
    if pnl_ratio < 0:
        consecutive_losses += 1
    else:
        consecutive_losses = 0

def upbit_client():
    return pyupbit.Upbit(ACCESS_KEY, SECRET_KEY)

upbit = upbit_client()

def get_balance(asset):
    try:
        b = upbit.get_balance(asset)
        return 0.0 if b is None else float(b)
    except Exception as e:
        log.info(f"[잔고 오류] {asset}: {e}")
        return 0.0

def get_avg_buy_price_from_balances(ticker: str):
    """업비트 잔고의 평단가(avg_buy_price) 조회"""
    try:
        curr = ticker.split("-")[1]
        bals = upbit.get_balances()
        if isinstance(bals, list):
            for b in bals:
                if b.get("currency") == curr:
                    ap = b.get("avg_buy_price")
                    return float(ap) if ap not in (None, "") else None
    except Exception:
        return None
    return None

def buy_crypto_currency(ticker, amount_krw):
    if DRY_RUN:
        log.info(f"[DRY_RUN][BUY] {ticker} {amount_krw:.0f} KRW")
        return {"dry": True}
    try:
        return upbit.buy_market_order(ticker, amount_krw)
    except Exception as e:
        log.info(f"[{ticker}] 매수 에러: {e}")
        return None

def sell_crypto_currency(ticker, amount_coin):
    if DRY_RUN:
        log.info(f"[DRY_RUN][SELL] {ticker} {amount_coin}")
        return {"dry": True}
    try:
        return upbit.sell_market_order(ticker, amount_coin)
    except Exception as e:
        log.info(f"[{ticker}] 매도 에러: {e}")
        return None

def calc_total_equity():
    try:
        krw = get_balance("KRW") or 0.0
    except Exception:
        krw = 0.0
    equity = float(krw)
    for t in set(entry_prices.keys()):
        try:
            coin = t.split('-')[1]
            bal = get_balance(coin)
            if bal and bal > 1e-10:
                px = pyupbit.get_current_price(t)
                if px:
                    equity += float(bal) * float(px)
        except Exception:
            continue
    return equity

def get_initial_balance_for_backtest():
    eq = calc_total_equity()
    return max(300_000, min(10_000_000, int(eq)))

def update_profit_reserve():
    global equity_hwm, reserved_profit
    eq = calc_total_equity()
    if eq > equity_hwm:
        equity_hwm = eq
    threshold = equity_hwm * (1 + PROFIT_SKIM_TRIGGER)
    if eq >= threshold:
        skim_amount = (eq - equity_hwm) * PROFIT_SKIM_RATIO
        if skim_amount > 0:
            reserved_profit += skim_amount
            equity_hwm = eq
            log.info(f"[RESERVE] Skim +{skim_amount:.0f}원 | reserve={reserved_profit:.0f}, HWM={equity_hwm:.0f}")
    if equity_hwm > 0:
        dd = (equity_hwm - eq) / equity_hwm
        if dd >= RESERVE_RELEASE_DD and reserved_profit > 0:
            release = reserved_profit * 0.5
            reserved_profit -= release
            log.info(f"[RESERVE] DD {dd*100:.2f}% → Release {release:.0f}원 | reserve={reserved_profit:.0f}")
    return eq

def get_dd_stage_params():
    """
    stage0: DD<5%     → 기본
    stage1: DD≥5%     → 현금비중 0.80
    stage2: DD≥10%    → 포지션수 -1
    stage3: DD≥15%    → 신규매수 차단 + 현금비중 0.70
    """
    eq = calc_total_equity()
    dd = 0.0 if equity_hwm <= 0 else (equity_hwm - eq) / equity_hwm
    stage = 0
    use_cash = USE_CASH_RATIO_BASE
    max_pos = MAX_ACTIVE_POSITIONS_BASE
    buy_block = False
    if dd >= 0.15:
        stage = 3; use_cash = 0.70; max_pos = max(1, MAX_ACTIVE_POSITIONS_BASE-2); buy_block = True
    elif dd >= 0.10:
        stage = 2; use_cash = 0.75; max_pos = max(1, MAX_ACTIVE_POSITIONS_BASE-1)
    elif dd >= 0.05:
        stage = 1; use_cash = 0.80
    log.info(f"[DD-MONITOR] DD={dd*100:.2f}% (Stage {stage}) | use_cash={use_cash:.2f}, max_pos={max_pos}, buy_block={buy_block}")
    return dd, stage, use_cash, max_pos, buy_block

def reconcile_positions_from_balance():
    to_drop = []
    for t in list(entry_prices.keys()):
        try:
            coin = t.split('-')[1]
            bal = get_balance(coin)
            if not bal or bal < 1e-8:
                to_drop.append(t)
        except Exception:
            to_drop.append(t)
    if to_drop:
        with state_lock:
            for t in to_drop:
                entry_prices.pop(t, None)
                highest_prices.pop(t, None)
        log.info(f"[RECONCILE] 실보유 0인 티커 정리: {to_drop}")

def get_held_tickers_from_balance():
    held = set()
    for t in entry_prices.keys():
        try:
            coin = t.split('-')[1]
            bal = get_balance(coin)
            if bal and bal > 1e-8:
                held.add(t)
        except Exception:
            pass
    return held

# =============== 레짐/브레드스 ===============
def get_asset_regime(ticker):
    try:
        df = safe_get_ohlcv(ticker, interval="minute60", count=200)
        if not is_valid_df(df, min_len=100):
            return "neutral"
        mac = get_macd_from_df(df.copy())
        macd, signal = mac['macd'].iloc[-1], mac['signal'].iloc[-1]
        rsi = get_rsi_from_df(df.copy())['rsi'].iloc[-1]
        if (macd > signal) and (rsi > 55): return "bull"
        if (macd < signal) and (rsi < 45): return "bear"
        return "neutral"
    except Exception:
        return "neutral"

def compute_breadth_above_ma20(top_list):
    if not top_list:
        log.info("[BREADTH] top_list 비어 있음 → 0.0")
        return 0.0
    count_above, total = 0, 0
    for t in top_list:
        try:
            df = safe_get_ohlcv(t, interval="minute60", count=60)
            if not is_valid_df(df, min_len=25):
                continue
            ma20 = df['close'].rolling(window=20).mean()
            if pd.notna(ma20.iloc[-1]):
                total += 1
                if df['close'].iloc[-1] > ma20.iloc[-1]:
                    count_above += 1
        except Exception:
            continue
    if total == 0:
        return 0.0
    return count_above / total

def composite_market_regime(top_list):
    btc_reg = get_asset_regime("KRW-BTC")
    eth_reg = get_asset_regime("KRW-ETH")
    breadth = compute_breadth_above_ma20(top_list)
    # 초기 breadth 0.0 보정: BTC/ETH 모두 bull이면 neutral로 완화
    if btc_reg == "bull" and eth_reg == "bull" and breadth == 0.0:
        regime = "neutral"
    elif btc_reg == "bear" or breadth < 0.40:
        regime = "bear"
    elif (btc_reg == "bull") and (eth_reg == "bull" or breadth > 0.60):
        regime = "bull"
    else:
        regime = "neutral"
    log.info(f"[REGIME] BTC={btc_reg} ETH={eth_reg} breadth={breadth*100:.1f}% → regime={regime}")
    return regime

# =============== 하이브리드 모델 (CNN + LSTM + Transformer) ===============
class CNNBlock(nn.Module):
    def __init__(self, in_ch=6, channels=16, k=3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, channels, kernel_size=k, padding=k//2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=k, padding=k//2)
        self.pool = nn.AdaptiveAvgPool1d(1)
    def forward(self, x):         # x: (B, T, F)
        x = x.transpose(1, 2)     # (B, F, T)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)  # (B, C)
        return x

class LSTMBlock(nn.Module):
    def __init__(self, in_dim=6, hid=32, layers=1):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hid, num_layers=layers, batch_first=True)
    def forward(self, x):         # (B, T, F)
        o, _ = self.lstm(x)
        return o[:, -1, :]        # (B, hid)

class TransBlock(nn.Module):
    def __init__(self, in_dim=6, d_model=32, heads=4, layers=1):
        super().__init__()
        self.emb = nn.Linear(in_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=heads, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=layers)
    def forward(self, x):         # (B, T, F)
        x = self.emb(x)
        x = self.enc(x)
        return x[:, -1, :]        # (B, d_model)

class HybridModel(nn.Module):
    def __init__(self, in_dim=6, seq_len=30):
        super().__init__()
        self.cnn = CNNBlock(in_ch=in_dim, channels=16, k=3)
        self.lstm = LSTMBlock(in_dim=in_dim, hid=32, layers=1)
        self.tran = TransBlock(in_dim=in_dim, d_model=32, heads=4, layers=1)
        mix_dim = 16 + 32 + 32
        self.head = nn.Sequential(
            nn.Linear(mix_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, x):         # (B, T, F)
        c = self.cnn(x)
        l = self.lstm(x)
        t = self.tran(x)
        h = torch.cat([c, l, t], dim=1)
        out = self.head(h)
        return out                 # (B, 1) in [0,1]

# =============== 데이터셋/학습 ===============
class TradingDataset(Dataset):
    def __init__(self, data, seq_len=30):
        self.data = data
        self.seq_len = seq_len
    def __len__(self):
        return max(0, len(self.data) - self.seq_len)
    def __getitem__(self, idx):
        x = self.data.iloc[idx:idx+self.seq_len][['macd','signal','rsi','adx','atr','return']].values
        y = self.data.iloc[idx + self.seq_len]['future_return']
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def train_hybrid_model(ticker, epochs=40):
    log.info(f"모델 학습 시작: {ticker}")
    data = get_features(ticker, normalize=True)
    if data is None or data.empty:
        log.info(f"경고: {ticker} 데이터 비어 있음. 학습 스킵"); return None
    ds = TradingDataset(data, seq_len=30)
    if len(ds) == 0:
        log.info(f"경고: {ticker} 데이터셋 너무 작음. 학습 스킵"); return None
    dl = DataLoader(ds, batch_size=32, shuffle=True, num_workers=0)
    model = HybridModel(in_dim=6, seq_len=30)
    crit = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    for ep in range(1, epochs+1):
        for xb, yb in dl:
            opt.zero_grad()
            pred = model(xb).view(-1)
            loss = crit(pred, yb.view(-1))
            loss.backward()
            opt.step()
        if ep % 5 == 0 or ep == epochs:
            log.info(f"[{ticker}] Epoch {ep}/{epochs} | Loss: {loss.item():.4f}")
    log.info(f"모델 학습 완료: {ticker}")
    return model

# =============== 백테스트 & 워크포워드 ===============
def backtest_series(data, model, init_bal, t_buy, t_sell, fee=0.0005, slip_bp=10):
    if data is None or data.empty: return 1.0
    seq = 30
    slip = slip_bp/10000.0
    bal, pos, entry = init_bal, 0.0, 0.0
    hi = 0.0
    for i in range(seq, len(data)-1):
        X = torch.tensor(data.iloc[i-seq:i][['macd','signal','rsi','adx','atr','return']].values, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            s = model(X).item()
        px = data.iloc[i]['close']
        if pos == 0 and s > t_buy:
            fill = px * (1 + slip)
            pos = bal / fill
            entry = fill
            hi = entry
            bal = 0
        elif pos > 0:
            hi = max(hi, px)
            unrl = (px - entry)/entry
            peak_drop = (hi - px)/hi if hi>0 else 0
            if unrl < STOP_LOSS_THRESHOLD:
                bal = pos * px * (1 - fee); pos = 0
            elif peak_drop > 0.02 and s < t_sell:
                bal = pos * px * (1 - fee); pos = 0
    final = bal + pos * data.iloc[-1]['close']
    return final / init_bal

def run_bt_wf(ticker, model, init_bal, regime):
    t_buy, t_sell = compute_ml_threshold(ticker, regime, use_lax=False)
    perf_bt = backtest(ticker, model, initial_balance=init_bal)
    df_all = get_features(ticker)
    if df_all is None or df_all.empty or len(df_all) < 200:
        perf_wf = 1.0
    else:
        cut = int(len(df_all)*0.8)
        wf = df_all.iloc[cut:].copy()
        perf_wf = backtest_series(wf, model, init_bal, t_buy, t_sell)
    return perf_bt, perf_wf

def backtest(ticker, model, initial_balance=1_000_000, fee=0.0005, slip_bp=10):
    data = get_features(ticker)
    if data is None or data.empty:
        return 1.0
    t_buy, t_sell = 0.55, 0.50
    return backtest_series(data, model, initial_balance, t_buy, t_sell, fee, slip_bp)

# =============== 문턱(적응형+완화안) ===============
def compute_ml_threshold(ticker, regime, use_lax=False):
    base = ML_BASE_THRESHOLD
    hist = ml_hist[ticker]
    if len(hist) >= 60:
        q = float(np.quantile(hist, 0.75))
        base = max(0.40, min(0.65, q))
    if regime == "bull":
        base -= 0.03
    elif regime == "bear":
        base += 0.04
    base = max(0.38, min(0.70, base))
    t_buy = base
    if use_lax:
        t_buy = max(0.36, t_buy * TBUY_LAX_FACTOR)
    t_sell = max(0.0, t_buy - TSELL_GAP)
    return t_buy, t_sell

def get_dynamic_tp_by_regime(regime: str) -> float:
    if regime == "bull":
        return 0.35
    elif regime == "bear":
        return 0.15
    return 0.25

# =============== 급등 감지/스프레드/랭킹 ===============
def detect_surge_tickers(threshold=0.03, interval="minute5", lookback=3):
    tickers = pyupbit.get_tickers(fiat="KRW")
    surged = []
    for t in tickers:
        try:
            df = pyupbit.get_ohlcv(t, interval=interval, count=lookback+1)
            if df is None or len(df) < lookback+1: continue
            chg = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]
            if chg >= threshold: surged.append(t)
        except Exception:
            continue
    if surged:
        log.info(f"[SURGE] 감지된 급등 코인: {surged}")
    return surged

def get_spread_bp(ticker):
    try:
        ob = pyupbit.get_orderbook(ticker)
        if not ob or 'orderbook_units' not in ob[0]: return None
        u = ob[0]['orderbook_units'][0]
        ask, bid = float(u['ask_price']), float(u['bid_price'])
        spread = (ask - bid) / ((ask + bid)/2)
        return spread * 10000.0
    except Exception:
        return None

def rank_universe(candidates, surge_dict):
    scored = []
    for t in candidates:
        sp = get_spread_bp(t)
        sp_score = 0.0 if sp is None else max(-2.0, min(2.0, (15.0 - sp)/5.0))
        surge_bonus = 0.5 if t in surge_dict else 0.0
        scored.append((t, sp_score + surge_bonus))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [t for t,_ in scored]

# =============== 동적 상위 N ===============
def compute_top_n(current_top=None):
    equity = calc_total_equity()
    scale = min(2.0, max(0.8, equity / 1_500_000))
    try:
        regime = composite_market_regime(current_top or [])
    except Exception:
        regime = "neutral"
    regime_k = 1.2 if regime=="bull" else (0.8 if regime=="bear" else 1.0)
    base = MAX_ACTIVE_POSITIONS_BASE * TOP_POOL_MULTIPLIER + TOP_POOL_BASE
    n = int(base * scale * regime_k)
    n = max(20, min(60, n))
    log.info(f"[TOP-N] equity≈{equity:.0f}, regime={regime}, scale={scale:.2f}, n={n}")
    return n

def get_top_tickers(n=None):
    if n is None or n <= 0:
        n = compute_top_n(current_top=[])
    n = max(20, min(60, int(n)))
    tickers = pyupbit.get_tickers(fiat="KRW")
    scores = []
    for t in tickers:
        df = safe_get_ohlcv(t, interval="day", count=3)
        if not is_valid_df(df, min_len=3): continue
        v = float((df['close'] * df['volume']).mean())
        scores.append((t, np.log1p(v)))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [t for t,_ in scores[:n]]

# =============== ATR 기반 포지션 사이징 (Full-Capital Adaptive) ===============
def calc_atr_position_budget(remaining_krw, remaining_slots, atr_abs, px, equity, base_risk=0.006, high_conf=False):
    if atr_abs is None or atr_abs <= 0 or px <= 0:
        return (remaining_krw * USE_CASH_RATIO_BASE) / max(1, remaining_slots)
    k = 1.5
    est_stop_ratio = (atr_abs * k) / px
    if est_stop_ratio <= 0:
        return (remaining_krw * USE_CASH_RATIO_BASE) / max(1, remaining_slots)
    risk_unit = base_risk * (1.3 if high_conf else 1.0)
    budget_by_risk = (equity * risk_unit) / est_stop_ratio
    budget_hardcap = (equity * POS_RISK_CAP) / est_stop_ratio
    equal_split = (remaining_krw * USE_CASH_RATIO_BASE) / max(1, remaining_slots)
    target = max(MIN_ORDER_KRW, min(budget_by_risk, 1.7*equal_split, 1.5*budget_hardcap))
    return target

# =============== 부분 익절/매도 로직 ===============
def try_partial_take_profit(ticker, change_ratio, coin_balance, now):
    did = False
    if change_ratio >= PARTIAL_TP2 and coin_balance > 0:
        amt = coin_balance * TP2_RATIO
        if sell_crypto_currency(ticker, amt):
            did = True; log.info(f"[{ticker}] 부분익절2: +{PARTIAL_TP2*100:.0f}% → {TP2_RATIO*100:.0f}% 매도")
    elif change_ratio >= PARTIAL_TP1 and coin_balance > 0:
        amt = coin_balance * TP1_RATIO
        if sell_crypto_currency(ticker, amt):
            did = True; log.info(f"[{ticker}] 부분익절1: +{PARTIAL_TP1*100:.0f}% → {TP1_RATIO*100:.0f}% 매도")
    if did:
        with state_lock:
            recent_trades[ticker] = now
    return did

def should_sell(ticker, current_price, ml_signal, t_sell, regime):
    if ticker not in entry_prices: return False
    entry = entry_prices[ticker]
    highest_prices[ticker] = max(highest_prices.get(ticker, entry), current_price)
    chg = (current_price - entry) / entry
    peak_drop = (highest_prices[ticker] - current_price) / highest_prices[ticker]
    weak_ml = (ml_signal < t_sell)
    if chg < -0.05: log.info(f"[{ticker}] 🚨 -5% 손절 발동"); return True
    tp_dyn = get_dynamic_tp_by_regime(regime)
    if chg >= tp_dyn:
        log.info(f"[{ticker}] 레짐 기반 강제 익절 도달(+{tp_dyn*100:.0f}%)")
        # True를 바로 반환하지 않는 이유: 메인에서 실시간 변화율로 한 번 더 강제청산 판단
    elif chg >= 0.15:
        if weak_ml or ml_signal < 0.6: log.info(f"[{ticker}] ✅ +15% & ML 약함 → 익절"); return True
        else: return False
    elif chg >= 0.10:
        if weak_ml or ml_signal < 0.5: log.info(f"[{ticker}] ✅ +10% & ML 약함 → 익절"); return True
    trail = TRAIL_DROP_BULL if regime=="bull" else TRAIL_DROP_BEAR
    if peak_drop > trail and (weak_ml or ml_signal < 0.5):
        log.info(f"[{ticker}] 📉 트레일링 스탑! 고점대비 {peak_drop*100:.2f}%"); return True
    try:
        if chg > 0.05 and ml_signal > 0.6:
            df = safe_get_ohlcv(ticker, interval="minute5", count=200)
            if is_valid_df(df, 50):
                df = get_macd_from_df(df)
                if df['macd'].iloc[-1] > df['signal'].iloc[-1]:
                    return False
    except Exception:
        pass
    return False

# =============== 메인 루프 ===============
def main():
    print("자동매매 시작!")
    models = {}
    threading.Thread(target=auto_save_state, daemon=True).start()

    # 초기 상위 코인 & 즉시 업데이트 시간 세팅
    top_tickers = get_top_tickers()
    log.info(f"[{datetime.now()}] 상위 코인 초기화(N={len(top_tickers)}): {top_tickers}")
    global last_top_update
    last_top_update = datetime.now()

    # 초기 레짐
    regime = composite_market_regime(top_tickers)

    # 초기 학습 (BT+WF, 점진적 완화)
    def add_if_pass(ticker, model, perf_bt, perf_wf, strict=True):
        if strict:
            cond = (perf_bt >= 1.05 and perf_wf >= 1.02)
        else:
            cond = (perf_bt >= 1.02 and perf_wf >= 1.00)
        if cond:
            models[ticker] = model
            last_trained_time[ticker] = datetime.now()
            log.info(f"[{ticker}] 모델 채택 (일반:{perf_bt:.2f} / 워크:{perf_wf:.2f}, strict={strict})")
            return True
        else:
            log.info(f"[{ticker}] 모델 제외 (일반:{perf_bt:.2f} / 워크:{perf_wf:.2f}, strict={strict})")
            return False

    passed_any = False
    for t in top_tickers:
        m = train_hybrid_model(t, epochs=40)
        if m is None: continue
        init_bal = get_initial_balance_for_backtest()
        bt, wf = run_bt_wf(t, m, init_bal, regime)
        ok = add_if_pass(t, m, bt, wf, strict=True)
        passed_any = passed_any or ok

    if not passed_any:
        log.info("[FILTER] 엄격 기준 통과 없음 → 완화 기준으로 재평가(MTF)")
        for t in top_tickers:
            if t in models: continue
            m = train_hybrid_model(t, epochs=25)
            if m is None: continue
            init_bal = get_initial_balance_for_backtest()
            bt, wf = run_bt_wf(t, m, init_bal, regime)
            add_if_pass(t, m, bt, wf, strict=False)

    reconcile_positions_from_balance()
    last_reconcile = datetime.min
    recent_surge = {}

    try:
        while True:
            now = datetime.now()
            reset_daily_if_needed()
            eq = update_profit_reserve()

            # DD 파라미터
            dd, dd_stage, USE_CASH_RATIO_EFF, MAX_ACTIVE_POS_EFF, BUY_BLOCK_DD = get_dd_stage_params()

            # 상위 코인 6시간마다 업데이트+필요시 재학습
            if (now - last_top_update) >= timedelta(hours=6):
                top_tickers = get_top_tickers()
                log.info(f"[{now}] 상위 코인 업데이트(N={len(top_tickers)}): {top_tickers}")
                last_top_update = now
                regime = composite_market_regime(top_tickers)
                for t in top_tickers:
                    if (t not in models) or (datetime.now() - last_trained_time.get(t, datetime.min) > TRAINING_INTERVAL):
                        m = train_hybrid_model(t, epochs=30)
                        if m is None: continue
                        init_bal = get_initial_balance_for_backtest()
                        bt, wf = run_bt_wf(t, m, init_bal, regime)
                        if not add_if_pass(t, m, bt, wf, strict=True):
                            add_if_pass(t, m, bt, wf, strict=False)

            # 유령 보유 정리(30분)
            if (now - last_reconcile) >= timedelta(minutes=30):
                reconcile_positions_from_balance()
                last_reconcile = now

            # 급등 감지
            surged = detect_surge_tickers(threshold=0.03)
            for t in surged:
                if t not in recent_surge:
                    recent_surge[t] = now
                    log.info(f"[{now}] 급상승 감지: {t}")
                    if t not in models:
                        m = train_hybrid_model(t, epochs=15)
                        if m is not None:
                            init_bal = get_initial_balance_for_backtest()
                            bt, wf = run_bt_wf(t, m, init_bal, regime)
                            if not add_if_pass(t, m, bt, wf, strict=True):
                                add_if_pass(t, m, bt, wf, strict=False)

            # 레짐 갱신
            regime = composite_market_regime(top_tickers)

            # 하드 블록(DD/일손실)
            risk_block = (pnl_today <= -DAILY_MAX_LOSS) or (consecutive_losses >= MAX_CONSECUTIVE_LOSSES)
            held = get_held_tickers_from_balance()

            # 동적 슬롯 (고신뢰 포커스 1개 추가 가능)
            max_slots = MAX_ACTIVE_POS_EFF
            if FOCUS_SLOT_ENABLE:
                max_slots = min(MAX_ACTIVE_POS_EFF + 1, 4)

            slots_available = max(0, max_slots - len(held))

            # 신규 진입 대상(랭킹 힌트용)
            target_pool = set(top_tickers) | set(surged) | set(models.keys())
            universe_new = list(target_pool - held)

            ranked_hint = rank_universe(universe_new, recent_surge)
            log.info(f"[BUY-HINT] ranked(top5)={ranked_hint[:5]} slots={slots_available} regime={regime} risk_block={risk_block}")

            # 자금
            krw_now = get_balance("KRW") or 0.0
            usable_krw = max(0.0, krw_now - reserved_profit)
            remaining_krw = usable_krw
            remaining_slots = slots_available
            log.info(f"[RESERVE] KRW={krw_now:.0f}, reserve={reserved_profit:.0f}, usable={usable_krw:.0f}, HWM={equity_hwm:.0f}")

            # 매수/매도 루프(보유 + 후보)
            targets = set(held) | set(universe_new)
            for t in list(targets):
                # -------- Patch #1: 쿨다운은 신규 매수만 제한 --------
                in_position = t in entry_prices
                cooldown = SURGE_COOLDOWN_TIME if t in recent_surge else COOLDOWN_TIME
                if (now - recent_trades.get(t, datetime.min) < cooldown) and (not in_position):
                    continue

                if t not in models:
                    log.info(f"[{t}] 모델 없음 → 스킵"); continue

                # 피처/지표 최신
                df = safe_get_ohlcv(t, interval="minute5", count=200)
                if not is_valid_df(df, 50): continue
                df = get_macd_from_df(df); df = get_rsi_from_df(df)
                df = get_adx_from_df(df); df = get_atr_from_df(df)
                feats = get_features(t, normalize=False)
                if feats is None or feats.empty: continue

                macd = feats['macd'].iloc[-1]; signal = feats['signal'].iloc[-1]
                rsi = feats['rsi'].iloc[-1]; adx = feats['adx'].iloc[-1]
                atr_abs = feats['atr'].iloc[-1]; px = feats['close'].iloc[-1]

                # 모델 신호
                X = torch.tensor(feats[['macd','signal','rsi','adx','atr','return']].tail(30).values,
                                 dtype=torch.float32).unsqueeze(0)
                model = models[t]
                model.eval()
                with torch.no_grad():
                    ml = model(X).item()
                ml_hist[t].append(ml)

                T_buy_strict, T_sell = compute_ml_threshold(t, regime, use_lax=False)
                T_buy_lax, _ = compute_ml_threshold(t, regime, use_lax=True)

                log.info(f"[DEBUG] {t} | ML={ml:.4f} T_buy={T_buy_strict:.3f}/T_lax={T_buy_lax:.3f}/T_sell={T_sell:.3f} "
                         f"MACD={macd:.4f}/{signal:.4f} RSI={rsi:.1f} ADX={adx:.1f} ATR={atr_abs:.6f} PX={px:.2f}")

                # -------- Patch #2: 실시간 -5% 즉시 손절 --------
                if t in entry_prices:
                    rt_px = pyupbit.get_current_price(t)
                    if rt_px is None: rt_px = px
                    avg_entry_rt = entry_prices[t]
                    change_rt = (rt_px - avg_entry_rt) / max(1e-12, avg_entry_rt)
                    if change_rt <= -0.05:
                        log.info(f"[{t}] 🚨 실시간 -5% 손절(rt) | entry={avg_entry_rt:.4f}, px={rt_px:.4f}")
                        coin = t.split('-')[1]
                        bal = get_balance(coin)
                        if bal and bal > 0:
                            _ = sell_crypto_currency(t, bal)
                            with state_lock:
                                entry_prices.pop(t, None)
                                highest_prices.pop(t, None)
                                recent_trades[t] = now
                                pos_plan.pop(t, None)
                            record_trade_pnl(change_rt)
                        continue  # 이번 티커 처리 종료

                # ===== 매수 로직 =====
                ATR_TH = 0.015 * px
                conds = [
                    ("MACD", macd > signal, f"{macd:.4f}>{signal:.4f}"),
                    ("RSI", rsi < (58 if regime=='bull' else 55), f"{rsi:.1f}<{'58' if regime=='bull' else '55'}"),
                    ("ADX", adx > (18 if regime=='bull' else 20), f"{adx:.1f}>{'18' if regime=='bull' else '20'}"),
                    ("ATR", atr_abs > ATR_TH, f"{atr_abs:.6f}>{ATR_TH:.6f}")
                ]

                allow_bear_buy = False
                if regime == "bear" and BEAR_ALLOW_BUYS and not risk_block and remaining_slots > 0:
                    try:
                        if (rsi < 35 and macd > signal and ml > max(T_buy_strict, 0.55)):
                            allow_bear_buy = True
                        else:
                            atr_prev = feats['atr'].iloc[-5] if len(feats) > 5 else atr_abs
                            if atr_abs > 1.2 * atr_prev:
                                allow_bear_buy = True
                            else:
                                btc = safe_get_ohlcv("KRW-BTC", interval="minute5", count=200)
                                if is_valid_df(btc, 50):
                                    btc_ret = btc['close'].pct_change().iloc[-1]
                                    alt_ret = feats['close'].pct_change().iloc[-1]
                                    if (alt_ret - btc_ret) > 0.01:
                                        allow_bear_buy = True
                    except Exception:
                        allow_bear_buy = False

                tech_ok = all(ok for _, ok, _ in conds)
                strict_gate = (ml > T_buy_strict)
                lax_gate = (ml > T_buy_lax)
                pass_model_filter = (t in models)

                can_new_buy = (not risk_block) and (remaining_slots > 0) and pass_model_filter and tech_ok and \
                              (strict_gate or lax_gate or allow_bear_buy)

                if t not in entry_prices and can_new_buy:
                    if remaining_krw > MIN_ORDER_KRW:
                        equity = calc_total_equity()
                        high_conf = (ml > 0.70 and adx > 25)
                        target = calc_atr_position_budget(
                            remaining_krw, remaining_slots, atr_abs, px, equity,
                            base_risk=0.006, high_conf=high_conf
                        )
                        first_amt = min(target * 0.6, remaining_krw * USE_CASH_RATIO_EFF)
                        if first_amt >= MIN_ORDER_KRW:
                            if buy_crypto_currency(t, first_amt):
                                # -------- Patch #3: 평단 = 업비트 avg_buy_price --------
                                avg_px = get_avg_buy_price_from_balances(t)
                                with state_lock:
                                    entry_prices[t] = (avg_px if avg_px else px)
                                    highest_prices[t] = entry_prices[t]
                                    recent_trades[t] = now
                                remaining_krw -= first_amt
                                remaining_slots -= 1
                                pos_plan[t] = {"target": target, "filled": first_amt, "tr":[0.2,0.2], "last": now}
                                log.info(f"[{t}] 1차 매수: {first_amt:.0f}원 / target≈{target:.0f} "
                                         f"| 남은KRW≈{remaining_krw:.0f}, 남은슬롯={remaining_slots}")
                    else:
                        log.info(f"[{t}] 매수 불가(KRW<{MIN_ORDER_KRW})")
                elif t not in entry_prices:
                    reasons = ", ".join([name for name, ok, _ in conds if not ok])
                    if not reasons:
                        reasons = "문턱미달" if not (strict_gate or lax_gate or allow_bear_buy) else "리스크/슬롯"
                    log.info(f"[{t}] 신규 매수 스킵 → {reasons}")

                # ===== 매도/추가매수 로직 =====
                if t in entry_prices:
                    # 실시간 TP/SL 강제청산 결정을 위해 rt_px 재활용
                    rt_px = pyupbit.get_current_price(t)
                    if rt_px is None: rt_px = px

                    entry = entry_prices[t]
                    highest_prices[t] = max(highest_prices.get(t, entry), rt_px)
                    chg = (px - entry)/entry
                    change_rt = (rt_px - entry) / max(1e-12, entry)

                    _, T_sell_eff = compute_ml_threshold(t, regime, use_lax=False)
                    will_sell = should_sell(t, px, ml, T_sell_eff, regime)

                    # -------- Patch #4: 강제청산도 실시간 변화율 기준 --------
                    tp_dyn = get_dynamic_tp_by_regime(regime)
                    force_liq = (change_rt <= -0.05) or (change_rt >= tp_dyn)

                    if will_sell or force_liq:
                        try:
                            coin = t.split('-')[1]
                            bal = get_balance(coin)
                        except Exception:
                            bal = 0.0
                        if bal and bal > 0:
                            # 부분익절 우선
                            if try_partial_take_profit(t, chg, bal, now):
                                bal = get_balance(coin)
                            reason = []
                            if change_rt <= -0.05: reason.append("강제손절(rt)")
                            if change_rt >= tp_dyn: reason.append("강제익절(rt)")
                            if will_sell and not reason: reason.append("전략매도")
                            log.info(f"[{t}] 매도 실행: {', '.join(reason)} | 실시간수익률 {change_rt*100:.2f}%")
                            sold = False
                            for k in range(2):
                                try:
                                    o = sell_crypto_currency(t, bal)
                                    if o: sold = True; break
                                    time.sleep(1.0)
                                except Exception as e:
                                    log.info(f"[{t}] 매도 오류 재시도: {e}")
                                    time.sleep(1.0)
                            if sold:
                                time.sleep(0.7)
                                remain = get_balance(coin)
                                if remain is None or remain < 1e-8:
                                    with state_lock:
                                        entry_prices.pop(t, None)
                                        highest_prices.pop(t, None)
                                        recent_trades[t] = now
                                        pos_plan.pop(t, None)
                                    record_trade_pnl(change_rt)
                                    log.info(f"[{t}] ✅ 매도 완료 및 상태 정리")
                                else:
                                    log.info(f"[{t}] ⚠️ 잔여 수량 감지({remain}) 다음 루프에서 처리")
                        else:
                            log.info(f"[{t}] 매도 불가: 보유=0")
                    else:
                        # 스케일-인(추가매수)
                        plan = pos_plan.get(t)
                        if plan and plan["tr"]:
                            tranche = plan["tr"][0]
                            add_amt = plan["target"] * tranche
                            buy_amt = max(MIN_ORDER_KRW, min(add_amt, remaining_krw * USE_CASH_RATIO_EFF))
                            if ml > T_buy_strict and buy_amt >= MIN_ORDER_KRW:
                                if buy_crypto_currency(t, buy_amt):
                                    with state_lock:
                                        # 평단 재동기화
                                        avg_px = get_avg_buy_price_from_balances(t)
                                        if avg_px: entry_prices[t] = avg_px
                                        highest_prices[t] = max(highest_prices.get(t, rt_px), rt_px)
                                        recent_trades[t] = now
                                    plan["filled"] += buy_amt
                                    plan["tr"].pop(0)
                                    plan["last"] = now
                                    remaining_krw -= buy_amt
                                    log.info(f"[{t}] 추가 매수: {buy_amt:.0f}원 (잔여 트랜치 {len(plan['tr'])}) "
                                             f"| 남은KRW≈{remaining_krw:.0f}")

            time.sleep(1.0)

    except KeyboardInterrupt:
        print("프로그램이 종료되었습니다.")

# =============== 진입점 ===============
if __name__ == "__main__":
    main()