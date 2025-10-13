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

# ===================== 실행/환경 기본 =====================
DRY_RUN = False
BEAR_ALLOW_BUYS = True                   # 하락장 예외 진입 허용
torch.set_num_threads(1)                 # t3.medium CPU 스레드 고정
torch.set_num_interop_threads(1)

# Upbit 키 (환경변수로 주입 권장)
ACCESS_KEY = os.getenv("UPBIT_ACCESS_KEY", "")
SECRET_KEY = os.getenv("UPBIT_SECRET_KEY", "")

# ===================== 전략 파라미터 =====================
# 학습/신호 스케줄
last_trained_time = {}
TRAINING_INTERVAL = timedelta(hours=6)

# 문턱/리스크
ML_BASE_THRESHOLD = 0.50                 # 기본 진입 문턱 (적응형에서 보정됨)
STOP_LOSS_THRESHOLD = -0.052             # 백테스트 기본 하드스톱(버퍼 0.2% 포함)
COOLDOWN_TIME = timedelta(minutes=30)
SURGE_COOLDOWN_TIME = timedelta(minutes=60)

# (신규) 레짐 기반 하드스톱 사용
USE_REGIME_STOP = True
REGIME_STOP_MAP = {
    "bull":    -0.06,   # 상승장: 더 여유
    "neutral": -0.052,  # 중립: 기본(버퍼 포함)
    "bear":    -0.045,  # 하락장: 더 타이트
}

# 포지션/현금
MAX_ACTIVE_POSITIONS_BASE = 2           # 기본 보유 슬롯 2개
FOCUS_SLOT_ENABLE = True                 # 고신뢰 1개 추가(최대 4개)
USE_CASH_RATIO_BASE = 1.00
MIN_ORDER_KRW = 6000

# 동적 상위 N(유동 후보군)
TOP_POOL_MULTIPLIER = 12
TOP_POOL_BASE = 4

# 하이브리드 문턱 완화(유연형)
TBUY_LAX_FACTOR = 0.95                   # 완화 진입: T_buy * 0.95
TSELL_GAP = 0.04                         # 히스테리시스: T_sell = T_buy - 0.04 (적응형)

# 자본/리저브/드로우다운
DAILY_MAX_LOSS = 0.02
MAX_CONSECUTIVE_LOSSES = 3
PROFIT_SKIM_TRIGGER = 0.03
PROFIT_SKIM_RATIO = 0.25
RESERVE_RELEASE_DD = 0.02
POS_RISK_CAP = 0.50                      # 포지션당 최대 자본 비중 50%

# 부분 익절/트레일링
PARTIAL_TP1, TP1_RATIO = 0.12, 0.20
PARTIAL_TP2, TP2_RATIO = 0.25, 0.10
TRAIL_DROP_BULL, TRAIL_DROP_BEAR = 0.04, 0.025

# K-of-N 동적 게이트 (기술조건 중 K개 이상 통과)
# 기술조건 후보: MACD Cross, ADX Strong, RSI OK, ATR OK  => 총 N=4
KOFN_CONFIG = {
    "bull":    {"K": 2, "N": 4},
    "neutral": {"K": 3, "N": 4},
    "bear":    {"K": 3, "N": 4},
}

# 1분 추론 적용 범위
INFER_1M_TOPK = 10   # 상위 10개 티커 + 실보유 티커는 1분봉으로 추론 시도

# ---------- (신규) Priority Buy 설정 ----------
PRIORITY_BUY_ENABLE = True
PRIORITY_BUY_SLOTS  = 1                  # 리스크블록 중에도 허용할 우선 슬롯 수
PRIORITY_BUY_ML_BONUS = 0.05             # 우선 진입은 T_buy보다 이만큼 더 요구
HIGHCONF_RISK_MULT = 1.6                 # 고신뢰일 때 리스크 단위 배수
PRIORITY_RISK_MULT = 2.0                 # Priority일 때 리스크 단위 배수(상한 캡은 POS_RISK_CAP)

# ===================== 상태 =====================
entry_prices = {}
entry_times  = {}        # << 진입 시간 분리 (부분익절로 리셋되지 않음)
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

# ===================== 로깅 =====================
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
entry_times  = load_pickle("entry_times.pkl", {})   # << 추가 복원
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
                atomic_save(entry_times,  "entry_times.pkl")   # << 추가 저장
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

# ===================== 안전 OHLCV & 검증 =====================
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

# ===================== 지표/피처 =====================
def get_macd_from_df(df):
    df = df.copy()
    df['short_ema'] = df['close'].ewm(span=12, adjust=False).mean()
    df['long_ema']  = df['close'].ewm(span=26, adjust=False).mean()
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
    df['L-C'] = (df['low']  - df['close'].shift(1)).abs()
    df['TR']  = df[['H-L','H-C','L-C']].max(axis=1)
    df['+DM'] = (df['high'] - df['high'].shift(1)).clip(lower=0)
    df['-DM'] = (df['low'].shift(1) - df['low']).clip(lower=0)
    df['+DM'] = df['+DM'].where(df['+DM'] > df['-DM'], 0)
    df['-DM'] = df['-DM'].where(df['-DM'] > df['+DM'], 0)
    df['TR_smooth'] = df['TR'].rolling(window=period).sum()
    df['+DM_smooth'] = df['+DM'].rolling(window=period).sum()
    df['-DM_smooth'] = df['-DM'].rolling(window=period).sum()
    df['+DI'] = 100 * (df['+DM_smooth'] / df['TR_smooth'])
    df['-DI'] = 100 * (df['-DM_smooth'] / df['TR_smooth'])
    denom = (df['+DI'] + df['-DI']).replace(0, np.nan)
    df['DX'] = 100 * (df['+DI'] - df['-DI']).abs() / denom
    df['adx'] = df['DX'].rolling(window=period).mean()
    return df

def get_atr_from_df(df, period=14):
    df = df.copy()
    df['H-L'] = df['high'] - df['low']
    df['H-C'] = (df['high'] - df['close'].shift(1)).abs()
    df['L-C'] = (df['low']  - df['close'].shift(1)).abs()
    df['TR']  = df[['H-L','H-C','L-C']].max(axis=1)
    df['atr'] = df['TR'].rolling(window=period).mean()
    return df

def get_features(ticker, interval="minute5", normalize=True, count=1000):
    df = safe_get_ohlcv(ticker, interval=interval, count=count)
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

# ===================== 자본/리스크/유틸 =====================
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
    return max(300_000, min(10_000_000, int(eq) if eq > 0 else 1_000_000))

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
                entry_times.pop(t, None)
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

# ===================== 레짐/브레드스 =====================
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
    # 초기 breadth=0 편향 보정: BTC/ETH 둘 다 bull인데 breadth=0이면 neutral
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

# ===================== 하이브리드 모델 (CNN + LSTM + Transformer) =====================
class CNNBlock(nn.Module):
    def __init__(self, in_ch=6, channels=16, k=3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, channels, kernel_size=k, padding=k//2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=k, padding=k//2)
        self.pool  = nn.AdaptiveAvgPool1d(1)
    def forward(self, x):               # x: (B, T, F)
        x = x.transpose(1, 2)           # (B, F, T)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)    # (B, C)
        return x

class LSTMBlock(nn.Module):
    def __init__(self, in_dim=6, hid=32, layers=1):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hid, num_layers=layers, batch_first=True)
    def forward(self, x):               # (B, T, F)
        o, _ = self.lstm(x)
        return o[:, -1, :]              # (B, hid)

class TransBlock(nn.Module):
    def __init__(self, in_dim=6, d_model=32, heads=4, layers=1):
        super().__init__()
        self.emb = nn.Linear(in_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=heads, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=layers)
    def forward(self, x):               # (B, T, F)
        x = self.emb(x)
        x = self.enc(x)
        return x[:, -1, :]              # (B, d_model)

class HybridModel(nn.Module):
    def __init__(self, in_dim=6, seq_len=30):
        super().__init__()
        self.cnn  = CNNBlock(in_ch=in_dim, channels=16, k=3)
        self.lstm = LSTMBlock(in_dim=in_dim, hid=32, layers=1)
        self.tran = TransBlock(in_dim=in_dim, d_model=32, heads=4, layers=1)
        mix_dim = 16 + 32 + 32
        self.head = nn.Sequential(
            nn.Linear(mix_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, x):               # (B, T, F)
        c = self.cnn(x)
        l = self.lstm(x)
        t = self.tran(x)
        h = torch.cat([c, l, t], dim=1)
        out = self.head(h)
        return out                       # (B, 1) in [0,1]

# ===================== 데이터셋/학습 =====================
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
    data = get_features(ticker, interval="minute5", normalize=True, count=1000)
    if data is None or data.empty:
        log.info(f"경고: {ticker} 데이터 비어 있음. 학습 스킵"); return None
    ds = TradingDataset(data, seq_len=30)
    if len(ds) == 0:
        log.info(f"경고: {ticker} 데이터셋 너무 작음. 학습 스킵"); return None
    dl = DataLoader(ds, batch_size=32, shuffle=True, num_workers=0)
    model = HybridModel(in_dim=6, seq_len=30)
    crit = nn.MSELoss()
    opt  = torch.optim.Adam(model.parameters(), lr=0.001)
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

# ===================== 문턱/레짐별 하드스톱 =====================
def compute_ml_threshold(ticker, regime, use_lax=False):
    # 최근 히스토리 기반 적응형
    base = ML_BASE_THRESHOLD
    hist = ml_hist[ticker]
    if len(hist) >= 60:
        q = float(np.quantile(hist, 0.75))
        base = max(0.40, min(0.65, q))
    # 레짐 보정
    if regime == "bull":
        base -= 0.03
    elif regime == "bear":
        base += 0.04
    base = max(0.38, min(0.70, base))
    t_buy = base
    if use_lax:
        t_buy = max(0.36, t_buy * TBUY_LAX_FACTOR)  # 완화
    t_sell = max(0.0, t_buy - TSELL_GAP)
    return t_buy, t_sell

def get_dynamic_tp_by_regime(regime: str) -> float:
    if regime == "bull":
        return 0.35
    elif regime == "bear":
        return 0.15
    return 0.25

def get_dynamic_stop_by_regime(regime: str) -> float:
    if USE_REGIME_STOP:
        return REGIME_STOP_MAP.get(regime, STOP_LOSS_THRESHOLD)
    return STOP_LOSS_THRESHOLD

# ===================== 급등/스프레드/랭킹 =====================
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

# ===================== 동적 상위 N =====================
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

# ===================== ATR 기반 포지션 사이징 =====================
def calc_atr_position_budget(
    remaining_krw, remaining_slots, atr_abs, px, equity,
    use_cash_ratio, base_risk=0.010, high_conf=False, priority=False, ml=None
):
    """
    - ATR 추정 스톱폭으로 리스크 단위 기반 예산 산출
    - high_conf/priority일 때 리스크 단위↑ (단, POS_RISK_CAP 상한 준수)
    """
    if atr_abs is None or atr_abs <= 0 or px <= 0:
        return (remaining_krw * use_cash_ratio) / max(1, remaining_slots)

    k = 1.5
    est_stop_ratio = (atr_abs * k) / px
    if est_stop_ratio <= 0:
        return (remaining_krw * use_cash_ratio) / max(1, remaining_slots)

    risk_unit = base_risk
    if high_conf:
        risk_unit *= HIGHCONF_RISK_MULT
    if priority:
        risk_unit *= PRIORITY_RISK_MULT

    # ML 강도에 따른 미세 가중 (0.60 기준으로 위쪽만 +최대 +60%)
    if ml is not None:
        boost = max(0.0, min(0.6, (ml - 0.60) * 0.8))
        risk_unit *= (1.0 + boost)

    budget_by_risk = (equity * risk_unit) / est_stop_ratio
    budget_hardcap = (equity * POS_RISK_CAP)  # 절대 상한
    equal_split = (remaining_krw * use_cash_ratio) / max(1, remaining_slots)
    equal_cap_factor = 2.0 if priority else 1.7

    target = max(MIN_ORDER_KRW, min(budget_by_risk, equal_cap_factor*equal_split, budget_hardcap))
    return target

# ===================== ATR 적응형 임계값 =====================
def atr_adaptive_gate(px, feats, regime):
    """
    ATR OK 판정:
      - 기준: px * base_pct * regime_adj 와 최근 ATR 중앙값(q50) 중 '작은 쪽'을 문턱으로 사용
      - 변동성 높을수록(상위 분위수) 약간 완화
    """
    atr_now = float(feats['atr'].iloc[-1])
    atr_hist = feats['atr'].tail(60).dropna()
    q50 = float(np.median(atr_hist)) if len(atr_hist) > 0 else atr_now
    base_pct = 0.012  # 1.2% 기본
    regime_adj = 1.0 if regime=="bull" else (1.1 if regime=="neutral" else 1.2)
    th_pct = base_pct * regime_adj
    th_abs = min(q50, px * th_pct)
    # 상위 분위수일 때 완화
    if len(atr_hist) >= 20:
        q75 = float(np.quantile(atr_hist, 0.75))
        if atr_now >= q75:
            th_abs *= 0.9
    passed = atr_now > th_abs
    return passed, atr_now, th_abs

# ===================== 부분 익절/트레일링/전체 매도 =====================
def try_partial_take_profit(ticker, change_ratio, coin_balance, now):
    """
    개선된 부분익절 로직 + ML 히스토리 + ATR 기반 조정
    """
    did = False
    try:
        ml_last = ml_hist[ticker][-1] if ticker in ml_hist and len(ml_hist[ticker]) > 0 else 0.5
        entry_time = entry_times.get(ticker, None)   # << 진입 시각은 entry_times에서
        time_since_entry_min = (now - entry_time).total_seconds()/60.0 if entry_time else 9999.0

        feats = None
        try:
            feats = get_features(ticker, interval="minute5", normalize=False, count=200)
        except Exception:
            feats = None
        atr_now = float(feats['atr'].iloc[-1]) if is_valid_df(feats, min_len=20) else None

        # 1) 급등 스파이크 감지
        if time_since_entry_min < 15 and change_ratio >= 0.12:
            log.info(f"[{ticker}] 부분익절 보류: 진입후 {time_since_entry_min:.1f}분, 급등 {change_ratio*100:.1f}%")
            return False

        # 2) ML 신호 강하면 부분익절 건너뜀 (수익 더 태움)
        if ml_last >= 0.65:
            log.info(f"[{ticker}] 부분익절 건너뜀: ML 강함({ml_last:.2f})")
            return False

        # 3) ATR 변동성 조정
        mult = 1.0
        if atr_now is not None and atr_now > 0:
            if is_valid_df(feats, 60) and atr_now > 1.5 * np.median(feats['atr'].tail(60)):
                mult = 0.6

        # TP2 우선 적용
        px_now = pyupbit.get_current_price(ticker)
        if px_now is None: return False
        if change_ratio >= PARTIAL_TP2 and coin_balance > 0:
            amt = coin_balance * (TP2_RATIO * mult)
            if amt * px_now >= MIN_ORDER_KRW:
                if sell_crypto_currency(ticker, amt):
                    did = True
                    log.info(f"[{ticker}] 부분익절2 실행: {TP2_RATIO*100*mult:.0f}% 매도 (ML={ml_last:.2f})")
        elif change_ratio >= PARTIAL_TP1 and coin_balance > 0:
            amt = coin_balance * (TP1_RATIO * mult)
            if amt * px_now >= MIN_ORDER_KRW:
                if sell_crypto_currency(ticker, amt):
                    did = True
                    log.info(f"[{ticker}] 부분익절1 실행: {TP1_RATIO*100*mult:.0f}% 매도 (ML={ml_last:.2f})")

        if did:
            with state_lock:
                recent_trades[ticker] = now   # 체결 타임스탬프만 갱신(진입시각은 그대로)

    except Exception as e:
        log.info(f"[{ticker}] 부분익절 예외: {e}")
        return False
    return did

def try_trailing_stop(ticker, coin_balance, now, regime="neutral"):
    """
    트레일링 스탑 로직
    """
    try:
        px_now = pyupbit.get_current_price(ticker)
        if px_now is None: return False
        with state_lock:
            hi = highest_prices.get(ticker, px_now)
            highest_prices[ticker] = max(hi, px_now)
            hi = highest_prices[ticker]

        drop_pct = (hi - px_now) / hi
        drop_threshold = TRAIL_DROP_BULL if regime=="bull" else TRAIL_DROP_BEAR

        if drop_pct >= drop_threshold and coin_balance > 0:
            if sell_crypto_currency(ticker, coin_balance):
                log.info(f"[{ticker}] 트레일링스탑 발동: {drop_pct*100:.2f}% 하락")
                with state_lock:
                    entry_prices.pop(ticker, None)
                    highest_prices.pop(ticker, None)
                    recent_trades.pop(ticker, None)
                    entry_times.pop(ticker, None)
                return True
    except Exception as e:
        log.info(f"[{ticker}] 트레일링스탑 예외: {e}")
    return False

def try_hard_stop(ticker, coin_balance, now, regime="neutral"):
    """
    레짐/하드스톱 기반 매도
    """
    try:
        px_now = pyupbit.get_current_price(ticker)
        if px_now is None or ticker not in entry_prices: return False
        entry_px = entry_prices[ticker]
        stop_loss = get_dynamic_stop_by_regime(regime)
        pnl = (px_now - entry_px)/entry_px
        if pnl <= stop_loss and coin_balance > 0:
            if sell_crypto_currency(ticker, coin_balance):
                log.info(f"[{ticker}] 하드스톱 매도: PnL={pnl*100:.2f}%, Stop={stop_loss*100:.2f}%")
                with state_lock:
                    entry_prices.pop(ticker, None)
                    highest_prices.pop(ticker, None)
                    recent_trades.pop(ticker, None)
                    entry_times.pop(ticker, None)
                return True
    except Exception as e:
        log.info(f"[{ticker}] 하드스톱 예외: {e}")
    return False

def manage_position(ticker, now, regime="neutral"):
    """
    혼합전략 포지션 관리: 부분익절 → 트레일링 → 하드스톱
    """
    coin = ticker.split('-')[1]
    coin_balance = get_balance(coin)
    if coin_balance <= 0: return

    px_now = pyupbit.get_current_price(ticker)
    if px_now is None: return

    entry_px = entry_prices.get(ticker, px_now)
    change_ratio = (px_now - entry_px) / entry_px

    # 1) 부분익절 시도
    try_partial_take_profit(ticker, change_ratio, coin_balance, now)

    # 2) 트레일링 스탑 시도(전량 청산 가능)
    if try_trailing_stop(ticker, coin_balance, now, regime=regime):
        return

    # 3) 하드스톱 적용(전량 청산 가능)
    try_hard_stop(ticker, coin_balance, now, regime=regime)

def should_sell(ticker, current_price, ml_signal, t_sell, regime):
    if ticker not in entry_prices: return False
    entry = entry_prices[ticker]
    highest_prices[ticker] = max(highest_prices.get(ticker, entry), current_price)
    chg = (current_price - entry) / entry
    peak_drop = (highest_prices[ticker] - current_price) / highest_prices[ticker]
    weak_ml = (ml_signal < t_sell)

    # === 레짐 기반 하드 스톱 (버퍼 포함 / 즉시) ===
    stop_thr = get_dynamic_stop_by_regime(regime)  # bull:-6%, neutral:-5.2%, bear:-4.5%
    if chg <= stop_thr:
        log.info(f"[{ticker}] 🚨 하드 스톱 발동({stop_thr*100:.1f}%) → 즉시 손절")
        return True

    # === 레짐 기반 강제 익절 ===
    tp_dyn = get_dynamic_tp_by_regime(regime)
    if chg >= tp_dyn:
        log.info(f"[{ticker}] 🎯 레짐 기반 강제 익절(+{tp_dyn*100:.0f}%) 도달 → 익절")
        return True

    # === 이익권에서의 관리 ===
    if chg >= 0.15:
        if weak_ml or ml_signal < 0.6:
            log.info(f"[{ticker}] ✅ +15% & ML 약함 → 익절"); return True
        else:
            return False
    if chg >= 0.10:
        if weak_ml or ml_signal < 0.5:
            log.info(f"[{ticker}] ✅ +10% & ML 약함 → 익절"); return True

    # === 트레일링 스탑 ===
    trail = TRAIL_DROP_BULL if regime=="bull" else TRAIL_DROP_BEAR
    if peak_drop > trail and (weak_ml or ml_signal < 0.5):
        log.info(f"[{ticker}] 📉 트레일링 스탑! 고점대비 {peak_drop*100:.2f}%"); return True

    # === 간단 보조 확인 ===
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

# ===================== 백테스트 & 워크포워드 =====================
def backtest_series(data, model, init_bal, t_buy, t_sell, fee=0.0005, slip_bp=10, stop_thr=STOP_LOSS_THRESHOLD):
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
            if unrl <= stop_thr:
                bal = pos * px * (1 - fee); pos = 0
            elif peak_drop > 0.02 and s < t_sell:
                bal = pos * px * (1 - fee); pos = 0
    final = bal + pos * data.iloc[-1]['close']
    return final / init_bal

def run_bt_wf(ticker, model, init_bal, regime):
    # 적응형 문턱(보수)
    t_buy, t_sell = compute_ml_threshold(ticker, regime, use_lax=False)
    stop_thr = get_dynamic_stop_by_regime(regime)  # 실거래와 동일한 하드스톱
    # 5분 전체 백테스트
    data_bt = get_features(ticker, interval="minute5", normalize=False, count=1000)
    if data_bt is None or data_bt.empty:
        perf_bt = 1.0
    else:
        perf_bt = backtest_series(data_bt, model, init_bal, t_buy, t_sell, stop_thr=stop_thr)
    # 5분 워크포워드(최근 20%)
    if data_bt is None or data_bt.empty or len(data_bt) < 200:
        perf_wf = 1.0
    else:
        cut = int(len(data_bt)*0.8)
        wf = data_bt.iloc[cut:].copy()
        perf_wf = backtest_series(wf, model, init_bal, t_buy, t_sell, stop_thr=stop_thr)
    return perf_bt, perf_wf

def backtest(ticker, model, initial_balance=1_000_000, fee=0.0005, slip_bp=10):
    data = get_features(ticker, interval="minute5", normalize=False, count=1000)
    if data is None or data.empty:
        return 1.0
    # 보수 문턱 + 기본 하드스톱(버퍼)로 간소 백테스트
    t_buy, t_sell = 0.55, 0.50
    return backtest_series(data, model, initial_balance, t_buy, t_sell, fee, slip_bp, stop_thr=STOP_LOSS_THRESHOLD)

# ===================== 메인 루프 =====================
def main():
    print("자동매매 시작!")
    models = {}
    # 자동 저장 스레드
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
        log.info("[FILTER] 엄격 기준 통과 없음 → 완화 기준으로 재평가")
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
            priority_set = set(ranked_hint[:5])  # 상위 5개를 우선 후보로 사용

            log.info(f"[BUY-HINT] ranked(top5)={ranked_hint[:5]} slots={slots_available} regime={regime} risk_block={risk_block}")

            # 자금
            krw_now = get_balance("KRW") or 0.0
            usable_krw = max(0.0, krw_now - reserved_profit)
            remaining_krw = usable_krw
            remaining_slots = slots_available
            log.info(f"[RESERVE] KRW={krw_now:.0f}, reserve={reserved_profit:.0f}, usable={usable_krw:.0f}, HWM={equity_hwm:.0f}")

            # Priority 슬롯 사용 현황
            priority_used = sum(1 for p in pos_plan.values() if p.get("priority", False))
            priority_slots_avail = max(0, PRIORITY_BUY_SLOTS - priority_used)

            # 5분+1분 하이브리드 추론 대상 선정
            top_for_1m = set(top_tickers[:INFER_1M_TOPK]) | held

            # 매수/매도 루프(보유 + 후보)
            targets = set(held) | set(universe_new)
            for t in list(targets):
                # ===== 쿨다운: '신규 진입'만 제한, 보유 포지션은 응급 처리 위해 무시 =====
                cooldown = SURGE_COOLDOWN_TIME if t in recent_surge else COOLDOWN_TIME
                in_cooldown = (now - recent_trades.get(t, datetime.min) < cooldown)
                if in_cooldown and (t not in entry_prices):
                    continue

                if t not in models:
                    log.info(f"[{t}] 모델 없음 → 스킵"); continue

                # -------- 1) 피처/지표 최신 (1m 선호, 실패시 5m 폴백) --------
                feats_1m = get_features(t, interval="minute1", normalize=False, count=600) if t in top_for_1m else pd.DataFrame()
                feats_5m = get_features(t, interval="minute5", normalize=False, count=600)

                use_1m = is_valid_df(feats_1m, 60)
                feats = feats_1m if use_1m else feats_5m
                if feats is None or feats.empty or len(feats) < 60:
                    continue

                macd = feats['macd'].iloc[-1]; signal = feats['signal'].iloc[-1]
                rsi  = feats['rsi'].iloc[-1];  adx    = feats['adx'].iloc[-1]
                atr_abs = feats['atr'].iloc[-1]; px = feats['close'].iloc[-1]

                # -------- 2) 모델 신호 --------
                X = torch.tensor(feats[['macd','signal','rsi','adx','atr','return']].tail(30).values,
                                 dtype=torch.float32).unsqueeze(0)
                model = models[t]
                model.eval()
                with torch.no_grad():
                    ml = model(X).item()
                ml_hist[t].append(ml)

                # 문턱(보수/완화) 계산
                T_buy_strict, T_sell = compute_ml_threshold(t, regime, use_lax=False)
                T_buy_lax, _         = compute_ml_threshold(t, regime, use_lax=True)

                # -------- 3) ATR 적응형 게이트 --------
                atr_ok, atr_now, atr_th = atr_adaptive_gate(px, feats, regime)

                # -------- 4) 기술조건 + K-of-N --------
                macd_cross  = (macd > signal)
                adx_strong  = (adx  > (18 if regime=='bull' else 20))
                rsi_ok      = (rsi  < (58 if regime=='bull' else 55))  # 과매수 회피

                conds_map = {
                    "MACD": macd_cross,
                    "ADX":  adx_strong,
                    "RSI":  rsi_ok,
                    "ATR":  atr_ok
                }
                K_req = KOFN_CONFIG[regime]["K"]; N_all = KOFN_CONFIG[regime]["N"]
                tech_pass = sum(1 for ok in conds_map.values() if ok)
                kpass = (tech_pass >= K_req)

                # 상세 디버그 로그
                log.info(
                    (f"[DEBUG] {t} | TF={'1m' if use_1m else '5m'} | "
                     f"ML={ml:.4f} T_buy={T_buy_strict:.3f}/T_lax={T_buy_lax:.3f}/T_sell={T_sell:.3f} | "
                     f"MACD={macd:.4f}/{signal:.4f} ({'X' if macd_cross else '-'}) "
                     f"RSI={rsi:.1f} ({'OK' if rsi_ok else 'NO'}) "
                     f"ADX={adx:.1f} ({'STRONG' if adx_strong else 'WEAK'}) "
                     f"ATR={atr_now:.6f}>(thr {atr_th:.6f}) ({'OK' if atr_ok else 'NO'}) | "
                     f"K-of-N: {tech_pass}/{N_all} (need {K_req})")
                )

                # -------- 5) 하락장 예외(옵션) --------
                allow_bear_buy = False
                if regime == "bear" and BEAR_ALLOW_BUYS and remaining_slots > 0:
                    try:
                        if (rsi < 35 and macd_cross and ml > max(T_buy_strict, 0.55)):
                            allow_bear_buy = True
                        else:
                            atr_hist = feats['atr'].tail(60).dropna()
                            if len(atr_hist) >= 20 and atr_now > 1.2 * float(np.median(atr_hist)):
                                allow_bear_buy = True
                            else:
                                btc = safe_get_ohlcv("KRW-BTC", interval="minute5", count=200)
                                if is_valid_df(btc, 50):
                                    btc_ret = btc['close'].pct_change().iloc[-1]
                                    alt_ret = feats['close'].pct_change().iloc[-1]
                                    if (alt_ret - btc_ret) > 0.01 and (macd_cross or adx_strong):
                                        allow_bear_buy = True
                    except Exception:
                        allow_bear_buy = False

                # -------- 6) Priority Buy 판정(리스크블록 우회) --------
                priority_ml   = (ml >= max(T_buy_strict + PRIORITY_BUY_ML_BONUS, 0.72))
                priority_tech = (kpass and adx_strong and atr_ok and macd_cross)
                priority_rank = (t in priority_set)
                allow_priority_buy = PRIORITY_BUY_ENABLE and priority_slots_avail > 0 and (priority_ml and priority_tech and (priority_rank or regime=="bull"))

                # -------- 7) 최종 진입 게이트 --------
                strict_gate = (ml > T_buy_strict)
                lax_gate    = (ml > T_buy_lax)
                pass_model  = (t in models)  # BT+WF 통과 모델만

                can_new_buy_normal = (not risk_block) and (remaining_slots > 0) and pass_model and ((kpass and (strict_gate or lax_gate)) or allow_bear_buy)
                can_new_buy_priority = (risk_block) and allow_priority_buy and (remaining_slots > 0) and pass_model

                can_new_buy = can_new_buy_normal or can_new_buy_priority

                if t not in entry_prices and can_new_buy:
                    if remaining_krw > MIN_ORDER_KRW:
                        equity = calc_total_equity()
                        high_conf = (ml > 0.70 and adx > 25 and macd_cross)
                        is_priority = can_new_buy_priority
                        target = calc_atr_position_budget(
                            remaining_krw, remaining_slots, atr_abs, px, equity,
                            use_cash_ratio=USE_CASH_RATIO_EFF,
                            base_risk=0.010,
                            high_conf=(high_conf or is_priority),
                            priority=is_priority,
                            ml=ml
                        )
                        # 러너는 크게: priority/high_conf면 초기 90%, 아니면 60%
                        first_frac = 0.9 if (high_conf or is_priority) else 0.6
                        first_amt = min(target * first_frac, remaining_krw * USE_CASH_RATIO_EFF)
                        if first_amt >= MIN_ORDER_KRW:
                            if buy_crypto_currency(t, first_amt):
                                with state_lock:
                                    entry_prices[t] = px
                                    highest_prices[t] = px
                                    recent_trades[t] = now
                                    if t not in entry_times:   # 진입 시각 최초 기록
                                        entry_times[t] = now
                                remaining_krw -= first_amt
                                remaining_slots -= 1
                                # priority면 트랜치 한 번만(10%) 남겨서 빠르게 채움
                                tr_list = [0.1] if (high_conf or is_priority) else [0.2, 0.2]
                                pos_plan[t] = {"target": target, "filled": first_amt, "tr": tr_list, "last": now, "priority": is_priority}
                                if is_priority:
                                    priority_slots_avail -= 1
                                log.info(
                                    f"[BUY] {t} | amt={first_amt:.0f}원 (target≈{target:.0f}) "
                                    f"| K-of-N {tech_pass}/{N_all} (need {K_req}), "
                                    f"ML={'strict' if strict_gate else ('lax' if lax_gate else 'no')} "
                                    f"| bear_ex={'Y' if allow_bear_buy else 'N'} "
                                    f"| priority={'Y' if is_priority else 'N'} "
                                    f"| 남은KRW≈{remaining_krw:.0f}, 남은슬롯={remaining_slots}"
                                )
                    else:
                        log.info(f"[{t}] 매수 불가(KRW<{MIN_ORDER_KRW})")

                elif t not in entry_prices:
                    # 진입 스킵 사유 출력
                    reasons = []
                    if risk_block and not allow_priority_buy:
                        reasons.append("리스크블록")
                    if not kpass:
                        reasons.append(f"K-of-N미달({tech_pass}/{K_req})")
                    if not (strict_gate or lax_gate) and not allow_bear_buy and not allow_priority_buy:
                        reasons.append("ML문턱미달")
                    if remaining_slots <= 0:
                        reasons.append("슬롯0")
                    if not reasons:
                        reasons.append("조건불충족")
                    log.info(f"[SKIP] {t} 신규매수 스킵 → {', '.join(reasons)}")

                # -------- 8) 매도 로직 --------
                if t in entry_prices:
                    # (A) 우선 포지션 관리 루틴 실행: 부분익절 → 트레일링 → 하드스톱
                    manage_position(t, now, regime=regime)

                    # manage_position에서 전량 청산했을 수 있음 → 보유 재확인
                    try:
                        coin = t.split('-')[1]
                        bal_check = get_balance(coin)
                        if bal_check is None or bal_check <= 0:
                            continue
                    except Exception:
                        pass

                    # (B) 추가 전략 매도(ML 약화/강제 익절) 판단
                    px = feats['close'].iloc[-1]
                    entry = entry_prices[t]
                    chg = (px - entry)/entry
                    _, T_sell_eff = compute_ml_threshold(t, regime, use_lax=False)
                    will_sell = should_sell(t, px, ml, T_sell_eff, regime)

                    tp_dyn   = get_dynamic_tp_by_regime(regime)
                    stop_thr = get_dynamic_stop_by_regime(regime)
                    force_liq = (chg <= stop_thr) or (chg >= tp_dyn)

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
                            if chg <= stop_thr: reason.append(f"강제손절({stop_thr*100:.1f}%)")
                            if chg >= tp_dyn:  reason.append(f"강제익절(+{tp_dyn*100:.0f}%)")
                            if will_sell and not reason: reason.append("전략매도")
                            log.info(f"[SELL] {t} | {'/'.join(reason)} | PnL {chg*100:.2f}%")

                            # 체결 신뢰도 향상: 3회 재시도
                            sold = False
                            for k in range(3):
                                try:
                                    o = sell_crypto_currency(t, bal)
                                    if o: sold = True; break
                                    time.sleep(1.0)
                                except Exception as e:
                                    log.info(f"[{t}] 매도 오류 재시도({k+1}/3): {e}")
                                    time.sleep(1.0)
                            if sold:
                                time.sleep(0.7)
                                remain = get_balance(coin)
                                if remain is None or remain < 1e-8:
                                    with state_lock:
                                        entry_prices.pop(t, None)
                                        highest_prices.pop(t, None)
                                        recent_trades[t] = now
                                        entry_times.pop(t, None)
                                        pos_plan.pop(t, None)
                                    record_trade_pnl(chg)
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
                            # 추가매수는 '신호 유지' + '기술조건 K-of-N 충족' 시
                            if ml > T_buy_strict and kpass and buy_amt >= MIN_ORDER_KRW:
                                if buy_crypto_currency(t, buy_amt):
                                    with state_lock:
                                        highest_prices[t] = max(highest_prices.get(t, px), px)
                                        recent_trades[t] = now
                                    plan["filled"] += buy_amt
                                    plan["tr"].pop(0)
                                    plan["last"] = now
                                    remaining_krw -= buy_amt
                                    log.info(f"[ADD] {t} | amt={buy_amt:.0f}원 (잔여 트랜치 {len(plan['tr'])}) "
                                             f"| 남은KRW≈{remaining_krw:.0f}")

            time.sleep(1.0)

    except KeyboardInterrupt:
        print("프로그램이 종료되었습니다.")

# ===================== 진입점 =====================
if __name__ == "__main__":
    main()