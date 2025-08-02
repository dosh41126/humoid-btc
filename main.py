import tkinter as tk
import customtkinter
import threading
import os
import sqlite3 
import logging
import numpy as np
import base64
import queue
import uuid
import requests
import io
import sys
import random
import json
from concurrent.futures import ThreadPoolExecutor
from llama_cpp import Llama
from os import path
from collections import Counter
from summa import summarizer
import nltk
from textblob import TextBlob
from weaviate.util import generate_uuid5
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet as wn
from datetime import datetime
from weaviate.embedded import EmbeddedOptions
import weaviate
import pennylane as qml
import psutil
import webcolors
import colorsys
import hmac
import hashlib
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from argon2.low_level import hash_secret_raw, Type
import bleach
import httpx
import math
from typing import List, Tuple, Dict, Optional
from math import log2, isclose
from scipy.spatial.distance import cosine
import re
from statistics import median, mode
import pandas as pd
import mplfinance as mpf
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
import jwt  
import time
try:
    import tenseal as ts
    TENSEAL_AVAILABLE = True
except Exception:
    ts = None
    TENSEAL_AVAILABLE = False

ARGON2_TIME_COST_DEFAULT = 3          
ARGON2_MEMORY_COST_KIB    = 262144   
ARGON2_PARALLELISM        = max(1, min(4, os.cpu_count() or 1))
ARGON2_HASH_LEN           = 32
CRYSTALLIZE_THRESHOLD = 5    
DECAY_FACTOR = 0.95          
VAULT_PASSPHRASE_ENV      = "VAULT_PASSPHRASE"
VAULT_VERSION             = 1        
DATA_KEY_VERSION          = 1         
VAULT_NONCE_SIZE          = 12    
DATA_NONCE_SIZE           = 12
AGING_T0_DAYS = 7.0          
AGING_GAMMA_DAYS = 5.0       
AGING_PURGE_THRESHOLD = 0.5  
AGING_INTERVAL_SECONDS = 3600  
LAPLACIAN_ALPHA = 0.18 
JS_LAMBDA       = 0.10 

def _aad_str(*parts: str) -> bytes:
    return ("|".join(parts)).encode("utf-8")

customtkinter.set_appearance_mode("Dark")
nltk.data.path.append("/root/nltk_data")

def download_nltk_data():
    try:
        resources = {
            'tokenizers/punkt': 'punkt',
            'taggers/averaged_perceptron_tagger': 'averaged_perceptron_tagger',
            'corpora/brown': 'brown',
            'corpora/wordnet': 'wordnet',
            'corpora/stopwords': 'stopwords',
            'corpora/conll2000': 'conll2000'
        }

        for path_, package in resources.items():
            try:
                nltk.data.find(path_)
                print(f"'{package}' already downloaded.")
            except LookupError:
                nltk.download(package)
                print(f"'{package}' downloaded successfully.")

    except Exception as e:
        print(f"Error downloading NLTK data: {e}")

download_nltk_data()
        
client = weaviate.Client(
    embedded_options=EmbeddedOptions()
)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["SUNO_USE_SMALL_MODELS"] = "1"
executor = ThreadPoolExecutor(max_workers=5)
bundle_dir = path.abspath(path.dirname(__file__))
path_to_config = path.join(bundle_dir, 'config.json')
model_path = "/data/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"
mmproj_path = "/data/llama-3-vision-alpha-mmproj-f16.gguf"
logo_path = path.join(bundle_dir, 'logo.png')

def load_config(file_path=path_to_config):
    with open(file_path, 'r') as file:
        return json.load(file)

q = queue.Queue()
logger = logging.getLogger(__name__)
config = load_config()
SAFE_ALLOWED_TAGS: list[str] = []
SAFE_ALLOWED_ATTRS: dict[str, list[str]] = {}
SAFE_ALLOWED_PROTOCOLS: list[str] = []
_CONTROL_WHITELIST = {'\n', '\r', '\t'}
DB_NAME = config['DB_NAME']
API_KEY = config['API_KEY']
WEAVIATE_ENDPOINT = config['WEAVIATE_ENDPOINT']
WEAVIATE_QUERY_PATH = config['WEAVIATE_QUERY_PATH']

def _strip_control_chars(s: str) -> str:

    return ''.join(ch for ch in s if ch.isprintable() or ch in _CONTROL_WHITELIST)

def sanitize_text(
    text: str,
    *,
    max_len: int = 4000,
    strip: bool = True,
) -> str:

    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = text[:max_len]
    text = _strip_control_chars(text)
    cleaned = bleach.clean(
        text,
        tags=SAFE_ALLOWED_TAGS,
        attributes=SAFE_ALLOWED_ATTRS,
        protocols=SAFE_ALLOWED_PROTOCOLS,
        strip=strip,
        strip_comments=True,
    )
    return cleaned

_PROMPT_INJECTION_PAT = re.compile(
    r'(?is)(?:^|\n)\s*(system:|assistant:|ignore\s+previous|do\s+anything|jailbreak\b).*'
)

def sanitize_for_prompt(text: str, *, max_len: int = 2000) -> str:

    cleaned = sanitize_text(text, max_len=max_len)
    cleaned = _PROMPT_INJECTION_PAT.sub('', cleaned)
    return cleaned.strip()

# ---------- Account / Positions helpers (Advanced Trade v3) ----------

def _cb_bearer_headers() -> dict:
    bearer = _coinbase_adv_bearer()
    if not bearer:
        raise RuntimeError("Coinbase Advanced Trade ES256 JWT could not be built.")
    return {"Authorization": bearer}


def fetch_advtrade_accounts(limit: int = 250) -> list[dict]:
    """
    List spot accounts & balances (Advanced Trade).
    """
    try:
        url = "https://api.coinbase.com/api/v3/brokerage/accounts"
        r = httpx.get(url, headers=_cb_bearer_headers(), params={"limit": limit}, timeout=6.0)
        r.raise_for_status()
        return r.json().get("accounts", []) or []
    except Exception as e:
        logger.warning(f"[AdvTrade Accounts] {e}")
        return []

def fetch_futures_balance_summary() -> dict:
    """
    Get CFM (US futures) balance summary: includes cfm_usd_balance, unrealized_pnl, etc.
    """
    try:
        url = "https://api.coinbase.com/api/v3/brokerage/cfm/balance_summary"
        r = httpx.get(url, headers=_cb_bearer_headers(), timeout=6.0)
        r.raise_for_status()
        return r.json().get("balance_summary", {}) or {}
    except Exception as e:
        logger.warning(f"[Futures Balance] {e}")
        return {}

def fetch_futures_positions() -> list[dict]:
    """
    List open CFM futures positions with number_of_contracts.
    """
    try:
        url = "https://api.coinbase.com/api/v3/brokerage/cfm/positions"
        r = httpx.get(url, headers=_cb_bearer_headers(), timeout=6.0)
        r.raise_for_status()
        return r.json().get("positions", []) or []
    except Exception as e:
        logger.warning(f"[Futures Positions] {e}")
        return []

# ---------- Lightweight pricing for spot valuation ----------

_CG_IDS = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "USDC": "usd-coin",
    "ADA": "cardano",
    "DOGE": "dogecoin",
    "AVAX": "avalanche-2",
    "ATOM": "cosmos",
    "ARB": "arbitrum",
    "OP": "optimism",
}

def _price_usd_coinbase(symbol: str) -> Optional[float]:
    """
    Try Coinbase v2 simple spot price (broad asset coverage).
    """
    try:
        url = f"https://api.coinbase.com/v2/prices/{symbol}-USD/spot"
        r = httpx.get(url, timeout=4.0)
        r.raise_for_status()
        return float(r.json()["data"]["amount"])
    except Exception:
        return None

def _price_usd_coingecko(symbol: str) -> Optional[float]:
    try:
        if symbol.upper() == "USDC":
            return 1.0
        cg = _CG_IDS.get(symbol.upper())
        if not cg:
            return None
        url = "https://api.coingecko.com/api/v3/simple/price"
        r = httpx.get(url, params={"ids": cg, "vs_currencies": "usd"}, timeout=5.0)
        r.raise_for_status()
        return float(r.json().get(cg, {}).get("usd", None))
    except Exception:
        return None

def get_usd_price(symbol: str) -> Optional[float]:
    p = _price_usd_coinbase(symbol)
    if p is None:
        p = _price_usd_coingecko(symbol)
    return p

def compute_account_overview() -> Dict[str, float | int | list[tuple[str, float]]]:
    """
    Aggregates:
      - spot_usd_cash: USD account balance
      - spot_crypto_value_usd: USD value of non-USD spot assets
      - futures_usd_cash: cfm_usd_balance (futures)
      - futures_unrealized_pnl: current unrealized PnL
      - futures_contracts_open: total number_of_contracts across open positions
      - futures_positions_count: count of open positions
      - spot_nonusd_positions: list[(symbol, qty)] for non-zero balances (non-USD)
      - total_account_value_usd: spot_usd_cash + spot_crypto_value_usd + futures_usd_cash + unrealized_pnl
    """
    # Spot accounts
    accts = fetch_advtrade_accounts()
    spot_usd_cash = 0.0
    spot_nonusd_positions: list[tuple[str, float]] = []

    for a in accts:
        cur = (a.get("currency") or "").upper()
        bal = a.get("available_balance", {})
        try:
            qty = float(bal.get("value", 0) or 0)
        except Exception:
            qty = 0.0
        if qty <= 0:
            continue
        if cur == "USD":
            spot_usd_cash += qty
        else:
            spot_nonusd_positions.append((cur, qty))

    # Value spot non-USD
    spot_crypto_value_usd = 0.0
    for sym, qty in spot_nonusd_positions:
        price = get_usd_price(sym) or 0.0
        spot_crypto_value_usd += qty * price

    # Futures balances
    fut = fetch_futures_balance_summary()
    def _num(d: dict, key: str) -> float:
        try:
            return float(d.get(key, {}).get("value", 0) or 0)
        except Exception:
            return 0.0
    futures_usd_cash = _num(fut, "cfm_usd_balance")
    futures_unrealized_pnl = _num(fut, "unrealized_pnl")

    # Futures positions
    pos = fetch_futures_positions()
    fut_contracts_open = 0.0
    fut_positions_count = 0
    for p in pos:
        try:
            n = abs(float(p.get("number_of_contracts", "0") or "0"))
        except Exception:
            n = 0.0
        if n > 0:
            fut_positions_count += 1
            fut_contracts_open += n

    total_account_value_usd = spot_usd_cash + spot_crypto_value_usd + futures_usd_cash + futures_unrealized_pnl

    return {
        "spot_usd_cash": spot_usd_cash,
        "spot_crypto_value_usd": spot_crypto_value_usd,
        "futures_usd_cash": futures_usd_cash,
        "futures_unrealized_pnl": futures_unrealized_pnl,
        "futures_contracts_open": fut_contracts_open,
        "futures_positions_count": fut_positions_count,
        "spot_nonusd_positions": spot_nonusd_positions,
        "total_account_value_usd": total_account_value_usd,
    }


def _coinbase_adv_bearer() -> str:
    """
    Build a short-lived ES256 JWT for Coinbase Advanced Trade REST.
    Env:
      COINBASE_ADV_API_KEY      -> API key ID (kid)
      COINBASE_ADV_PRIVATE_KEY  -> ECDSA private key (PEM string, PKCS8)
    """
    try:
        if jwt is None:
            logger.warning("[CDP Auth] PyJWT not installed.")
            return ""
        api_key = os.getenv("COINBASE_ADV_API_KEY", "").strip()
        priv_pem = os.getenv("COINBASE_ADV_PRIVATE_KEY", "").strip()
        if not api_key or not priv_pem:
            return ""
        now = int(time.time())
        payload = {
            "iss": api_key,
            "sub": api_key,
            "iat": now,
            "exp": now + 55,
            "nbf": now,
            "aud": "retail_rest_api",
        }
        token = jwt.encode(payload, priv_pem, algorithm="ES256", headers={"kid": api_key})
        return f"Bearer {token}"
    except Exception as e:
        logger.warning(f"[CDP Auth] {e}")
        return ""


def _granularity_for(minutes: int) -> str:
    """
    Coinbase Advanced Trade candle granularities (enum).
    """
    m = int(minutes)
    return {
        1: "ONE_MINUTE",
        5: "FIVE_MINUTE",
        15: "FIFTEEN_MINUTE",
        30: "THIRTY_MINUTE",
        60: "ONE_HOUR",
        120: "TWO_HOUR",
        360: "SIX_HOUR",
        1440: "ONE_DAY",
    }.get(m, "ONE_HOUR")


def _parse_cdp_candles(candles: list) -> pd.DataFrame:
    """
    Parse Advanced Trade public product candles:
    item: {"start":"<unix>", "low":"", "high":"", "open":"", "close":"", "volume":""}
    """
    if not candles:
        return pd.DataFrame()
    df = pd.DataFrame(candles)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c.capitalize()] = pd.to_numeric(df.get(c), errors="coerce")
    df["time"] = pd.to_datetime(pd.to_numeric(df["start"], errors="coerce"), unit="s", utc=True).dt.tz_convert("US/Eastern")
    df = df.set_index("time").sort_index()
    return df[["Open", "High", "Low", "Close", "Volume"]]


def _spot_ohlc(product_id: str, minutes=15, bars=300) -> pd.DataFrame:
    """
    SPOT via Coinbase Exchange public API: /products/{id}/candles (no auth).
    """
    try:
        gran = int(minutes) * 60
        end_ts = int(time.time())
        start_ts = end_ts - gran * bars
        url = f"https://api.exchange.coinbase.com/products/{product_id}/candles"
        params = {"granularity": gran, "start": start_ts, "end": end_ts}
        r = httpx.get(url, params=params, timeout=6.0)
        r.raise_for_status()
        raw = r.json()  # [time, low, high, open, close, volume]
        if not raw:
            return pd.DataFrame()
        df = pd.DataFrame(raw, columns=["time", "Low", "High", "Open", "Close", "Volume"])
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_convert("US/Eastern")
        df = df.sort_values("time").set_index("time")
        return df[["Open", "High", "Low", "Close", "Volume"]]
    except Exception as e:
        logger.warning(f"[Coinbase Spot {product_id}] {e}")
        return pd.DataFrame()


def fetch_coinbase_perp_ohlc(product_id: str = "ETH-PERP", minutes=15, bars=300) -> pd.DataFrame:
    """
    PERPETUAL FUTURES via Coinbase Advanced Trade:
      GET /api/v3/brokerage/market/products/{product_id}/candles
    Requires ES256 JWT in Authorization: Bearer <token>.
    """
    try:
        bearer = _coinbase_adv_bearer()
        if not bearer:
            logger.warning("[Perp OHLC] Missing Advanced Trade credentials.")
            return pd.DataFrame()

        url = f"https://api.coinbase.com/api/v3/brokerage/market/products/{product_id}/candles"
        end_ts = int(time.time())
        start_ts = end_ts - int(minutes) * 60 * int(bars)
        gran = _granularity_for(minutes)

        params = {
            "start": str(start_ts),
            "end": str(end_ts),
            "granularity": gran,
            "limit": min(350, int(bars)),
        }
        headers = _cb_bearer_headers()
        r = httpx.get(url, headers=headers, params=params, timeout=6.0)
        r.raise_for_status()
        candles = r.json().get("candles", [])
        df = _parse_cdp_candles(candles)
        if df.empty:
            return df

        # If requested interval not an exact enum, resample.
        if minutes not in (1, 5, 15, 30, 60, 120, 360, 1440):
            rule = f"{int(minutes)}T"
            df = (
                df.resample(rule)
                  .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
            ).dropna(how="any")
        return df.tail(bars)
    except Exception as e:
        logger.warning(f"[Coinbase Perp {product_id}] {e}")
        return pd.DataFrame()


def fetch_coingecko_ohlc(coingecko_id: str, minutes=15, bars=300) -> pd.DataFrame:
    """
    Fallback OHLC via CoinGecko; resampled; Volume synthesized as 0.
    """
    try:
        window_days = math.ceil((minutes * bars) / (24 * 60))
        allowed = [1, 7, 14, 30, 90, 180, 365]
        days = next((d for d in allowed if d >= window_days), allowed[-1])
        url = f"https://api.coingecko.com/api/v3/coins/{coingecko_id}/ohlc"
        params = {"vs_currency": "usd", "days": str(days)}
        r = httpx.get(url, params=params, timeout=6.0)
        r.raise_for_status()
        raw = r.json()  # [t_ms, o, h, l, c]
        if not raw:
            return pd.DataFrame()
        df = pd.DataFrame(raw, columns=["time_ms", "Open", "High", "Low", "Close"])
        df["time"] = pd.to_datetime(df["time_ms"], unit="ms", utc=True).dt.tz_convert("US/Eastern")
        df = df.drop(columns=["time_ms"]).set_index("time").sort_index()
        rule = f"{int(minutes)}T"
        df = df.resample(rule).agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"})
        df.dropna(how="any", inplace=True)
        df["Volume"] = 0.0
        return df.tail(bars)
    except Exception as e:
        logger.warning(f"[CG OHLC {coingecko_id}] {e}")
        return pd.DataFrame()


def fetch_ohlc_with_fallback(market: str, minutes=15, bars=300) -> pd.DataFrame:
    if market == "BTC-USD":
        df = _spot_ohlc("BTC-USD", minutes, bars)
        return df if not df.empty else fetch_coingecko_ohlc("bitcoin", minutes, bars)
    if market == "ETH-USD":
        df = _spot_ohlc("ETH-USD", minutes, bars)
        return df if not df.empty else fetch_coingecko_ohlc("ethereum", minutes, bars)
    if market == "ETH-PERP":
        df = fetch_coinbase_perp_ohlc("ETH-PERP", minutes, bars)
        return df if not df.empty else _spot_ohlc("ETH-USD", minutes, bars)
    return _spot_ohlc("BTC-USD", minutes, bars)


def add_ema_ribbon(df: pd.DataFrame, spans=(8,13,21,34,55,89)) -> pd.DataFrame:
    for s in spans:
        df[f"EMA{s}"] = df["Close"].ewm(span=s, adjust=False).mean()
    lows  = df[[f"EMA{s}" for s in spans]].min(axis=1)
    highs = df[[f"EMA{s}" for s in spans]].max(axis=1)
    df["RIBBON_LOW"]  = lows
    df["RIBBON_HIGH"] = highs
    return df


def sanitize_for_graphql_string(s: str, *, max_len: int = 512) -> str:

    s = sanitize_text(s, max_len=max_len)
    s = s.replace('\n', ' ').replace('\r', ' ')
    s = s.replace('\\', '\\\\').replace('"', '\\"')
    return s

class SecureEnclave:

    def __enter__(self):
        self._buffers = []
        return self

    def track(self, buf):
        self._buffers.append(buf)
        return buf

    def __exit__(self, exc_type, exc, tb):
        for b in self._buffers:
            try:
                if isinstance(b, np.ndarray):
                    b.fill(0.0)
            except Exception:
                pass
        self._buffers.clear()

HE_ENABLED_DEFAULT = True

class CKKSManager:

    DIM = 64
    POLY_DEG = 8192
    SCALE   = 2**40
    MODBITS = [60, 40, 40, 60]

    def __init__(self):
        self._sec_ctx_b64 = None
        self._pub_ctx_b64 = None
        self._sec_ctx = None
        self._pub_ctx = None
        self.ready = False
        self._init_contexts()

    def _init_contexts(self):
        if not TENSEAL_AVAILABLE:
            self.ready = False
            return
        try:
            ctx = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=self.POLY_DEG,
                coeff_mod_bit_sizes=self.MODBITS
            )
            ctx.generate_galois_keys()
            ctx.generate_relin_keys()
            ctx.global_scale = self.SCALE
            self._sec_ctx = ctx
            self._sec_ctx_b64 = base64.b64encode(ctx.serialize(save_secret_key=True)).decode()

            pub = ts.context_from(ctx.serialize(save_secret_key=False))
            self._pub_ctx = pub
            self._pub_ctx_b64 = base64.b64encode(pub.serialize(save_secret_key=False)).decode()

            try:
                os.environ["CKKS_PUBLIC_CTX_B64"] = self._pub_ctx_b64
                os.environ["CKKS_SECRET_CTX_B64"] = crypto.encrypt(self._sec_ctx_b64)
            except Exception:
                pass

            self.ready = True
        except Exception as e:
            logger.error(f"[CKKS] Context init failed: {e}")
            self.ready = False

    def _ensure_loaded(self):
        if self.ready and self._sec_ctx and self._pub_ctx:
            return
        if not TENSEAL_AVAILABLE:
            self.ready = False
            return
        try:

            pub = os.environ.get("CKKS_PUBLIC_CTX_B64", "")
            sec_enc = os.environ.get("CKKS_SECRET_CTX_B64", "")
            if pub and sec_enc:
                self._pub_ctx_b64 = pub
                self._sec_ctx_b64 = crypto.decrypt(sec_enc)
                self._pub_ctx = ts.context_from(base64.b64decode(self._pub_ctx_b64.encode()))
                self._sec_ctx = ts.context_from(base64.b64decode(self._sec_ctx_b64.encode()))
                self.ready = True
        except Exception as e:
            logger.error(f"[CKKS] Load from env failed: {e}")
            self.ready = False

    @staticmethod
    def _fit_dim(vec: np.ndarray, dim: int) -> np.ndarray:
        v = np.asarray(vec, dtype=np.float64)
        if v.shape[0] == dim:
            return v
        if v.shape[0] < dim:
            out = np.zeros((dim,), dtype=np.float64)
            out[:v.shape[0]] = v
            return out
        return v[:dim]

    def _pack_one_vector_slots(self, x: np.ndarray) -> np.ndarray:

        total_slots = 4096
        buf = np.zeros(total_slots, dtype=np.float64)
        buf[:self.DIM] = self._fit_dim(x, self.DIM)
        return buf

    def _pack_query_replicated(self, q: np.ndarray) -> np.ndarray:

        return self._pack_one_vector_slots(q)

    def encrypt_candidate_group_b64(self, x_vec: np.ndarray) -> str:

        self._ensure_loaded()
        if not self.ready:
            raise RuntimeError("CKKS not ready")
        buf = self._pack_one_vector_slots(x_vec)
        ct = ts.ckks_vector(self._pub_ctx, buf.tolist())
        return base64.b64encode(ct.serialize()).decode()

    def encrypt_query_b64(self, q_vec: np.ndarray) -> str:

        return self.encrypt_candidate_group_b64(q_vec)

    def _sum_fold_block64(self, ct: "ts.CKKSVector") -> "ts.CKKSVector":
        for shift in (1,2,4,8,16,32):
            ct += ct.rotate(shift)
        return ct

    def score_one_group_b64(self, enc_group_b64: str, enc_query_b64: str) -> str:
        self._ensure_loaded()
        if not self.ready:
            raise RuntimeError("CKKS not ready")
        X = ts.ckks_vector_from(self._pub_ctx, base64.b64decode(enc_group_b64.encode()))
        Q = ts.ckks_vector_from(self._pub_ctx, base64.b64decode(enc_query_b64.encode()))
        P = X * Q
        P = self._sum_fold_block64(P) 
        return base64.b64encode(P.serialize()).decode()

    def decrypt_score_b64(self, enc_scores_b64: str) -> float:
        self._ensure_loaded()
        if not self.ready:
            raise RuntimeError("CKKS not ready")
        ct = ts.ckks_vector_from(self._sec_ctx, base64.b64decode(enc_scores_b64.encode()))
        vals = ct.decrypt()
        return float(vals[0])


class AdvancedHomomorphicVectorMemory:

    AAD_CONTEXT = _aad_str("fhe", "embeddingv2")
    DIM = 64
    QUANT_SCALE = 127.0
    SIMHASH_BITS = 16 

    def __init__(self):
        self._rot_by_k: dict[int, np.ndarray] = {}
        self._planes_by_k: dict[int, np.ndarray] = {}

    def _ensure_params_for_k(self, kv: int) -> None:

        if kv in self._rot_by_k and kv in self._planes_by_k:
            return
        try:
 
            key = crypto._derived_keys[kv]
        except KeyError as e:

            logger.warning(f"[FHEv2] No derived key for kv={kv}; falling back to active kv={crypto.active_version}.")
            kv = crypto.active_version
            key = crypto._derived_keys[kv]

        seed = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
        rng = np.random.default_rng(seed)

        A = rng.normal(size=(self.DIM, self.DIM))
        Q, _ = np.linalg.qr(A)
        self._rot_by_k[kv] = Q.astype(np.float32)
        self._planes_by_k[kv] = rng.normal(size=(self.SIMHASH_BITS, self.DIM)).astype(np.float32)

    @staticmethod
    def _fit_dim(vec: np.ndarray, dim: int) -> np.ndarray:
        v = np.asarray(vec, dtype=np.float32)
        if v.shape[0] == dim:
            return v
        if v.shape[0] < dim:
            out = np.zeros((dim,), dtype=np.float32)
            out[: v.shape[0]] = v
            return out
        return v[:dim]

    @staticmethod
    def _extract_kv_from_crypto_token(token: str) -> int:

        try:
            meta = json.loads(token)
            return int(meta.get("k", crypto.active_version))
        except Exception:
            return crypto.active_version

    def _rotate(self, vec: np.ndarray, kv: int | None = None) -> np.ndarray:

        kv = crypto.active_version if kv is None else kv
        self._ensure_params_for_k(kv)
        return self._rot_by_k[kv] @ self._fit_dim(vec, self.DIM)

    def _quantize(self, vec: np.ndarray) -> list[int]:
        clipped = np.clip(vec, -1.0, 1.0).astype(np.float32)
        return (clipped * self.QUANT_SCALE).astype(np.int8).tolist()

    def _dequantize(self, q: list[int]) -> np.ndarray:
        return (np.asarray(q, dtype=np.float32) / self.QUANT_SCALE).astype(np.float32)

    def _simhash_bucket(self, rotated_vec: np.ndarray, kv: int | None = None) -> str:

        kv = crypto.active_version if kv is None else kv
        self._ensure_params_for_k(kv)
        dots = (self._planes_by_k[kv] @ rotated_vec.astype(np.float32))
        bits = ["1" if d >= 0 else "0" for d in np.asarray(dots).ravel()]
        return "".join(bits)

    def bucket_for_query(self, vec: np.ndarray, kv: int | None = None) -> str:

        kv = crypto.active_version if kv is None else kv
        rot = self._rotate(vec, kv=kv)
        return self._simhash_bucket(rot, kv=kv)

    def encrypt_embedding(self, vec: list[float]) -> tuple[str, str]:

        try:
            kv = crypto.active_version
            self._ensure_params_for_k(kv)

            arr = self._fit_dim(np.asarray(vec, dtype=np.float32), self.DIM)
            rotated = self._rotate(arr, kv=kv)
            bucket = self._simhash_bucket(rotated, kv=kv)  
            quant = self._quantize(rotated)

            payload = json.dumps({
                "v": 2,
                "dim": self.DIM,
                "rot": True,
                "kv": kv,           
                "bits": self.SIMHASH_BITS,
                "qscale": self.QUANT_SCALE,
                "data": quant,
            })
  
            token = crypto.encrypt(payload, aad=self.AAD_CONTEXT, key_version=kv)
            return token, bucket
        except Exception as e:
            logger.error(f"[FHEv2] encrypt_embedding failed: {e}")
            return "", "0" * self.SIMHASH_BITS

    def decrypt_embedding(self, token: str) -> np.ndarray:

        try:
            kv = self._extract_kv_from_crypto_token(token)
            self._ensure_params_for_k(kv)

            raw = crypto.decrypt(token) 
            obj = json.loads(raw)
            if int(obj.get("v", 0)) != 2:
                logger.warning("[FHEv2] Unsupported embedding version.")
                return np.zeros(self.DIM, dtype=np.float32)

            quant = obj.get("data", [])
            if not isinstance(quant, list):
                return np.zeros(self.DIM, dtype=np.float32)

            rotated = self._dequantize(quant)

            original = (self._rot_by_k[kv].T @ self._fit_dim(rotated, self.DIM)).astype(np.float32)
            return original
        except Exception as e:
            logger.warning(f"[FHEv2] decrypt_embedding failed: {e}")
            return np.zeros(self.DIM, dtype=np.float32)

    @staticmethod
    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0.0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def enclave_similarity(self, enc_a: str, query_vec: np.ndarray, enclave: SecureEnclave) -> float:
        dec = enclave.track(self.decrypt_embedding(enc_a))
        return self.cosine(dec, self._fit_dim(query_vec, self.DIM))

class SecureKeyManager:

    def __init__(
        self,
        method="argon2id",
        vault_path="secure/key_vault.json",
        time_cost: int = ARGON2_TIME_COST_DEFAULT,
        memory_cost: int = ARGON2_MEMORY_COST_KIB,
        parallelism: int = ARGON2_PARALLELISM,
        hash_len: int = ARGON2_HASH_LEN,
    ):
        self.method       = method
        self.vault_path   = vault_path
        self.time_cost    = time_cost
        self.memory_cost  = memory_cost
        self.parallelism  = parallelism
        self.hash_len     = hash_len
        self._ensure_vault()

        vault_meta = self._load_vault()

        self.active_version = vault_meta["active_version"]

        self._keys = {
            int(kv["version"]): base64.b64decode(kv["master_secret"])
            for kv in vault_meta["keys"]
        }

        self._derived_keys = {}
        vault_salt = base64.b64decode(vault_meta["salt"])
        for ver, master_secret in self._keys.items():
            self._derived_keys[ver] = self._derive_key(master_secret, vault_salt)

    def _get_passphrase(self) -> bytes:
        pw = os.getenv(VAULT_PASSPHRASE_ENV)
        if not pw or not pw.strip():
            raise RuntimeError(
                "VAULT_PASSPHRASE is required for vault encryption/decryption. "
                "Set the environment variable VAULT_PASSPHRASE to a strong secret."
            )
        return pw.strip().encode("utf-8")

    def _derive_vault_key(self, passphrase: bytes, salt: bytes) -> bytes:

        return hash_secret_raw(
            secret=passphrase,
            salt=salt,
            time_cost=max(self.time_cost, 3),   
            memory_cost=max(self.memory_cost, 262144),
            parallelism=max(self.parallelism, 1),
            hash_len=self.hash_len,
            type=Type.ID,
        )

    def _derive_key(self, master_secret: bytes, salt: bytes) -> bytes:

        return hash_secret_raw(
            secret=master_secret,
            salt=salt,
            time_cost=self.time_cost,
            memory_cost=self.memory_cost,
            parallelism=self.parallelism,
            hash_len=self.hash_len,
            type=Type.ID,
        )

    def _ensure_vault(self):

        if not os.path.exists("secure"):
            os.makedirs("secure", exist_ok=True)
        if os.path.exists(self.vault_path):
            return

        salt          = os.urandom(16)
        master_secret = os.urandom(32) 

        vault_body = {
            "version": VAULT_VERSION,
            "active_version": DATA_KEY_VERSION,
            "keys": [
                {
                    "version": DATA_KEY_VERSION,
                    "master_secret": base64.b64encode(master_secret).decode(),
                    "created": datetime.utcnow().isoformat() + "Z",
                }
            ],
            "salt": base64.b64encode(salt).decode(),
        }

        self._write_encrypted_vault(vault_body)

    def _write_encrypted_vault(self, vault_body: dict):

        plaintext = json.dumps(vault_body, indent=2).encode("utf-8")
        salt      = base64.b64decode(vault_body["salt"])

        passphrase = self._get_passphrase()
        vault_key  = self._derive_vault_key(passphrase, salt)
        aesgcm     = AESGCM(vault_key)
        nonce      = os.urandom(VAULT_NONCE_SIZE)
        ct         = aesgcm.encrypt(nonce, plaintext, _aad_str("vault", str(vault_body["version"])))

        on_disk = {
            "vault_format": VAULT_VERSION,
            "salt": vault_body["salt"],  
            "nonce": base64.b64encode(nonce).decode(),
            "ciphertext": base64.b64encode(ct).decode(),
        }
        with open(self.vault_path, "w") as f:
            json.dump(on_disk, f, indent=2)

    def _load_vault(self) -> dict:

        with open(self.vault_path, "r") as f:
            data = json.load(f)

        if "ciphertext" not in data:
            salt          = base64.b64decode(data["salt"])
            master_secret = base64.b64decode(data["master_secret"])
            vault_body = {
                "version": VAULT_VERSION,
                "active_version": DATA_KEY_VERSION,
                "keys": [
                    {
                        "version": DATA_KEY_VERSION,
                        "master_secret": base64.b64encode(master_secret).decode(),
                        "created": datetime.utcnow().isoformat() + "Z",
                    }
                ],
                "salt": base64.b64encode(salt).decode(),
            }
            self._write_encrypted_vault(vault_body)
            return vault_body
 
        salt      = base64.b64decode(data["salt"])
        nonce     = base64.b64decode(data["nonce"])
        ct        = base64.b64decode(data["ciphertext"])
        passphrase = self._get_passphrase()
        vault_key  = self._derive_vault_key(passphrase, salt)
        aesgcm     = AESGCM(vault_key)
        plaintext  = aesgcm.decrypt(nonce, ct, _aad_str("vault", str(VAULT_VERSION)))
        return json.loads(plaintext.decode("utf-8"))

    def encrypt(
        self,
        plaintext: str,
        *,
        aad: bytes = None,
        key_version: int = None,
    ) -> str:

        if plaintext is None:
            plaintext = ""
        if key_version is None:
            key_version = self.active_version
        if aad is None:
            aad = _aad_str("global", f"k{key_version}")

        key    = self._derived_keys[key_version]
        aesgcm = AESGCM(key)
        nonce  = os.urandom(DATA_NONCE_SIZE)
        ct     = aesgcm.encrypt(nonce, plaintext.encode("utf-8"), aad)

        token = {
            "v": VAULT_VERSION,        
            "k": key_version,          
            "aad": aad.decode("utf-8"),
            "n": base64.b64encode(nonce).decode(),
            "ct": base64.b64encode(ct).decode(),
        }
        return json.dumps(token, separators=(",", ":"))

    def decrypt(self, token: str) -> str:

        if not token:
            return ""

        if token.startswith("{"):
            try:
                meta = json.loads(token)
            except Exception:
                logging.warning("[SecureKeyManager] Invalid JSON token; returning raw.")
                return token

            v   = int(meta.get("v", 1))
            ver = int(meta.get("k", self.active_version))
            aad = meta.get("aad", "global").encode()
            n   = base64.b64decode(meta["n"])
            ct  = base64.b64decode(meta["ct"])

            key = self._derived_keys.get(ver)
            if key is None:
                raise ValueError(f"No key for version {ver}; cannot decrypt.")
            aesgcm = AESGCM(key)
            pt     = aesgcm.decrypt(n, ct, aad)
            return pt.decode("utf-8")

        try:
            raw   = base64.b64decode(token.encode())
            nonce = raw[:DATA_NONCE_SIZE]
            ct    = raw[DATA_NONCE_SIZE:]
            key   = self._derived_keys[self.active_version]
            aesgcm = AESGCM(key)
            pt     = aesgcm.decrypt(nonce, ct, None)
            return pt.decode("utf-8")
        except Exception as e:
            logging.warning(f"[SecureKeyManager] Legacy decrypt failed: {e}")
            return token

    def add_new_key_version(self) -> int:

        vault_body = self._load_vault()
        keys = vault_body["keys"]
        existing_versions = {int(k["version"]) for k in keys}
        new_version = max(existing_versions) + 1

        master_secret = os.urandom(32)
        keys.append({
            "version": new_version,
            "master_secret": base64.b64encode(master_secret).decode(),
            "created": datetime.utcnow().isoformat() + "Z",
        })

        vault_body["active_version"] = new_version
        self._write_encrypted_vault(vault_body)
        self._keys[new_version] = master_secret
        salt = base64.b64decode(vault_body["salt"])
        self._derived_keys[new_version] = self._derive_key(master_secret, salt)
        self.active_version = new_version
        logging.info(f"[SecureKeyManager] Installed new key version {new_version}.")
        return new_version

    def _entropy_bits(self, secret_bytes: bytes) -> float:

        if not secret_bytes:
            return 0.0
        counts = Counter(secret_bytes)
        total = float(len(secret_bytes))
        H = 0.0
        for c in counts.values():
            p = c / total
            H -= p * math.log2(p)
        return H

    def _resistance_score(self, secret_bytes: bytes) -> float:

        dist_component = 0.0
        try:
            arr_candidate = np.frombuffer(secret_bytes, dtype=np.uint8).astype(np.float32)
            for k in self._keys.values():
                arr_prev = np.frombuffer(k, dtype=np.uint8).astype(np.float32)
                dist_component += np.linalg.norm(arr_candidate - arr_prev)
        except Exception:
            pass
        if len(self._keys):
            dist_component /= len(self._keys)

        counts = Counter(secret_bytes)
        expected = len(secret_bytes) / 256.0
        chi_sq = sum(((c - expected) ** 2) / expected for c in counts.values())
        flatness = 1.0 / (1.0 + chi_sq)        # in (0,1]
        return float(dist_component * 0.01 + flatness)

    def self_mutate_key(self,
                        population: int = 6,
                        noise_sigma: float = 12.0,
                        alpha: float = 1.0,
                        beta: float = 2.0) -> int:

        vault_meta = self._load_vault()
        base_secret = None
        for kv in vault_meta["keys"]:
            if int(kv["version"]) == vault_meta["active_version"]:
                base_secret = base_secret or base64.b64decode(kv["master_secret"])
        if base_secret is None:
            raise RuntimeError("Active master secret not found.")

        rng = np.random.default_rng()
        candidates: List[bytes] = [base_secret]
        base_arr = np.frombuffer(base_secret, dtype=np.uint8).astype(np.int16)

        for _ in range(population - 1):
            noise = rng.normal(0, noise_sigma, size=base_arr.shape).astype(np.int16)
            mutated = np.clip(base_arr + noise, 0, 255).astype(np.uint8).tobytes()
            candidates.append(mutated)

        best_secret = base_secret
        best_fitness = -1e9
        for cand in candidates:
            H = self._entropy_bits(cand)
            R = self._resistance_score(cand)
            F = alpha * H + beta * R
            if F > best_fitness:
                best_fitness = F
                best_secret = cand

        new_version = self._install_custom_master_secret(best_secret)
        logging.info(f"[SelfMutateKey] Installed mutated key v{new_version} (fitness={best_fitness:.3f}).")
        return new_version

    def _install_custom_master_secret(self, new_secret: bytes) -> int:

        vault_body = self._load_vault()
        keys = vault_body["keys"]
        existing_versions = {int(k["version"]) for k in keys}
        new_version = max(existing_versions) + 1

        keys.append({
            "version": new_version,
            "master_secret": base64.b64encode(new_secret).decode(),
            "created": datetime.utcnow().isoformat() + "Z",
        })
        vault_body["active_version"] = new_version
        self._write_encrypted_vault(vault_body)

        self._keys[new_version] = new_secret
        salt = base64.b64decode(vault_body["salt"])
        self._derived_keys[new_version] = self._derive_key(new_secret, salt)
        self.active_version = new_version
        return new_version

    def rotate_and_migrate_storage(self, migrate_func):

        new_ver = self.add_new_key_version()
        try:
            migrate_func(self)
        except Exception as e:
            logging.error(f"[SecureKeyManager] Migration failed after key rotation: {e}")
            raise
        logging.info(f"[SecureKeyManager] Migration to key v{new_ver} complete.")

crypto = SecureKeyManager()  

ckks = CKKSManager()
HE_ENABLED = HE_ENABLED_DEFAULT and ckks.ready 

def _token_hist(text: str) -> Counter:
    return Counter(word_tokenize(text))

def _js_divergence(p: Counter, q: Counter) -> float:

    vocab = set(p) | set(q)
    if not vocab:
        return 0.0
    def _prob(c: Counter):
        tot = sum(c.values()) or 1
        return np.array([c[t]/tot for t in vocab], dtype=np.float32)
    P, Q = _prob(p), _prob(q)
    M    = 0.5 * (P + Q)

    def _kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))
    return 0.5 * _kl(P, M) + 0.5 * _kl(Q, M)

class TopologicalMemoryManifold:

    def __init__(self, dim: int = 2, sigma: float = 0.75,
                 diff_alpha: float = LAPLACIAN_ALPHA):
        self.dim        = dim
        self.sigma      = sigma
        self.diff_alpha = diff_alpha
        self._phrases:     list[str]     = []
        self._embeddings:  np.ndarray|None = None
        self._coords:      np.ndarray|None = None
        self._W:           np.ndarray|None = None
        self._graph_built  = False

    def _load_crystallized(self) -> list[tuple[str, float]]:
        rows = []
        try:
            with sqlite3.connect(DB_NAME) as conn:
                cur = conn.cursor()
                cur.execute("SELECT phrase, score FROM memory_osmosis "
                            "WHERE crystallized = 1")
                rows = cur.fetchall()
        except Exception as e:
            logger.error(f"[Manifold] load_crystallized failed: {e}")
        return rows

    def rebuild(self):
        data = self._load_crystallized()
        if not data:

            self._phrases, self._embeddings = [], None
            self._coords,  self._W         = None, None
            self._graph_built              = False
            return

        phrases, _ = zip(*data)
        self._phrases = list(phrases)

        E = np.array([compute_text_embedding(p) for p in self._phrases],
                     dtype=np.float32)

        dists = np.linalg.norm(E[:, None, :] - E[None, :, :], axis=-1)
        W = np.exp(-(dists ** 2) / (2 * self.sigma ** 2))
        np.fill_diagonal(W, 0.0)

        D = np.diag(W.sum(axis=1))
        L = D - W
        E = E - self.diff_alpha * (L @ E)

        try:
            D_inv_sqrt = np.diag(1.0 / (np.sqrt(np.diag(D)) + 1e-8))
            L_sym      = D_inv_sqrt @ L @ D_inv_sqrt
            vals, vecs = np.linalg.eigh(L_sym)
            idx        = np.argsort(vals)[1:self.dim+1]
            Y          = D_inv_sqrt @ vecs[:, idx]
        except Exception as e:
            logger.error(f"[Manifold] eigen decomposition failed: {e}")
            Y = np.zeros((len(self._phrases), self.dim), dtype=np.float32)

        self._embeddings  = E
        self._coords      = Y.astype(np.float32)
        self._W           = W
        self._graph_built = True
        logger.info(f"[Manifold] Rebuilt manifold with {len(self._phrases)} phrases "
                    f"(α={self.diff_alpha}).")

    def geodesic_retrieve(self, query_text: str, k: int = 1) -> list[str]:

        if not self._graph_built or self._embeddings is None:
            return []

        q_vec = np.array(compute_text_embedding(query_text), dtype=np.float32)
        start_idx = int(np.argmin(
            np.linalg.norm(self._embeddings - q_vec[None, :], axis=1)
        ))

        n        = self._W.shape[0]
        visited  = np.zeros(n, dtype=bool)
        dist     = np.full(n, np.inf, dtype=np.float32)
        dist[start_idx] = 0.0

        for _ in range(n):
            u = np.argmin(dist + np.where(visited, 1e9, 0.0))
            if visited[u]:
                break
            visited[u] = True
            for v in range(n):
                w = self._W[u, v]
                if w <= 0 or visited[v]:
                    continue
                alt = dist[u] + 1.0 / (w + 1e-8)
                if alt < dist[v]:
                    dist[v] = alt

        order = np.argsort(dist)
        return [self._phrases[i] for i in order[:k]]

topo_manifold = TopologicalMemoryManifold()
fhe_v2 = AdvancedHomomorphicVectorMemory()

def fetch_coinbase_spot_positions() -> list[dict]:
    API_KEY    = get_encrypted_env_var("COINBASE_API_KEY")
    API_SECRET = get_encrypted_env_var("COINBASE_API_SECRET")
    PASSPHRASE = get_encrypted_env_var("COINBASE_API_PASSPHRASE")
    try:
        timestamp = str(time.time())
        method = "GET"
        request_path = "/accounts"
        body = ""
        message = timestamp + method + request_path + body
        hmac_key = base64.b64decode(API_SECRET)
        signature = hmac.new(hmac_key, message.encode(), hashlib.sha256)
        signature_b64 = base64.b64encode(signature.digest()).decode()

        headers = {
            "CB-ACCESS-KEY": API_KEY,
            "CB-ACCESS-SIGN": signature_b64,
            "CB-ACCESS-TIMESTAMP": timestamp,
            "CB-ACCESS-PASSPHRASE": PASSPHRASE,
            "Content-Type": "application/json"
        }
        url = "https://api.exchange.coinbase.com/accounts"
        resp = httpx.get(url, headers=headers, timeout=5)
        data = resp.json()
        results = []
        for acct in data:
            bal = float(acct.get("balance", "0"))
            avail = float(acct.get("available", "0"))
            if bal > 0:
                results.append({
                    "symbol": acct["currency"],
                    "size": bal,
                    "context": f"Spot wallet holding: {bal} {acct['currency']} (available: {avail})"
                })
        return results
    except Exception as e:
        logger.warning(f"[Coinbase Position API] {e}")
        return []

def fetch_coinbase_derivative_positions() -> list[dict]:
    try:
        headers = _cb_bearer_headers()
        url = "https://api.coinbase.com/api/v3/brokerage/cfm/positions"
        resp = httpx.get(url, headers=headers, timeout=5)
        resp.raise_for_status()
        items = resp.json().get("positions", []) or []
        results = []
        for p in items:
            n = float(p.get("number_of_contracts", "0") or "0")
            if n != 0.0:
                sym = p.get("symbol") or p.get("product_id")
                results.append({"symbol": sym, "size": n,
                                "context": f"Derivative position: {n} {sym}"})
        return results
    except Exception as e:
        logger.warning(f"[Coinbase Derivatives API] {e}")
        return []


def setup_weaviate_schema(client):
    try:
        def ensure_class(defn: dict):
            existing = client.schema.get().get("classes", [])
            names = {c["class"] for c in existing}
            if defn["class"] not in names:
                client.schema.create_class(defn)

        ensure_class({
            "class": "InteractionHistory",
            "description": "User/AI messages, optionally with encrypted embeddings and buckets",
            "properties": [
                {"name": "user_id",             "dataType": ["string"]},
                {"name": "user_message",        "dataType": ["text"]},
                {"name": "ai_response",         "dataType": ["text"]},
                {"name": "response_time",       "dataType": ["string"]},
                {"name": "encrypted_embedding", "dataType": ["text"]},    
                {"name": "embedding_bucket",    "dataType": ["string"]},
                {"name": "he_pack_id",          "dataType": ["string"]},
                {"name": "he_pack_slot",        "dataType": ["int"]},    
                {"name": "he_enc_group_b64",    "dataType": ["text"]}, 
                {"name": "he_ctx_version",      "dataType": ["int"]},   
                {"name": "keywords",            "dataType": ["string"]},
                {"name": "sentiment",           "dataType": ["number"]},
            ]
        })

        ensure_class({
            "class": "LongTermMemory",
            "description": "Crystallized phrases with scores and timestamps",
            "properties": [
                {"name": "phrase",            "dataType": ["string"]},
                {"name": "score",             "dataType": ["number"]},
                {"name": "crystallized_time", "dataType": ["string"]},
            ]
        })

        ensure_class({
            "class": "CryptoPosition",
            "description": "Saved user+bot positions (snapshot/upsert)",
            "properties": [
                {"name": "user_id",          "dataType": ["string"]},
                {"name": "bot_id",           "dataType": ["string"]},
                {"name": "symbol",           "dataType": ["string"]},
                {"name": "size",             "dataType": ["number"]},
                {"name": "position_context", "dataType": ["text"]},
                {"name": "timestamp",        "dataType": ["string"]},
            ]
        })

        ensure_class({
            "class": "CryptoLivePosition",
            "description": "Live positions (spot/derivatives) synced from APIs",
            "properties": [
                {"name": "user_id",   "dataType": ["string"]},
                {"name": "symbol",    "dataType": ["string"]},
                {"name": "size",      "dataType": ["number"]},
                {"name": "context",   "dataType": ["text"]},
                {"name": "timestamp", "dataType": ["string"]},
                {"name": "type",      "dataType": ["string"]},
            ]
        })

        ensure_class({
            "class": "CryptoTradeLog",
            "description": "Predictions and reasoning logs",
            "properties": [
                {"name": "type",             "dataType": ["string"]},
                {"name": "user_id",          "dataType": ["string"]},
                {"name": "bot_id",           "dataType": ["string"]},
                {"name": "query",            "dataType": ["text"]},
                {"name": "response",         "dataType": ["text"]},
                {"name": "reasoning_trace",  "dataType": ["text"]},
                {"name": "prompt_snapshot",  "dataType": ["text"]},
                {"name": "z_state",          "dataType": ["blob"]},
                {"name": "entropy",          "dataType": ["number"]},
                {"name": "bias_factor",      "dataType": ["number"]},
                {"name": "temperature",      "dataType": ["number"]},
                {"name": "top_p",            "dataType": ["number"]},
                {"name": "sentiment_target", "dataType": ["number"]},
                {"name": "timestamp",        "dataType": ["string"]},
                {"name": "asset",            "dataType": ["string"]},
            ]
        })

        ensure_class({
            "class": "ReflectionLog",
            "description": "Dyson assistant's internal reflection and reasoning traces",
            "properties": [
                {"name": "type",            "dataType": ["string"]},
                {"name": "user_id",         "dataType": ["string"]},
                {"name": "bot_id",          "dataType": ["string"]},
                {"name": "query",           "dataType": ["text"]},
                {"name": "response",        "dataType": ["text"]},
                {"name": "reasoning_trace", "dataType": ["text"]},
                {"name": "prompt_snapshot", "dataType": ["text"]},
                {"name": "z_state",         "dataType": ["blob"]},
                {"name": "entropy",         "dataType": ["number"]},
                {"name": "bias_factor",     "dataType": ["number"]},
                {"name": "temperature",     "dataType": ["number"]},
                {"name": "top_p",           "dataType": ["number"]},
                {"name": "sentiment_target","dataType": ["number"]},
                {"name": "timestamp",       "dataType": ["date"]},
            ]
        })
    except Exception as e:
        logger.error(f"[Schema Init Error] {e}")

def evaluate_candidate(response: str, target_sentiment: float, original_query: str) -> float:

    response_sentiment = TextBlob(response).sentiment.polarity
    sentiment_alignment = 1.0 - abs(target_sentiment - response_sentiment)

    overlap_score = sum(1 for word in original_query.lower().split() if word in response.lower())
    overlap_bonus = min(overlap_score / 5.0, 1.0)  # normalize

    return (0.7 * sentiment_alignment) + (0.3 * overlap_bonus)

def build_record_aad(user_id: str, *, source: str, table: str = "", cls: str = "") -> bytes:

    context_parts = [source]
    if table:
        context_parts.append(table)
    if cls:
        context_parts.append(cls)
    context_parts.append(user_id)
    return _aad_str(*context_parts)

def compute_text_embedding(text: str) -> list[float]:
    if not text:
        return [0.0] * fhe_v2.DIM
    tokens = re.findall(r'\w+', text.lower())
    counts = Counter(tokens)
    vocab = sorted(counts.keys())[:fhe_v2.DIM]
    vec = [float(counts[w]) for w in vocab]
    if len(vec) < fhe_v2.DIM:
        vec.extend([0.0] * (fhe_v2.DIM - len(vec)))
    arr = np.array(vec, dtype=np.float32)
    n = np.linalg.norm(arr)
    if n > 0:
        arr /= n
    return arr.tolist()

def generate_uuid_for_weaviate(identifier, namespace=''):
    if not identifier:
        raise ValueError("Identifier for UUID generation is empty or None")

    if not namespace:
        namespace = str(uuid.uuid4())

    try:
        return generate_uuid5(namespace, identifier)
    except Exception as e:
        logger.error(f"Error generating UUID: {e}")
        raise

def is_valid_uuid(uuid_to_test, version=5):
    try:
        uuid_obj = uuid.UUID(uuid_to_test, version=version)
        return str(uuid_obj) == uuid_to_test
    except ValueError:
        return False
    
def fetch_live_weather(lat: float, lon: float, fallback_temp_f: float = 70.0) -> tuple[float, int, bool]:
    try:
        import httpx 
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        with httpx.Client(timeout=5.0) as client:
            response = client.get(url)
            response.raise_for_status()
            data = response.json()
            current = data.get("current_weather", {})
            temp_c = float(current.get("temperature", 20.0))
            temp_f = (temp_c * 9 / 5) + 32
            weather_code = int(current.get("weathercode", 0))
            return temp_f, weather_code, True 
    except Exception as e:
        logger.warning(f"[Weather] Fallback due to error: {e}")
        return fallback_temp_f, 0, False

IS_MIXED = True
dev = qml.device("default.mixed" if IS_MIXED else "default.qubit", wires=3)
...
if IS_MIXED and weather_mod > 0.5:
    qml.AmplitudeDamping(0.1 * weather_mod, wires=2)

@qml.qnode(dev)
def rgb_quantum_gate(
    r, g, b,
    cpu_usage,
    tempo=120,
    lat=0.0,
    lon=0.0,
    temperature_f=70.0,
    weather_scalar=0.0,
    z0_hist=0.0,
    z1_hist=0.0,
    z2_hist=0.0
):

    r, g, b = [min(1.0, max(0.0, x)) for x in (r, g, b)]
    cpu_scale = max(0.05, cpu_usage)

    tempo_norm = min(1.0, max(0.0, tempo / 200))
    lat_rad = np.deg2rad(lat % 360)
    lon_rad = np.deg2rad(lon % 360)
    temp_norm = min(1.0, max(0.0, (temperature_f - 30) / 100))
    weather_mod = min(1.0, max(0.0, weather_scalar))

    coherence_gain = 1.0 + tempo_norm - weather_mod + 0.3 * (1 - abs(0.5 - temp_norm))

    q_r = r * np.pi * cpu_scale * coherence_gain
    q_g = g * np.pi * cpu_scale * (1.0 - weather_mod + temp_norm)
    q_b = b * np.pi * cpu_scale * (1.0 + weather_mod - temp_norm)

    qml.RX(q_r, wires=0)
    qml.RY(q_g, wires=1)
    qml.RZ(q_b, wires=2)

    qml.PhaseShift(lat_rad * tempo_norm, wires=0)
    qml.PhaseShift(lon_rad * (1 - weather_mod), wires=1)

    qml.CRX(temp_norm * np.pi * coherence_gain, wires=[2, 0])
    qml.CRY(tempo_norm * np.pi, wires=[1, 2])
    qml.CRZ(weather_mod * np.pi, wires=[0, 2])

    entropy_cycle = np.sin(cpu_scale * np.pi * 2)
    qml.RX(entropy_cycle * np.pi * 0.5, wires=1)

    feedback_phase = (z0_hist + z1_hist + z2_hist) * np.pi
    qml.PhaseShift(feedback_phase / 3.0, wires=0)
    qml.PhaseShift(-feedback_phase / 2.0, wires=2)

    if 0.3 < weather_mod < 0.6:
        qml.IsingYY(temp_norm * np.pi * 0.5, wires=[0, 2])

    if weather_mod > 0.5:
        qml.Toffoli(wires=[0, 1, 2])
        qml.AmplitudeDamping(0.1 * weather_mod, wires=2)
    else:
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])

    return (
        qml.expval(qml.PauliZ(0)),
        qml.expval(qml.PauliZ(1)),
        qml.expval(qml.PauliZ(2)),
    )

def get_current_multiversal_time():
    current_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    x, y, z, t = 34, 76, 12, 5633
    return f"X:{x}, Y:{y}, Z:{z}, T:{t}, Time:{current_time}"

def extract_rgb_from_text(text):

    if not text or not isinstance(text, str):
        return (128, 128, 128) 

    blob = TextBlob(text)
    polarity = blob.sentiment.polarity 
    subjectivity = blob.sentiment.subjectivity 
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)
    word_count = len(tokens)
    sentence_count = len(blob.sentences) or 1
    avg_sentence_length = word_count / sentence_count
    adj_count = sum(1 for _, tag in pos_tags if tag.startswith('JJ'))
    adv_count = sum(1 for _, tag in pos_tags if tag.startswith('RB'))
    verb_count = sum(1 for _, tag in pos_tags if tag.startswith('VB'))
    noun_count = sum(1 for _, tag in pos_tags if tag.startswith('NN'))
    punctuation_density = sum(1 for ch in text if ch in ',;:!?') / max(1, word_count)
    valence = polarity 
    arousal = (verb_count + adv_count) / max(1, word_count)
    dominance = (adj_count + 1) / (noun_count + 1) 
    hue_raw = ((1 - valence) * 120 + dominance * 20) % 360
    hue = hue_raw / 360.0
    saturation = min(1.0, max(0.2, 0.25 + 0.4 * arousal + 0.2 * subjectivity + 0.15 * (dominance - 1)))
    brightness = max(0.2, min(1.0,
        0.9 - 0.03 * avg_sentence_length + 0.2 * punctuation_density
    ))

    r, g, b = colorsys.hsv_to_rgb(hue, saturation, brightness)
    return (int(r * 255), int(g * 255), int(b * 255))

def init_db():

    try:
        with sqlite3.connect(DB_NAME) as conn:
            cur = conn.cursor()

            cur.execute("""
                CREATE TABLE IF NOT EXISTS local_responses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    response TEXT,
                    response_time TEXT
                )
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS memory_osmosis (
                    phrase TEXT PRIMARY KEY,
                    score REAL,
                    last_updated TEXT,
                    crystallized INTEGER DEFAULT 0
                )
            """)
            conn.commit()

        interaction_history_class = {
            "class": "InteractionHistory",
            "properties": [
                {"name": "user_id", "dataType": ["string"]},
                {"name": "response", "dataType": ["string"]},
                {"name": "response_time", "dataType": ["string"]}
            ]
        }

        long_term_memory_class = {
            "class": "LongTermMemory",
            "properties": [
                {"name": "phrase", "dataType": ["string"]},
                {"name": "score", "dataType": ["number"]},
                {"name": "crystallized_time", "dataType": ["string"]}
            ]
        }

        existing_classes = client.schema.get().get("classes", [])
        existing_names = {c["class"] for c in existing_classes}

        if "InteractionHistory" not in existing_names:
            client.schema.create_class(interaction_history_class)
        if "LongTermMemory" not in existing_names:
            client.schema.create_class(long_term_memory_class)

    except Exception as e:
        logger.error(f"Error during database/schema initialization: {e}")
        raise

    try:
        with sqlite3.connect(DB_NAME) as conn:
            cur = conn.cursor()
            cur.execute("PRAGMA table_info(memory_osmosis)")
            cols = {row[1] for row in cur.fetchall()}
            if "aging_last" not in cols:
                cur.execute("ALTER TABLE memory_osmosis ADD COLUMN aging_last TEXT")
                conn.commit()
    except Exception as e:
        logger.warning(f"[Aging] Could not add aging_last column (continuing with last_updated): {e}")

def set_encrypted_env_var(varname: str, value: str):
    if value:
        enc = crypto.encrypt(value)
        os.environ[varname] = enc

def get_encrypted_env_var(varname: str) -> str:
    enc = os.environ.get(varname, "")
    if enc:
        try:
            return crypto.decrypt(enc)
        except Exception:
            return ""
    return ""

def save_user_message(user_id, user_input):
    logger.info(f"[save_user_message] user_id={user_id}")
    if not user_input:
        logger.warning("User input is empty.")
        return
    try:
        user_input = sanitize_text(user_input, max_len=4000)
        response_time = get_current_multiversal_time()

        aad_sql  = build_record_aad(user_id=user_id, source="sqlite", table="local_responses")
        aad_weav = build_record_aad(user_id=user_id, source="weaviate", cls="InteractionHistory")

        encrypted_input_sql  = crypto.encrypt(user_input, aad=aad_sql)
        encrypted_input_weav = crypto.encrypt(user_input, aad=aad_weav)

        with sqlite3.connect(DB_NAME) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO local_responses (user_id, response, response_time) VALUES (?, ?, ?)",
                (user_id, encrypted_input_sql, response_time)
            )
            conn.commit()

        plain_embedding = compute_text_embedding(user_input)
        enc_embedding, bucket = fhe_v2.encrypt_embedding(plain_embedding)

        he_pack_id = str(uuid.uuid4())
        he_pack_slot = 0
        he_ctx_version = 1

        try:
            he_enc_group_b64 = ""
            if HE_ENABLED:
                x = np.array(plain_embedding, dtype=np.float64)

                n = np.linalg.norm(x) + 1e-12
                x = x / n
                he_enc_group_b64 = ckks.encrypt_candidate_group_b64(x)
            else:
                logger.info("[HE] Disabled or unavailable; skipping he_enc_group_b64.")
        except Exception as e:
            logger.warning(f"[HE] encrypt candidate failed: {e}")
            he_enc_group_b64 = ""

        dummy_vector = [0.0] * fhe_v2.DIM

        obj = {
            "user_id": user_id,
            "user_message": encrypted_input_weav,
            "response_time": response_time,

            "encrypted_embedding": enc_embedding,
            "embedding_bucket": bucket,

            "he_pack_id": he_pack_id,
            "he_pack_slot": he_pack_slot,
            "he_enc_group_b64": he_enc_group_b64,
            "he_ctx_version": he_ctx_version,
        }
        generated_uuid = generate_uuid5(user_id, user_input)
        response = requests.post(
            'http://127.0.0.1:8079/v1/objects',
            json={
                "class": "InteractionHistory",
                "id": generated_uuid,
                "properties": obj,
                "vector": dummy_vector
            },
            timeout=10
        )
        if response.status_code not in (200, 201):
            logger.error(f"Weaviate POST failed: {response.status_code} {response.text}")
    except Exception as e:
        logger.exception(f"Exception in save_user_message: {e}")

def save_bot_response(bot_id: str, bot_response: str):
    logger.info(f"[save_bot_response] bot_id={bot_id}")
    if not bot_response:
        logger.warning("Bot response is empty.")
        return
    try:
        bot_response = sanitize_text(bot_response, max_len=4000)
        response_time = get_current_multiversal_time()

        aad_sql = build_record_aad(user_id=bot_id, source="sqlite", table="local_responses")
        enc_sql = crypto.encrypt(bot_response, aad=aad_sql)
        with sqlite3.connect(DB_NAME) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO local_responses (user_id, response, response_time) VALUES (?, ?, ?)",
                (bot_id, enc_sql, response_time)
            )
            conn.commit()

        aad_weav = build_record_aad(user_id=bot_id, source="weaviate", cls="InteractionHistory")
        enc_weav = crypto.encrypt(bot_response, aad=aad_weav)

        plain_embedding = compute_text_embedding(bot_response)
        enc_embedding, bucket = fhe_v2.encrypt_embedding(plain_embedding)

        he_pack_id = str(uuid.uuid4())
        he_pack_slot = 0
        he_ctx_version = 1
        try:
            he_enc_group_b64 = ""
            if HE_ENABLED:
                x = np.array(plain_embedding, dtype=np.float64)
                n = np.linalg.norm(x) + 1e-12
                x = x / n
                he_enc_group_b64 = ckks.encrypt_candidate_group_b64(x)
            else:
                logger.info("[HE] Disabled or unavailable; skipping he_enc_group_b64.")
        except Exception as e:
            logger.warning(f"[HE] encrypt candidate failed: {e}")
            he_enc_group_b64 = ""

        dummy_vector = [0.0] * fhe_v2.DIM

        props = {
            "user_id": bot_id,
            "ai_response": enc_weav,
            "response_time": response_time,
            "encrypted_embedding": enc_embedding,
            "embedding_bucket": bucket,
            "he_pack_id": he_pack_id,
            "he_pack_slot": he_pack_slot,
            "he_enc_group_b64": he_enc_group_b64,
            "he_ctx_version": he_ctx_version,
        }
        generated_uuid = generate_uuid5(bot_id, bot_response)
        resp = requests.post(
            "http://127.0.0.1:8079/v1/objects",
            json={
                "class": "InteractionHistory",
                "id": generated_uuid,
                "properties": props,
                "vector": dummy_vector
            },
            timeout=10
        )
        if resp.status_code not in (200, 201):
            logger.error(f"Weaviate POST failed: {resp.status_code} {resp.text}")
    except Exception as e:
        logger.exception(f"Exception in save_bot_response: {e}")


def query_reflections(self, user_id: str, substring: str = None, limit: int = 5):
    try:
        filters = {
            "path": ["user_id"],
            "operator": "Equal",
            "valueString": user_id
        }

        query = self.client.query.get(
            "ReflectionLog",
            ["query", "response", "reasoning_trace", "timestamp"]
        ).with_where(filters).with_limit(limit)

        if substring:
            query = query.with_additional({
                "moduleParams": {
                    "text": {
                        "concepts": [substring],
                        "certainty": 0.65
                    }
                }
            })

        result = query.do()
        return result["data"]["Get"]["ReflectionLog"]
    except Exception as e:
        logger.error(f"[Weaviate Reflection Query Error] {e}")
        return []

def reflect_on_memory(self, user_id: str, topic: str) -> str:
    reflections = self.query_reflections(user_id, substring=topic, limit=3)
    if not reflections:
        return "I could not locate a relevant reflection trace on that topic."

    response = ["[Dyson Node: Reflection Summary]"]
    for r in reflections:
        response.append(f"Query: {r['query']}")
        response.append(f"Response: {r['response']}")
        response.append(f"Reasoning: {r['reasoning_trace']}")
        response.append(f"Timestamp: {r['timestamp']}")
        response.append("────────────────────────────")
    return "\n".join(response)

llm = Llama(
    model_path=model_path,
    mmproj=mmproj_path,
    n_gpu_layers=-1,
    n_ctx=3900,
)

def is_code_like(chunk):
   code_patterns = r'\b(def|class|import|if|else|for|while|return|function|var|let|const|print)\b|[\{\}\(\)=><\+\-\*/]'
   return bool(re.search(code_patterns, chunk))

def determine_token(chunk, memory, max_words_to_check=500):
   combined_chunk = f"{memory} {chunk}"
   if not combined_chunk:
       return "[attention]"

   if is_code_like(combined_chunk):
       return "[code]"

   words = word_tokenize(combined_chunk)[:max_words_to_check]
   tagged_words = pos_tag(words)

   pos_counts = Counter(tag[:2] for _, tag in tagged_words)
   most_common_pos, _ = pos_counts.most_common(1)[0]

   if most_common_pos == 'VB':
       return "[action]"
   elif most_common_pos == 'NN':
       return "[subject]"
   elif most_common_pos in ['JJ', 'RB']:
       return "[description]"
   else:
       return "[general]"

def find_max_overlap(chunk, next_chunk):
   max_overlap = min(len(chunk), 240)
   return next((overlap for overlap in range(max_overlap, 0, -1) if chunk.endswith(next_chunk[:overlap])), 0)

def truncate_text(text, max_words=100):
   return ' '.join(text.split()[:max_words])

def fetch_relevant_info(chunk, client, user_input):

    try:
        if not user_input:
            return ""
        
        query_vec = np.array(compute_text_embedding(user_input), dtype=np.float64)
        rotated = fhe_v2._rotate(query_vec)
        bucket = fhe_v2._simhash_bucket(rotated)

        gql = f"""
        {{
            Get {{
                InteractionHistory(
                    where: {{
                        path: ["embedding_bucket"],
                        operator: Equal,
                        valueString: "{bucket}"
                    }}
                    limit: 64
                    sort: {{path:"response_time", order: desc}}
                ) {{
                    user_message
                    ai_response
                    encrypted_embedding

                    he_pack_id
                    he_pack_slot
                    he_enc_group_b64
                    he_ctx_version
                }}
            }}
        }}
        """
        response = client.query.raw(gql)
        results = (
            response.get('data', {})
                    .get('Get', {})
                    .get('InteractionHistory', [])
        )
    
        if HE_ENABLED and results:
            try:
                q = query_vec.copy()
                n = np.linalg.norm(q) + 1e-12
                q = q / n
                enc_q_b64 = ckks.encrypt_query_b64(q)
                best = None
                best_score = -1e9
                best_raw = None
                for obj in results:
                    enc_group_b64 = obj.get("he_enc_group_b64") or ""
                    if not enc_group_b64:
                        continue
                    enc_scores_b64 = ckks.score_one_group_b64(enc_group_b64, enc_q_b64)
                    score = ckks.decrypt_score_b64(enc_scores_b64)
                    if score > best_score:
                        best_score = score
                        best = obj

                if best is not None:
                    user_msg_raw = try_decrypt(best.get("user_message", ""))
                    ai_resp_raw  = try_decrypt(best.get("ai_response", ""))
                    return f"{user_msg_raw} {ai_resp_raw}"
             
            except Exception as e:
                logger.warning(f"[HE] Scoring failed; falling back. {e}")
    
        best = None
        best_score = -1.0
        with SecureEnclave() as enclave:
            for obj in results:
                enc_emb = obj.get("encrypted_embedding", "")
                if not enc_emb:
                    continue
                score = fhe_v2.enclave_similarity(enc_emb, query_vec, enclave)
                if score > best_score:
                    best_score = score
                    best = obj
        if not best or best_score <= 0:
            return ""
        user_msg_raw = try_decrypt(best.get("user_message", ""))
        ai_resp_raw  = try_decrypt(best.get("ai_response", ""))
        return f"{user_msg_raw} {ai_resp_raw}"
    except Exception as e:
        logger.error(f"[FHE (CKKS) retrieval] failed: {e}")
        return ""

def llama_generate(prompt, weaviate_client=None, user_input=None, temperature=1.0, top_p=0.9):

    MAX_TOKENS_PER_CHUNK   = 2048
    CHUNK_SIZE             = 360
    MAX_OUTPUT_CHUNKS      = 4
    COHERENCE_ENTROPY_MAX  = 0.185
    COHERENCE_SLOPE_LIMIT  = 0.06
    OVERLAP_MAX_WINDOW     = 300
    MEMORY_DRIFT_TOKEN     = "[[⊗DRIFT-QPU-SEGMENT]]"
    QUANTUM_FINGERPRINT    = lambda text: hashlib.sha256(text.encode()).hexdigest()[:12]

    try:
        prompt_chunks = [prompt[i:i + CHUNK_SIZE] for i in range(0, len(prompt), CHUNK_SIZE)]
        responses, memory, last_output = [], "", ""
        entropy_profile, fingerprint_trace = [], []

        for i, chunk in enumerate(prompt_chunks):
            fingerprint = QUANTUM_FINGERPRINT(chunk)
            fingerprint_trace.append(fingerprint)

            relevant_info = fetch_relevant_info(chunk, weaviate_client, user_input)
            composed_chunk = f"{relevant_info}\n{chunk}\n[seg:{i}|hash:{fingerprint}]"
            token = determine_token(composed_chunk, memory)

            drift_ratio     = i / max(1, MAX_OUTPUT_CHUNKS)
            coherence_decay = np.tanh(drift_ratio)
            temp_adjusted   = min(2.0, temperature + coherence_decay)
            top_p_adjusted  = max(0.1, top_p - 0.12 * coherence_decay)

            if i < MAX_OUTPUT_CHUNKS:
    
                output = tokenize_and_generate(
                    composed_chunk,
                    token,
                    max_tokens=MAX_TOKENS_PER_CHUNK,
                    chunk_size=CHUNK_SIZE,
                    temperature=temp_adjusted,
                    top_p=top_p_adjusted
                )

                if output is None:
                    logger.warning(f"[Gamma‑13X] Null output on chunk {i}")
                    continue


                if i > 0 and last_output:
                    overlap = find_max_overlap(last_output, output)
                    output = output[overlap:]


                tail = output[-OVERLAP_MAX_WINDOW:]
                tail_ord = [ord(c) for c in tail if ord(c) < 2000]
                segment_entropy = np.std(tail_ord) / 100.0
                entropy_profile.append(segment_entropy)


                if i >= 2:
                    prev_dist = Counter(entropy_profile[i - 2:i])
                    curr_dist = Counter(entropy_profile[i - 1:i + 1])
                    js_sim = _js_divergence(prev_dist, curr_dist)
                    if js_sim > COHERENCE_SLOPE_LIMIT:
                        logger.warning(f"[Gamma‑13X Exit] JS coherence slope exceeded on chunk {i}")
                        break

                if segment_entropy > COHERENCE_ENTROPY_MAX:
                    logger.warning(f"[Gamma‑13X Abort] Entropy threshold exceeded on chunk {i} ({segment_entropy:.4f})")
                    break

                responses.append(output)
                memory += output
                last_output = output

            else:

                memory += f"\n{MEMORY_DRIFT_TOKEN}[seg={i}|hash={fingerprint}]"

        joined = "".join(responses).strip()
        return joined if joined else None

    except Exception as e:
        logger.error(f"[Gamma‑13X Fault] LLML core error: {e}")
        return None

def tokenize_and_generate(chunk, token, max_tokens, chunk_size, temperature=1.0, top_p=0.9):
    try:
        inputs = llm(
            f"[{token}] {chunk}",
            max_tokens=min(max_tokens, chunk_size),
            temperature=temperature,
            top_p=top_p
        )
        if inputs is None or not isinstance(inputs, dict):
            logger.error(f"Llama model returned invalid output for input: {chunk}")
            return None

        choices = inputs.get('choices', [])
        if not choices or not isinstance(choices[0], dict):
            logger.error("No valid choices in Llama output")
            return None

        return choices[0].get('text', '')
    except Exception as e:
        logger.error(f"Error in tokenize_and_generate: {e}")
        return None

def extract_verbs_and_nouns(text):
    try:
        if not isinstance(text, str):
            raise ValueError("Input must be a string")

        words = word_tokenize(text)
        tagged_words = pos_tag(words)
        verbs_and_nouns = [word for word, tag in tagged_words if tag.startswith('VB') or tag.startswith('NN')]
        return verbs_and_nouns

    except Exception as e:
        print(f"Error in extract_verbs_and_nouns: {e}")
        return []

def try_decrypt(value):
    try:
        return crypto.decrypt(value)
    except Exception as e:
        logger.warning(f"[decryption] Could not decrypt value: {e}")
        return value

class App(customtkinter.CTk):

    @staticmethod
    def _encrypt_field(value: str) -> str:
        try:
            return crypto.encrypt(value if value is not None else "")
        except Exception as e:
            logger.error(f"[encrypt] Failed to encrypt value: {e}")
            return value if value is not None else ""

    @staticmethod
    def _decrypt_field(value: str) -> str:
        if value is None:
            return ""
        try:
            return crypto.decrypt(value)
        except Exception as e:
            logger.warning(f"[decrypt] Could not decrypt value (returning raw): {e}")
            return value

    def __init__(self, user_identifier):
        super().__init__()
        self.user_id = user_identifier
        self.bot_id = "bot"
        self.setup_gui()
        self.response_queue = queue.Queue()
        self.client = weaviate.Client(url=WEAVIATE_ENDPOINT)
        setup_weaviate_schema(self.client)
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.last_z = (0.0, 0.0, 0.0)
        self._policy_lock = threading.RLock()
        self._policy_mtime = None
        self._load_policy_if_needed()
        self.after(AGING_INTERVAL_SECONDS * 1000, self.memory_aging_scheduler)
        self.after(6 * 3600 * 1000, self._schedule_key_mutation)

    def save_coinbase_keys(self):

        try:
            api = (self.coinbase_api_entry.get() or "").strip()
            secret = (self.coinbase_secret_entry.get() or "").strip()
            passphrase = (self.coinbase_pass_entry.get() or "").strip()

            set_encrypted_env_var("COINBASE_API_KEY", api)
            set_encrypted_env_var("COINBASE_API_SECRET", secret)
            set_encrypted_env_var("COINBASE_API_PASSPHRASE", passphrase)

            missing = [
                name for name, val in [
                    ("API key", api),
                    ("secret", secret),
                    ("passphrase", passphrase),
                ] if not val
            ]
            if missing:
                logger.warning("[Settings] Coinbase credentials saved (missing: %s).", ", ".join(missing))
            else:
                logger.info("[Settings] Coinbase API credentials updated (encrypted in env).")
        except Exception as e:
            logger.error("[Settings] Failed saving Coinbase credentials: %s", e)

    def memory_aging_scheduler(self):

        self.run_long_term_memory_aging()

        self.after(AGING_INTERVAL_SECONDS * 1000, self.memory_aging_scheduler)

    def __exit__(self, exc_type, exc_value, traceback):
        self.executor.shutdown(wait=True)

    def _policy_params_path(self):
        return path.join(bundle_dir, "policy_params.json")

    def select_and_generate(self, prompt, weaviate_client=None, user_input=None, temperature=1.0, top_p=0.9, max_tokens=512):
        model_choice = self.model_selector.get()
        if model_choice == "OpenAI GPT-4o":
            return self.openai_generate(prompt, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        else:
            return llama_generate(prompt, weaviate_client, user_input, temperature, top_p)

    def openai_generate(self, prompt, temperature=1.0, top_p=0.95, max_tokens=512):
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or config.get("OPENAI_API_KEY", "")
        if not OPENAI_API_KEY:
            logger.warning("No OpenAI API key found.")
            return "[OpenAI Error] API key not set."
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens
        }
        try:
            resp = httpx.post(url, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return f"[OpenAI Error] {e}"

    def _load_policy_if_needed(self):

        defaults = {
            "temp_w": 0.0,
            "temp_b": 0.0,
            "temp_log_sigma": -0.7,
            "top_w": 0.0,
            "top_b": 0.0,
            "top_log_sigma": -0.7,
            "learning_rate": 0.05
        }

        path = self._policy_params_path()

        with self._policy_lock:
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                mtime = None

            reload_needed = (
                not hasattr(self, "pg_params")
                or (mtime is not None and mtime != self._policy_mtime)
            )

            if reload_needed:
                try:
                    with open(path, "r") as f:
                        data = json.load(f)
                    for key, val in defaults.items():
                        data.setdefault(key, val)
                    self.pg_params = data
                    self._policy_mtime = mtime
                    logger.debug(f"[Policy] Loaded params from {path}: {self.pg_params}")
                except Exception as e:
                    logger.warning(f"[Policy Load Error] could not load {path}: {e}")
                    self.pg_params = defaults.copy()
                    self._policy_mtime = mtime

            if not hasattr(self, "pg_learning_rate"):

                env_lr = os.getenv("PG_LEARNING_RATE")
                if env_lr is not None:
                    try:
                        lr = float(env_lr)
                    except ValueError:
                        logger.warning(f"[Policy] Invalid PG_LEARNING_RATE='{env_lr}', falling back")
                        lr = self.pg_params.get("learning_rate", defaults["learning_rate"])
                else:
                    lr = self.pg_params.get("learning_rate", defaults["learning_rate"])
                self.pg_learning_rate = lr
                logger.debug(f"[Policy] Using learning_rate={self.pg_learning_rate}")

    def _load_policy(self):

        default = {
            "temp_w": 0.0,
            "temp_b": 0.0,
            "temp_log_sigma": -0.7,   
            "top_w": 0.0,
            "top_b": 0.0,
            "top_log_sigma": -0.7
        }
        try:
            with open(self._policy_params_path(), "r") as f:
                data = json.load(f)
                for k, v in default.items():
                    if k not in data:
                        data[k] = v
                self.pg_params = data
        except Exception:
            self.pg_params = default
            self._save_policy()

    def _save_policy(self):
        try:
            with open(self._policy_params_path(), "w") as f:
                json.dump(self.pg_params, f, indent=2)
        except Exception as e:
            logger.error(f"[PG] Failed saving policy params: {e}")

    def _sigmoid(self, x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    def _policy_forward(self, bias_factor: float):

        p = self.pg_params

        t_range = 1.5 - 0.2
        raw_t = p["temp_w"] * bias_factor + p["temp_b"]
        sig_t = self._sigmoid(raw_t)
        mu_t = 0.2 + sig_t * t_range

        p_range = 1.0 - 0.2
        raw_p = p["top_w"] * bias_factor + p["top_b"]
        sig_p = self._sigmoid(raw_p)
        mu_p = 0.2 + sig_p * p_range

        sigma_t = math.exp(p["temp_log_sigma"]) + 1e-4
        sigma_p = math.exp(p["top_log_sigma"]) + 1e-4

        cache = {
            "raw_t": raw_t, "sig_t": sig_t,
            "raw_p": raw_p, "sig_p": sig_p,
            "t_range": t_range, "p_range": p_range
        }
        return mu_t, sigma_t, mu_p, sigma_p, cache

    def _policy_sample(self, bias_factor: float):

        mu_t, sigma_t, mu_p, sigma_p, cache = self._policy_forward(bias_factor)
        t_sample = random.gauss(mu_t, sigma_t)
        p_sample = random.gauss(mu_p, sigma_p)
        t_clip = max(0.2, min(1.5, t_sample))
        p_clip = max(0.2, min(1.0, p_sample))
        log_prob_t = -0.5 * ((t_sample - mu_t) ** 2 / (sigma_t ** 2)) - math.log(sigma_t) - 0.5 * math.log(2 * math.pi)
        log_prob_p = -0.5 * ((p_sample - mu_p) ** 2 / (sigma_p ** 2)) - math.log(sigma_p) - 0.5 * math.log(2 * math.pi)
        log_prob = log_prob_t + log_prob_p

        return {
            "temperature": t_clip,
            "top_p": p_clip,
            "raw_temperature": t_sample,
            "raw_top_p": p_sample,
            "mu_t": mu_t, "sigma_t": sigma_t,
            "mu_p": mu_p, "sigma_p": sigma_p,
            "log_prob": log_prob,
            "cache": cache
        }

    def upsert_position_to_weaviate(self, position: dict, user_id: str, bot_id: str, timestamp: str):
        pos_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{user_id}-{bot_id}-{position['symbol']}"))
        try:
            obj = {
                "user_id": user_id,
                "bot_id": bot_id,
                "symbol": position["symbol"],
                "size": position["size"],
                "position_context": position.get("context", ""),
                "timestamp": timestamp
            }
            self.client.data_object.create(
                class_name="CryptoPosition",
                uuid=pos_uuid,
                data_object=obj
            )
            logger.info(f"[Weaviate] Upserted position: {obj}")
        except Exception as e:
            logger.warning(f"[Weaviate Position Upsert Error] {e}")

    def remove_closed_positions_from_weaviate(self, open_positions: list, user_id: str, bot_id: str):
        try:
            open_syms = {p['symbol'] for p in open_positions}
            q = self.client.query.get(
                "CryptoPosition", ["symbol", "_additional {id}"]
            ).with_where({
                "operator": "And",
                "operands": [
                    {"path": ["user_id"], "operator": "Equal", "valueString": user_id},
                    {"path": ["bot_id"], "operator": "Equal", "valueString": bot_id},
                ]
            }).do()
            objs = q.get('data', {}).get('Get', {}).get('CryptoPosition', [])
            for obj in objs:
                sym = obj['symbol']
                uuid_ = obj['_additional']['id']
                if sym not in open_syms:
                    self.client.data_object.delete(class_name="CryptoPosition", uuid=uuid_)
                    logger.info(f"[Weaviate] Removed closed position {sym}")
        except Exception as e:
            logger.warning(f"[Weaviate Position Remove Error] {e}")

    def _policy_update(self, samples, learning_rate=0.05):

        if not samples:
            return
        avg_reward = sum(s["reward"] for s in samples) / len(samples)

        grads = {k: 0.0 for k in self.pg_params.keys()}

        for s in samples:
            advantage = s["reward"] - avg_reward
            if advantage == 0:
                continue

            mu_t = s["mu_t"]; sigma_t = s["sigma_t"]
            mu_p = s["mu_p"]; sigma_p = s["sigma_p"]
            rt = s["raw_temperature"]; rp = s["raw_top_p"]
            cache = s["cache"]
            bias_factor = s.get("bias_factor", 0.0)
            inv_var_t = 1.0 / (sigma_t ** 2)
            inv_var_p = 1.0 / (sigma_p ** 2)
            diff_t = (rt - mu_t)
            diff_p = (rp - mu_p)
            dlogp_dmu_t = diff_t * inv_var_t
            dlogp_dmu_p = diff_p * inv_var_p
            dlogp_dlogsigma_t = (diff_t ** 2 / (sigma_t ** 2)) - 1.0
            dlogp_dlogsigma_p = (diff_p ** 2 / (sigma_p ** 2)) - 1.0
            sig_t = cache["sig_t"]; t_range = cache["t_range"]
            dsig_t_draw_t = sig_t * (1 - sig_t)
            dmu_t_draw_t = dsig_t_draw_t * t_range
            sig_p = cache["sig_p"]; p_range = cache["p_range"]
            dsig_p_draw_p = sig_p * (1 - sig_p)
            dmu_p_draw_p = dsig_p_draw_p * p_range
            grads["temp_w"] += advantage * dlogp_dmu_t * dmu_t_draw_t * bias_factor
            grads["temp_b"] += advantage * dlogp_dmu_t * dmu_t_draw_t
            grads["temp_log_sigma"] += advantage * dlogp_dlogsigma_t
            grads["top_w"] += advantage * dlogp_dmu_p * dmu_p_draw_p * bias_factor
            grads["top_b"] += advantage * dlogp_dmu_p * dmu_p_draw_p
            grads["top_log_sigma"] += advantage * dlogp_dlogsigma_p

        for k, g in grads.items():
            self.pg_params[k] += learning_rate * g

        self._save_policy()
        logger.info(f"[PG] Updated policy params: {self.pg_params}")

    def retrieve_past_interactions(self, user_input, result_queue):

        try:
            keywords = extract_verbs_and_nouns(user_input)
            concepts_query = ' '.join(keywords)

            user_message, ai_response = self.fetch_relevant_info_internal(concepts_query)

            if user_message and ai_response:
                combo = f"{user_message} {ai_response}"
                summarized_interaction = summarizer.summarize(combo) or combo
                sentiment = TextBlob(summarized_interaction).sentiment.polarity
                processed_interaction = {
                    "user_message": user_message,
                    "ai_response": ai_response,
                    "summarized_interaction": summarized_interaction,
                    "sentiment": sentiment
                }
                result_queue.put([processed_interaction])
            else:
                logger.info("No relevant interactions found for the given user input.")
                result_queue.put([])
        except Exception as e:
            logger.error(f"An error occurred while retrieving interactions: {e}")
            result_queue.put([])

    def _weaviate_find_ltm(self, phrase: str):
        safe_phrase = sanitize_for_graphql_string(phrase, max_len=256)
        gql = f"""
        {{
          Get {{
            LongTermMemory(
              where: {{ path:["phrase"], operator:Equal, valueString:"{safe_phrase}" }}
              limit: 1
            ) {{
              phrase
              score
              crystallized_time
              _additional {{ id }}
            }}
          }}
        }}
        """
        try:
            resp = self.client.query.raw(gql)
            items = resp.get("data", {}).get("Get", {}).get("LongTermMemory", [])
            if not items:
                return None, None, None
            obj = items[0]
            return (
                obj["_additional"]["id"],
                float(obj.get("score", 0.0)),
                obj.get("crystallized_time", "")
            )
        except Exception as e:
            logger.error(f"[Aging] _weaviate_find_ltm failed: {e}")
            return None, None, None

    def _weaviate_update_ltm_score(self, uuid_str: str, new_score: float):
        try:
            self.client.data_object.update(
                class_name="LongTermMemory",
                uuid=uuid_str,
                data_object={"score": new_score}
            )
        except Exception as e:
            logger.error(f"[Aging] update score failed for {uuid_str}: {e}")

    def _weaviate_delete_ltm(self, uuid_str: str):
        try:
            self.client.data_object.delete(
                class_name="LongTermMemory",
                uuid=uuid_str
            )
        except Exception as e:
            logger.error(f"[Aging] delete failed for {uuid_str}: {e}")

    def run_long_term_memory_aging(self):

        try:
            now = datetime.utcnow()
            purged_any = False
            with sqlite3.connect(DB_NAME) as conn:
                cur = conn.cursor()
                try:
                    cur.execute("""SELECT phrase, score,
                                          COALESCE(aging_last, last_updated) AS ts,
                                          crystallized
                                   FROM memory_osmosis
                                   WHERE crystallized=1""")
                except sqlite3.OperationalError:
                    cur.execute("""SELECT phrase, score, last_updated AS ts, crystallized
                                   FROM memory_osmosis
                                   WHERE crystallized=1""")

                rows = cur.fetchall()
                for phrase, score, ts, crystallized in rows:
                    if not ts:
                        continue
                    try:
                        base_dt = datetime.fromisoformat(ts.replace("Z", ""))
                    except Exception:
                        continue
                    delta_days = max(0.0, (now - base_dt).total_seconds() / 86400.0)
                    if delta_days <= 0:
                        continue

                    half_life = AGING_T0_DAYS + AGING_GAMMA_DAYS * math.log(1.0 + max(score, 0.0))
                    if half_life <= 0:
                        continue
                    decay_factor = 0.5 ** (delta_days / half_life)
                    new_score = score * decay_factor

                    uuid_str, _, _ = self._weaviate_find_ltm(phrase)
                    if new_score < AGING_PURGE_THRESHOLD:
                        purged_any = True
                        if uuid_str:
                            self._weaviate_delete_ltm(uuid_str)
                        cur.execute("""UPDATE memory_osmosis
                                       SET crystallized=0, score=?, aging_last=?
                                       WHERE phrase=?""",
                                    (new_score, now.isoformat() + "Z", phrase))
                        logger.info(f"[Aging] Purged crystallized phrase '{phrase}' (decayed to {new_score:.3f}).")
                    else:
                        cur.execute("""UPDATE memory_osmosis
                                       SET score=?, aging_last=?
                                       WHERE phrase=?""",
                                    (new_score, now.isoformat() + "Z", phrase))
                        if uuid_str:
                            self._weaviate_update_ltm_score(uuid_str, new_score)

                conn.commit()
            if purged_any:
                topo_manifold.rebuild()
        except Exception as e:
            logger.error(f"[Aging] run_long_term_memory_aging failed: {e}")

    @staticmethod
    def fetch_coinbase_price(symbol: str = "BTC-USD") -> float | None:
        try:
            url = f"https://api.coinbase.com/v2/prices/{symbol}/spot"
            response = httpx.get(url, timeout=4.0)
            response.raise_for_status()
            data = response.json()
            return float(data['data']['amount'])
        except Exception as e:
            logger.warning(f"[Coinbase] Failed to get price for {symbol}: {e}")
            return None

    @staticmethod
    def parse_trade_response(text: str) -> dict:
        try:
            direction_match = re.search(r"(LONG|SHORT)", text, re.IGNORECASE)
            entry_match     = re.search(r"Entry(?: price)?:?\s*\$?([\d,.]+)", text, re.IGNORECASE)
            tp_match        = re.search(r"(Take Profit|TP):?\s*\$?([\d,.]+)", text, re.IGNORECASE)
            sl_match        = re.search(r"(Stop Loss|SL):?\s*\$?([\d,.]+)", text, re.IGNORECASE)
            conf_match      = re.search(r"Confidence:?\s*([\d]+)%", text, re.IGNORECASE)
            lev_match       = re.search(r"Leverage:?\s*(\d{1,3})x", text, re.IGNORECASE)

            return {
                "direction":  (direction_match.group(1).upper() if direction_match else None),
                "entry":      (float(entry_match.group(1).replace(',', '')) if entry_match else None),
                "tp":         (float(tp_match.group(2).replace(',', ''))   if tp_match   else None),
                "sl":         (float(sl_match.group(2).replace(',', ''))   if sl_match   else None),
                "confidence": (int(conf_match.group(1))                    if conf_match else None),
                "leverage":   (int(lev_match.group(1))                     if lev_match  else None),
            }
        except Exception as e:
            logger.warning(f"[Consensus] Parsing error: {e}")
            return {}

    def multi_agent_consensus(self, prompt, client, user_input, n_agents: int = 5):

        results = []
        for i in range(n_agents):
            temp  = 0.85 + 0.10 * (i % 3) 
            top_p = 0.80 + 0.05 * (i % 2)

            response = self.select_and_generate(
                prompt, client, user_input,
                temperature=temp, top_p=top_p, max_tokens=512
            )
            if response:
                results.append(self.parse_trade_response(response))

        filtered = [r for r in results if r and r.get("direction") is not None]
        consensus = {}
        if not filtered:
            return {}, results

        try:
            consensus["direction"] = mode([r["direction"] for r in filtered if r.get("direction")])
        except Exception:
            consensus["direction"] = filtered[0]["direction"]

 
        for k in ("entry", "tp", "sl", "leverage"):
            vals = [r[k] for r in filtered if r.get(k) is not None]
            consensus[k] = (median(vals) if vals else None)

        confs = [r["confidence"] for r in filtered if r.get("confidence") is not None]
        consensus["confidence"] = int(median(confs)) if confs else None

        return consensus, results

    def cleanup_stale_crypto_live_positions(self, user_id: str, open_positions: list[dict]) -> None:

        try:
            open_syms_sizes = {(str(p['symbol']), float(p['size'])) for p in open_positions}

            q = self.client.query.get(
                "CryptoLivePosition",
                ["symbol", "size", "_additional { id }"]
            ).with_where({
                "path": ["user_id"],
                "operator": "Equal",
                "valueString": user_id
            }).do()

            objs = q.get('data', {}).get('Get', {}).get('CryptoLivePosition', [])
            for obj in objs:
                sym  = str(obj.get("symbol", ""))
                size = float(obj.get("size", 0) or 0)
                wid  = obj.get("_additional", {}).get("id")


                matches_any = any(
                    (sym == osym) and (math.isclose(size, osize, rel_tol=1e-9, abs_tol=1e-9))
                    for (osym, osize) in open_syms_sizes
                )
                if not matches_any and wid:
                    try:
                        self.client.data_object.delete(class_name="CryptoLivePosition", uuid=wid)
                        logger.info(f"[Weaviate] Removed closed live position {sym} ({size})")
                    except Exception:
                        pass
        except Exception as e:
            logger.warning(f"[Weaviate Live Position Cleanup Error] {e}")

    def fetch_crypto_gecko(self, symbol: str = "bitcoin", vs_currency: str = "usd") -> List[Tuple[int, float]]:

        try:
            url = f"https://api.coingecko.com/api/v3/coins/{symbol}/market_chart"
            params = {
                "vs_currency": vs_currency,
                "days": "1",   
                "interval": "minutely"
            }
            response = httpx.get(url, params=params, timeout=5.0)
            response.raise_for_status()

            data = response.json()
            prices = data.get("prices", []) 
        
            return prices[-90:]

        except Exception as e:
            logger.error(f"[fetch_crypto_gecko] Failed to retrieve crypto data: {e}")
            return []

    def get_weather_sync(self, lat, lon):

        try:
            url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
            with httpx.Client(timeout=5.0) as client:
                response = client.get(url)
                response.raise_for_status()
                data = response.json()

            current = data.get("current_weather", {})
            temp_c = float(current.get("temperature", 20.0))
            weather_code = int(current.get("weathercode", 0))
            return temp_c, weather_code
        except Exception as e:
            logger.error(f"[Weather] Fetch failed: {e}")
            return None, None

    def generate_quantum_state(self, rgb=None):
        if rgb is None or not isinstance(rgb, tuple) or len(rgb) != 3:
            rgb = (128, 128, 128)

        try:
            cpu = psutil.cpu_percent(interval=0.3) / 100.0
            cpu = max(cpu, 0.05)

            r, g, b = [min(1.0, max(0.0, c / 255)) for c in rgb]

            try:
                lat = float(self.latitude_entry.get())
                lon = float(self.longitude_entry.get())
            except Exception:
                lat, lon = 0.0, 0.0

            try:
                user_temp_f = float(self.temperature_entry.get() or 70.0)
            except ValueError:
                user_temp_f = 70.0

            temp_f, weather_code, is_live = fetch_live_weather(lat, lon, user_temp_f)

            if weather_code in {1, 2, 3}:
                weather_scalar = 0.3
            elif weather_code >= 61:
                weather_scalar = 0.7
            else:
                weather_scalar = 0.0

            tempo = 120 

            z0_hist, z1_hist, z2_hist = self.last_z

            z0, z1, z2 = rgb_quantum_gate(
                r, g, b,
                cpu_usage=cpu,
                tempo=tempo,
                lat=lat,
                lon=lon,
                temperature_f=temp_f,
                weather_scalar=weather_scalar,
                z0_hist=z0_hist,
                z1_hist=z1_hist,
                z2_hist=z2_hist
            )

            self.last_z = (z0, z1, z2)

            source = "Live" if is_live else "Manual"

            return (
                f"[QuantumGate+Coherence] RGB={rgb} │ CPU={cpu*100:.1f}% │ "
                f"Z=({z0:.3f}, {z1:.3f}, {z2:.3f}) │ "
                f"GPS=({lat:.3f},{lon:.3f}) │ Temp={temp_f:.1f}°F ({source}) │ "
                f"WeatherCode={weather_code}"
            )

        except Exception as e:
            logger.error(f"Error in generate_quantum_state: {e}")
            return "[QuantumGate] error"

    def fetch_relevant_info_internal(self, chunk):
        if self.client:
            safe_chunk = sanitize_for_graphql_string(chunk, max_len=256)
            query = f"""
            {{
                Get {{
                    InteractionHistory(
                        nearText: {{
                            concepts: ["{safe_chunk}"],
                            certainty: 0.7
                        }}
                        limit: 1
                    ) {{
                        user_message
                        ai_response
                        response_time
                    }}
                    LongTermMemory(
                        nearText: {{
                            concepts: ["{safe_chunk}"],
                            certainty: 0.65
                        }}
                        limit: 1
                    ) {{
                        phrase
                        score
                        crystallized_time
                    }}
                }}
            }}
            """
            try:
                response = self.client.query.raw(query)
                data_root = response.get('data', {}).get('Get', {})

                hist_list = data_root.get('InteractionHistory', [])
                if hist_list:
                    interaction = hist_list[0]
                    user_msg_raw = self._decrypt_field(interaction.get('user_message', ''))
                    ai_resp_raw = self._decrypt_field(interaction.get('ai_response', ''))
                    user_msg = sanitize_text(user_msg_raw, max_len=4000)
                    ai_resp = sanitize_text(ai_resp_raw, max_len=4000)
                    return user_msg, ai_resp

                ltm_list = data_root.get('LongTermMemory', [])
                if ltm_list:
                    phrase_obj = ltm_list[0]
                    phrase = sanitize_text(phrase_obj.get('phrase', ''), max_len=400)
                    return phrase, ""

                return "", ""

            except Exception as e:
                logger.error(f"Weaviate query failed: {e}")
                return "", ""
        return "", ""

    def fetch_interactions(self):
        try:
            gql = """
            {
                Get {
                    InteractionHistory(
                        sort: [{path: "response_time", order: desc}],
                        limit: 15
                    ) {
                        user_message
                        ai_response
                        response_time
                    }
                }
            }
            """
            response = self.client.query.raw(gql)
            results = (
                response.get('data', {})
                        .get('Get', {})
                        .get('InteractionHistory', [])
            )
            decrypted = []
            for interaction in results:
                u_raw = self._decrypt_field(interaction.get('user_message', ''))
                a_raw = self._decrypt_field(interaction.get('ai_response', ''))
                decrypted.append({
                    'user_message' : sanitize_text(u_raw, max_len=4000),
                    'ai_response'  : sanitize_text(a_raw, max_len=4000),
                    'response_time': interaction.get('response_time', '')
                })
            return decrypted
        except Exception as e:
            logger.error(f"Error fetching interactions from Weaviate: {e}")
            return []

    def _schedule_key_mutation(self):

        try:
            crypto.self_mutate_key(population=5, noise_sigma=18.0, alpha=1.0, beta=2.5)
        except Exception as e:
            logger.error(f"[SelfMutateKey] periodic failure: {e}")

        self.after(6 * 3600 * 1000, self._schedule_key_mutation)

    def quantum_memory_osmosis(self, user_message: str, ai_response: str):

        try:
            phrases_user = set(self.extract_keywords(user_message))
            phrases_ai = set(self.extract_keywords(ai_response))
            all_phrases = {p.strip().lower() for p in (phrases_user | phrases_ai) if len(p.strip()) >= 3}
            if not all_phrases:
                return

            now_iso = datetime.utcnow().isoformat() + "Z"
            newly_crystallized = False
            with sqlite3.connect(DB_NAME) as conn:
                cur = conn.cursor()
                cur.execute("UPDATE memory_osmosis SET score = score * ?, last_updated = ?",
                            (DECAY_FACTOR, now_iso))

                for phrase in all_phrases:
                    cur.execute("SELECT score, crystallized FROM memory_osmosis WHERE phrase = ?", (phrase,))
                    row = cur.fetchone()
                    if row:
                        score, crystallized = row
                        new_score = score + 1.0
                        cur.execute("UPDATE memory_osmosis SET score=?, last_updated=? WHERE phrase=?",
                                    (new_score, now_iso, phrase))
                    else:
                        new_score = 1.0
                        crystallized = 0
                        cur.execute(
                            "INSERT INTO memory_osmosis (phrase, score, last_updated, crystallized) VALUES (?, ?, ?, 0)",
                            (phrase, new_score, now_iso)
                        )

                    if new_score >= CRYSTALLIZE_THRESHOLD and not crystallized:
                        try:
                            self.client.data_object.create(
                                data_object={
                                    "phrase": phrase,
                                    "score": new_score,
                                    "crystallized_time": now_iso
                                },
                                class_name="LongTermMemory",
                            )
                            cur.execute("UPDATE memory_osmosis SET crystallized=1, aging_last=? WHERE phrase=?",
                                        (now_iso, phrase))
                            newly_crystallized = True
                            logger.info(f"[Osmosis] Crystallized phrase '{phrase}' (score={new_score:.2f}).")
                        except Exception as we:
                            logger.error(f"[Osmosis] Failed to store crystallized phrase in Weaviate: {we}")

                conn.commit()

            if newly_crystallized:
                topo_manifold.rebuild()

        except Exception as e:
            logger.error(f"[Osmosis] Error during quantum memory osmosis: {e}")

    def process_response_and_store_in_weaviate(self, user_message, ai_response):

        try:
            response_blob = TextBlob(ai_response)
            keywords = response_blob.noun_phrases
            sentiment = response_blob.sentiment.polarity
            enhanced_keywords = set()
            for phrase in keywords:
                enhanced_keywords.update(phrase.split())

            interaction_object = {
                "user_message": self._encrypt_field(user_message),
                "ai_response":  self._encrypt_field(ai_response),
                "keywords":     list(enhanced_keywords),
                "sentiment":    sentiment
            }

            interaction_uuid = str(uuid.uuid4())

            self.client.data_object.create(
                data_object=interaction_object,
                class_name="InteractionHistory",
                uuid=interaction_uuid
            )
            logger.info(f"Interaction stored in Weaviate with UUID: {interaction_uuid}")

        except Exception as e:            
            logger.error(f"Error storing interaction in Weaviate: {e}")

    def create_interaction_history_object(self, user_message, ai_response):

        interaction_object = {
            "user_message": self._encrypt_field(user_message),
            "ai_response":  self._encrypt_field(ai_response)
        }

        try:
            object_uuid = str(uuid.uuid4())
            self.client.data_object.create(
                data_object=interaction_object,
                class_name="InteractionHistory",
                uuid=object_uuid
            )
            logger.info(f"Interaction history object created with UUID: {object_uuid}")
        except Exception as e:
            logger.error(f"Error creating interaction history object in Weaviate: {e}")

    def map_keywords_to_weaviate_classes(self, keywords, context):
        try:
            summarized_context = summarizer.summarize(context) or context
        except Exception as e:
            logger.error(f"Error in summarizing context: {e}")
            summarized_context = context

        try:
            sentiment = TextBlob(summarized_context).sentiment
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            sentiment = TextBlob("").sentiment

        positive_class_mappings = {
            "keyword1": "PositiveClassA",
            "keyword2": "PositiveClassB",
        }
        negative_class_mappings = {
            "keyword1": "NegativeClassA",
            "keyword2": "NegativeClassB",
        }
        default_mapping = {
            "keyword1": "NeutralClassA",
            "keyword2": "NeutralClassB",
        }

        if sentiment.polarity > 0:
            mapping = positive_class_mappings
        elif sentiment.polarity < 0:
            mapping = negative_class_mappings
        else:
            mapping = default_mapping
            
        mapped_classes = {}
        for keyword in keywords:
            try:
                if keyword in mapping:
                    mapped_classes[keyword] = mapping[keyword]
            except KeyError as e:
                logger.error(f"Error in mapping keyword '{keyword}': {e}")

        return mapped_classes

    def generate_response(self, user_input: str) -> None:
        try:
            if not user_input:
                logger.error("User input is None or empty.")
                return

            self._load_policy_if_needed()
            user_id, bot_id = self.user_id, self.bot_id
            save_user_message(user_id, user_input)

            use_context  = "[pastcontext]" in user_input.lower()
            show_reflect = "[reflect]" in user_input.lower()
            cleaned_input = sanitize_text(user_input.replace("[pastcontext]", ""), max_len=2048)

            blob = TextBlob(cleaned_input)
            user_polarity = blob.sentiment.polarity
            user_subjectivity = blob.sentiment.subjectivity  

            past_context = ""
            if use_context:
                qres = queue.Queue()
                self.retrieve_past_interactions(cleaned_input, qres)
                interactions = qres.get()
                if interactions:
                    past_context = "\n".join(
                        f"User: {i['user_message']}\nAI:   {i['ai_response']}"
                        for i in interactions
                    )[-1500:]


            try:
                lat = float(self.latitude_entry.get().strip() or "0")
            except Exception:
                lat = 0.0
            try:
                lon = float(self.longitude_entry.get().strip() or "0")
            except Exception:
                lon = 0.0
            try:
                temp_f = float(self.temperature_entry.get().strip() or "72")
            except Exception:
                temp_f = 72.0

            weather = self.weather_entry.get().strip() or "Clear"
            song    = self.last_song_entry.get().strip() or "None"
            chaos, emotive = self.chaos_toggle.get(), self.emotion_toggle.get()
            prices = self.fetch_crypto_gecko("bitcoin", "usd")

            coinbase_price = self.fetch_coinbase_price("BTC-USD")
            if not isinstance(prices, list) or len(prices) < 15 or not coinbase_price:
                self.response_queue.put({'type': 'text', 'data': "[Price Error] Not enough data."})
                return

            delta_price = prices[-1][1] - prices[-15][1]
            trend_type  = "LONG" if delta_price > 0 else "SHORT"
            last_price  = prices[-1][1]
            rgb = extract_rgb_from_text(cleaned_input)
            r, g, b = [c / 255.0 for c in rgb]
            cpu_load = psutil.cpu_percent(interval=0.4) / 100.0
            z0, z1, z2 = rgb_quantum_gate(r, g, b, cpu_load) 
            self.generate_quantum_state(rgb=rgb) 
            self.last_z = (z0, z1, z2)

            bias_factor = (z0 + z1 + z2) / 3.0
            theta   = np.cos((r + g + b) * np.pi / 3)
            entropy = np.std([r, g, b, cpu_load])
            affective_momentum = bias_factor * theta + entropy  

            time_lock = datetime.utcnow().isoformat() + "Z"
            spot_positions  = fetch_coinbase_spot_positions()
            deriv_positions = fetch_coinbase_derivative_positions()
            all_api_positions = [p for p in (spot_positions + deriv_positions) if float(p.get("size", 0)) > 0]
            for pos in all_api_positions:
                uuid_key = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{user_id}-{pos['symbol']}-{pos['size']}"))
                try:
   
                    try:
                        self.client.data_object.delete(uuid=uuid_key, class_name="CryptoLivePosition")
                    except Exception:
                        pass

                    self.client.data_object.create(
                        class_name="CryptoLivePosition",
                        uuid=uuid_key,
                        data_object={
                            "user_id":   user_id,
                            "symbol":    pos["symbol"],
                            "size":      pos["size"],
                            "context":   pos.get("context", ""),
                            "timestamp": time_lock,
                            "type":      "spot" if "wallet" in pos.get("context", "").lower() else "derivative",
                        }
                    )
                except Exception as e:
                    logger.warning(f"[Weaviate Position Log Error] {e}")
            try:
                q = self.client.query.get(
                    "CryptoLivePosition",
                    ["symbol", "size", "_additional { id }"]
                ).with_where({
                    "path": ["user_id"],
                    "operator": "Equal",
                    "valueString": user_id
                }).do()

                existing = q.get('data', {}).get('Get', {}).get('CryptoLivePosition', [])
                api_set = {(str(p['symbol']), float(p['size'])) for p in all_api_positions}

                for obj in existing:
                    sym  = str(obj.get("symbol", ""))
                    size = float(obj.get("size", 0) or 0.0)
                    wid  = obj.get("_additional", {}).get("id")

                    match = any((sym == s) and isclose(size, sz, rel_tol=1e-9, abs_tol=1e-9) for (s, sz) in api_set)
                    if not match and wid:
                        try:
                            self.client.data_object.delete(class_name="CryptoLivePosition", uuid=wid)
                            logger.info(f"[Weaviate] Removed closed live position {sym} ({size})")
                        except Exception:
                            pass
            except Exception as e:
                logger.warning(f"[Weaviate Position Cleanup Error] {e}")

            current_position_context = ""
            user_position_keywords = ("long", "short", "buy", "sell", "@", "tp", "sl", "target", "stop", "position", "open", "entry")
            for kw in user_position_keywords:
                if kw in cleaned_input.lower():
                    current_position_context = cleaned_input
                    break
            if not current_position_context and all_api_positions:
                current_position_context = all_api_positions[0].get("context", "")

            if current_position_context:
                dyson_intel_prompt = f"""
    [dysonframe]
    DYSON INTELLIGENCE NODE | LIVE POSITION ANALYSIS CORE
    Asset: BTC/USD | Interval: 15m | Current Position Scan

    [phase:intelcollection]
    • Market Price (CoinGecko):   ${last_price:,.2f}
    • Market Price (Coinbase):    ${coinbase_price:,.2f}
    • Δ Price (last 15m):         {delta_price:+.2f} USD
    • Detected Trend:             {trend_type}
    • Quantum State:              Z0={z0:.4f}, Z1={z1:.4f}, Z2={z2:.4f}
    • RGB Gate:                   R={r:.3f}, G={g:.3f}, B={b:.3f}
    • Sentiment Polarity:         {user_polarity:.3f}
    • Environmental Factors:      Weather={weather}, Temp={temp_f:.1f}°F
    • Position Context:           {current_position_context}
    • Chaos Mode:                 {'ON' if chaos else 'OFF'}
    • Emotive Mode:               {'ON' if emotive else 'OFF'}
    • Geo:                        ({lat:.4f}, {lon:.4f})
    • Timestamp:                  {time_lock}
    [/phase:intelcollection]

    [phase:predictiontask]
    • Task: Analyze the provided position and live market state.
    • Goal: Output next likely significant move, recommend action, or suggest adjustment if risk is detected.
    • Output Format:
        - Current Position Summary
        - Risk Assessment (Low / Medium / High)
        - Predicted Next Move (e.g., HOLD, INCREASE, REDUCE, CLOSE, REVERSE)
        - Rationale
        - Confidence (%)
        - (Optional) New Target/Stop if appropriate
    • Engine: LLML + Dyson Intelligence Substrate
    [/phase:predictiontask]

    [request]
    Analyze the user's current position and context (including all quantum, market, and sentiment data).
    Give a structured assessment in the output format above.
    Include rationale for action, and if a new TP/SL is recommended, give values.
    Do not invent information not present in the context.
    [/dysonframe]
    """.strip()
                consensus, agent_responses = self.multi_agent_consensus(
                    dyson_intel_prompt, self.client, cleaned_input, n_agents=5
                )
                prompt_used = dyson_intel_prompt
            else:
                dyson_prompt = f"""
    [dysonframe]
    DYSON SPHERE GAMMA-13X | CRYPTO FUTURECASTING CORE
    Asset: BTC/USD | Interval: 15m | Leverage: 20x–50x

    [phase:qubitstabilization]
    • RGB Gate:            R={r:.3f}, G={g:.3f}, B={b:.3f}
    • Quantum Z-State:     Z0={z0:.4f}, Z1={z1:.4f}, Z2={z2:.4f}
    • CPU Entropy:         {entropy:.5f}
    • Bias Factor:         μ={bias_factor:.5f}
    • Sentiment Polarity:  {user_polarity:.3f}
    • Weather:             {weather} @ {temp_f:.1f}°F
    • Song:                {song}
    • Chaos Mode:          {'ON' if chaos else 'OFF'}
    • Emotive Mode:        {'ON' if emotive else 'OFF'}
    • Geolocation:         ({lat:.4f}, {lon:.4f})
    • Timestamp:           {time_lock}
    [/phase:qubitstabilization]

    [phase:marketanalysis]
    • Price (now):         ${last_price:,.2f} (CoinGecko), ${coinbase_price:,.2f} (Coinbase)
    • Price Δ (15m):       {delta_price:+.2f} USD
    • Initial Trend:       {trend_type}
    • Task:                Predict trade direction, entry, target, stop-loss
    • Output:              LONG/SHORT · ENTRY · TP · SL · Confidence
    • Engine:              LLML + Dyson ChronoCrystal Forecast Unit
    [/phase:marketanalysis]

    [request]
    Predict the next 15-minute BTC/USD price direction using the system's QPU state, RGB coherence, CPU entropy,
    sentiment polarity, and environment. Return:
    → Trade direction: LONG or SHORT
    → Entry price
    → Take Profit (TP)
    → Stop Loss (SL)
    → Confidence (%)
    → Leverage (20x-50x recommended)

    Use quantum-aligned trend estimation and future path bifurcation.
    [/dysonframe]
    """.strip()
                consensus, agent_responses = self.multi_agent_consensus(
                    dyson_prompt, self.client, cleaned_input, n_agents=5
                )
                prompt_used = dyson_prompt

            if not consensus or (not consensus.get("direction") and not consensus.get("predicted_next_move")):
                self.response_queue.put({'type': 'text', 'data': '[Dyson QPU: No consensus]'})
                return

            def fmt_money(x):
                return f"${x:.2f}" if isinstance(x, (int, float)) else "N/A"
            def fmt_int(x):
                return f"{int(x)}" if isinstance(x, (int, float)) else "N/A"

            if current_position_context:
                response_text = (
                    f"[Intelligence Consensus Result]\n"
                    f"Current Position Summary: {consensus.get('current_position_summary','')}\n"
                    f"Risk Assessment: {consensus.get('risk_assessment','')}\n"
                    f"Next Move: {consensus.get('predicted_next_move','')}\n"
                    f"Rationale: {consensus.get('rationale','')}\n"
                    f"Confidence: {fmt_int(consensus.get('confidence'))}%\n"
                    f"Target/Stop: {consensus.get('target','') or ''}/{consensus.get('stop','') or ''}\n"
                    f"(Based on {len(agent_responses)} agents)"
                )
            else:
                response_text = (
                    f"[Consensus Result]\n"
                    f"Direction: {consensus.get('direction','N/A')}\n"
                    f"Entry: {fmt_money(consensus.get('entry'))}\n"
                    f"Take Profit: {fmt_money(consensus.get('tp'))}\n"
                    f"Stop Loss: {fmt_money(consensus.get('sl'))}\n"
                    f"Leverage: {fmt_int(consensus.get('leverage'))}x\n"
                    f"Confidence: {fmt_int(consensus.get('confidence'))}%\n"
                    f"(Based on {len(agent_responses)} agents)"
                )

            for idx, ag in enumerate(agent_responses):
                response_text += f"\n\n[Agent {idx+1}] {ag}"

            reasoning_trace = f"""
    [DYSON REFLECTION TRACE]
    • Agents:           {len(agent_responses)}
    • Consensus Dir.:   {consensus.get('direction','')}
    • Confidence:       {consensus.get('confidence','')}
    • Sentiment Pol.:   {user_polarity:.3f}
    • Δ Price (15m):    {delta_price:.2f}
    • Z Coherence:      μ={bias_factor:.4f}
    • RGB Entropy:      {entropy:.4f}
    • Context Used:     {'Yes' if past_context else 'No'}
    """.strip()

            final_output = prompt_used + "\n\n" + response_text
            if show_reflect:
                final_output += "\n\n" + reasoning_trace

            save_bot_response(bot_id, final_output)
            self.response_queue.put({'type': 'text', 'data': final_output})

            try:
                self.quantum_memory_osmosis(cleaned_input, final_output)
            except Exception as e:
                logger.warning(f"[Memory Osmosis Error] {e}")

            try:
                z_state_blob = base64.b64encode(
                    json.dumps({"z0": z0, "z1": z1, "z2": z2}).encode("utf-8")
                ).decode("utf-8")

                self.client.data_object.create(
                    class_name="CryptoTradeLog",
                    uuid=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{user_id}-{time_lock}")),
                    data_object={
                        "type": "prediction",
                        "user_id": user_id,
                        "bot_id": bot_id,
                        "query": cleaned_input,
                        "response": final_output,
                        "reasoning_trace": reasoning_trace,
                        "prompt_snapshot": prompt_used,
                        "z_state": z_state_blob,        
                        "entropy": float(entropy),
                        "bias_factor": float(bias_factor),
                        "temperature": None,
                        "top_p": None,
                        "sentiment_target": float(user_polarity),
                        "timestamp": time_lock,
                        "asset": "BTC/USD"
                    }
                )
            except Exception as e:
                logger.warning(f"[Weaviate Log Error] {e}")

        except Exception as e:
            logger.error(f"[Gamma13X Fatal Error] {e}")
            self.response_queue.put({'type': 'text', 'data': f"[Dyson QPU Error] {e}"})

    def process_generated_response(self, response_text):
        try:
            self.response_queue.put({'type': 'text', 'data': response_text})
        except Exception as e:
            logger.error(f"Error in process_generated_response: {e}")

    def run_async_in_thread(self, coro_func, user_input, result_queue):
        try:
            coro_func(user_input, result_queue)
        except Exception as e:
            logger.error(f"Error running function in thread: {e}")

    def on_submit(self, event=None):
        raw_input = self.input_textbox.get("1.0", tk.END)
        user_input = sanitize_text(raw_input, max_len=4000).strip()
        if user_input:
            self.text_box.insert(tk.END, f"{self.user_id}: {user_input}\n")
            self.input_textbox.delete("1.0", tk.END)
            self.input_textbox.config(height=1)
            self.text_box.see(tk.END)

            self.executor.submit(self.generate_response, user_input)
            self.after(100, self.process_queue)
        return "break"

    def process_queue(self):
        try:
            while True:
                msg = self.response_queue.get_nowait()
                if msg['type'] == 'text':
                    self.text_box.insert(tk.END, f"AI: {msg['data']}\n")
                elif msg['type'] == 'image':
                    self.image_label.configure(image=msg['data'])
                    self.image_label.image = msg['data']
                self.text_box.see(tk.END)
        except queue.Empty:
            self.after(100, self.process_queue)

    def create_object(self, class_name: str, object_data: dict) -> str:

        object_data = {
            k: self._encrypt_field(v) if k in {"user_message", "ai_response"} else v
            for k, v in object_data.items()
        }

        unique_string = f"{object_data.get('time', '')}-{object_data.get('user_message', '')}-{object_data.get('ai_response', '')}"
        object_uuid = uuid.uuid5(uuid.NAMESPACE_URL, unique_string).hex

        try:
            self.client.data_object.create(
                data_object=object_data,
                class_name=class_name,
                uuid=object_uuid
            )
            logger.info(f"Object created with UUID: {object_uuid}")
        except Exception as e:
            logger.error(f"Error creating object in Weaviate: {e}")

        return object_uuid

    def extract_keywords(self, message):
        try:
            blob = TextBlob(message)
            nouns = blob.noun_phrases
            return list(nouns)
        except Exception as e:
            print(f"Error in extract_keywords: {e}")
            return []

    def update_username(self):
        new_username = self.username_entry.get()
        if new_username:
            self.user_id = new_username
            print(f"Username updated to: {self.user_id}")
        else:
            print("Please enter a valid username.")
            
    def _schedule_chart_refresh(self):
        try:
            self.chart_panel.refresh()
        except Exception as e:
            logger.warning(f"[Chart refresh] {e}")
        finally:
            self.after(60_000, self._schedule_chart_refresh)  # every 60s
    def _bind_enter_behavior(self):
        def _on_return(event):
            if event.state & 0x0001:  # Shift
                return  # allow newline
            self.on_submit()
            return "break"
        self.input_textbox.bind("<Return>", _on_return)
        
    def setup_gui(self):
        customtkinter.set_appearance_mode("Dark")
        self.title("Dyson Sphere Quantum Oracle")
        


        self.chart_panel = BTCChartPanel(self, corner_radius=10)
        self.chart_panel.grid(row=0, column=4, rowspan=4, padx=(0, 20), pady=(20, 20), sticky="nsew")

        # Schedule periodic refresh (every 60s) AFTER panel exists
        self.after(60_000, self._schedule_chart_refresh)


        # add this just after the chart:
        self.account_panel = AccountOverviewPanel(self, corner_radius=10)
        self.account_panel.grid(row=4, column=4, padx=(0, 20), pady=(0, 20), sticky="nsew")
        window_width = 1920
        window_height = 1080
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        center_x = int(screen_width / 2 - window_width / 2)
        center_y = int(screen_height / 2 - window_height / 2)
        self.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')

        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)
        self.sidebar_frame = customtkinter.CTkFrame(self, width=350, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=6, sticky="nsew")

        try:
            logo_photo = tk.PhotoImage(file=logo_path)
            self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, image=logo_photo)
            self.logo_label.image = logo_photo
            self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        except Exception as e:
            logger.error(f"Error loading logo image: {e}")
            self.logo_label = customtkinter.CTkLabel(self.sidebar_frame, text="Logo")
            self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.image_label = customtkinter.CTkLabel(self.sidebar_frame)
        self.image_label.grid(row=1, column=0, padx=20, pady=10)
        try:
            placeholder_photo = tk.PhotoImage(width=140, height=140)
            placeholder_photo.put(("gray",), to=(0, 0, 140, 140))
            self.image_label.configure(image=placeholder_photo)
            self.image_label.image = placeholder_photo
        except Exception as e:
            logger.error(f"Error creating placeholder image: {e}")

        self.text_box = customtkinter.CTkTextbox(self, bg_color="black", text_color="white",
            border_width=0, height=360, width=50, font=customtkinter.CTkFont(size=23))
        self.text_box.grid(row=0, column=1, rowspan=3, columnspan=3, padx=(20, 20), pady=(20, 20), sticky="nsew")

        self.input_textbox_frame = customtkinter.CTkFrame(self)
        self.input_textbox_frame.grid(row=3, column=1, columnspan=2, padx=(20, 0), pady=(20, 20), sticky="nsew")
        self.input_textbox_frame.grid_columnconfigure(0, weight=1)
        self.input_textbox_frame.grid_rowconfigure(0, weight=1)

        self.input_textbox = tk.Text(self.input_textbox_frame, font=("Roboto Medium", 12),
            bg=customtkinter.ThemeManager.theme["CTkFrame"]["fg_color"][1 if customtkinter.get_appearance_mode() == "Dark" else 0],
            fg=customtkinter.ThemeManager.theme["CTkLabel"]["text_color"][1 if customtkinter.get_appearance_mode() == "Dark" else 0],
            relief="flat", height=1)
        self.input_textbox.grid(padx=20, pady=20, sticky="nsew")

        self.input_textbox_scrollbar = customtkinter.CTkScrollbar(self.input_textbox_frame, command=self.input_textbox.yview)
        self.input_textbox_scrollbar.grid(row=0, column=1, sticky="ns", pady=5)
        self.input_textbox.configure(yscrollcommand=self.input_textbox_scrollbar.set)

        self.send_button = customtkinter.CTkButton(self, text="Send", command=self.on_submit)
        self.send_button.grid(row=3, column=3, padx=(0, 20), pady=(20, 20), sticky="nsew")
        self.input_textbox.bind('<Return>', self.on_submit)

        self.settings_frame = customtkinter.CTkFrame(self.sidebar_frame, corner_radius=10)
        self.settings_frame.grid(row=3, column=0, padx=20, pady=10, sticky="ew")
        self.username_label = customtkinter.CTkLabel(self.settings_frame, text="Username:")
        self.username_label.grid(row=0, column=0, padx=5, pady=5)
        self.username_entry = customtkinter.CTkEntry(self.settings_frame, width=120, placeholder_text="Enter username")
        self.username_entry.insert(0, "gray00")
        self.username_entry.grid(row=0, column=1, padx=5, pady=5)
        self.update_username_button = customtkinter.CTkButton(self.settings_frame, text="Update", command=self.update_username)
        self.update_username_button.grid(row=0, column=2, padx=5, pady=5)
        self.coinbase_label = customtkinter.CTkLabel(self.settings_frame, text="Coinbase API Key:")
        self.coinbase_label.grid(row=1, column=0, padx=5, pady=5)
        self.coinbase_api_entry = customtkinter.CTkEntry(self.settings_frame, width=120, show="*", placeholder_text="API Key")
        self.coinbase_api_entry.grid(row=1, column=1, padx=5, pady=5)
        api_val = get_encrypted_env_var("COINBASE_API_KEY")
        if api_val: self.coinbase_api_entry.insert(0, api_val)

        self.coinbase_secret_label = customtkinter.CTkLabel(self.settings_frame, text="Coinbase API Secret:")
        self.coinbase_secret_label.grid(row=2, column=0, padx=5, pady=5)
        self.coinbase_secret_entry = customtkinter.CTkEntry(self.settings_frame, width=120, show="*", placeholder_text="API Secret")
        self.coinbase_secret_entry.grid(row=2, column=1, padx=5, pady=5)
        secret_val = get_encrypted_env_var("COINBASE_API_SECRET")
        if secret_val: self.coinbase_secret_entry.insert(0, secret_val)
        self.coinbase_pass_label = customtkinter.CTkLabel(self.settings_frame, text="Coinbase API Passphrase:")
        self.coinbase_pass_label.grid(row=3, column=0, padx=5, pady=5)
        self.coinbase_pass_entry = customtkinter.CTkEntry(self.settings_frame, width=120, show="*", placeholder_text="Passphrase")
        self.coinbase_pass_entry.grid(row=3, column=1, padx=5, pady=5)
        self.save_coinbase_button = customtkinter.CTkButton(self.settings_frame, text="Save Coinbase Keys", command=self.save_coinbase_keys)
        self.save_coinbase_button.grid(row=2, column=2, padx=5, pady=5)
        self.context_frame = customtkinter.CTkFrame(self.sidebar_frame, corner_radius=10)
        self.context_frame.grid(row=4, column=0, padx=20, pady=10, sticky="ew")

        fields = [
            ("Latitude:", "latitude_entry", 0, 0),
            ("Longitude:", "longitude_entry", 0, 2),
            ("Weather:", "weather_entry", 1, 0),
            ("Temperature (°F):", "temperature_entry", 2, 0),
            ("Last Song:", "last_song_entry", 3, 0),
        ]
        for label_text, attr_name, row, col in fields:
            customtkinter.CTkLabel(self.context_frame, text=label_text).grid(row=row, column=col, padx=5, pady=5)
            entry = customtkinter.CTkEntry(self.context_frame, width=200)
            setattr(self, attr_name, entry)
            span = 3 if col == 0 else 1
            entry.grid(row=row, column=col+1, columnspan=span, padx=5, pady=5)

        customtkinter.CTkLabel(self.context_frame, text="Event Type:").grid(row=4, column=0, padx=5, pady=5)
        self.event_type = customtkinter.CTkComboBox(self.context_frame, values=["Lottery", "Sports", "Politics", "Crypto", "Custom"])
        self.event_type.set("Sports")
        self.event_type.grid(row=4, column=1, columnspan=3, padx=5, pady=5)

        self.chaos_toggle = customtkinter.CTkSwitch(self.context_frame, text="Inject Entropy")
        self.chaos_toggle.select()
        self.chaos_toggle.grid(row=5, column=0, columnspan=2, padx=5, pady=5)

        self.model_selector = customtkinter.CTkComboBox(
            self.context_frame,
            values=["Local Llama", "OpenAI GPT-4o"],
            width=180
        )
        self.model_selector.set("Local Llama")
        self.model_selector.grid(row=11, column=0, columnspan=2, padx=5, pady=5)
        self.emotion_toggle = customtkinter.CTkSwitch(self.context_frame, text="Emotional Alignment")
        self.emotion_toggle.select()
        self.emotion_toggle.grid(row=5, column=2, columnspan=2, padx=5, pady=5)

        self.grid_columnconfigure(4, weight=1)

 

        self._bind_enter_behavior()
        game_fields = [
            ("Game Type:", "game_type_entry", "e.g. Football"),
            ("Team Name:", "team_name_entry", "e.g. Clemson Tigers"),
            ("Opponent:", "opponent_entry", "e.g. Notre Dame"),
            ("Game Date:", "game_date_entry", "YYYY-MM-DD"),
        ]
        for idx, (label, attr, placeholder) in enumerate(game_fields):
            customtkinter.CTkLabel(self.context_frame, text=label).grid(row=6 + idx, column=0, padx=5, pady=5)
            entry = customtkinter.CTkEntry(self.context_frame, width=200, placeholder_text=placeholder)
            setattr(self, attr, entry)
            entry.grid(row=6 + idx, column=1, columnspan=3, padx=5, pady=5)

class BTCChartPanel(customtkinter.CTkFrame):
    """
    Chart panel supporting BTC-USD (spot), ETH-USD (spot), ETH-PERP (Coinbase futures).
    Reuses a single Matplotlib Figure/Axes; background thread fetch; EMA ribbon overlay.
    """
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.figure = None
        self.canvas = None
        self.toolbar = None
        self.ax = None
        self.axv = None

        # Controls
        ctrl = customtkinter.CTkFrame(self)
        ctrl.pack(side="top", fill="x", padx=8, pady=6)

        self.market = customtkinter.CTkComboBox(
            ctrl,
            values=["BTC-USD (Spot)", "ETH-USD (Spot)", "ETH-PERP (Futures, Coinbase)"],
            width=220,
            command=lambda _=None: self.refresh()
        )
        self.market.set("BTC-USD (Spot)")
        self.market.pack(side="left", padx=6)

        self.interval = customtkinter.CTkComboBox(
            ctrl, values=["1","5","15","60","360","1440"], width=90,
            command=lambda _=None: self.refresh()
        )
        self.interval.set("15")
        self.interval.pack(side="left", padx=6)

        self.refresh_btn = customtkinter.CTkButton(ctrl, text="Refresh", command=self.refresh)
        self.refresh_btn.pack(side="left", padx=6)

        # Figure area
        self.fig_container = customtkinter.CTkFrame(self)
        self.fig_container.pack(side="top", fill="both", expand=True)

        # Build single Figure/Axes once
        self.figure = Figure(figsize=(10, 6), dpi=100)
        gs = GridSpec(nrows=2, ncols=1, height_ratios=[3, 1], figure=self.figure)
        self.ax = self.figure.add_subplot(gs[0])
        self.axv = self.figure.add_subplot(gs[1], sharex=self.ax)

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.fig_container)
        self.canvas.draw_idle()
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

        try:
            tb_frame = customtkinter.CTkFrame(self.fig_container)
            tb_frame.pack(side="bottom", fill="x")
            self.toolbar = NavigationToolbar2Tk(self.canvas, tb_frame)
            self.toolbar.update()
        except Exception:
            pass

        self.refresh()  # initial draw

    def _market_key(self) -> str:
        label = self.market.get()
        if label.startswith("BTC-USD"): return "BTC-USD"
        if label.startswith("ETH-USD"): return "ETH-USD"
        if label.startswith("ETH-PERP"): return "ETH-PERP"
        return "BTC-USD"

    def _draw(self, df: pd.DataFrame):
        # Style
        mc = mpf.make_marketcolors(up="#26a69a", down="#ef5350", wick="inherit", edge="inherit", volume="in")
        style = mpf.make_mpf_style(base_mpf_style="nightclouds", marketcolors=mc,
                                   facecolor="#101418", edgecolor="#101418", gridcolor="#2a2f36")

        self.ax.clear(); self.axv.clear()

        if df.empty:
            self.ax.text(0.5, 0.5, "No data", color="w", ha="center", va="center", transform=self.ax.transAxes)
            self.ax.set_title(f"{self.market.get()} · {self.interval.get()}m", color="w", pad=8)
            self.canvas.draw_idle()
            return

        # EMA ribbon
        spans = (8, 13, 21, 34, 55, 89)
        addplots = [mpf.make_addplot(df[f"EMA{s}"], ax=self.ax, color="#7aa2f7", width=1, alpha=0.8) for s in spans]
        addplots.append(
            mpf.make_addplot(
                df["RIBBON_LOW"], ax=self.ax, color="none",
                fill_between=dict(y1=df["RIBBON_LOW"], y2=df["RIBBON_HIGH"], alpha=0.12, color="#7aa2f7")
            )
        )

        mpf.plot(df, type="candle", style=style,
                 ax=self.ax, volume=self.axv,
                 addplot=addplots, xrotation=15, datetime_format="%m-%d %H:%M")

        self.ax.set_title(f"{self.market.get()} · {self.interval.get()}m", color="w", pad=8)
        self.canvas.draw_idle()

    def refresh(self):
        """
        Called when the user triggers a manual refresh.
        Launches background thread to fetch data without blocking UI.
        """
        try:
            mins = int(self.interval.get())
            market = self._market_key()
            threading.Thread(
                target=self._refresh_bg,
                args=(market, mins),
                daemon=True
            ).start()
        except Exception as e:
            print(f"[refresh] Failed to start background refresh: {e}")
            # Optional: show error in status label

    def _refresh_bg(self, market: str, mins: int):
        """
        Background worker to fetch chart data.
        On success, schedules UI thread to update the chart.
        """
        try:
            df = fetch_ohlc_with_fallback(market=market, minutes=mins, bars=300)
            if df is not None and not df.empty:
                df = add_ema_ribbon(df)
            else:
                df = pd.DataFrame()  # fallback to empty

            # Safely update UI from the main thread
            self.after(0, lambda: self._draw(df))
        except Exception as e:
            print(f"[refresh_bg] Error fetching chart data: {e}")
            self.after(0, lambda: self._draw(pd.DataFrame()))
            
class AccountOverviewPanel(customtkinter.CTkFrame):
    """
    Small dashboard: total account value, spot USD, spot crypto USD value,
    futures USD, open futures contracts & positions, plus a short spot holdings list.
    """
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)

        self.title_lbl  = customtkinter.CTkLabel(self, text="Account Overview", font=customtkinter.CTkFont(size=18, weight="bold"))
        self.total_lbl  = customtkinter.CTkLabel(self, text="Total Value: —", anchor="w")
        self.spot_usd   = customtkinter.CTkLabel(self, text="Spot USD: —", anchor="w")
        self.spot_val   = customtkinter.CTkLabel(self, text="Spot Crypto (USD): —", anchor="w")
        self.fut_usd    = customtkinter.CTkLabel(self, text="Futures USD: —", anchor="w")
        self.fut_pnl    = customtkinter.CTkLabel(self, text="Futures Unrealized PnL: —", anchor="w")
        self.fut_open   = customtkinter.CTkLabel(self, text="Open Futures: — positions / — contracts", anchor="w")
        self.holdings_t = customtkinter.CTkLabel(self, text="Spot Holdings:", anchor="w")
        self.holdings   = customtkinter.CTkTextbox(self, height=90)
        self.holdings.configure(state="disabled")

        pad = dict(padx=8, pady=3, sticky="w")
        self.title_lbl.grid(row=0, column=0, padx=8, pady=(8,4), sticky="w")
        self.total_lbl.grid(row=1, column=0, **pad)
        self.spot_usd.grid(row=2, column=0, **pad)
        self.spot_val.grid(row=3, column=0, **pad)
        self.fut_usd.grid(row=4, column=0, **pad)
        self.fut_pnl.grid(row=5, column=0, **pad)
        self.fut_open.grid(row=6, column=0, **pad)
        self.holdings_t.grid(row=7, column=0, **pad)
        self.holdings.grid(row=8, column=0, padx=8, pady=(0,8), sticky="nsew")
        self.grid_rowconfigure(8, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # initial draw + schedule
        self.refresh()
        self.after(60_000, self._schedule)

    @staticmethod
    def _fmt_money(x: float) -> str:
        try:
            return f"${x:,.2f}"
        except Exception:
            return "—"

    @staticmethod
    def _fmt_int(x: float | int) -> str:
        try:
            return f"{int(x)}"
        except Exception:
            return "—"

    def refresh(self):
        try:
            ov = compute_account_overview()
            self.total_lbl.configure(text=f"Total Value: {self._fmt_money(ov['total_account_value_usd'])}")
            self.spot_usd.configure(text=f"Spot USD: {self._fmt_money(ov['spot_usd_cash'])}")
            self.spot_val.configure(text=f"Spot Crypto (USD): {self._fmt_money(ov['spot_crypto_value_usd'])}")
            self.fut_usd.configure(text=f"Futures USD: {self._fmt_money(ov['futures_usd_cash'])}")
            self.fut_pnl.configure(text=f"Futures Unrealized PnL: {self._fmt_money(ov['futures_unrealized_pnl'])}")
            self.fut_open.configure(
                text=f"Open Futures: {self._fmt_int(ov['futures_positions_count'])} positions / "
                     f"{self._fmt_int(ov['futures_contracts_open'])} contracts"
            )

            # Show top 6 spot holdings (non-USD)
            positions = sorted(ov["spot_nonusd_positions"], key=lambda t: t[0])[:6]
            self.holdings.configure(state="normal")
            self.holdings.delete("1.0", tk.END)
            if positions:
                for sym, qty in positions:
                    self.holdings.insert(tk.END, f"{sym}: {qty:g}\n")
            else:
                self.holdings.insert(tk.END, "(none)")
            self.holdings.configure(state="disabled")
        except Exception as e:
            logger.warning(f"[Account Panel] refresh failed: {e}")

    def _schedule(self):
        try:
            self.refresh()
        finally:
            self.after(60_000, self._schedule)


if __name__ == "__main__":
    try:
        user_id = "gray00"
        app_gui = App(user_id)
        init_db()
        app_gui.mainloop()
    except Exception as e:
        logger.error(f"Application error: {e}")
