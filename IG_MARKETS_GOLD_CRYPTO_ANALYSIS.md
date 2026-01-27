# IG Markets - Gold & Crypto Instruments Analysis

## Executive Summary

Investigation of Gold and Solana availability on IG Markets for streaming integration.

**Key Finding:** ⚠️ **While streaming data is available for these instruments, they are NOT tradeable via OTC/API** (`otcTradeable: false`).

---

## Gold Instruments

### ✅ Recommended for Streaming

| Epic | Name | Price | Streaming | OTC Tradeable | Status |
|------|------|-------|-----------|---------------|--------|
| **CS.D.CFEGOLD.CEE.IP** | Spot Gold ($1) | $5,075.63 | ✅ Yes | ❌ No | TRADEABLE |
| **CS.D.CFEGOLD.CNE.IP** | Gold (€1 Mini) | $5,075.65 | ✅ Yes | ❌ No | TRADEABLE |

### 🔶 Weekend Gold (Limited Use)

| Epic | Name | Price | Streaming | Status |
|------|------|-------|-----------|--------|
| IX.D.SUNGOLD.CEE.IP | Weekend Spot Gold ($1) | $5,073.10 | ✅ Yes | EDITS_ONLY |
| IX.D.SUNGOLD.CFD.IP | Weekend Spot Gold ($100) | $5,073.10 | ✅ Yes | EDITS_ONLY |

**Recommendation:** **CS.D.CFEGOLD.CEE.IP** (Spot Gold $1)
- Most liquid
- Standard contract size
- Full trading hours (except weekends)

---

## Cryptocurrency - Solana

### ✅ Recommended for Streaming

| Epic | Name | Price | Streaming | OTC Tradeable | Status |
|------|------|-------|-----------|---------------|--------|
| **CS.D.SOLUSD.CFD.IP** | Solana ($1) | $124.65 | ✅ Yes | ❌ No | TRADEABLE |

### ❌ Not Suitable (No Streaming)

| Epic | Name | Streaming |
|------|------|-----------|
| UD.D.SOLZUS.CASH.IP | Solana ETF | ❌ No |
| UB.D.HSDTUS.CASH.IP | Solana Company | ❌ No |
| EG.D.ASOLNA.CASH.IP | 21Shares Solana Staking ETP | ✅ Yes (but EDITS_ONLY) |

**Recommendation:** **CS.D.SOLUSD.CFD.IP** (Solana $1)
- Only crypto-direct Solana instrument with streaming
- 24/7 trading
- Tight spreads ($0.70)

---

## Other Crypto Options (Bonus)

### Bitcoin (Highly Liquid)

| Epic | Name | Price | Streaming | Status |
|------|------|-------|-----------|--------|
| **CS.D.BITCOIN.CEEM.IP** | Bitcoin ($0.1) | $88,163 | ✅ Yes | TRADEABLE |
| **CS.D.BITCOIN.CEE.IP** | Bitcoin ($1) | $88,163 | ✅ Yes | TRADEABLE |
| CS.D.BITCOIN.CNEM.IP | Bitcoin (€0.1) | $88,163 | ✅ Yes | TRADEABLE |
| CS.D.BITCOIN.CNE.IP | Bitcoin (€1) | $88,163 | ✅ Yes | TRADEABLE |

### Other Cryptos with Streaming

| Epic | Name | Price | Streaming |
|------|------|-------|-----------|
| CS.D.ETHUSD.CFD.IP | Ethereum ($1) | Available | ✅ Yes |
| CS.D.BCHUSD.CFD.IP | Bitcoin Cash ($1) | $579.44 | ✅ Yes |
| CS.D.ETHXBT.CFD.IP | Ether/Bitcoin ($1) | 0.03301 | ✅ Yes |

---

## ⚠️ CRITICAL LIMITATION: Not Tradeable via API

**All Gold and Crypto instruments have `otcTradeable: false`**

This means:
- ✅ **You CAN stream live price data** (1-minute candles via Lightstreamer)
- ❌ **You CANNOT place trades via API/OTC** (fastapi-dev won't work)
- ⚠️ Trades must be placed manually through IG web platform

### Why This Matters:

1. **Streaming Only**: Good for portfolio tracking, market analysis, correlation studies
2. **No Automated Trading**: Cannot integrate with your trading-ui or task-worker scanner
3. **Manual Execution Required**: If you want to trade based on signals, must execute manually

---

## Implementation Guide

### Adding to Streaming Service (Data Collection Only)

**File:** `stream-app/config.py`

Add to `ACTIVE_EPICS`:
```python
ACTIVE_EPICS = [
    "CS.D.GBPUSD.MINI.IP",
    "CS.D.USDJPY.MINI.IP",
    "CS.D.AUDUSD.MINI.IP",
    "CS.D.USDCAD.MINI.IP",
    "CS.D.EURJPY.MINI.IP",
    "CS.D.AUDJPY.MINI.IP",
    "CS.D.NZDUSD.MINI.IP",
    "CS.D.USDCHF.MINI.IP",
    "CS.D.EURUSD.CEEM.IP",
    "CS.D.EURGBP.MINI.IP",
    "CS.D.CFEGOLD.CEE.IP",      # ✨ Gold
    "CS.D.SOLUSD.CFD.IP",       # ✨ Solana
    "CS.D.BITCOIN.CEE.IP"       # ✨ Bitcoin (optional)
]
```

Then restart:
```bash
docker restart fastapi-stream
```

---

## Resource Impact

Adding 2-3 crypto/commodity instruments:

| Metric | Current (10 pairs) | +3 instruments | Change |
|--------|-------------------|----------------|--------|
| CPU | 1.35% | ~1.75% | +0.4% |
| RAM | 85MB | ~110MB | +25MB |
| Network | 13.5KB/min | 17.5KB/min | +4KB/min |
| DB Growth | 2,880 candles/day | 3,744 candles/day | +864/day |
| IG Subscription | 25% (10/40) | 32.5% (13/40) | +7.5% |

**Verdict:** ✅ Negligible impact - plenty of headroom

---

## Use Cases

### ✅ Good Use Cases:
1. **Portfolio Diversification Tracking**: Monitor gold/crypto alongside forex
2. **Correlation Analysis**: Study relationship between crypto/gold and forex pairs
3. **Market Sentiment**: Gold as risk-off indicator, crypto for risk appetite
4. **Historical Data**: Build backtesting database for multi-asset strategies

### ❌ Not Suitable For:
1. Automated trading (not API tradeable)
2. Strategy scanner integration
3. Order execution via fastapi-dev
4. Trailing stop management

---

## Recommendation

### For Streaming Only (Data Collection):
✅ **YES** - Add these epics:
- `CS.D.CFEGOLD.CEE.IP` (Gold)
- `CS.D.SOLUSD.CFD.IP` (Solana)
- `CS.D.BITCOIN.CEE.IP` (Bitcoin) - optional but recommended

**Benefits:**
- Minimal resource cost
- Rich multi-asset dataset
- Correlation analysis capabilities
- Future-proofing if IG enables API trading

### For Automated Trading:
❌ **NO** - Not currently possible via IG Markets API

**Alternative:**
- Use streaming for analysis/signals
- Execute trades manually on IG platform
- Or find alternative broker with crypto/commodity API access

---

## Testing Tool

A search tool has been created: `worker/app/search_ig_markets.py`

Usage:
```bash
docker exec task-worker python /app/search_ig_markets.py <search_term>
```

Examples:
```bash
docker exec task-worker python /app/search_ig_markets.py gold
docker exec task-worker python /app/search_ig_markets.py ethereum
docker exec task-worker python /app/search_ig_markets.py oil
```

---

## Next Steps

**If you want to proceed with streaming (data collection only):**

1. Decide which instruments to add (gold, solana, bitcoin, etc.)
2. Update `stream-app/config.py` with epic codes
3. Restart `fastapi-stream` container
4. Monitor logs for successful subscription
5. Verify data flowing to `ig_candles` table

**If you need automated trading:**
- Investigate alternative brokers with commodity/crypto API support
- Or accept manual execution for these instruments
- Keep streaming for analysis, trade manually

---

Generated: 2026-01-26 19:45 CET
Script: `worker/app/search_ig_markets.py`
