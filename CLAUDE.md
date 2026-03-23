# CNN-LSTM MES Scalping System

## Project Overview
Clean-slate deep learning scalping system for MES (Micro E-mini S&P 500 Futures). Replaces rule-based strategies with a unified CNN-LSTM architecture for entry and exit, gated by an HMM regime detector.

- **Instrument**: MES (Micro E-mini S&P 500 Futures, $1.25/tick)
- **Strategy**: L1-first, L2-later data approach using CNN-LSTM (DeepLOB-inspired)
- **Language**: Python 3.11+ / asyncio
- **Deep Learning**: PyTorch 2.x, TorchScript for production
- **Data Storage**: Parquet (columnar, partitioned by date)
- **Market Data L1**: Databento BBO feed
- **Market Data L2**: EdgeClear + Rithmic (MBO-level, future phase)
- **Execution**: Tradovate API initially ($0.70 RT), EdgeClear later ($0.44 RT)
- **Infrastructure**: Vultr Chicago VPS (co-located near CME Aurora)

## Architecture Pipeline
1. **Market Data Ingestion** — L1 BBO quotes + trades (Databento) → later L2 full depth (EdgeClear+Rithmic)
2. **Feature Pipeline** — Stationary, z-score normalized features at 1-second resolution
3. **HMM Regime Classifier** — 3-state: trending / mean-reverting / choppy
4. **CNN-LSTM Entry Model** — Directional probability + expected return magnitude
5. **Filter Gate** — Rule-based safety (spread, VPIN, time-of-day, news blackout)
6. **CNN-LSTM Exit Model** — Remaining Favorable Excursion (RFE) + adverse excursion probability
7. **Circuit Breakers** — 8-tick max loss, 5-min max hold, daily loss limit

## L1 Features
- Order Flow Imbalance (OFI)
- VPIN (Volume-synchronized probability of informed trading)
- Micro-Price
- Cumulative Volume Delta
- Bid-Ask Spread
- Trade Arrival Rate
- Volume Profile (POC, VAH, VAL)
- Hurst Exponent
- Realized Volatility
- Lee-Ready Trade Classification
- Trade Size Distribution
- Return Autocorrelation

## L2 Features (Phase 9)
- Multi-Level OFI (MLOFI) across 10 levels
- Book Depth Ratio
- Order Book Slope
- Queue Position Estimation
- Depth-Weighted Imbalance
- Absorption Ratio

## Model Architecture
- **Input**: `[batch, 100, num_features]` — 100 one-second timesteps
- **Backbone**: Conv1D blocks → Inception module → LSTM (DeepLOB-inspired)
- **Entry heads**: directional probability (sigmoid) + expected return magnitude (linear)
- **Exit heads**: Remaining Favorable Excursion (RFE) + adverse excursion probability
- **Entry threshold**: directional prob > 0.65, expected return > 1.5 ticks, HMM confidence > 0.5
- **Exit threshold**: RFE < 0.5 ticks OR adverse excursion prob > 0.7 OR circuit breaker
- **Regularization**: dropout 0.2-0.3, label smoothing, walk-forward validation

## Performance Targets
| Metric              | Target       |
|---------------------|--------------|
| Profit Factor       | > 1.5        |
| Sharpe Ratio (daily)| > 1.5        |
| Win Rate            | > 52%        |
| Avg Win / Avg Loss  | > 1.3        |
| Max Drawdown        | < $500       |
| Trades / Day        | 5-20         |
| Trade Duration      | 30s - 3min   |

## Development Phases
1. L1 Data Infrastructure (2-3 weeks)
2. L1 Feature Engineering (3-4 weeks)
3. HMM Regime Detector (2-3 weeks)
4. CNN-LSTM Entry Model (4-6 weeks)
5. CNN-LSTM Exit Model (3-4 weeks)
6. Filter Gate & Circuit Breakers (1-2 weeks)
7. Execution Optimization & Slippage Reduction (2-3 weeks) — **highest ROI phase**
8. Integrated Backtesting Framework (3-4 weeks)
9. L2 Data Upgrade & Retrain (4-5 weeks)
10. Retraining & Drift Detection (2-3 weeks)
11. Paper Trading & Live Integration (4-6 weeks)
12. Optimization & Iteration (ongoing)

## Data Layout
```
data/
├── l1/                    # Level 1 BBO + trades (Parquet)
│   ├── year=2025/
│   └── year=2026/
├── l2/                    # Level 2 full depth (Databento MBP-10, zstd compressed)
│   └── GLBX-20260307-.../  # MES Sept-Nov 2025
├── parquet/               # Historical data partitioned by year
│   ├── year=2011/
│   └── ... through year=2026/
```

## Research Foundations
- **DeepLOB** (Zhang et al., 2019) — CNN-LSTM on limit order books, 83.4% F1
- **Deep Order Flow Imbalance** (Kolm, Turiel & Westray, 2021) — LSTM on order flow beats raw LOB
- **HMM-LSTM Hybrid** (Kemper, 2025) — 50% volatility reduction using entropy-weighted Bayesian model averaging
- **Order Flow Image Representation** (2024) — CNN on order flow visualizations

## Technology Stack
- **HMM**: hmmlearn + custom BOCPD
- **Monitoring**: Grafana + Prometheus
- **Model Registry**: MLflow or DVC
- **Model size**: ~60K params, single-digit ms GPU inference, <50ms CPU

---

# Backtesting Skepticism Checklist

Every backtest result should be challenged against these failure modes. If a backtest looks too good, it probably is. Apply these checks ruthlessly before trusting any result.

## Look-Ahead Bias
- **Feature calculation**: Ensure no feature uses future data. Z-score normalization must use only past data (Welford's online algorithm, not batch normalization over the full dataset).
- **Label construction**: If labeling trades (e.g., "price went up 2 ticks in next 30s"), make sure the label window doesn't overlap with the next trade's entry window.
- **Train/test contamination**: Walk-forward only. Never shuffle time series data. Never let training data appear after test data chronologically.
- **HMM regime labels**: If the HMM is fit on the full dataset, regime labels leak future information. Fit incrementally or on training data only.

## Survivorship & Selection Bias
- **Strategy selection**: If you tested 50 parameter combos and picked the best one, the reported Sharpe is inflated. Use Monte Carlo permutation tests (10K shuffles, beat the 95th percentile).
- **Time period cherry-picking**: A strategy that works Sept-Nov 2025 may fail in other regimes. Validate across multiple market conditions (trending, choppy, high-vol, low-vol).
- **Feature selection**: If features were selected based on test-set performance, results are overstated. Use walk-forward feature importance.

## Execution Realism
- **Fill assumptions**: Limit orders do NOT always fill. A backtest that assumes fills at the limit price whenever price touches it will massively overstate performance. Model fill probability based on queue position and historical BBO refresh rates.
- **Slippage**: Model at least 4 scenarios (0.25, 0.5, 0.75, 1.0 tick). At $1.25/tick and 1,500+ trades, even 0.25 ticks of unmodeled slippage = ~$480 of phantom profit.
- **Spread crossing**: If your entry signal fires when spread > 1 tick, you're paying more to enter. Filter or model this.
- **Latency**: Add realistic latency (50ms minimum for order submission + exchange matching). Signals that depend on sub-10ms reaction times are not achievable on a VPS.
- **Partial fills**: Especially relevant for limit orders in thin markets. Don't assume you always get the full contract.
- **Market impact**: Less relevant for 1-lot MES, but still worth noting if scaling up.

## Commission Drag
- **Always include commissions**: $0.70 RT (Tradovate) or $0.44 RT (EdgeClear). A strategy with 1,500 trades/month and $0.70 RT commission = $1,050/month in commissions alone.
- **Break-even calculation**: At $0.70 RT, you need > 0.56 ticks average profit per trade just to break even. At $0.44 RT, you need > 0.35 ticks.

## Overfitting Red Flags
- **Suspiciously high Sharpe**: Daily Sharpe > 3.0 on a scalping strategy is almost certainly overfit. Be suspicious above 2.0.
- **Perfect regime alignment**: If the HMM magically avoids every drawdown, it's likely fit on the test data.
- **Too many parameters**: More features and hyperparameters = more degrees of freedom to overfit. Track effective degrees of freedom relative to sample size.
- **Fragile thresholds**: If changing entry threshold from 0.65 to 0.63 dramatically changes results, the edge is probably noise. Robust strategies degrade gracefully.
- **Drawdown-free equity curves**: Real strategies have drawdowns. A smooth equity curve in backtest means something is wrong.
- **Win rate > 65%**: For a scalping strategy with 1:1.3 risk/reward, a sustained 65%+ win rate is suspicious. Achievable win rates are typically 52-60%.

## Data Quality
- **Gaps and staleness**: Market data feeds have gaps. If your backtest silently forward-fills stale quotes, features like OFI and VPIN become meaningless during those periods.
- **Corporate actions / rollovers**: MES is a futures contract that rolls quarterly. Make sure continuous contract construction doesn't introduce artificial price jumps.
- **Timestamp alignment**: L1 quote timestamps and trade timestamps may not be perfectly synchronized. Sub-second features are sensitive to this.
- **Holiday / half-day sessions**: Reduced liquidity sessions behave differently. Either exclude them or model them separately.

## Statistical Validation
- **Monte Carlo permutation test**: Shuffle trade labels 10K times. Your strategy's Sharpe must beat the 95th percentile of random shuffles to claim statistical significance.
- **Out-of-sample degradation**: Expect 30-50% performance degradation from in-sample to out-of-sample. If degradation is less than 20%, you may still be overfit (the "test set" may not be truly independent).
- **Number of trades**: With < 200 trades, almost no statistic is reliable. Target 500+ trades for any meaningful conclusion.
- **Autocorrelation of returns**: If trade returns are autocorrelated, standard Sharpe calculations are biased. Check and adjust.

---

# Development Guidelines

## Code Standards
- Python 3.11+ with type hints on all function signatures
- Use `asyncio` for all I/O-bound operations (data feeds, order submission)
- Parquet for all data storage — partition by date, use pyarrow
- All features must pass Augmented Dickey-Fuller stationarity test before use
- Use Welford's online algorithm for running statistics (mean, variance) — never batch normalize over future data
- **Never use Python loops over data** — always use NumPy vectorized operations. Replace `for` loops over arrays with reshape/broadcasting/advanced indexing. Use `np.cumsum`, `np.where`, `np.lib.stride_tricks` instead of manual iteration. This applies to feature computation, R/S calculations, window extraction, and label construction. The only acceptable loops are over a small fixed set (e.g., iterating over 3-4 chunk sizes).

## Project Conventions
- Data lives in `data/` (gitignored) — never commit market data
- Features should be computed at 1-second resolution
- All prices in ticks (not dollars) internally — convert only for display/logging
- Use walk-forward validation exclusively — never random train/test splits on time series
- Every model change must be validated against the Monte Carlo permutation test

## Risk-First Development
- Implement circuit breakers before any live testing
- Every new feature/model must be tested against the slippage scenarios (0.25, 0.5, 0.75, 1.0 tick)
- Daily loss limits are non-negotiable — hardcode, don't make configurable
- Kill switch must work independently of the main trading loop
- Log every order, fill, and signal for post-trade analysis

## Common Pitfalls to Avoid
- Don't optimize for Sharpe alone — a strategy can have high Sharpe but unacceptable tail risk
- Don't add complexity without proving the simpler version is insufficient first
- Don't skip the HMM gate to "get more trades" — choppy regime trades are where most losses come from
- Don't trust a backtest that hasn't been run with realistic fill simulation
- Don't retrain on a drawdown — wait for the full retraining window to avoid panic-fitting

## Databento Data
- L2 data is in DBN format (`.dbn.zst`) — use the `databento` Python package to read
- L1 data is already in Parquet format — use `pyarrow` or `pandas`
- MES continuous contract symbol: `MES.c.0`
- L2 schema: `mbp-10` (market by price, 10 depth levels)
- Dataset: `GLBX.MDP3` (CME Globex)

## Execution Notes
- Slippage recovery is the highest-ROI optimization (Phase 7)
- Every 0.25 ticks recovered at 1,533 trades = ~$480
- Current estimated slippage drag: 0.5 ticks = $1,916/month
- Limit orders for profit targets (zero slippage), market orders only for stops
- Broker migration from Tradovate ($0.70 RT) to EdgeClear ($0.44 RT) saves ~$400/month at 1,500 trades
