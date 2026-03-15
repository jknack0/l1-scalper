
CNN-LSTM MES Scalping System
Research-Backed Development Roadmap
with Claude Code Implementation Prompts
Micro E-mini S&P 500 Futures  |  Deep Learning Signal Generation & Exit Management
March 2026
Architecture: HMM Regime → CNN-LSTM Entry → CNN-LSTM Exit → Circuit Breakers
Data Strategy: L1 (Databento TBBO) → Build & Validate → L2 (EdgeClear + Rithmic MBO) → Retrain & Upgrade
Stack: Python / asyncio / PyTorch / Databento TBBO / Tradovate → EdgeClear+Rithmic / Vultr Chicago VPS

Executive Summary
This roadmap defines a clean-slate development plan for an MES futures scalping system powered by deep learning. The system replaces traditional rule-based strategy logic — named setups like ORB, VWAP Reversion, CVD Divergence — with a unified CNN-LSTM architecture for both trade entry and exit decisions, gated by an HMM regime detector. Instead of hand-coding entry conditions for each pattern, the CNN-LSTM learns to recognize predictive microstructure patterns directly from data, including patterns that don’t correspond to any named technical setup.
The system follows a deliberate L1-first, L2-later data strategy. Phases 1 through 7 build, train, validate, and backtest the full pipeline using Level 1 data (best bid/offer + trades) from Databento’s TBBO feed. Tradovate’s API handles execution only — it does not provide market depth data suitable for feature computation. This keeps data costs low, simplifies the feature pipeline, and gets a working system to live trading faster. Phase 8 then upgrades the data infrastructure to full Level 2 depth via EdgeClear + Rithmic for MBO-level order book data, extends the feature pipeline with multi-level features that require depth, retrains both models on the richer feature set, and validates the improvement before promoting to production.
The approach is grounded in peer-reviewed research:
DeepLOB (Zhang et al., 2019): Demonstrated that CNN-LSTM models trained on limit order book data learn universal microstructural features that generalize across instruments they weren’t trained on. The architecture uses ~60K parameters, making it lightweight enough for real-time inference.
Deep Order Flow Imbalance (Kolm, Turiel & Westray, 2021): Showed that LSTM networks trained on order flow imbalance (a stationary transformation of the order book) significantly outperform models trained on raw order book states. Predictive accuracy peaks at approximately two average price changes, establishing the practical forecasting horizon for scalping.
HMM-LSTM Hybrid (Kemper, 2025): Validated the HMM-LSTM hybrid approach for regime-adaptive risk management, achieving over 50% volatility reduction using entropy-weighted Bayesian model averaging to fuse regime probabilities. The key insight: use HMM posteriors as continuous position sizing weights, not hard regime switches.
The roadmap is structured into 11 phases. Each phase includes detailed explanations of what’s being built and why, specific deliverables, validation criteria, and a Claude Code implementation prompt that can be used to build that phase exactly as specified.

System Architecture Overview
The architecture collapses the traditional Signal → Filter → Strategy pipeline into a streamlined flow with two neural network models and a minimal rule-based safety layer. There are no named strategies. The CNN-LSTM is the strategy.
Runtime Data Flow
1. Market Data Ingestion
Initially operates on Level 1 data from Databento’s TBBO (Top of Book with BBO) feed. TBBO delivers a BBO snapshot (best bid price/size, best ask price/size) attached to every trade event, giving you the exact market context at the moment each trade executes. This is ideal for features like Lee-Ready trade classification (which needs the mid-price at trade time), CVD, and VPIN. For features that need to track quote changes between trades (OFI, micro-price), you supplement with Databento’s MBP-1 (Market by Price, 1 level) or their bbo-1s schema for 1-second BBO snapshots. Tradovate’s API handles order execution only. In Phase 8, the system upgrades to full Level 2 depth via EdgeClear + Rithmic, unlocking MBO-level order book data.
2. Feature Pipeline
Raw data is transformed into stationary, z-score normalized feature vectors at 1-second resolution. Stationarity is critical: raw prices and sizes are non-stationary (they drift over time), which makes neural networks learn spurious patterns that don’t generalize. By transforming inputs into things like order flow imbalance (the change in supply/demand, not the absolute level), VPIN (a ratio), and z-scored spreads, every feature oscillates around zero with roughly constant variance, which is what LSTMs are designed to process.
L1 features (Phases 1-7): OFI at best bid/ask, VPIN, bid-ask spread, cumulative volume delta, micro-price, trade arrival rate, realized volatility, volume profile metrics, Hurst exponent, return autocorrelation, trade size distribution, and Lee-Ready classification. L2 features (Phase 8+): multi-level OFI, book depth ratio across 10+ levels, queue position estimation, depth-weighted imbalance, order book slope, absorption ratio.
3. HMM Regime Classifier
A 3-state Student-t HMM with Bayesian Online Changepoint Detection (BOCPD) runs on 1-minute resampled bars. It emits a posterior probability vector [P(trending), P(mean-reverting), P(choppy)] that gates downstream model selection and position sizing. The Student-t emission distribution (instead of Gaussian) handles the fat-tailed returns that MES exhibits during news events and volatility spikes, preventing the HMM from over-signaling regime changes during normal tail events. BOCPD provides online transition detection so you don’t need to retrain the HMM on a fixed window — it adapts in real time.
4. CNN-LSTM Entry Model
Consumes a rolling window of feature vectors (100 timesteps at 1-second resolution). The CNN layers extract local patterns — things like a sudden OFI spike coinciding with a spread narrowing, or a VPIN surge preceding a trade burst. The LSTM layer captures how these local patterns evolve over the 100-second window — whether the OFI spike is sustained or fading, whether the VPIN regime is escalating or stabilizing. The model has two output heads: a directional probability (sigmoid, trained with binary cross-entropy) and an expected return magnitude (linear, trained with MSE). Both must clear their thresholds for an entry to trigger. Separate model weights are maintained per regime, selected by the HMM posterior.
5. Filter Gate
Rule-based safety checks that the models cannot override. These exist because neural networks can’t anticipate situations they’ve never seen in training data — extreme spread blowouts, flash crashes, or illiquid pre-market conditions. The filters include: spread width (no entry when spread > 2 ticks), VPIN toxicity threshold (no entry when VPIN > 0.8), time-of-day restrictions (avoid first/last 5 minutes of RTH when the book is thin), and news event blackouts (configurable calendar for FOMC, CPI, NFP). These are intentionally not learned because you want them to be interpretable, auditable, and impossible for a model update to accidentally remove.
6. CNN-LSTM Exit Model
Once in a position, a second CNN-LSTM runs continuously on the same feature stream plus trade-specific context (unrealized P&L, bars since entry, how the entry model’s confidence has evolved). It predicts remaining favorable excursion (RFE) — how many more ticks of profit are likely before the move reverses — and adverse excursion probability — the likelihood of a drawdown exceeding X ticks in the next N bars. Exit when RFE drops below commission threshold or adverse probability exceeds limit. This replaces static profit targets, stop losses, and trailing stops with an adaptive exit that sees the current microstructure.
7. Circuit Breakers
Hard limits that override all model outputs: max loss per trade (8 ticks), max hold time (5 minutes), daily loss limit (configurable), and a position kill switch. These exist for tail scenarios the models can’t anticipate — the exit model might predict positive RFE during a flash crash because it’s never seen one in training. The circuit breakers are your last line of capital protection.

Key Research Foundations
DeepLOB: Universal Feature Extraction from Order Books
Zhang et al. (2019) introduced a CNN-LSTM architecture specifically designed for limit order book data. The model processes raw LOB snapshots through three convolutional blocks (each with Conv1D → BatchNorm → ReLU), an inception module that runs parallel convolutions at different kernel sizes to capture multi-scale patterns, and an LSTM layer for temporal dependencies. On the FI-2010 benchmark (Finnish stocks), it achieves 83.4% F1 on the 3-class mid-price prediction task. More importantly, when trained on one year of London Stock Exchange data, it generalizes to instruments it was never trained on with only marginal accuracy loss. This suggests the CNN-LSTM learns universal microstructural features — things like how order flow imbalance relates to price movement — that transfer across instruments. The architecture has approximately 60,000 parameters, orders of magnitude smaller than language models, making it fast enough for real-time inference even on CPU.
Deep Order Flow Imbalance: Stationary Inputs Beat Raw States
Kolm, Turiel & Westray (2021) tested deep learning models on 115 Nasdaq stocks using nanosecond-precision order book data from LOBSTER. Their key finding: LSTM networks trained on order flow (a stationary transformation computed from changes in the order book) significantly outperform models trained on raw order book states (the actual bid/ask prices and sizes). This matters because raw LOB states are non-stationary — the bid price of MES today is very different from a year ago, but the pattern of how OFI spikes precede price moves is consistent. By feeding the model stationary inputs, you dramatically reduce the risk of overfitting to specific price levels. They also showed that predictive accuracy peaks at approximately two average price changes and decays beyond that, establishing a practical ceiling on the forecasting horizon for any scalping model.
HMM-LSTM Hybrid for Regime-Adaptive Risk
Kemper (2025) combined interpretable HMMs with LSTM networks using entropy-weighted Bayesian model averaging to fuse their regime probability outputs. Applied to semiconductor equities over a five-year out-of-sample period, the framework achieved over 50% volatility reduction and 15-17 percentage points of drawdown improvement versus passive strategies without materially sacrificing returns. The critical insight for our architecture: rather than hard-switching between regime-specific models (which causes whipsaw at regime boundaries), use the HMM posterior probabilities as continuous position sizing weights. This means during a regime transition, both models contribute proportionally, softening the handoff.
Order Flow Image Representation for Scalping
A 2024 study on encoding order flow as images for CNN processing was specifically motivated by discretionary scalpers who visually read the DOM and time & sales. The researchers encoded LOB states and trades as 2D images and trained CNNs to predict short-term volatility. The models learned to detect events like large bid orders being filled by aggressive market sell orders — exactly the patterns human scalpers look for. This validates our CNN component: if a CNN can learn the same patterns that experienced traders identify visually in the order flow, it can do so at scale across every tick without fatigue or emotional bias. Interestingly, adding LSTM to the CNN in this study did not improve results, which the authors attributed to breaking correlations between consecutive images during normalization — a useful cautionary note about feature preprocessing.
Production Feature Engineering for Scalping Engines
A March 2026 production case study documented 14 ML features in a Rust crypto scalping engine, tracing each to foundational research. The most relevant architectural insights: (1) z-score normalization over a 100-sample window using Welford’s online algorithm is essential to prevent the LSTM from being dominated by whichever feature has the largest absolute scale; (2) during exchange reconnections, stale LOB data produces constant feature vectors that the model interprets as a signal, so you need a staleness check that zeros features when data age exceeds your update interval; (3) the system’s value lies not in individual features but in their interactions — gating Ornstein-Uhlenbeck z-scores by Hurst exponent, feeding Lee-Ready classified trades into VPIN, scaling everything by microstructure regime confidence, and using ensemble disagreement as a kill switch.

11-Phase Development Roadmap
Phase 1: L1 Data Infrastructure & Historical Dataset
Duration: 2-3 weeks
What we’re building and why:
The data pipeline is the foundation everything else depends on. We’re starting with Level 1 data (best bid/offer + trades) from Databento’s TBBO schema because it’s the most cost-effective path to a working system. TBBO attaches a BBO snapshot to every trade event, giving you the exact market state at the moment each trade executes. This is perfect for trade classification (Lee-Ready needs the mid-price at trade time), CVD, and VPIN.
For features that track quote changes between trades (OFI, micro-price), TBBO alone has a gap: you only see the BBO when a trade happens, not when quotes update without a trade. During active MES RTH sessions this gap is small (trades fire every few hundred milliseconds), but during quieter periods it matters. The solution is to also pull Databento’s MBP-1 (1-level market-by-price) feed which updates on every quote change, and merge the two streams by timestamp. This gives you complete BBO coverage: quote changes from MBP-1, trade events with BBO context from TBBO.
Tradovate’s API is used for order execution only. It does not provide depth data or market data suitable for feature computation through its API. All feature data comes from Databento, maintaining a clean separation between your data infrastructure and execution infrastructure. This separation also means you can swap execution venues (e.g., migrate to EdgeClear in Phase 8) without touching the data pipeline.
Deliverables:
Databento historical data pull: minimum 2 years of MES TBBO + MBP-1 data stored in Parquet format, partitioned by trading date. TBBO provides trade events with BBO context; MBP-1 provides standalone quote updates.
Real-time data capture daemon: asyncio service on Vultr Chicago VPS subscribing to Databento’s live MES TBBO and MBP-1 feeds, merging by timestamp, and writing to Parquet with sub-second latency.
Data validation suite: integrity checks for gaps (missing RTH minutes), duplicates, timestamp monotonicity (no out-of-order events), and quote consistency (crossed quotes where bid > ask indicate data corruption).
Schema definition: canonical Parquet schema for merged BBO+trade records: timestamp (nanosecond), bid_price, bid_size, ask_price, ask_size, trade_price, trade_size, trade_side (aggressor), event_type (quote_update | trade).
L2 upgrade stub: abstract DataProvider interface so the feature pipeline consumes a standard format regardless of whether L1 or L2 data is behind it. When Phase 8 adds depth levels, only the DataProvider implementation changes, not the feature computation code.
Validation:
Gap-free coverage for all RTH sessions (9:30-16:00 ET) across the 2-year historical window
BBO reconstruction matches CME settlement prices at daily close within rounding tolerance
Trade-through detection: any trade executing outside the BBO is flagged (indicates data issue or latent quote)
Merge quality: confirm MBP-1 quote updates and TBBO trade events align within 1ms for overlapping timestamps

🤖 Claude Code Prompt: Phase 1 — L1 Data Infrastructure
You are building Phase 1 of a CNN-LSTM MES scalping system. This phase sets up the
L1 data infrastructure using Databento's TBBO and MBP-1 feeds for MES futures.
 
PROJECT STRUCTURE:
Create a Python project at ./scalp-bot/ with the following structure:
  scalp-bot/
    pyproject.toml          (Python 3.11+, deps: databento, pyarrow, asyncio, click)
    src/
      data/
        __init__.py
        schemas.py          (Parquet schema definitions, canonical record types)
        provider.py          (Abstract DataProvider interface for L1/L2 abstraction)
        databento_client.py  (Databento API wrapper for historical + live feeds)
        capture_daemon.py    (Async live capture service, TBBO + MBP-1 merge)
        historical_pull.py   (CLI script to download 2 years of MES TBBO + MBP-1)
        validation.py        (Data integrity checks: gaps, duplicates, crossed quotes)
        merge.py             (Merge TBBO trade events with MBP-1 quote updates by timestamp)
      config/
        settings.py          (Databento API key, MES symbol, date ranges, paths)
 
REQUIREMENTS:
1. schemas.py: Define a canonical record dataclass with fields:
   - timestamp (int64 nanoseconds since epoch)
   - bid_price (float64), bid_size (int32)
   - ask_price (float64), ask_size (int32)
   - trade_price (float64, NaN if quote-only update)
   - trade_size (int32, 0 if quote-only update)
   - trade_side (int8: 1=buyer aggressor, -1=seller aggressor, 0=unknown/quote-only)
   - event_type (str: "quote" | "trade")
   Define the Parquet schema using pyarrow with appropriate types.
 
2. provider.py: Abstract base class DataProvider with methods:
   - get_historical(symbol, start_date, end_date) -> Iterator[RecordBatch]
   - subscribe_live(symbol, callback) -> None
   - get_schema() -> pa.Schema
   This interface will be implemented for L1 (Databento) now and L2 (Rithmic) in Phase 8.
 
3. databento_client.py: Implement DataProvider for Databento:
   - Historical: use databento.Historical client to fetch MES TBBO + MBP-1 schemas
   - Live: use databento.Live client for real-time subscription
   - Handle Databento's specific schema mapping to our canonical format
 
4. merge.py: Merge logic for TBBO + MBP-1 streams:
   - TBBO records become event_type="trade" rows with full BBO context
   - MBP-1 records that don't coincide with a trade become event_type="quote" rows
   - Dedup: if a TBBO trade and MBP-1 quote have the same timestamp, keep the TBBO
     record (it has both trade and BBO data)
   - Output: single sorted-by-timestamp Parquet file per trading date
 
5. historical_pull.py: CLI tool (click) that:
   - Accepts start_date, end_date, output_dir params
   - Pulls MES.CME TBBO + MBP-1 data from Databento for each trading day
   - Merges and writes one Parquet file per day to output_dir/YYYY-MM-DD.parquet
   - Supports resumption (skips dates that already have output files)
   - Default: pull 2 years ending yesterday
 
6. capture_daemon.py: Async service that:
   - Subscribes to Databento live TBBO + MBP-1 for MES
   - Merges events in real-time using a small time-based buffer (50ms) to handle
     out-of-order arrivals between the two feeds
   - Writes completed 1-minute Parquet files to a dated directory structure
   - Logs connection status, events/second, and any detected gaps
   - Graceful shutdown on SIGTERM
 
7. validation.py: Functions to check a Parquet file for:
   - Timestamp gaps > 5 seconds during RTH (9:30-16:00 ET) = warning
   - Timestamp gaps > 30 seconds during RTH = error
   - Duplicate timestamps with identical data = dedup
   - Non-monotonic timestamps = error
   - Crossed quotes (bid_price >= ask_price) = error
   - Trade-through (trade_price outside bid/ask range) = warning
   Run on all historical files after pull completes.
 
DATABENTO SPECIFICS:
- Dataset: "GLBX.MDP3" (CME Globex)
- Symbol: "MES.FUT" or the specific front-month contract
- Schemas: "tbbo" for trade+BBO, "mbp-1" for top-of-book quotes
- Use SType.CONTINUOUS for front-month rolling
- Store Databento API key in environment variable DATABENTO_API_KEY
 
TESTING:
- Write unit tests for merge logic with synthetic TBBO + MBP-1 data
- Write integration test that pulls 1 day of historical data and validates
- Test capture daemon startup/shutdown cycle with live feed (30 second test)



Phase 2: L1 Feature Engineering Pipeline
Duration: 3-4 weeks
What we’re building and why:
The feature pipeline transforms raw BBO quotes and trades into the stationary, normalized feature vectors that the CNN-LSTM will consume. This is arguably the most important phase of the project, because the quality of features determines the ceiling of model performance. No amount of model architecture tuning can compensate for noisy or non-stationary inputs.
The core principle comes from Kolm et al. (2021): models trained on stationary transformations of the order book dramatically outperform models trained on raw order book states. Raw prices are non-stationary (MES might trade at 5200 today and 5400 next month), but the pattern of how order flow imbalance spikes precede price moves is approximately stationary. Every feature we compute is either naturally stationary (like bid-ask spread in ticks, which oscillates around a constant mean) or explicitly transformed to be stationary (like OFI, which is a differenced quantity).
Each feature is z-score normalized over a rolling 100-sample window using Welford’s online algorithm. This is critical for the LSTM: without normalization, the feature with the largest absolute scale (usually raw bid/ask sizes) dominates the network’s attention, and all other features are effectively ignored. Welford’s algorithm computes the rolling mean and variance in a single pass without storing the full window, making it numerically stable and memory-efficient for real-time computation.
L1 features (BBO + trades only):
Feature
Source
How It Works
Order Flow Imbalance (OFI)
Cont et al., 2014
Tracks changes at the best bid/ask: if the bid price ticks up or bid size increases, that’s positive OFI (buying pressure). If the ask price ticks down or ask size increases, that’s negative OFI (selling pressure). Computed from consecutive BBO snapshots. The single most predictive L1 feature for short-term price direction.
VPIN
Easley et al., 2012
Volume-Synchronized Probability of Informed Trading. Buckets trades into fixed-volume bins, classifies each trade as buy/sell (via Lee-Ready), and measures the absolute imbalance between buy and sell volume in each bucket. High VPIN = informed traders are active = higher adverse selection risk. Computed from trade data + Lee-Ready classification.
Micro-Price
Stoikov, 2018
Adjusts the mid-price based on the bid/ask size imbalance: if the bid has 50 contracts and the ask has 10, the fair price is closer to the ask (more demand than supply). Formula: mid + spread * (bid_size - ask_size) / (bid_size + ask_size) / 2. A higher-frequency fair value estimate than the raw mid.
Cumulative Volume Delta (CVD)
Kirilenko et al., 2017
Net buyer volume minus seller volume over a rolling window. Uses Lee-Ready classified trades. Sustained positive CVD = persistent buying pressure. Divergence between CVD and price direction is a classic reversal signal.
Bid-Ask Spread
Standard
Current spread in ticks (ask - bid). MES typically has a 1-tick spread during RTH. Widens during news events and low liquidity. A direct measure of market uncertainty and transaction cost.
Trade Arrival Rate
Hawkes process lit.
Number of trades per second over a rolling 30-second window. Proxy for market activity regime. Spikes during momentum moves, drops during consolidation. Helps the model distinguish between active and quiet markets.
Volume Profile (POC/VAH/VAL)
Market Profile
Session volume distribution: POC is the price with the most volume (fair value), VAH/VAL define the 70% value area. Computed from all trades in the current RTH session. Price near POC = balanced, price at VAH/VAL = extended, potential reversion.
Hurst Exponent
Mandelbrot, 1968
Rolling estimate (rescaled range method) over 200 trade-price observations. H > 0.5 = trending, H < 0.5 = mean-reverting, H = 0.5 = random walk. Directly informs whether momentum or reversion strategies have an edge in the current window.
Realized Volatility
Standard
Standard deviation of 1-second log returns over a rolling 60-second window. Feeds into the HMM regime detector and helps the model calibrate expected move sizes.
Lee-Ready Classification
Lee & Ready, 1991
Classifies each trade as buyer or seller-initiated by comparing trade price to the mid-price at trade time (from TBBO’s BBO context). Trades above mid = buyer initiated, below = seller initiated. At mid = use tick rule (compare to previous trade). Feeds into CVD and VPIN.
Trade Size Distribution
Standard
Rolling percentile rank of current trade size vs the last 500 trades. Large trades (95th+ percentile) are more likely institutional. Helps detect hidden institutional flow.
Return Autocorrelation
Standard
Rolling autocorrelation (lag 1) of 1-second returns over a 60-second window. Positive autocorrelation = momentum regime, negative = mean-reversion. Complements Hurst for regime detection.


L2 features (deferred to Phase 8):
Multi-Level OFI (MLOFI), Book Depth Ratio, Order Book Slope, Queue Position Estimation, Depth-Weighted Imbalance, and Absorption Ratio. These all require price/size information at multiple book levels, which L1 data doesn’t provide. They’re documented in Phase 8.
Normalization details:
Every feature is z-score normalized: z = (x - μ) / σ where μ and σ are computed over a rolling 100-sample window using Welford’s online algorithm. This produces numerically stable estimates of mean and variance without storing the full window. The 100-sample window at 1-second resolution means features are normalized over the last ~100 seconds, long enough to capture the local distribution but short enough to adapt to intraday volatility changes. A staleness check zeros all features when the most recent data point is older than 2 seconds (indicating a feed gap). This prevents the model from acting on stale features during reconnections.
Output format:
Each 1-second timestep produces a fixed-dimension feature vector of ~12 values (one per L1 feature). These are stacked into rolling windows of 100 timesteps to form the CNN-LSTM input tensor of shape [batch_size, 100, 12]. The 100-second lookback was chosen based on DeepLOB’s finding that LOB sequence length only marginally impacts performance beyond ~50-100 events, and because MES’s typical microstructure patterns (order flow surges, spread regime changes) play out over 10-120 seconds.
Validation:
Augmented Dickey-Fuller test on every feature confirms stationarity (p < 0.01)
Z-score distributions: verify mean ≈ 0, std ≈ 1 across the full historical dataset
Look-ahead bias audit: every feature must use only data available at computation time. OFI uses the prior BBO change, not the current one. VPIN uses completed volume buckets, not the in-progress bucket.
Feature correlation matrix: check for redundant features (|r| > 0.9) that add noise without information

🤖 Claude Code Prompt: Phase 2 — L1 Feature Engineering Pipeline
You are building Phase 2 of a CNN-LSTM MES scalping system. This phase creates the
feature engineering pipeline that transforms L1 data (BBO + trades) into stationary,
normalized feature vectors for the CNN-LSTM.
 
PROJECT STRUCTURE (extend ./scalp-bot/):
  src/
    features/
      __init__.py
      base.py              (Abstract Feature base class, Welford normalizer)
      ofi.py               (Order Flow Imbalance from BBO changes)
      vpin.py              (Volume-Synchronized Probability of Informed Trading)
      microprice.py        (Stoikov micro-price from bid/ask imbalance)
      cvd.py               (Cumulative Volume Delta with Lee-Ready classification)
      spread.py            (Bid-ask spread in ticks)
      trade_rate.py        (Trade arrival rate - trades per second, rolling 30s)
      volume_profile.py    (POC, VAH, VAL from session trade distribution)
      hurst.py             (Rolling Hurst exponent via rescaled range)
      realized_vol.py      (Rolling realized volatility from 1s log returns)
      lee_ready.py         (Trade classification: buyer vs seller initiated)
      trade_size_dist.py   (Rolling percentile rank of trade size)
      return_autocorr.py   (Rolling lag-1 autocorrelation of 1s returns)
      pipeline.py          (FeaturePipeline: orchestrates all features, outputs tensor)
      normalizer.py        (WelfordNormalizer: online z-score with staleness check)
 
REQUIREMENTS:
 
1. base.py - Abstract Feature class:
   class Feature(ABC):
       name: str
       @abstractmethod
       def update(self, record: CanonicalRecord) -> Optional[float]:
           """Process one record, return feature value or None if not ready."""
       @abstractmethod
       def reset(self) -> None:
           """Reset state for new session."""
 
2. normalizer.py - Welford's online z-score normalizer:
   class WelfordNormalizer:
       def __init__(self, window: int = 100):
       def update(self, value: float) -> float:
           """Add value, return z-score. Uses Welford's online algorithm."""
       def is_valid(self) -> bool:
           """True if we have enough samples for meaningful normalization."""
   
   Implementation must use Welford's numerically stable single-pass algorithm:
   - Maintain: count, mean, M2 (sum of squared differences)
   - On each value: delta = value - mean; mean += delta/count; delta2 = value - mean; M2 += delta*delta2
   - Variance = M2 / count
   - For rolling window: use a circular buffer of size=window, subtract old values
 
   Also implement StalenessCheck:
   - Track timestamp of last update
   - zero_if_stale(value, current_ts, max_age_ns=2_000_000_000) -> float
     Returns 0.0 if last update was > max_age_ns ago, else returns value
 
3. ofi.py - Order Flow Imbalance:
   Computed from consecutive BBO snapshots (event_type="quote" records):
   - delta_bid = (bid_price_now > bid_price_prev) * bid_size_now
                - (bid_price_now < bid_price_prev) * bid_size_prev
                + (bid_price_now == bid_price_prev) * (bid_size_now - bid_size_prev)
   - delta_ask = similar logic for ask side (but negative = selling pressure)
   - OFI = delta_bid - delta_ask
   This is the Cont et al. (2014) definition. Only uses L1 (best bid/ask).
 
4. vpin.py - VPIN:
   - Bucket trades into fixed-volume bins (bucket_size = median 1-minute volume)
   - Each trade classified via Lee-Ready (import from lee_ready.py)
   - For each completed bucket: buy_volume, sell_volume
   - VPIN = rolling mean of |buy_vol - sell_vol| / bucket_size over last 50 buckets
   - Only emits a value when a bucket completes
 
5. microprice.py - Stoikov Micro-Price:
   micro = mid_price + spread * (bid_size - ask_size) / (2 * (bid_size + ask_size))
   Then take the log-return of microprice vs previous microprice as the feature.
 
6. cvd.py - Cumulative Volume Delta:
   - Maintain rolling window (300 seconds) of classified trade volumes
   - CVD = sum(buy_volume) - sum(sell_volume) in window
   - Feature value = CVD (will be z-score normalized downstream)
 
7. spread.py: (ask_price - bid_price) in ticks. MES tick = 0.25 points.
 
8. trade_rate.py: Count trades in rolling 30-second window. Output: trades/second.
 
9. volume_profile.py:
   - Maintain histogram of trade volume at each price level for current RTH session
   - POC = price with highest volume
   - Value Area = price range containing 70% of volume, centered on POC
   - Feature outputs: (current_price - POC) in ticks, (current_price - VAH) in ticks,
     (current_price - VAL) in ticks
 
10. hurst.py: Rolling Hurst exponent via rescaled range (R/S) method over last 200
    trade prices. Output range: 0 to 1. H>0.5 = trending, H<0.5 = mean-reverting.
 
11. realized_vol.py: Standard deviation of 1-second log returns over rolling 60s window.
 
12. lee_ready.py: Classify each trade:
    - If trade_price > mid_price at trade time: buyer initiated (+1)
    - If trade_price < mid_price: seller initiated (-1)
    - If trade_price == mid_price: use tick rule (compare to previous trade price)
    Uses the BBO context from TBBO records.
 
13. trade_size_dist.py: Rolling percentile rank of current trade size vs last 500 trades.
    Output: 0.0 to 1.0 (0.95 = trade is in 95th percentile of recent sizes).
 
14. return_autocorr.py: Lag-1 autocorrelation of 1-second mid-price returns over
    rolling 60-second window. Range: -1 to 1.
 
15. pipeline.py - FeaturePipeline:
    class FeaturePipeline:
        def __init__(self, features: List[Feature], window_size: int = 100):
        def process_record(self, record: CanonicalRecord) -> Optional[np.ndarray]:
            """Process one record. Returns normalized feature vector if all features ready.
            Maintains rolling window of feature vectors for CNN-LSTM input."""
        def get_window(self) -> Optional[np.ndarray]:
            """Returns shape [window_size, num_features] tensor if window is full."""
        def reset_session(self) -> None:
            """Call at start of each RTH session to reset session-specific features."""
    
    The pipeline:
    - Resamples to 1-second bars (takes last value per second for each feature)
    - Normalizes each feature independently via WelfordNormalizer
    - Applies staleness check
    - Maintains a circular buffer of window_size normalized feature vectors
    - Returns the full window as a numpy array when requested
 
TESTING:
- Unit test each feature with synthetic data (known inputs -> expected outputs)
- Test OFI: simulate a bid price tick up -> verify positive OFI
- Test VPIN: simulate 50 buy trades, 50 sell trades -> verify VPIN near 0
- Test Lee-Ready: trade above mid -> buyer, below -> seller, at mid -> tick rule
- Test WelfordNormalizer: verify mean~0, std~1 after 200 random samples
- Integration test: feed 1 day of historical data through pipeline, verify output shape
  is [num_seconds_in_day, 12] with values roughly in [-3, 3] range
- Test staleness: simulate 5-second gap, verify features zero out during gap



Phase 3: HMM Regime Detector
Duration: 2-3 weeks
What we’re building and why:
The HMM is the outer gatekeeper that tells the CNN-LSTM models what kind of market they’re operating in. Markets cycle between trending periods (where momentum strategies work), mean-reverting periods (where fading moves works), and choppy/noisy periods (where nothing works). A single model trained on all regimes learns an average behavior that’s suboptimal in every regime. By detecting the current regime first and routing to a regime-specific model (or weighting models by regime probability), each model only needs to learn the patterns relevant to its regime.
We use a 3-state HMM because your prior analysis on 133M bars revealed a near-2-state structure where RANGE_BOUND dominates (~53%) and TRENDING shows near-zero self-transition. The third state captures the choppy/high-noise periods where we want to reduce or eliminate trading. The Student-t emission distribution handles MES’s fat-tailed return distribution better than Gaussian — without it, the HMM interprets normal tail events as regime changes and over-signals transitions.
BOCPD (Bayesian Online Changepoint Detection) runs on top of the HMM to detect regime transitions in real time without requiring fixed-window retraining. When BOCPD detects a changepoint, it increases the uncertainty in the HMM’s posterior, triggering the confidence cooldown that reduces position sizing until the new regime stabilizes.
Architecture:
3-state HMM with Student-t emissions, fitted on 1-minute resampled bars
Input features: realized_vol, return_autocorr, spread (mean over 1-min bar), trade_arrival_rate (mean over 1-min bar)
BOCPD layer for online transition detection
Output: posterior probability vector [P(trending), P(mean_reverting), P(choppy)] at each 1-minute bar
Regime-to-behavior mapping:
Trending (State 0): Characterized by high return autocorrelation (positive), elevated but not extreme volatility, and sustained OFI in one direction. The entry model should be biased toward trend continuation signals.
Mean-Reverting (State 1): Characterized by negative return autocorrelation, moderate volatility, and oscillating OFI. The entry model should favor counter-trend signals near extremes (VWAP deviation, value area boundaries).
Choppy/Low-Signal (State 2): Characterized by low autocorrelation magnitude (near zero), low or spiky volatility, and no sustained OFI direction. Position sizing scaled to near-zero. The system mostly sits flat.
Position sizing integration:
Rather than hard-switching between regime-specific models (which causes whipsaw at boundaries), use the posterior as a continuous sizing weight. If the HMM says P(trending)=0.7, P(mean_reverting)=0.2, P(choppy)=0.1, the trending entry model runs at 70% position size, the mean-reverting model at 20%, and no trading from the choppy model. Add a confidence cooldown: after any regime transition (dominant state changes), reduce all position sizing by 50% for N bars (e.g., 5 minutes) until the dominant posterior stabilizes above 0.6.
Validation:
Backtest regime labels against known market events: FOMC days should show regime transitions, quiet summer afternoons should be mostly mean-reverting or choppy
Verify choppy regimes correlate with historically unprofitable periods (test with a simple OFI-threshold strategy)
Regime persistence: trending states should last 5-30 minutes on average (not flickering every bar)
Transition matrix stability: retrain on two non-overlapping 6-month periods, verify similar transition probabilities

🤖 Claude Code Prompt: Phase 3 — HMM Regime Detector
You are building Phase 3 of a CNN-LSTM MES scalping system. This phase creates the
HMM regime detector that classifies market conditions into trending, mean-reverting,
or choppy states.
 
PROJECT STRUCTURE (extend ./scalp-bot/):
  src/
    regime/
      __init__.py
      hmm.py               (3-state Student-t HMM with hmmlearn)
      bocpd.py             (Bayesian Online Changepoint Detection)
      regime_detector.py   (Combined HMM + BOCPD, emits posterior probabilities)
      position_sizer.py    (Converts regime posteriors to position size weights)
      trainer.py           (Offline HMM training on historical 1-min bars)
      visualizer.py        (Plot regime labels overlaid on price for debugging)
 
REQUIREMENTS:
 
1. hmm.py - 3-state HMM:
   - Use hmmlearn.hmm.GaussianHMM as base (Student-t via custom emission override)
   - OR use pomegranate for native Student-t support
   - States: TRENDING=0, MEAN_REVERTING=1, CHOPPY=2
   - Input features per 1-min bar (computed from the FeaturePipeline output):
     * realized_vol: mean of realized_vol feature over the 1-min bar
     * return_autocorr: mean of return_autocorr feature over the 1-min bar
     * spread_mean: mean bid-ask spread over the 1-min bar
     * trade_rate_mean: mean trade arrival rate over the 1-min bar
   - fit(data: np.ndarray) -> None  # shape [n_bars, 4]
   - predict_proba(observation: np.ndarray) -> np.ndarray  # shape [3] posteriors
   - save/load model to disk
 
2. bocpd.py - Bayesian Online Changepoint Detection:
   - Implement the Adams & MacKay (2007) algorithm
   - Hazard function: constant rate (1/250 = expect regime change every ~4 hours)
   - Observation model: normal-inverse-gamma conjugate prior
   - detect(observation: np.ndarray) -> float  # changepoint probability 0-1
   - When changepoint_prob > 0.7, signal a regime transition
 
3. regime_detector.py - Combined system:
   class RegimeDetector:
       def __init__(self, hmm_model_path: str):
       def update(self, one_min_features: np.ndarray) -> RegimeState:
           """Process one 1-minute bar, return current regime state."""
       
   @dataclass
   class RegimeState:
       posteriors: np.ndarray          # [P(trending), P(mean_reverting), P(choppy)]
       dominant_regime: int            # argmax of posteriors
       confidence: float               # max(posteriors)
       changepoint_prob: float         # from BOCPD
       bars_since_transition: int      # stability counter
       in_cooldown: bool               # True if recently transitioned
 
4. position_sizer.py:
   class RegimePositionSizer:
       def __init__(self, cooldown_bars: int = 5, min_confidence: float = 0.6):
       def get_weights(self, state: RegimeState) -> Dict[str, float]:
           """Returns {'trending': 0.0-1.0, 'mean_reverting': 0.0-1.0, 'choppy': 0.0}
           
           Logic:
           - Base weights = posteriors (trending weight = P(trending), etc.)
           - Choppy weight always 0 (we don't trade in chop)
           - If in_cooldown: multiply all weights by 0.5
           - If confidence < min_confidence: multiply all weights by confidence/min_confidence
           - Normalize so trending + mean_reverting <= 1.0
           """
 
5. trainer.py - Offline training script:
   - Load historical 1-min bars from Parquet files
   - Resample the 1-second feature data to 1-minute using mean aggregation
   - Fit the 3-state HMM using EM algorithm
   - Cross-validate: train on first 70% of data, evaluate regime stability on last 30%
   - Save trained model to disk
   - Generate regime label report: % time in each state, avg duration per state,
     transition matrix
 
6. visualizer.py:
   - Plot price chart with background colored by dominant regime
     (green=trending, blue=mean_reverting, red=choppy)
   - Plot posterior probabilities over time as stacked area chart
   - Mark BOCPD-detected changepoints with vertical lines
   - Save as PNG for visual debugging
 
TESTING:
- Synthetic test: generate 3 distinct regimes (trending = autocorrelated returns,
  mean-reverting = negatively autocorrelated, choppy = white noise). Verify HMM
  recovers the correct labels.
- Test BOCPD: inject a sudden volatility spike, verify changepoint detection
- Test position_sizer: verify cooldown reduces weights, choppy always returns 0
- Integration: train on 6 months of historical data, visualize regimes on 1 month
  of out-of-sample data, sanity check that regime labels match intuition



Phase 4: CNN-LSTM Entry Model (L1 Features)
Duration: 4-6 weeks
What we’re building and why:
This is the core of the system: a CNN-LSTM neural network that replaces all named strategy logic with a single learned entry signal. Instead of hand-coding rules like “enter long when OFI > X and price is within Y ticks of POC,” the model learns these patterns (and subtler variations of them) directly from labeled training data. The CNN layers act as automatic feature extractors that detect local microstructure patterns in the 100-second input window — things like an OFI spike coinciding with a spread narrowing, or a VPIN surge followed by a trade-rate acceleration. The LSTM layer then captures how these local patterns evolve over time — whether the signal is building or fading.
The architecture is inspired by DeepLOB (Zhang et al., 2019) but adapted for our L1 feature vector input rather than raw order book states. Key adaptations: we use 1D convolutions along the time axis (since our input is already a feature vector, not a spatial grid of price levels), dilated convolutions to capture multi-scale patterns without increasing parameter count, and an inception module that runs parallel convolutions at different kernel sizes.
The model has two output heads: a directional probability (sigmoid, 0-1, trained with binary cross-entropy) and an expected return magnitude (linear, in ticks, trained with MSE). Both must clear their thresholds for an entry to trigger. The directional head tells you which way, the magnitude head tells you whether the expected move is large enough to justify the commission cost. Separate model weights are maintained for the trending and mean-reverting regimes, selected by the HMM posterior.
Architecture details:
Input: Shape [batch, 100, 12] — 100 timesteps at 1-second resolution, 12 L1 features per timestep
Conv Block 1: Conv1D(in=12, out=32, kernel=3, dilation=1) → BatchNorm → ReLU → Dropout(0.2). Captures 3-second local patterns.
Conv Block 2: Conv1D(in=32, out=32, kernel=3, dilation=2) → BatchNorm → ReLU → Dropout(0.2). Captures 7-second patterns (effective receptive field grows with dilation).
Conv Block 3: Conv1D(in=32, out=32, kernel=3, dilation=4) → BatchNorm → ReLU → Dropout(0.2). Captures 15-second patterns.
Inception Module: Parallel Conv1D with kernel sizes 1, 3, 5 on the Conv Block 3 output. Each branch produces 16 channels. Concatenated to 48 channels. This captures multiple time scales simultaneously — the model learns which scale is informative per regime.
LSTM: 2-layer LSTM with hidden_size=64, dropout=0.3 between layers. Processes the 100-step sequence of CNN features. The final hidden state encodes the temporal evolution of all patterns detected by the CNN.
Direction Head: Linear(64, 1) → Sigmoid. Output: P(price goes up in next K bars).
Magnitude Head: Linear(64, 1). Output: expected return in ticks over next K bars (can be negative).
Training:
Direction label: Smoothed mid-price change over next K bars (K=20 seconds default for MES). Smoothed per DeepLOB: moves < 0.5 ticks labeled neutral and excluded. Up/down only.
Magnitude label: Actual forward return in ticks over next K bars.
Regime segmentation: Historical data segmented by Phase 3 HMM labels. Separate model weights trained for trending and mean-reverting regimes. No model for choppy (we don’t trade).
Walk-forward: Train months 1-6, validate month 7, test month 8. Roll forward in 1-month steps.
Optimizer: Adam, lr=0.01, epsilon=1.0 (per DeepLOB). Batch size 32.
Regularization: Dropout 0.2 (CNN) / 0.3 (LSTM), label smoothing 0.2 on cross-entropy, early stopping after 20 epochs without validation improvement.
Entry decision: Fire when direction_prob > 0.65 AND expected_magnitude > 1.5 ticks AND HMM dominant posterior > 0.5 AND all filter gates pass.
Validation:
Out-of-sample F1 > 0.55 on directional classification (realistic for microstructure data)
Sharpe > 1.0 for simple enter-on-signal, exit-after-K-bars strategy (before exit model optimization)
Compare against: (1) random entry baseline, (2) simple OFI-threshold baseline, (3) buy-and-hold
Feature importance: permutation importance to verify the model uses multiple features, not just one

🤖 Claude Code Prompt: Phase 4 — CNN-LSTM Entry Model
You are building Phase 4 of a CNN-LSTM MES scalping system. This phase creates the
CNN-LSTM entry model that generates trade signals from L1 feature vectors.
 
PROJECT STRUCTURE (extend ./scalp-bot/):
  src/
    models/
      __init__.py
      entry_model.py       (PyTorch CNN-LSTM with dual output heads)
      dataset.py           (PyTorch Dataset for walk-forward training)
      trainer.py           (Training loop with walk-forward validation)
      labeler.py           (Generate direction + magnitude labels from price data)
      evaluator.py         (Metrics: F1, Sharpe, precision, recall, feature importance)
      inference.py         (Real-time inference wrapper for live trading)
 
REQUIREMENTS:
 
1. entry_model.py - PyTorch CNN-LSTM:
   class EntryModel(nn.Module):
       def __init__(self, num_features=12, seq_len=100, cnn_channels=32,
                    lstm_hidden=64, lstm_layers=2, dropout_cnn=0.2, dropout_lstm=0.3):
       
       Architecture:
       - Conv Block 1: Conv1d(num_features, cnn_channels, kernel=3, dilation=1, padding="same")
                       -> BatchNorm1d -> ReLU -> Dropout(dropout_cnn)
       - Conv Block 2: Conv1d(cnn_channels, cnn_channels, kernel=3, dilation=2, padding="same")
                       -> BatchNorm1d -> ReLU -> Dropout(dropout_cnn)
       - Conv Block 3: Conv1d(cnn_channels, cnn_channels, kernel=3, dilation=4, padding="same")
                       -> BatchNorm1d -> ReLU -> Dropout(dropout_cnn)
       - Inception: 3 parallel Conv1d branches on Block 3 output:
           Branch A: Conv1d(cnn_channels, 16, kernel=1)
           Branch B: Conv1d(cnn_channels, 16, kernel=3, padding=1)
           Branch C: Conv1d(cnn_channels, 16, kernel=5, padding=2)
           Concatenate -> 48 channels
       - LSTM: nn.LSTM(input_size=48, hidden_size=lstm_hidden, num_layers=lstm_layers,
                       batch_first=True, dropout=dropout_lstm)
       - Direction head: nn.Linear(lstm_hidden, 1) -> Sigmoid
       - Magnitude head: nn.Linear(lstm_hidden, 1)
       
       IMPORTANT: Conv1d expects shape [batch, channels, seq_len].
       Input arrives as [batch, seq_len, features]. Transpose before CNN, transpose back for LSTM.
       
       def forward(self, x):
           # x shape: [batch, seq_len, num_features]
           x = x.permute(0, 2, 1)  # -> [batch, num_features, seq_len]
           # ... CNN blocks ...
           x = x.permute(0, 2, 1)  # -> [batch, seq_len, cnn_out_channels]
           lstm_out, (h_n, c_n) = self.lstm(x)
           last_hidden = h_n[-1]  # [batch, lstm_hidden]
           direction = torch.sigmoid(self.direction_head(last_hidden))
           magnitude = self.magnitude_head(last_hidden)
           return direction.squeeze(-1), magnitude.squeeze(-1)
 
2. labeler.py - Training label generation:
   def generate_labels(prices: np.ndarray, timestamps: np.ndarray,
                       horizon_seconds: int = 20, min_move_ticks: float = 0.5,
                       tick_size: float = 0.25) -> Tuple[np.ndarray, np.ndarray]:
       """
       For each timestep t:
       - forward_return = mid_price[t + horizon] - mid_price[t] (in ticks)
       - direction_label: 1 if forward_return > min_move_ticks,
                          0 if forward_return < -min_move_ticks,
                          NaN if |forward_return| <= min_move_ticks (excluded from training)
       - magnitude_label: forward_return in ticks (continuous)
       Returns (direction_labels, magnitude_labels)
       """
 
3. dataset.py - Walk-forward dataset:
   class ScalpDataset(torch.utils.data.Dataset):
       def __init__(self, features: np.ndarray, direction_labels: np.ndarray,
                    magnitude_labels: np.ndarray, window_size: int = 100):
           """
           features: shape [num_timesteps, num_features]
           Labels aligned by index. Filter out NaN direction labels.
           Each sample: features[i-window_size:i] -> (direction_label[i], magnitude_label[i])
           """
       def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
           """Returns (feature_window, direction_label, magnitude_label)"""
 
4. trainer.py - Walk-forward training:
   class WalkForwardTrainer:
       def __init__(self, model_class, train_months=6, val_months=1, test_months=1):
       
       def train_fold(self, train_data, val_data, regime_label: str) -> EntryModel:
           """
           Train one model on one regime's data for one fold.
           
           Loss = BCE(direction_pred, direction_label) * label_smoothing
                + MSE(magnitude_pred, magnitude_label) * 0.5
           
           Optimizer: Adam(lr=0.01, eps=1.0)
           Batch size: 32
           Early stopping: 20 epochs without validation improvement
           Label smoothing: 0.2 (targets become 0.1 and 0.9 instead of 0 and 1)
           """
       
       def run_walk_forward(self, all_data, regime_labels) -> List[FoldResult]:
           """
           For each fold:
           1. Split data by time into train/val/test
           2. Segment by regime label (from Phase 3 HMM)
           3. Train separate models for 'trending' and 'mean_reverting'
           4. Evaluate on test set
           5. Roll forward 1 month
           Returns metrics per fold.
           """
 
5. evaluator.py:
   - F1 score, precision, recall on direction classification (threshold=0.5)
   - Sharpe ratio of simple strategy: enter on signal (direction_prob > 0.65 AND
     magnitude > 1.5 ticks), exit after K bars
   - Permutation feature importance: for each feature, shuffle it and measure
     accuracy drop. Report ranked importance.
   - Comparison baselines: random entry, OFI-threshold entry (enter when |OFI_z| > 2)
 
6. inference.py - Real-time inference wrapper:
   class EntryInference:
       def __init__(self, trending_model_path: str, mean_reverting_model_path: str):
       def predict(self, feature_window: np.ndarray, regime_state: RegimeState
                  ) -> EntrySignal:
           """
           Run both models, weight by regime posteriors.
           
           EntrySignal:
               direction_prob: float (weighted average of both models' direction probs)
               magnitude: float (weighted average of both models' magnitude preds)
               should_enter: bool (direction_prob > 0.65 AND magnitude > 1.5)
               side: int (1 for long if direction_prob > 0.65, -1 for short if < 0.35)
           """
 
TESTING:
- Unit test model forward pass with random input -> verify output shapes
- Test labeler: known price sequence -> verify correct labels
- Test dataset: verify window extraction and label alignment
- Train on 1 month of data as smoke test, verify loss decreases
- Integration: full walk-forward on 6 months, verify F1 > 0.5 on validation



Phase 5: CNN-LSTM Exit Model (L1 Features)
Duration: 3-4 weeks
What we’re building and why:
The exit model replaces your ExitEngine’s declarative rules (static_target, static_stop, trailing_stop, time_stop, etc.) with a neural network that sees the current microstructure and makes adaptive exit decisions. A static 2-tick target exits too early when the model detects the move has 4+ ticks remaining, and a static 4-tick stop holds too long when the model detects adverse flow building. The exit model adapts in real time because it sees the same feature stream as the entry model plus trade-specific context (how long you’ve been in, current P&L, whether the entry signal is still valid).
The training approach is deliberately decoupled from the entry model: we train on ALL bars as hypothetical entry points, labeling each with the forward RFE and adverse excursion from that bar. This means the exit model doesn’t need to know why you entered — it just learns to predict whether the current microstructure supports holding. This decoupling prevents error compounding (entry model mistakes propagating into exit model training) and means you can train the exit model independently and in parallel with the entry model.
Additional input features (trade-specific context):
Unrealized P&L in ticks (current price minus entry price, signed by direction)
Bars elapsed since entry (normalized by max hold time)
Entry model’s current directional probability (has the entry signal strengthened or weakened?)
Entry model’s current magnitude prediction (is the expected remaining move growing or shrinking?)
Output heads:
RFE (Remaining Favorable Excursion): Linear output predicting additional ticks of profit remaining. When RFE < 0.5 ticks (below commission), exit.
Adverse Excursion Probability: Sigmoid output predicting P(drawdown > X ticks in next N bars). When this > 0.7, exit.
Exit fires when ANY of: RFE < 0.5 ticks, adverse prob > 0.7, or circuit breaker triggers (8-tick loss, 5-min hold).

🤖 Claude Code Prompt: Phase 5 — CNN-LSTM Exit Model
You are building Phase 5 of a CNN-LSTM MES scalping system. This phase creates the
CNN-LSTM exit model that manages open positions with adaptive exit decisions.
 
PROJECT STRUCTURE (extend ./scalp-bot/):
  src/
    models/
      exit_model.py        (PyTorch CNN-LSTM for exit signals)
      exit_labeler.py      (Generate RFE + adverse excursion labels)
      exit_dataset.py      (Dataset for exit model training)
      exit_trainer.py      (Training loop, same walk-forward as entry)
      exit_inference.py    (Real-time exit inference wrapper)
 
REQUIREMENTS:
 
1. exit_model.py:
   Same CNN-LSTM backbone as EntryModel but with:
   - num_features = 12 (L1 features) + 4 (trade context) = 16
   - Trade context features appended to each timestep in the window:
     * unrealized_pnl_ticks (float, signed)
     * bars_since_entry (float, normalized 0-1 by max_hold=300 bars)
     * entry_direction_prob (float, current entry model output)
     * entry_magnitude (float, current entry model magnitude output)
   
   Output heads:
   - RFE head: nn.Linear(lstm_hidden, 1)  # remaining favorable excursion in ticks
   - Adverse head: nn.Linear(lstm_hidden, 1) -> Sigmoid  # P(adverse > X ticks)
 
2. exit_labeler.py:
   def generate_exit_labels(prices: np.ndarray, adverse_threshold_ticks: float = 4.0,
                            horizon_bars: int = 60, tick_size: float = 0.25
                           ) -> Tuple[np.ndarray, np.ndarray]:
       """
       For EVERY bar (not just entry points):
       
       RFE label: From bar t, find the maximum favorable excursion (MFE) before
       the price reverses by reversal_threshold (2 ticks) from the MFE peak.
       RFE = MFE_remaining = MFE_total - unrealized_pnl_at_bar_t
       For training, since we treat every bar as a hypothetical long entry:
       RFE = max(prices[t:t+horizon] - prices[t]) in ticks
       
       Adverse label: Binary - did price move against by > adverse_threshold_ticks
       within the next horizon_bars?
       adverse = 1 if min(prices[t:t+horizon] - prices[t]) < -adverse_threshold_ticks
       
       Returns (rfe_labels, adverse_labels)
       """
 
3. exit_dataset.py:
   Similar to ScalpDataset but:
   - Includes trade context features (set to 0 during training since we train on
     hypothetical entries - the model learns to use them during inference)
   - No NaN filtering (every bar is a valid training sample)
 
4. exit_trainer.py:
   Loss = MSE(rfe_pred, rfe_label) + BCE(adverse_pred, adverse_label)
   Same walk-forward protocol as entry trainer.
   No regime segmentation needed (exit model is regime-agnostic, the entry model
   handles regime filtering).
 
5. exit_inference.py:
   class ExitInference:
       def __init__(self, model_path: str):
       def predict(self, feature_window: np.ndarray,
                   trade_context: TradeContext) -> ExitSignal:
           """
           TradeContext: unrealized_pnl, bars_since_entry, current_entry_prob,
                         current_entry_magnitude
           
           Appends trade context to each row of feature_window.
           Runs model inference.
           
           ExitSignal:
               rfe: float (predicted remaining favorable excursion in ticks)
               adverse_prob: float (probability of adverse move)
               should_exit: bool (rfe < 0.5 OR adverse_prob > 0.7)
               reason: str ("rfe_exhausted" | "adverse_risk" | "hold")
           """
 
TESTING:
- Test exit_labeler on synthetic price series with known MFE/adverse outcomes
- Verify model forward pass shape with 16 input features
- Train on 1 month as smoke test
- Compare exit model P&L vs static exits (2-tick target / 4-tick stop) on backtest



Phase 6: Filter Gate & Circuit Breakers
Duration: 1-2 weeks
What we’re building and why:
This is the rule-based safety layer that exists because neural networks have blind spots. The models were trained on historical data and can’t anticipate situations they’ve never seen — extreme spread blowouts, flash crashes, or the specific behavior of MES during a surprise Fed announcement. The filters and circuit breakers provide hard guardrails that no model output can override. They’re intentionally not learned because you want them to be interpretable (you can explain exactly why a trade was blocked), auditable (a log shows which rule fired), and robust (a model retrain can’t accidentally remove a safety check).
Filter gates (pre-entry):
Spread filter: No entry when bid-ask spread > 2 ticks. Wide spreads mean your fill will be worse and the round-trip cost eats more of your expected edge.
VPIN filter: No entry when VPIN > 0.8. Extreme VPIN means informed traders are dominating flow and you’re likely on the wrong side of an information asymmetry.
Time filter: No trading in first 5 minutes and last 5 minutes of RTH (9:30-9:35 and 15:55-16:00 ET). The opening has erratic price discovery and the close has thin books.
News blackout: No trading from 2 minutes before to 5 minutes after scheduled macro events (FOMC, CPI, NFP, etc.). Configurable event calendar.
Regime confidence: No entry when HMM max posterior < 0.5. If the regime classifier is uncertain, the model selection is unreliable.
Circuit breakers (always active):
Max loss per trade: 8-tick hard stop. Market order. No override.
Max hold time: 5 minutes. Forced exit. Scalp trades shouldn’t last longer — if they do, the original signal was wrong.
Daily loss limit: $200/day default. Halts all trading for the session. Prevents tilt cascades.
Consecutive loss limit: 3 consecutive losers triggers a 15-minute cooldown. Catches model miscalibration or sudden regime changes.
Kill switch: Manual override via HTTP endpoint. Immediately flattens all positions and halts trading.

🤖 Claude Code Prompt: Phase 6 — Filter Gate & Circuit Breakers
You are building Phase 6 of a CNN-LSTM MES scalping system. This phase creates the
rule-based safety layer (filters and circuit breakers).
 
PROJECT STRUCTURE (extend ./scalp-bot/):
  src/
    safety/
      __init__.py
      filters.py           (Pre-entry filter gates)
      circuit_breakers.py  (Always-active position safety limits)
      event_calendar.py    (Macro event schedule for news blackouts)
      kill_switch.py       (HTTP endpoint for manual emergency shutdown)
    config/
      filters.yaml         (Configurable thresholds)
      events.yaml          (Scheduled macro events for the next 3 months)
 
REQUIREMENTS:
 
1. filters.py:
   class FilterGate:
       def __init__(self, config: FilterConfig):
       def check(self, market_state: MarketState, regime_state: RegimeState
                ) -> FilterResult:
           """
           Returns FilterResult(passed=bool, blocked_by=Optional[str], reason=str)
           
           Checks in order (short-circuit on first failure):
           1. spread_filter: market_state.spread_ticks <= config.max_spread (default 2)
           2. vpin_filter: market_state.vpin <= config.max_vpin (default 0.8)
           3. time_filter: current time not in excluded windows
           4. news_filter: no scheduled event within config.pre_event_minutes (2) or
              config.post_event_minutes (5)
           5. regime_filter: regime_state.confidence >= config.min_regime_confidence (0.5)
           """
   
   @dataclass
   class FilterConfig:
       max_spread_ticks: float = 2.0
       max_vpin: float = 0.8
       excluded_time_ranges: List[Tuple[time, time]] = [(time(9,30), time(9,35)),
                                                         (time(15,55), time(16,0))]
       pre_event_minutes: int = 2
       post_event_minutes: int = 5
       min_regime_confidence: float = 0.5
 
2. circuit_breakers.py:
   class CircuitBreakers:
       def __init__(self, config: BreakerConfig):
       def check_position(self, position: Position) -> BreakerResult:
           """Check active position against hard limits."""
       def check_session(self, session_stats: SessionStats) -> BreakerResult:
           """Check session-level limits."""
       def on_trade_closed(self, trade_result: TradeResult) -> None:
           """Update consecutive loss counter."""
       
   @dataclass
   class BreakerConfig:
       max_loss_ticks: int = 8          # hard stop per trade
       max_hold_seconds: int = 300      # 5 minutes
       daily_loss_limit_usd: float = 200.0
       consecutive_loss_limit: int = 3
       cooldown_seconds: int = 900      # 15 minutes after consecutive losses
   
   BreakerResult: (triggered=bool, action="flatten"|"halt"|"cooldown", reason=str)
 
3. event_calendar.py:
   - Load events from YAML (date, time, event_name, importance)
   - is_in_blackout(current_dt, pre_minutes, post_minutes) -> bool
   - next_event(current_dt) -> Optional[Event]
 
4. kill_switch.py:
   - Simple FastAPI/aiohttp endpoint: POST /kill -> flatten all, halt trading
   - GET /status -> returns current trading state (active/halted/cooldown)
   - POST /resume -> re-enable trading after manual halt
 
5. filters.yaml and events.yaml: YAML config files with defaults.
 
TESTING:
- Test each filter independently with mock market state
- Test circuit breaker: simulate 8-tick loss -> verify flatten triggered
- Test consecutive losses: simulate 3 losers -> verify cooldown activated
- Test news blackout: mock event at 14:00, verify blocked at 13:58 and 14:04
- Integration: feed historical data with known bad conditions (wide spread events),
  verify filter blocks entries during those periods



Phase 7: Integrated Backtesting Framework
Duration: 3-4 weeks
What we’re building and why:
The backtester simulates the entire system pipeline with realistic execution modeling. This is where you find out whether the model’s statistical accuracy translates to actual profits after commissions, slippage, and latency. A model with 60% directional accuracy can still lose money if the average loser is larger than the average winner, or if slippage eats the edge on every trade.
Key realism requirements: fills must simulate actual market execution (you buy at the ask, not the mid), commissions must be modeled ($0.70 RT per MES contract via Tradovate), slippage must be configurable (default 0.25 ticks), and latency between signal generation and order execution must be simulated (default 50ms for Vultr-to-Tradovate). The backtester replays tick-level data in chronological order, running every component of the pipeline exactly as it would run live.
Target metrics:
Metric
Target
Why This Target
Net Profit Factor
> 1.5
Gross profit / gross loss after all costs. 1.5 means you make $1.50 for every $1 lost.
Sharpe Ratio (daily)
> 1.5
Annualized risk-adjusted return. Below 1.0 is noise, 1.5+ is a real signal.
Win Rate
> 52%
With scalping’s small avg win/loss, you need slightly more than coin flip.
Avg Win / Avg Loss
> 1.3
Reward-to-risk. The exit model should cut losers faster than winners.
Max Drawdown
< $500
On a 1-contract MES account, this is manageable.
Avg Trades / Day
5-20
Enough for statistical significance within a few weeks of live trading.
Avg Trade Duration
30s-3min
Consistent with scalping timeframe and 5-min max hold.
Max Consecutive Losses
< 8
More than 8 in a row suggests model breakdown, not variance.


Walk-forward protocol: Train on 6 months, validate on 1 month, test on 1 month. Roll forward in 1-month increments. Report metrics on out-of-sample months only. Monte Carlo permutation test: shuffle trade labels 10,000 times and verify actual performance exceeds the 95th percentile of random performance. This confirms the model captures real signal, not just noise.

🤖 Claude Code Prompt: Phase 7 — Integrated Backtesting Framework
You are building Phase 7 of a CNN-LSTM MES scalping system. This phase creates the
integrated backtesting engine that simulates the full pipeline with realistic execution.
 
PROJECT STRUCTURE (extend ./scalp-bot/):
  src/
    backtest/
      __init__.py
      engine.py            (Main backtesting engine - tick-level event replay)
      execution.py         (Simulated order execution with slippage and commission)
      position.py          (Position tracking, P&L calculation)
      metrics.py           (Performance metrics calculation)
      walk_forward.py      (Walk-forward orchestration across time folds)
      report.py            (Generate backtest report with charts)
      monte_carlo.py       (Permutation test for statistical significance)
 
REQUIREMENTS:
 
1. engine.py - Main backtest engine:
   class BacktestEngine:
       def __init__(self, feature_pipeline: FeaturePipeline,
                    regime_detector: RegimeDetector,
                    entry_inference: EntryInference,
                    exit_inference: ExitInference,
                    filter_gate: FilterGate,
                    circuit_breakers: CircuitBreakers,
                    execution_config: ExecutionConfig):
       
       def run(self, data_path: str, start_date: str, end_date: str) -> BacktestResult:
           """
           Replay historical data tick-by-tick in chronological order:
           
           For each record:
           1. Update feature pipeline -> get feature vector
           2. Every 60 seconds: update regime detector with 1-min bar
           3. If no position:
              a. Get entry signal from entry_inference
              b. Check filter_gate
              c. If entry signal fires and filters pass: open position via execution sim
           4. If in position:
              a. Check circuit breakers (hard stop, max hold time, daily loss)
              b. Get exit signal from exit_inference (with trade context)
              c. If exit signal fires or breaker triggers: close position
           5. Log all decisions, signals, fills
           
           Return BacktestResult with all trades and metrics.
           """
 
2. execution.py:
   class SimulatedExecution:
       def __init__(self, slippage_ticks: float = 0.25,
                    commission_per_side: float = 0.35,
                    latency_ms: float = 50):
       
       def fill_market_order(self, side: int, current_bbo: BBO,
                             signal_timestamp: int) -> Fill:
           """
           Long entry: fill at ask_price + slippage_ticks * tick_size
           Short entry: fill at bid_price - slippage_ticks * tick_size
           Commission: $0.35 per side ($0.70 round trip)
           Latency: order executes at the BBO that exists latency_ms after signal
           """
 
3. position.py:
   class Position:
       entry_price, entry_time, side, entry_bar_index
       def unrealized_pnl_ticks(self, current_price) -> float
       def bars_held(self, current_bar_index) -> int
       def close(self, exit_price, exit_time, reason) -> TradeResult
   
   class TradeResult:
       entry_price, exit_price, side, pnl_ticks, pnl_usd, commission_usd,
       net_pnl_usd, duration_seconds, exit_reason, entry_signal, exit_signal
 
4. metrics.py:
   def calculate_metrics(trades: List[TradeResult]) -> BacktestMetrics:
       """
       Calculate: profit_factor, sharpe_ratio (daily, annualized),
       win_rate, avg_win_loss_ratio, max_drawdown_usd, avg_trades_per_day,
       avg_trade_duration_seconds, max_consecutive_losses, total_pnl_usd,
       total_commission_usd, total_trades
       
       MES specifics: tick_size = 0.25 points, point_value = $5,
       so 1 tick = $1.25
       """
 
5. walk_forward.py:
   Orchestrate walk-forward:
   - For each fold (6mo train, 1mo val, 1mo test):
     a. Train entry + exit models (call Phase 4/5 trainers)
     b. Run backtest on test month
     c. Collect metrics
   - Aggregate across all folds
   - Report: per-fold metrics + aggregate metrics
 
6. monte_carlo.py:
   def permutation_test(trades: List[TradeResult], n_permutations: int = 10000
                       ) -> PermutationResult:
       """
       Shuffle trade PnL values randomly n_permutations times.
       Calculate Sharpe for each shuffle.
       Return: actual_sharpe, p_value (fraction of shuffles >= actual),
               percentile_95_sharpe
       If actual_sharpe > percentile_95: model has real signal (p < 0.05)
       """
 
7. report.py:
   Generate HTML report with:
   - Equity curve plot
   - Monthly returns table
   - Trade distribution (PnL histogram)
   - Regime breakdown (% of trades per regime, metrics per regime)
   - Filter statistics (how many signals were blocked by each filter)
   - Drawdown chart
   - Monte Carlo result visualization
 
TESTING:
- Test execution: verify long fill at ask + slippage, short fill at bid - slippage
- Test position P&L: known entry/exit -> verify correct pnl_ticks and pnl_usd
- Test metrics: synthetic trades with known outcomes -> verify all metrics correct
- Test monte carlo: random trades -> verify p_value near 0.5 (no signal)
- Integration: run full backtest on 1 month of data, verify report generates



Phase 8: L2 Data Upgrade & Model Retraining
Duration: 4-5 weeks
What we’re building and why:
At this point you have a complete, backtested, L1-based system. Phase 8 upgrades the data infrastructure from L1 (Databento TBBO/MBP-1) to full Level 2 depth via EdgeClear + Rithmic. Rithmic provides MBO-level data: individual order events (add, modify, cancel, fill) with full queue visibility at every price level. This unlocks features that are impossible with BBO data alone — things like how deep the book is, whether large resting orders are being absorbed, and the shape of the order book’s price-impact curve.
Critically, this upgrade is structured as a measured improvement, not a leap of faith. You retrain both models on the expanded feature set and A/B test against the L1 baseline. If the L2 features don’t improve performance (possible — MES has relatively thin books compared to equity LOBs), you keep the L1 models and save on data costs. Feature importance analysis (SHAP or permutation importance) tells you exactly which L2 features are contributing and which are noise.
New L2 features:
Multi-Level OFI (MLOFI): Extends OFI across the top 10 book levels with exponential decay weighting (closer levels count more). Captures supply/demand shifts that aren’t visible at the BBO — e.g., a large resting bid 3 ticks below the market that’s gradually being filled. This is the highest-value L2 feature based on the research.
Book Depth Ratio: Total bid size / total ask size across all visible levels. If there’s 5x more size on the bid side than the ask, there’s structural demand support even if the BBO looks balanced.
Order Book Slope: Linear regression of cumulative size vs price distance from mid. Steep slope = thin book = vulnerable to momentum moves. Flat slope = deep book = likely to absorb and mean-revert. Directly informs whether a move has continuation potential.
Queue Position Estimation: From MBO event data, estimate where your hypothetical order would sit in the queue at the best bid/ask. Informs limit-vs-market order decisions for future execution optimization.
Depth-Weighted Imbalance: Like book depth ratio but with exponential decay: size at the best level counts most, deeper levels count progressively less. More nuanced than the simple ratio.
Absorption Ratio: Rate at which aggressive orders consume passive resting orders at a price level. High absorption = strong directional conviction behind the flow, not just a fleeting spike.
Model retraining: Both entry and exit CNN-LSTM models are retrained with the expanded feature vector (~18-20 features instead of 12). The CNN input dimension grows but the architecture stays the same. A/B comparison on identical out-of-sample periods determines whether L2 models replace L1 models in production. Promotion criteria: improvement on at least 2 of 3 metrics (profit factor, Sharpe, win rate) without degrading max drawdown.

🤖 Claude Code Prompt: Phase 8 — L2 Data Upgrade & Model Retraining
You are building Phase 8 of a CNN-LSTM MES scalping system. This phase upgrades from
L1 (Databento BBO) to L2 (EdgeClear + Rithmic MBO) data and retrains models.
 
PROJECT STRUCTURE (extend ./scalp-bot/):
  src/
    data/
      rithmic_client.py    (Rithmic API integration for L2/MBO data)
      l2_provider.py       (DataProvider implementation for Rithmic L2 data)
    features/
      mlofi.py             (Multi-Level Order Flow Imbalance)
      book_depth_ratio.py  (Bid vs ask depth across all levels)
      book_slope.py        (Order book price-impact curve slope)
      queue_position.py    (Queue position estimation from MBO events)
      depth_imbalance.py   (Depth-weighted bid/ask imbalance)
      absorption.py        (Passive order absorption rate)
      pipeline_l2.py       (Extended FeaturePipeline with L1 + L2 features)
    evaluation/
      ab_comparison.py     (A/B test L1 vs L2 models on same out-of-sample data)
      feature_importance.py (Permutation importance for L2 feature contribution)
 
REQUIREMENTS:
 
1. rithmic_client.py:
   - Connect to Rithmic via their Protocol Buffer API
   - Subscribe to MES L2 market data (full order book depth)
   - Subscribe to MBO events (individual order add/modify/cancel/fill)
   - Parse into canonical format with additional fields:
     * book_levels: List[BookLevel] (price, size, order_count per level, bid/ask side)
     * mbo_event: Optional[MBOEvent] (order_id, side, price, size, event_type)
 
2. l2_provider.py:
   - Implement the DataProvider interface from Phase 1
   - get_historical: load from Parquet L2 files (Databento MBP-10 historical + Rithmic live)
   - subscribe_live: Rithmic live feed
 
3. mlofi.py - Multi-Level OFI:
   - For each of the top N levels (default 10):
     * Compute OFI at that level (same Cont et al. formula as L1 OFI)
     * Weight by exp(-decay * level_distance), decay = 0.5 default
   - MLOFI = sum of weighted per-level OFIs
   - Only the L1 OFI (level 0) was available before. MLOFI adds levels 1-9.
 
4. book_depth_ratio.py:
   total_bid = sum(size for all bid levels)
   total_ask = sum(size for all ask levels)
   ratio = log(total_bid / total_ask)  # log ratio for symmetry around 0
 
5. book_slope.py:
   - For bid side: fit linear regression of cumulative_size vs ticks_from_mid
   - For ask side: same
   - Feature = bid_slope - ask_slope (positive = bid side is steeper = thinner support)
 
6. queue_position.py:
   - Track order events at best bid/ask via MBO feed
   - Estimate queue depth (total orders ahead at best price)
   - Feature = estimated_queue_depth / average_trade_size
   - This is informational for future limit order execution optimization
 
7. depth_imbalance.py:
   - For each level i: weight_i = exp(-0.3 * i)
   - weighted_bid = sum(bid_size[i] * weight_i for i in levels)
   - weighted_ask = sum(ask_size[i] * weight_i for i in levels)
   - imbalance = (weighted_bid - weighted_ask) / (weighted_bid + weighted_ask)
 
8. absorption.py:
   - Track volume consumed at each price level per second
   - absorption_rate = volume_consumed_at_best / total_resting_at_best
   - Rolling 10-second average
   - High absorption = strong directional intent
 
9. pipeline_l2.py:
   Extends FeaturePipeline from Phase 2:
   - Includes all 12 L1 features
   - Adds 6 L2 features: mlofi, book_depth_ratio, book_slope,
     queue_position, depth_imbalance, absorption
   - Total features: ~18
   - Same WelfordNormalizer and staleness checks
 
10. ab_comparison.py:
    def compare_models(l1_model_path, l2_model_path, test_data_path) -> ComparisonResult:
        """
        Run both models on identical out-of-sample data.
        Compare: profit_factor, sharpe, win_rate, max_drawdown
        
        Promotion criteria:
        - L2 must beat L1 on at least 2 of 3: profit_factor, sharpe, win_rate
        - L2 must not degrade max_drawdown (L2_drawdown <= L1_drawdown * 1.1)
        """
 
11. feature_importance.py:
    - Permutation importance: for each L2 feature, shuffle it and measure
      accuracy/Sharpe drop
    - If any L2 feature has importance < 0.01 (less than 1% contribution),
      consider dropping it to reduce noise
 
TESTING:
- Test MLOFI with synthetic 10-level book data
- Test book_slope with known linear book shape
- Test absorption with known fill sequence
- A/B comparison on historical data: train L1 and L2 models, compare metrics
- Feature importance: verify at least 2 L2 features rank in top 50% of importance



Phase 9: Automated Retraining & Drift Detection
Duration: 2-3 weeks
What we’re building and why:
Market microstructure isn’t static. New market makers enter, exchange rules change, participant behavior evolves. A model trained 6 months ago may not capture today’s patterns. The retraining pipeline automates weekly model updates on a rolling training window, with drift detection that catches when the model’s inputs or outputs have shifted significantly from their training distribution. This prevents slow performance degradation that you wouldn’t notice until the monthly P&L statement.
Components:
Weekly scheduled retrain of both entry and exit models on a rolling 6-month window
Feature drift monitoring: alert when any feature’s rolling mean or variance drifts > 2 std from training stats
Performance drift: rolling 20-trade Sharpe. Below 0.5 triggers out-of-cycle retrain
Model versioning: all checkpoints stored with metadata for rollback
A/B gate: new model must beat current model on 2-week backtest before promotion

🤖 Claude Code Prompt: Phase 9 — Automated Retraining & Drift Detection
You are building Phase 9 of a CNN-LSTM MES scalping system. This phase automates
model retraining and drift detection.
 
PROJECT STRUCTURE (extend ./scalp-bot/):
  src/
    retraining/
      __init__.py
      scheduler.py         (Cron-based weekly retraining orchestration)
      drift_detector.py    (Feature and performance drift monitoring)
      model_registry.py    (Version control for model checkpoints)
      promotion.py         (A/B comparison before promoting new model)
      alerts.py            (Alerting on drift detection, failed retrains)
 
REQUIREMENTS:
 
1. scheduler.py:
   - Cron job (Sunday midnight CT) that:
     a. Pulls latest 6 months of data
     b. Retrains entry + exit models using Phase 4/5 trainers
     c. Runs A/B comparison vs current production model
     d. If promotion criteria met: update production model symlink
     e. Log everything, alert on failure
 
2. drift_detector.py:
   class DriftDetector:
       def __init__(self, reference_stats: Dict[str, FeatureStats]):
           """reference_stats: mean/std per feature from training data"""
       def check_feature_drift(self, recent_features: np.ndarray) -> List[DriftAlert]:
           """Flag if any feature's rolling mean/std drifts > 2 std from reference"""
       def check_performance_drift(self, recent_trades: List[TradeResult]) -> Optional[DriftAlert]:
           """Flag if rolling 20-trade Sharpe < 0.5"""
 
3. model_registry.py:
   - Store models at: models/{model_type}/{version}/
   - Metadata: training date range, feature stats, validation metrics, git hash
   - Symlink: models/{model_type}/production -> current best version
   - rollback(model_type, version) -> restore previous version
 
4. promotion.py:
   - Backtest new model on most recent 2 weeks
   - Compare vs current production model on same period
   - Promote only if: 2 of 3 (profit_factor, sharpe, win_rate) improve
   - Log comparison results regardless of promotion decision
 
5. alerts.py:
   - Feature drift alert -> log + optional webhook (Slack, email)
   - Performance drift alert -> trigger out-of-cycle retrain
   - Retrain failure alert -> log + webhook



Phase 10: Paper Trading & Live Integration
Duration: 4-6 weeks
What we’re building and why:
Paper trading is the final validation before risking real capital. You run the full system on the Vultr VPS against live market data, submitting orders through Tradovate’s (or EdgeClear’s, if the Phase 8 migration is complete) paper trading environment. The goal is to confirm that backtest performance translates to live execution — checking for discrepancies in fill quality, latency, data feed reliability, and system stability under real market conditions including high-volatility events.
Paper trading (3-4 weeks minimum): Full system on live data. Compare results against backtest expectations. Flag any metric discrepancy > 20%. Stress test through at least 2 high-volatility sessions.
Live transition: Week 1: 1 contract, first 2 hours of RTH only. Week 2: 1 contract, full RTH. Week 3+: scale if metrics match paper within 15%. Kill switch: auto-shutdown if daily loss exceeds $150 in Weeks 1-2.

🤖 Claude Code Prompt: Phase 10 — Paper Trading & Live Integration
You are building Phase 10 of a CNN-LSTM MES scalping system. This phase deploys the
full system for paper trading and live execution.
 
PROJECT STRUCTURE (extend ./scalp-bot/):
  src/
    execution/
      __init__.py
      tradovate_client.py  (Tradovate API wrapper for order execution)
      order_manager.py     (Order lifecycle management)
      paper_mode.py        (Paper trading execution simulator with live data)
    orchestrator/
      __init__.py
      main.py              (Main event loop - ties everything together)
      dashboard.py         (Real-time monitoring: P&L, regime, model confidence)
      logger.py            (Full audit trail of every decision and fill)
    config/
      live.yaml            (Live trading configuration)
 
REQUIREMENTS:
 
1. tradovate_client.py:
   - Authenticate with Tradovate API (OAuth2)
   - Place market orders for MES (entry and exit)
   - Query positions, fills, account balance
   - Handle connection drops with automatic reconnection
   - Support both paper and live environments via config flag
 
2. order_manager.py:
   class OrderManager:
       def __init__(self, client: TradovateClient, breakers: CircuitBreakers):
       async def enter(self, signal: EntrySignal) -> Optional[Position]:
           """Submit market order, track fill, create Position object."""
       async def exit(self, position: Position, signal: ExitSignal) -> TradeResult:
           """Submit exit order, track fill, record trade result."""
       async def emergency_flatten(self) -> None:
           """Flatten all positions immediately (kill switch)."""
 
3. main.py - Main orchestrator:
   async def run():
       """
       Main event loop:
       1. Initialize all components (data, features, HMM, models, filters, breakers)
       2. Subscribe to live data feed
       3. For each incoming record:
          - Update feature pipeline
          - Every 60s: update regime detector
          - If flat: check entry signal + filters -> enter if triggered
          - If in position: check exit signal + breakers -> exit if triggered
       4. Handle graceful shutdown (SIGTERM -> flatten -> save state)
       """
 
4. dashboard.py:
   - Serve real-time metrics via HTTP (for Grafana or simple web UI):
     * Current P&L (session, daily)
     * Open position details
     * HMM regime state + posteriors
     * Entry/exit model confidence scores
     * Feature health (staleness, out-of-distribution flags)
     * Execution quality (fill slippage vs expected)
     * Latency percentiles (signal-to-fill)
 
5. logger.py:
   - Log every: data record, feature vector, regime update, entry/exit signal,
     filter decision, order submission, fill, trade result
   - Format: structured JSON, one file per day
   - Retention: 90 days
 
6. live.yaml:
   mode: paper  # or "live"
   symbol: MES
   max_contracts: 1
   session_start: "09:30"
   session_end: "16:00"
   daily_loss_limit_usd: 200
   week1_daily_loss_limit_usd: 150
 
TESTING:
- Dry run: replay 1 day of historical data through the full orchestrator
- Paper test: run against live feed for 30 minutes, verify all components working
- Latency test: measure signal-to-order-submission time, target < 100ms
- Kill switch test: trigger kill switch, verify positions flatten within 2 seconds
- Reconnection test: simulate feed disconnect, verify auto-reconnect and state recovery



Phase 11: Optimization, Monitoring & Iteration
Duration: Ongoing
What we’re building and why:
Once live, the system enters continuous improvement mode. This includes hyperparameter tuning (entry/exit thresholds, lookback windows, prediction horizons), architecture experiments (attention mechanisms, transformer replacements, ensemble methods), and operational hardening (automated failover, data feed redundancy).
Key optimization targets:
Entry threshold: grid search the 0.65 confidence cutoff on rolling out-of-sample data
Lookback window: test 50, 100, 200 timestep windows for entry and exit models
Prediction horizon: tune the K-bar forward-return horizon per regime
Position sizing: Kelly criterion scaling based on model confidence
Architecture experiments:
Self-attention layer between CNN and LSTM (per DeepLOBATT) for multi-horizon forecasting
Transformer encoder replacing LSTM for better long-range dependencies
Ensemble of 3 models (different seeds) for more stable signals
DQN-based exit: reinforcement learning agent that learns optimal exit timing
L2 depth experiments: 20+ level MLOFI, MBO event-level features vs aggregated snapshots
2D CNN on order book images: encode full L2 book as image, use 2D convolutions

🤖 Claude Code Prompt: Phase 11 — Optimization & Iteration
You are building Phase 11 of a CNN-LSTM MES scalping system. This phase is ongoing
optimization and experimentation on the live system.
 
PROJECT STRUCTURE (extend ./scalp-bot/):
  src/
    optimization/
      __init__.py
      threshold_search.py  (Grid search for entry/exit thresholds)
      window_search.py     (Optimal lookback window search)
      horizon_search.py    (Optimal prediction horizon per regime)
      kelly.py             (Kelly criterion position sizing)
    experiments/
      __init__.py
      attention_model.py   (DeepLOBATT-style attention layer)
      transformer_model.py (Transformer encoder replacing LSTM)
      ensemble.py          (Multi-model ensemble with seed diversity)
      dqn_exit.py          (DQN-based reinforcement learning exit agent)
    ops/
      failover.py          (Automated VPS failover)
      feed_redundancy.py   (Secondary data feed fallback)
      compliance_log.py    (Audit trail for all decisions and fills)
 
REQUIREMENTS:
 
1. threshold_search.py:
   def search_entry_threshold(backtest_data, thresholds=[0.55, 0.60, 0.65, 0.70, 0.75]):
       """Run backtest at each threshold, report Sharpe/PF/win_rate per threshold.
       Use rolling 3-month out-of-sample windows, not in-sample optimization."""
 
2. kelly.py:
   def kelly_size(win_rate: float, avg_win: float, avg_loss: float,
                  fraction: float = 0.25) -> float:
       """Quarter-Kelly position sizing based on recent performance.
       kelly = (win_rate * avg_win - (1-win_rate) * avg_loss) / avg_win
       Return: fraction * kelly (capped at 1.0, floored at 0.1)"""
 
3. attention_model.py:
   Add self-attention between CNN output and LSTM input:
   - Multi-head attention (4 heads, dim=48) on CNN feature sequence
   - Allows model to weight different timesteps in the window
   - Compare performance vs base model
 
4. ensemble.py:
   class EnsembleInference:
       def __init__(self, model_paths: List[str]):  # 3 models, different seeds
       def predict(self, features, regime_state) -> EntrySignal:
           """Average direction_prob and magnitude across models.
           Also compute std of predictions - high std = model disagreement = reduce size."""
 
5. failover.py:
   - Health check endpoint on primary VPS
   - Secondary VPS polls health every 10 seconds
   - If primary unresponsive for 30 seconds: secondary takes over
   - Requires shared model storage (S3 or similar) and position state sync
 
6. compliance_log.py:
   - Append-only log of every: signal, filter decision, order, fill, P&L
   - Fields: timestamp, event_type, details (JSON), model_version, regime_state
   - Queryable for debugging ("show me all trades where VPIN > 0.7")
   - Daily summary export



Timeline Summary
Phase
Duration
Data
Key Deliverable
1. L1 Data Infrastructure
2-3 wk
L1
2yr MES TBBO+MBP-1 dataset + live capture
2. L1 Feature Pipeline
3-4 wk
L1
12 stationary feature vectors at 1s resolution
3. HMM Regime Detector
2-3 wk
L1
3-state HMM with posterior probabilities
4. Entry Model (L1)
4-6 wk
L1
CNN-LSTM entry with dual output heads
5. Exit Model (L1)
3-4 wk
L1
CNN-LSTM exit with RFE + adverse prob
6. Filters & Breakers
1-2 wk
—
Rule-based safety layer
7. Backtesting (L1)
3-4 wk
L1
Walk-forward validated performance metrics
8. L2 Upgrade & Retrain
4-5 wk
L2
Rithmic MBO, 6 new features, retrained models
9. Retraining Pipeline
2-3 wk
L2
Automated weekly retrain + drift detection
10. Paper → Live
4-6 wk
L2
Live system on 1 MES contract
11. Optimization
Ongoing
L2
Continuous improvement


Total estimated time to live trading: 28-40 weeks (7-10 months)
Phases 4 and 5 can overlap (exit model trains independently). You can paper trade on L1 features after Phase 7 while Phase 8 runs in parallel, potentially shaving 4-5 weeks off the critical path.

Technology Stack
Component
Technology
Rationale
Language
Python 3.11+ / asyncio
Existing codebase, PyTorch ecosystem, async I/O for real-time
Deep Learning
PyTorch 2.x
Dynamic graphs for debugging, TorchScript for production inference
Data Storage
Parquet (by date)
Columnar, compressed, fast reads for PyTorch DataLoaders
Market Data (L1)
Databento TBBO + MBP-1
BBO+trades per event. Execution-independent data source.
Market Data (L2)
EdgeClear + Rithmic (Ph 8+)
Full MBO depth. Individual order events. Queue visibility.
Execution (L1)
Tradovate API
Already funded, $0.70 RT. Execution only, no depth data via API.
Execution (L2)
EdgeClear (Ph 8+)
Full depth + execution. Replaces Tradovate.
HMM
hmmlearn + custom BOCPD
Proven library. Student-t via custom emission override.
Normalization
Welford’s algorithm
Numerically stable online z-score. No look-ahead.
Infrastructure
Vultr Chicago VPS
Near CME Aurora. ~15.6ms measured to Tradovate.
Monitoring
Grafana + Prometheus
Real-time dashboards, alerting on drift and performance.
Model Registry
MLflow or DVC
Version control for checkpoints, training metadata, reproducibility.


Key References
Zhang, Z., Zohren, S., & Roberts, S. (2019). DeepLOB: Deep Convolutional Neural Networks for Limit Order Books. IEEE Transactions on Signal Processing.
Kolm, P. N., Turiel, J., & Westray, N. (2021). Deep Order Flow Imbalance: Extracting Alpha at Multiple Horizons from the Limit Order Book. Mathematical Finance.
Kemper, L. (2025). Hybrid Regime Detection and Risk Management in Semiconductor Equities: A Bayesian HMM-LSTM Framework. SSRN.
Cont, R., Kukanov, A., & Stoikov, S. (2014). The Price Impact of Order Book Events. Journal of Financial Economics.
Easley, D., López de Prado, M., & O’Hara, M. (2012). Flow Toxicity and Liquidity in a High-Frequency World. Review of Financial Studies.
Kirilenko, A. et al. (2017). The Flash Crash: High-Frequency Trading in an Electronic Market. The Journal of Finance.
Briola, A. et al. (2025). Deep Limit Order Book Forecasting: A Microstructural Guide. Quantitative Finance.
Stoikov, S. (2018). The Micro-Price: A High-Frequency Estimator of Future Prices. Quantitative Finance.
Xu, K., Gould, M. D., & Howison, S. D. Multi-Level Order-Flow Imbalance in a Limit Order Book. Working Paper.
Adams, R. P. & MacKay, D. J. C. (2007). Bayesian Online Changepoint Detection. arXiv.
