"""
Loss Prevention Filter (LPF) - Pattern-Based Trade Blocking

Rule-based penalty system that blocks high-probability losing trades.
Each rule assigns a penalty score. Penalties aggregate per-category
(max within each category, sum across categories). If total exceeds
threshold, trade is blocked.

Starts in monitor mode - logs decisions without blocking.
"""

import logging
import os
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional
from contextlib import contextmanager

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False

logger = logging.getLogger(__name__)


class LossPreventionFilter:
    """Evaluates signals against data-backed losing patterns and assigns penalty scores."""

    def __init__(self, backtest_mode: bool = False):
        self.backtest_mode = backtest_mode
        self._rules: List[Dict] = []
        self._config: Optional[Dict] = None
        self._loaded = False
        self._db_url = os.getenv(
            'STRATEGY_CONFIG_DATABASE_URL',
            'postgresql://postgres:postgres@postgres:5432/strategy_config'
        )
        self._load_config()

    @contextmanager
    def _get_connection(self):
        conn = None
        try:
            conn = psycopg2.connect(self._db_url)
            yield conn
        finally:
            if conn:
                conn.close()

    def _load_config(self):
        """Load rules and config from database."""
        if not PSYCOPG2_AVAILABLE:
            logger.warning("🛡️ LPF: psycopg2 not available - filter disabled")
            return

        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    # Load global config
                    cur.execute("SELECT * FROM loss_prevention_config WHERE id = 1")
                    row = cur.fetchone()
                    if row:
                        self._config = dict(row)
                    else:
                        logger.warning("🛡️ LPF: No config found in loss_prevention_config")
                        return

                    # Load enabled rules
                    cur.execute("""
                        SELECT rule_name, category, penalty, condition_config, apply_in_backtest
                        FROM loss_prevention_rules
                        WHERE is_enabled = TRUE
                        ORDER BY category, penalty DESC
                    """)
                    self._rules = [dict(r) for r in cur.fetchall()]

            self._loaded = True
            mode = self._config.get('block_mode', 'monitor')
            threshold = self._config.get('penalty_threshold', 0.60)
            logger.info(f"🛡️ LPF: Loaded {len(self._rules)} rules | mode={mode} | threshold={threshold}")

        except Exception as e:
            logger.error(f"🛡️ LPF: Failed to load config: {e}")
            self._loaded = False

    @property
    def is_enabled(self) -> bool:
        return bool(self._loaded and self._config and self._config.get('is_enabled', False))

    @property
    def block_mode(self) -> str:
        if not self._config:
            return 'monitor'
        return self._config.get('block_mode', 'monitor')

    @property
    def penalty_threshold(self) -> float:
        if not self._config:
            return 0.60
        return float(self._config.get('penalty_threshold', 0.60))

    def evaluate(self, signal: Dict, signal_timestamp: Optional[datetime] = None) -> Dict:
        """
        Evaluate a signal against all rules.

        Returns dict with:
            - allowed: bool
            - total_penalty: float
            - triggered_rules: list of {rule_name, category, penalty}
            - decision: 'allowed' | 'would_block' | 'blocked'
        """
        if not self.is_enabled:
            return {'allowed': True, 'total_penalty': 0.0, 'triggered_rules': [], 'decision': 'allowed'}

        triggered = []
        for rule in self._rules:
            # Skip backtest-excluded rules
            if self.backtest_mode and not rule.get('apply_in_backtest', True):
                continue

            if self._check_rule(rule, signal, signal_timestamp):
                triggered.append({
                    'rule_name': rule['rule_name'],
                    'category': rule['category'],
                    'penalty': float(rule['penalty'])
                })

        # Aggregate: max penalty per category, sum across categories
        category_penalties = {}
        for t in triggered:
            cat = t['category']
            pen = t['penalty']
            if cat not in category_penalties:
                category_penalties[cat] = pen
            else:
                # For boosts (negative), take min (most negative); for penalties, take max
                if pen < 0:
                    category_penalties[cat] = min(category_penalties[cat], pen)
                else:
                    category_penalties[cat] = max(category_penalties[cat], pen)

        total_penalty = sum(category_penalties.values())

        would_block = total_penalty >= self.penalty_threshold
        if would_block and self.block_mode == 'block':
            decision = 'blocked'
            allowed = False
        elif would_block:
            decision = 'would_block'
            allowed = True  # Monitor mode - allow but log
        else:
            decision = 'allowed'
            allowed = True

        result = {
            'allowed': allowed,
            'total_penalty': round(total_penalty, 2),
            'triggered_rules': triggered,
            'decision': decision,
            'category_penalties': {k: round(v, 2) for k, v in category_penalties.items()}
        }

        # Log decision
        epic = signal.get('epic', 'Unknown')
        signal_type = signal.get('signal_type', '?')
        rule_names = [t['rule_name'] for t in triggered]

        if decision == 'blocked':
            logger.warning(f"🛡️🚫 LPF BLOCKED: {epic} {signal_type} | penalty={total_penalty:.2f} | rules={rule_names}")
        elif decision == 'would_block':
            logger.warning(f"🛡️👁️ LPF WOULD BLOCK: {epic} {signal_type} | penalty={total_penalty:.2f} | rules={rule_names}")
        elif triggered:
            logger.info(f"🛡️✅ LPF ALLOWED: {epic} {signal_type} | penalty={total_penalty:.2f} | rules={rule_names}")

        # Log decision to database
        if self._config and self._config.get('log_decisions', True):
            self._log_decision(signal, result, signal_timestamp)

        return result

    def _check_rule(self, rule: Dict, signal: Dict, signal_timestamp: Optional[datetime] = None) -> bool:
        """Check if a rule matches the given signal."""
        cond = rule.get('condition_config', {})
        if isinstance(cond, str):
            cond = json.loads(cond)

        rule_type = cond.get('type', '')

        try:
            if rule_type == 'pair':
                return self._check_pair(cond, signal)
            elif rule_type == 'pair_and_confidence':
                return self._check_pair(cond, signal) and self._check_confidence_min(cond, signal)
            elif rule_type == 'pair_and_regime':
                return self._check_pair(cond, signal) and self._check_regime(cond, signal)
            elif rule_type == 'pair_and_hours':
                return self._check_pair(cond, signal) and self._check_hours(cond, signal, signal_timestamp)
            elif rule_type == 'confidence_range':
                return self._check_confidence_range(cond, signal)
            elif rule_type == 'multi_threshold':
                return self._check_multi_threshold(cond, signal)
            elif rule_type == 'day_of_week':
                return self._check_day_of_week(cond, signal, signal_timestamp)
            elif rule_type == 'hour_utc':
                return self._check_hours(cond, signal, signal_timestamp)
            elif rule_type == 'regime':
                return self._check_regime(cond, signal)
            elif rule_type == 'regime_and_alignment':
                return self._check_regime_and_alignment(cond, signal)
            elif rule_type == 'direction_and_indicator':
                return self._check_direction_and_indicator(cond, signal)
            elif rule_type == 'hour_and_regime':
                return self._check_hours(cond, signal, signal_timestamp) and self._check_regime(cond, signal)
            elif rule_type == 'session_and_regime':
                return self._check_session(cond, signal) and self._check_regime(cond, signal)
            elif rule_type == 'indicator_threshold':
                return self._check_indicator_threshold(cond, signal)
            elif rule_type == 'move_exhaustion':
                return self._check_move_exhaustion(cond, signal)
            else:
                logger.debug(f"🛡️ LPF: Unknown rule type '{rule_type}' in {rule['rule_name']}")
                return False
        except Exception as e:
            logger.debug(f"🛡️ LPF: Error checking rule {rule['rule_name']}: {e}")
            return False

    # ---- Rule checkers ----

    def _check_pair(self, cond: Dict, signal: Dict) -> bool:
        epic = signal.get('epic', '')
        return cond.get('epic_contains', '') in epic

    def _check_confidence_min(self, cond: Dict, signal: Dict) -> bool:
        conf = self._get_confidence(signal)
        if conf is None:
            return False
        return conf >= cond.get('min_confidence', 1.0)

    def _check_confidence_range(self, cond: Dict, signal: Dict) -> bool:
        conf = self._get_confidence(signal)
        if conf is None:
            return False
        min_c = cond.get('min_confidence', 0.0)
        max_c = cond.get('max_confidence', 1.0)
        return min_c <= conf < max_c

    def _check_regime(self, cond: Dict, signal: Dict) -> bool:
        regime = self._get_regime(signal)
        return regime == cond.get('regime', '')

    def _check_regime_and_alignment(self, cond: Dict, signal: Dict) -> bool:
        if not self._check_regime(cond, signal):
            return False
        aligned = signal.get('all_timeframes_aligned', None)
        if aligned is None:
            return False
        expected = cond.get('all_timeframes_aligned', True)
        return bool(aligned) == expected

    def _check_hours(self, cond: Dict, signal: Dict, signal_timestamp: Optional[datetime] = None) -> bool:
        hour = self._get_hour_utc(signal, signal_timestamp)
        if hour is None:
            return False
        return hour in cond.get('hours', [])

    def _check_day_of_week(self, cond: Dict, signal: Dict, signal_timestamp: Optional[datetime] = None) -> bool:
        day = self._get_day_of_week(signal, signal_timestamp)
        if day is None:
            return False
        return day in cond.get('days', [])

    def _check_multi_threshold(self, cond: Dict, signal: Dict) -> bool:
        conditions = cond.get('conditions', {})
        for field, threshold in conditions.items():
            if field == 'confidence':
                val = self._get_confidence(signal)
            else:
                val = signal.get(field)
            if val is None:
                return False
            try:
                if float(val) < float(threshold):
                    return False
            except (ValueError, TypeError):
                return False
        return True

    def _check_direction_and_indicator(self, cond: Dict, signal: Dict) -> bool:
        direction = signal.get('signal_type', '').upper()
        expected_dir = cond.get('direction', '').upper()
        if direction != expected_dir:
            return False
        return self._check_indicator_threshold(cond, signal)

    def _check_session(self, cond: Dict, signal: Dict) -> bool:
        session = signal.get('market_session', '') or signal.get('session', '')
        return str(session).lower() == cond.get('session', '').lower()

    def _check_indicator_threshold(self, cond: Dict, signal: Dict) -> bool:
        indicator = cond.get('indicator', '')
        val = signal.get(indicator)
        if val is None:
            return False
        try:
            val = float(val)
        except (ValueError, TypeError):
            return False

        if 'max_value' in cond:
            return val < float(cond['max_value'])
        if 'min_value' in cond:
            return val >= float(cond['min_value'])
        return False

    def _check_move_exhaustion(self, cond: Dict, signal: Dict) -> bool:
        """Check if multiple exhaustion dimensions converge against trade direction."""
        direction = signal.get('signal_type', '').upper()
        if direction == 'BEAR':
            direction = 'SELL'
        elif direction == 'BULL':
            direction = 'BUY'

        min_triggers = cond.get('min_triggers', 3)
        triggers = 0
        checked = 0

        # 1. RSI extreme against direction
        rsi = signal.get('rsi')
        if rsi is not None:
            checked += 1
            rsi_threshold = cond.get('rsi_threshold', 25)
            if direction == 'SELL' and float(rsi) < rsi_threshold:
                triggers += 1
            elif direction == 'BUY' and float(rsi) > (100 - rsi_threshold):
                triggers += 1

        # 2. Stochastic extreme against direction
        stoch = signal.get('stoch_k')
        if stoch is not None:
            checked += 1
            stoch_threshold = cond.get('stoch_threshold', 15)
            if direction == 'SELL' and float(stoch) < stoch_threshold:
                triggers += 1
            elif direction == 'BUY' and float(stoch) > (100 - stoch_threshold):
                triggers += 1

        # 3. EMA overextension (price too far from mean)
        ema_dist = signal.get('ema_distance_pips')
        if ema_dist is not None:
            checked += 1
            ema_max_pips = cond.get('ema_max_distance_pips', 40)
            if abs(float(ema_dist)) > ema_max_pips:
                triggers += 1

        # 4. Low efficiency ratio (choppy/exhausted market)
        er = signal.get('efficiency_ratio')
        if er is not None:
            checked += 1
            er_max = cond.get('er_max', 0.15)
            if float(er) < er_max:
                triggers += 1

        # 5. Declining volume
        vol_trend = signal.get('volume_trend', '')
        if vol_trend:
            checked += 1
            if vol_trend == 'decreasing':
                triggers += 1

        # 6. ATR percentile extreme (volatility spike = exhaustion)
        atr_pct = signal.get('atr_percentile')
        if atr_pct is not None:
            checked += 1
            atr_pct_min = cond.get('atr_pct_min', 0.85)
            if float(atr_pct) > atr_pct_min:
                triggers += 1

        # Need minimum data to make a decision
        if checked < 3:
            return False

        return triggers >= min_triggers

    # ---- Helpers ----

    def _get_confidence(self, signal: Dict) -> Optional[float]:
        conf = signal.get('confidence_score') or signal.get('confidence')
        if conf is None:
            return None
        try:
            conf = float(conf)
            # Normalize: if >1, treat as percentage
            if conf > 1.0:
                conf = conf / 100.0
            return conf
        except (ValueError, TypeError):
            return None

    def _get_regime(self, signal: Dict) -> str:
        regime = signal.get('market_regime_detected') or signal.get('market_regime') or ''
        return str(regime).lower().replace(' ', '_')

    def _get_hour_utc(self, signal: Dict, signal_timestamp: Optional[datetime] = None) -> Optional[int]:
        """Get hour UTC from signal timestamp (backtest-safe)."""
        if self.backtest_mode and signal_timestamp:
            return signal_timestamp.hour
        # Try signal's own timestamp fields
        for field in ['signal_timestamp', 'market_timestamp', 'timestamp']:
            ts = signal.get(field)
            if ts and isinstance(ts, datetime):
                return ts.hour
        # Fall back to current time (live mode)
        if not self.backtest_mode:
            return datetime.now(timezone.utc).hour
        return None

    def _get_day_of_week(self, signal: Dict, signal_timestamp: Optional[datetime] = None) -> Optional[int]:
        """Get day of week (0=Monday) from signal timestamp (backtest-safe)."""
        if self.backtest_mode and signal_timestamp:
            return signal_timestamp.weekday()
        for field in ['signal_timestamp', 'market_timestamp', 'timestamp']:
            ts = signal.get(field)
            if ts and isinstance(ts, datetime):
                return ts.weekday()
        if not self.backtest_mode:
            return datetime.now(timezone.utc).weekday()
        return None

    def _log_decision(self, signal: Dict, result: Dict, signal_timestamp: Optional[datetime] = None):
        """Log decision to loss_prevention_decisions table."""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO loss_prevention_decisions
                            (epic, signal_type, confidence, total_penalty, triggered_rules, decision, signal_timestamp)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (
                        signal.get('epic', ''),
                        signal.get('signal_type', ''),
                        self._get_confidence(signal),
                        result['total_penalty'],
                        json.dumps(result['triggered_rules']),
                        result['decision'],
                        signal_timestamp or datetime.now(timezone.utc)
                    ))
                conn.commit()
        except Exception as e:
            logger.debug(f"🛡️ LPF: Failed to log decision: {e}")
