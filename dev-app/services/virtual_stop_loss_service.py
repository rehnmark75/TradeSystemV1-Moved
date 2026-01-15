"""
Virtual Stop Loss Service for Scalping Mode

Monitors scalp positions in real-time using streaming prices
and closes positions programmatically when virtual SL threshold is breached.
This bypasses IG's minimum stop loss restrictions while maintaining
a broker-level safety net SL as crash protection.

Features:
- Real-time price monitoring via Lightstreamer MARKET subscription
- Configurable virtual SL distance per pair
- Automatic position sync from database
- Thread-safe position management
- Graceful reconnection handling
"""

import asyncio
import logging
from typing import Dict, Optional, List, Set
from dataclasses import dataclass, field
from datetime import datetime
from threading import Lock
import time

from services.db import SessionLocal
from services.models import TradeLog
from services.ig_orders import partial_close_position, get_point_value
from services.market_price_stream import MarketPriceStreamManager, MarketPrice
from config_virtual_stop import (
    VIRTUAL_STOP_LOSS_ENABLED,
    POSITION_SYNC_INTERVAL_SECONDS,
    CLOSE_ATTEMPT_COOLDOWN_SECONDS,
    get_virtual_sl_pips,
    is_virtual_sl_enabled,
    # Dynamic VSL imports
    DYNAMIC_VSL_ENABLED,
    SPREAD_AWARE_TRIGGERS_ENABLED,
    BASELINE_SPREAD_PIPS,
    MAX_SPREAD_PENALTY_PIPS,
    get_dynamic_vsl_config,
    is_dynamic_vsl_enabled,
)
# Import DEMO auth headers for closing trades (production headers are for streaming only)
from dependencies import get_ig_auth_headers

logger = logging.getLogger(__name__)


@dataclass
class ScalpPosition:
    """Tracked scalp position for virtual SL monitoring."""
    trade_id: int
    deal_id: str
    epic: str
    direction: str  # BUY or SELL
    entry_price: float
    virtual_sl_pips: float
    virtual_sl_price: float  # Initial/fixed VSL price
    position_size: float
    opened_at: datetime
    last_price: Optional[float] = None
    last_price_time: Optional[datetime] = None
    last_close_attempt: Optional[float] = None  # timestamp of last close attempt
    close_in_progress: bool = False

    # Dynamic VSL tracking fields
    peak_profit_pips: float = 0.0              # Maximum favorable excursion (MFE)
    current_stage: str = "initial"             # initial, breakeven, stage1, stage2
    breakeven_triggered: bool = False          # Has BE been triggered?
    early_lock_triggered: bool = False         # Has early_lock been triggered? (legacy)
    stage1_triggered: bool = False             # Has stage1 been triggered?
    stage2_triggered: bool = False             # Has stage2 been triggered?
    dynamic_vsl_price: Optional[float] = None  # Current dynamic VSL level (None = use fixed)


class VirtualStopLossService:
    """
    Real-time virtual stop loss monitoring for scalp trades.

    Architecture:
    1. Subscribes to MARKET price streams for active scalp positions
    2. Monitors adverse price movement in real-time (sub-second)
    3. Triggers position close when virtual SL breached
    4. Updates trade_log with closure details
    5. Syncs positions with database periodically
    """

    def __init__(self, auth_headers: dict):
        """
        Initialize VSL service.

        Args:
            auth_headers: IG API auth headers for streaming and trading
        """
        self.auth_headers = auth_headers
        self.stream_manager: Optional[MarketPriceStreamManager] = None
        self.positions: Dict[int, ScalpPosition] = {}  # trade_id -> ScalpPosition
        self.epic_to_trades: Dict[str, Set[int]] = {}  # epic -> set of trade_ids
        self._lock = Lock()
        self._running = False
        self._sync_task: Optional[asyncio.Task] = None
        self._stats = {
            "positions_tracked": 0,
            "vsl_triggered_count": 0,
            "price_updates_processed": 0,
            "last_sync_time": None,
        }

    async def start(self) -> bool:
        """
        Start the virtual stop loss service.

        Returns:
            True if started successfully
        """
        if not VIRTUAL_STOP_LOSS_ENABLED:
            logger.warning("[VSL] ⚠️ Virtual stop loss is disabled in config")
            return False

        logger.info("[VSL] 🚀 Starting Virtual Stop Loss Service...")

        try:
            # Initialize stream manager
            self.stream_manager = MarketPriceStreamManager(self.auth_headers)
            connected = await self.stream_manager.connect()

            if not connected:
                logger.error("[VSL] ❌ Failed to connect to price stream")
                return False

            self._running = True

            # Do initial position sync
            await self._sync_scalp_positions()

            # Start position sync loop
            self._sync_task = asyncio.create_task(self._position_sync_loop())

            logger.info("[VSL] ✅ Virtual Stop Loss Service started")
            logger.info(f"[VSL] 📊 Tracking {len(self.positions)} scalp positions")
            return True

        except Exception as e:
            logger.error(f"[VSL] ❌ Failed to start: {e}")
            return False

    async def stop(self):
        """Stop the virtual stop loss service."""
        logger.info("[VSL] 🛑 Stopping Virtual Stop Loss Service...")

        self._running = False

        # Cancel sync task
        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass

        # Disconnect stream
        if self.stream_manager:
            self.stream_manager.disconnect()

        logger.info("[VSL] Virtual Stop Loss Service stopped")
        logger.info(f"[VSL] Stats: {self._stats}")

    def add_scalp_position(self, trade: TradeLog) -> bool:
        """
        Add a scalp position for VSL monitoring.

        Args:
            trade: TradeLog record for the scalp trade

        Returns:
            True if added successfully
        """
        try:
            with self._lock:
                if trade.id in self.positions:
                    logger.debug(f"[VSL] Position {trade.id} already tracked")
                    return True

                # Check if VSL enabled for this pair
                if not is_virtual_sl_enabled(trade.symbol):
                    logger.info(f"[VSL] VSL disabled for {trade.symbol}, skipping")
                    return False

                # Get VSL configuration
                virtual_sl_pips = trade.virtual_sl_pips or get_virtual_sl_pips(trade.symbol)

                # Calculate virtual SL price
                point_value = get_point_value(trade.symbol)
                sl_distance = virtual_sl_pips * point_value

                direction = trade.direction.upper()
                if direction == "BUY":
                    # BUY: SL is below entry
                    virtual_sl_price = trade.entry_price - sl_distance
                else:  # SELL
                    # SELL: SL is above entry
                    virtual_sl_price = trade.entry_price + sl_distance

                # Create position tracker with persisted dynamic VSL state
                # This enables service restart resilience - state survives restarts!
                position = ScalpPosition(
                    trade_id=trade.id,
                    deal_id=trade.deal_id,
                    epic=trade.symbol,
                    direction=direction,
                    entry_price=trade.entry_price,
                    virtual_sl_pips=virtual_sl_pips,
                    virtual_sl_price=virtual_sl_price,
                    position_size=trade.current_size or 1.0,
                    opened_at=trade.timestamp or datetime.utcnow(),
                    # Load persisted dynamic VSL state from database
                    peak_profit_pips=getattr(trade, 'vsl_peak_profit_pips', None) or 0.0,
                    current_stage=getattr(trade, 'vsl_stage', None) or "initial",
                    breakeven_triggered=getattr(trade, 'vsl_breakeven_triggered', None) or False,
                    stage1_triggered=getattr(trade, 'vsl_stage1_triggered', None) or False,
                    stage2_triggered=getattr(trade, 'vsl_stage2_triggered', None) or False,
                    dynamic_vsl_price=getattr(trade, 'vsl_dynamic_sl_price', None),
                )

                self.positions[trade.id] = position

                # Track epic -> trades mapping
                if trade.symbol not in self.epic_to_trades:
                    self.epic_to_trades[trade.symbol] = set()
                self.epic_to_trades[trade.symbol].add(trade.id)

                # Subscribe to market prices if not already
                if self.stream_manager and trade.symbol not in self.stream_manager.get_subscribed_epics():
                    self.stream_manager.subscribe(
                        trade.symbol,
                        lambda price, epic=trade.symbol: self._on_price_update(price)
                    )

                self._stats["positions_tracked"] = len(self.positions)

                # Log position addition with state restoration info
                state_info = ""
                if position.breakeven_triggered or position.stage1_triggered or position.stage2_triggered:
                    state_info = f" [RESTORED: stage={position.current_stage}, peak={position.peak_profit_pips:.1f}pips"
                    if position.dynamic_vsl_price:
                        state_info += f", dyn_vsl={position.dynamic_vsl_price:.5f}"
                    state_info += "]"

                logger.info(f"[VSL] ✅ Added position: Trade {trade.id} {trade.symbol} {direction} "
                          f"@ {trade.entry_price:.5f}, VSL @ {virtual_sl_price:.5f} ({virtual_sl_pips} pips){state_info}")

                return True

        except Exception as e:
            logger.error(f"[VSL] ❌ Error adding position {trade.id}: {e}")
            return False

    def remove_position(self, trade_id: int):
        """
        Remove a position from VSL monitoring.

        Args:
            trade_id: Trade ID to remove
        """
        with self._lock:
            position = self.positions.pop(trade_id, None)
            if position:
                # Remove from epic mapping
                if position.epic in self.epic_to_trades:
                    self.epic_to_trades[position.epic].discard(trade_id)

                    # Unsubscribe if no more positions for this epic
                    if not self.epic_to_trades[position.epic]:
                        del self.epic_to_trades[position.epic]
                        if self.stream_manager:
                            self.stream_manager.unsubscribe(position.epic)

                self._stats["positions_tracked"] = len(self.positions)
                logger.info(f"[VSL] Removed position: Trade {trade_id}")

    def _on_price_update(self, price: MarketPrice):
        """
        Handle real-time price update - CHECK FOR VSL BREACH.

        This is called on every price tick from Lightstreamer.
        Must be fast and non-blocking.

        With Dynamic VSL enabled:
        1. Update dynamic VSL level based on profit (move to BE, stage1)
        2. Check for breach using the effective VSL (dynamic if set, else fixed)

        Args:
            price: MarketPrice with current BID/OFFER
        """
        try:
            self._stats["price_updates_processed"] += 1

            # Get positions for this epic
            with self._lock:
                trade_ids = list(self.epic_to_trades.get(price.epic, set()))

            for trade_id in trade_ids:
                with self._lock:
                    position = self.positions.get(trade_id)
                    if not position or position.close_in_progress:
                        continue

                    # Update last known price
                    position.last_price = price.mid
                    position.last_price_time = price.timestamp

                # Update dynamic VSL if enabled (may move VSL to BE or stage1)
                self._update_dynamic_vsl(position, price)

                # Determine effective VSL price (dynamic if set, else original fixed)
                effective_vsl_price = position.dynamic_vsl_price or position.virtual_sl_price

                # Check for VSL breach at effective level
                breached = self._check_vsl_breach_at_level(position, price, effective_vsl_price)

                if breached:
                    # Log which VSL level was breached
                    stage_info = f" (stage: {position.current_stage})" if position.dynamic_vsl_price else ""
                    logger.warning(f"[VSL BREACH] 🚨 Trade {position.trade_id} {position.epic} {position.direction}: "
                                 f"VSL @ {effective_vsl_price:.5f}{stage_info}")

                    # Check cooldown to prevent rapid-fire attempts
                    now = time.time()
                    if position.last_close_attempt:
                        if (now - position.last_close_attempt) < CLOSE_ATTEMPT_COOLDOWN_SECONDS:
                            continue

                    position.last_close_attempt = now
                    position.close_in_progress = True

                    # Trigger async close (don't block the callback)
                    asyncio.create_task(self._trigger_vsl_close(position, price))

        except Exception as e:
            logger.error(f"[VSL] Error processing price update: {e}")

    def _check_vsl_breach(self, position: ScalpPosition, price: MarketPrice) -> bool:
        """
        Check if virtual stop loss has been breached.

        Uses appropriate price for direction:
        - BUY positions: Check BID (you sell at BID to close)
        - SELL positions: Check OFFER (you buy at OFFER to close)

        Args:
            position: ScalpPosition to check
            price: Current MarketPrice

        Returns:
            True if VSL breached
        """
        if position.direction == "BUY":
            # For BUY: SL is below entry, breach when BID drops to/below SL
            check_price = price.bid
            breached = check_price <= position.virtual_sl_price
        else:  # SELL
            # For SELL: SL is above entry, breach when OFFER rises to/above SL
            check_price = price.offer
            breached = check_price >= position.virtual_sl_price

        if breached:
            logger.warning(f"[VSL BREACH] 🚨 Trade {position.trade_id} {position.epic} {position.direction}: "
                         f"Price {check_price:.5f} breached VSL {position.virtual_sl_price:.5f} "
                         f"(Entry: {position.entry_price:.5f}, VSL pips: {position.virtual_sl_pips})")

        return breached

    def _check_vsl_breach_at_level(self, position: ScalpPosition, price: MarketPrice, vsl_price: float) -> bool:
        """
        Check if virtual stop loss has been breached at a specific price level.

        Args:
            position: ScalpPosition to check
            price: Current MarketPrice
            vsl_price: The VSL price level to check against

        Returns:
            True if VSL breached
        """
        if position.direction == "BUY":
            check_price = price.bid
            breached = check_price <= vsl_price
        else:  # SELL
            check_price = price.offer
            breached = check_price >= vsl_price

        return breached

    def _calculate_profit_pips(self, position: ScalpPosition, price: MarketPrice) -> float:
        """
        Calculate current profit in pips for a position.

        Uses the appropriate price for direction:
        - BUY: Use BID (you sell at BID to close)
        - SELL: Use OFFER (you buy at OFFER to close)

        Args:
            position: ScalpPosition to calculate profit for
            price: Current MarketPrice

        Returns:
            Profit in pips (positive = profit, negative = loss)
        """
        point_value = get_point_value(position.epic)

        if position.direction == "BUY":
            # BUY: Profit when price > entry (use bid for closing)
            profit = (price.bid - position.entry_price) / point_value
        else:  # SELL
            # SELL: Profit when price < entry (use offer for closing)
            profit = (position.entry_price - price.offer) / point_value

        return profit

    def _get_effective_be_trigger(self, epic: str, current_spread_pips: float) -> float:
        """
        Get the effective breakeven trigger, adjusted for spread.

        When spread widens (news events, low liquidity), require more profit
        before triggering breakeven to avoid premature exits.

        Args:
            epic: Market epic
            current_spread_pips: Current spread in pips

        Returns:
            Effective breakeven trigger in pips
        """
        config = get_dynamic_vsl_config(epic)
        base_trigger = config['breakeven_trigger_pips']

        if not SPREAD_AWARE_TRIGGERS_ENABLED:
            return base_trigger

        # Add penalty if spread is wider than baseline
        if current_spread_pips > BASELINE_SPREAD_PIPS:
            spread_penalty = min(
                current_spread_pips - BASELINE_SPREAD_PIPS,
                MAX_SPREAD_PENALTY_PIPS
            )
            adjusted_trigger = base_trigger + spread_penalty
            logger.debug(f"[VSL] Spread-adjusted BE trigger for {epic}: "
                        f"{base_trigger} + {spread_penalty:.1f} = {adjusted_trigger:.1f} pips "
                        f"(spread: {current_spread_pips:.1f})")
            return adjusted_trigger

        return base_trigger

    def _update_dynamic_vsl(self, position: ScalpPosition, price: MarketPrice) -> None:
        """
        Update VSL level based on current profit - the core dynamic VSL logic.

        Stage progression (one-way, never goes back):
        1. Initial: VSL at -3 pips (or -4 for JPY) - the starting protection
        2. Breakeven: When +3 pips reached, VSL moves to entry +0.5 pip
        3. Stage 1: When +4.5 pips reached, VSL moves to entry +2 pips

        Args:
            position: ScalpPosition to update
            price: Current MarketPrice
        """
        if not is_dynamic_vsl_enabled():
            return

        # Calculate current profit
        current_profit_pips = self._calculate_profit_pips(position, price)

        # Update peak profit tracking (MFE)
        if current_profit_pips > position.peak_profit_pips:
            position.peak_profit_pips = current_profit_pips

        # Get dynamic VSL config for this pair
        config = get_dynamic_vsl_config(position.epic)
        point_value = get_point_value(position.epic)

        # Calculate current spread in pips
        spread_pips = (price.offer - price.bid) / point_value

        # Get spread-adjusted breakeven trigger
        effective_be_trigger = self._get_effective_be_trigger(position.epic, spread_pips)

        # Stage 2 check (highest priority - check this first)
        if not position.stage2_triggered and 'stage2_trigger_pips' in config:
            if current_profit_pips >= config['stage2_trigger_pips']:
                position.stage2_triggered = True
                position.current_stage = "stage2"
                lock_distance = config['stage2_lock_pips'] * point_value

                if position.direction == "BUY":
                    position.dynamic_vsl_price = position.entry_price + lock_distance
                else:  # SELL
                    position.dynamic_vsl_price = position.entry_price - lock_distance

                logger.info(f"[VSL] 🚀 Trade {position.trade_id} → STAGE 2: "
                           f"Profit={current_profit_pips:.1f} pips, "
                           f"Locking +{config['stage2_lock_pips']} pips @ {position.dynamic_vsl_price:.5f}")

                # Update stats
                self._stats["stage2_triggered_count"] = self._stats.get("stage2_triggered_count", 0) + 1

                # Persist state to database for restart resilience
                self._persist_vsl_state(position)
                return

        # Stage 1 check (second highest priority)
        if not position.stage1_triggered:
            if current_profit_pips >= config['stage1_trigger_pips']:
                position.stage1_triggered = True
                position.current_stage = "stage1"
                lock_distance = config['stage1_lock_pips'] * point_value

                if position.direction == "BUY":
                    position.dynamic_vsl_price = position.entry_price + lock_distance
                else:  # SELL
                    position.dynamic_vsl_price = position.entry_price - lock_distance

                logger.info(f"[VSL] 📈 Trade {position.trade_id} → STAGE 1: "
                           f"Profit={current_profit_pips:.1f} pips, "
                           f"Locking +{config['stage1_lock_pips']} pips @ {position.dynamic_vsl_price:.5f}")

                # Update stats
                self._stats["stage1_triggered_count"] = self._stats.get("stage1_triggered_count", 0) + 1

                # Persist state to database for restart resilience
                self._persist_vsl_state(position)
                return

        # Early lock check (between breakeven and stage1)
        # Triggers at +4 pips, moves VSL to -2.5 pips (reducing risk)
        if not position.early_lock_triggered and 'early_lock_trigger_pips' in config:
            if current_profit_pips >= config['early_lock_trigger_pips']:
                position.early_lock_triggered = True
                position.current_stage = "early_lock"
                lock_pips = config['early_lock_pips']  # Negative value (e.g., -2.5)
                lock_distance = lock_pips * point_value

                if position.direction == "BUY":
                    # BUY: negative lock_pips means below entry (e.g., entry - 2.5 pips)
                    position.dynamic_vsl_price = position.entry_price + lock_distance
                else:  # SELL
                    # SELL: negative lock_pips means above entry (e.g., entry + 2.5 pips)
                    position.dynamic_vsl_price = position.entry_price - lock_distance

                logger.info(f"[VSL] 🔒 Trade {position.trade_id} → EARLY LOCK: "
                           f"Profit={current_profit_pips:.1f} pips, "
                           f"VSL reduced to {lock_pips} pips @ {position.dynamic_vsl_price:.5f}")

                # Update stats
                self._stats["early_lock_triggered_count"] = self._stats.get("early_lock_triggered_count", 0) + 1

                # Persist state to database for restart resilience
                self._persist_vsl_state(position)
                return

        # Breakeven check (only if early_lock not triggered yet)
        if not position.breakeven_triggered:
            if current_profit_pips >= effective_be_trigger:
                position.breakeven_triggered = True
                position.current_stage = "breakeven"
                lock_distance = config['breakeven_lock_pips'] * point_value

                if position.direction == "BUY":
                    position.dynamic_vsl_price = position.entry_price + lock_distance
                else:  # SELL
                    position.dynamic_vsl_price = position.entry_price - lock_distance

                logger.info(f"[VSL] 🎯 Trade {position.trade_id} → BREAKEVEN: "
                           f"Profit={current_profit_pips:.1f} pips (trigger={effective_be_trigger:.1f}), "
                           f"VSL moved to {position.dynamic_vsl_price:.5f} (+{config['breakeven_lock_pips']} pips)")

                # Update stats
                self._stats["breakeven_triggered_count"] = self._stats.get("breakeven_triggered_count", 0) + 1

                # Persist state to database for restart resilience
                self._persist_vsl_state(position)

    def _persist_vsl_state(self, position: ScalpPosition) -> None:
        """
        Persist dynamic VSL state to database for restart resilience.

        This ensures that if the service restarts, the position will be
        restored with the correct stage, peak profit, and dynamic VSL price.

        Args:
            position: ScalpPosition with updated state to persist
        """
        try:
            with SessionLocal() as db:
                trade = db.query(TradeLog).filter(TradeLog.id == position.trade_id).first()
                if trade:
                    trade.vsl_stage = position.current_stage
                    trade.vsl_breakeven_triggered = position.breakeven_triggered
                    trade.vsl_stage1_triggered = position.stage1_triggered
                    trade.vsl_stage2_triggered = position.stage2_triggered
                    trade.vsl_peak_profit_pips = position.peak_profit_pips
                    trade.vsl_dynamic_sl_price = position.dynamic_vsl_price
                    db.commit()
                    logger.debug(f"[VSL] 💾 Persisted state for Trade {position.trade_id}: "
                               f"stage={position.current_stage}, peak={position.peak_profit_pips:.1f}pips")
        except Exception as e:
            logger.error(f"[VSL] ❌ Failed to persist state for Trade {position.trade_id}: {e}")

    async def _trigger_vsl_close(self, position: ScalpPosition, price: MarketPrice):
        """
        Execute position close when VSL breached.

        Args:
            position: ScalpPosition that triggered VSL
            price: Current MarketPrice at time of trigger
        """
        try:
            logger.info(f"[VSL CLOSE] 🔴 Triggering close for Trade {position.trade_id} {position.epic}")

            # Get DEMO auth headers for closing (production headers are for streaming only)
            # Trades are executed on demo account, so close must use demo credentials
            demo_headers = await get_ig_auth_headers()

            # Execute full position close using existing ig_orders function
            result = await partial_close_position(
                deal_id=position.deal_id,
                epic=position.epic,
                direction=position.direction,
                size_to_close=position.position_size,  # Close full position
                auth_headers=demo_headers  # Use DEMO headers, not production
            )

            if result.get('success'):
                logger.info(f"[VSL CLOSE] ✅ Trade {position.trade_id} closed successfully")
                self._stats["vsl_triggered_count"] += 1

                # Update database
                await self._update_trade_closed(position, price, reason="virtual_stop_loss")

                # Remove from tracking
                self.remove_position(position.trade_id)
            else:
                error = result.get('error', 'Unknown error')
                logger.error(f"[VSL CLOSE] ❌ Trade {position.trade_id} close failed: {error}")

                # Check if position already closed
                if result.get('already_closed'):
                    logger.info(f"[VSL] Position {position.trade_id} was already closed on broker")
                    await self._update_trade_closed(position, price, reason="broker_closed")
                    self.remove_position(position.trade_id)
                else:
                    # Reset close_in_progress to allow retry
                    position.close_in_progress = False

        except Exception as e:
            logger.error(f"[VSL CLOSE] ❌ Error closing Trade {position.trade_id}: {e}")
            position.close_in_progress = False

    async def _update_trade_closed(self, position: ScalpPosition, price: MarketPrice, reason: str):
        """
        Update trade_log with closure details including dynamic VSL stage info.

        Args:
            position: Closed ScalpPosition
            price: MarketPrice at closure
            reason: Closure reason string
        """
        try:
            with SessionLocal() as db:
                trade = db.query(TradeLog).filter(TradeLog.id == position.trade_id).first()
                if trade:
                    trade.status = "closed"
                    trade.closed_at = datetime.utcnow()
                    trade.virtual_sl_triggered = True
                    trade.virtual_sl_triggered_at = datetime.utcnow()

                    # Calculate exit price
                    if position.direction == "BUY":
                        exit_price = price.bid
                    else:
                        exit_price = price.offer
                    trade.exit_price_calculated = exit_price

                    # Calculate pips gained/lost
                    point_value = get_point_value(position.epic)
                    if position.direction == "BUY":
                        pips = (exit_price - position.entry_price) / point_value
                    else:
                        pips = (position.entry_price - exit_price) / point_value
                    trade.pips_gained = pips

                    # Record dynamic VSL stage info (if columns exist)
                    try:
                        if hasattr(trade, 'vsl_stage'):
                            trade.vsl_stage = position.current_stage
                        if hasattr(trade, 'vsl_breakeven_triggered'):
                            trade.vsl_breakeven_triggered = position.breakeven_triggered
                        if hasattr(trade, 'vsl_stage1_triggered'):
                            trade.vsl_stage1_triggered = position.stage1_triggered
                        if hasattr(trade, 'vsl_peak_profit_pips'):
                            trade.vsl_peak_profit_pips = position.peak_profit_pips
                    except Exception as stage_err:
                        logger.debug(f"[VSL] Could not update stage columns (may not exist yet): {stage_err}")

                    db.commit()

                    # Enhanced logging with stage info
                    stage_info = f", stage={position.current_stage}, peak={position.peak_profit_pips:.1f}pips" if position.breakeven_triggered else ""
                    logger.info(f"[VSL] 📝 Updated trade {position.trade_id}: "
                              f"exit={exit_price:.5f}, pips={pips:.1f}, reason={reason}{stage_info}")

        except Exception as e:
            logger.error(f"[VSL] Error updating trade {position.trade_id}: {e}")

    async def _position_sync_loop(self):
        """Periodically sync scalp positions from database."""
        while self._running:
            try:
                await asyncio.sleep(POSITION_SYNC_INTERVAL_SECONDS)
                await self._sync_scalp_positions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[VSL] Position sync error: {e}")
                await asyncio.sleep(10)  # Wait before retry

    async def _sync_scalp_positions(self):
        """Load active scalp positions from database."""
        try:
            with SessionLocal() as db:
                # Get active scalp trades
                scalp_trades = db.query(TradeLog).filter(
                    TradeLog.is_scalp_trade == True,
                    TradeLog.status.in_(['pending', 'tracking', 'break_even', 'trailing'])
                ).all()

                # Track which trades we found
                found_ids = set()

                # Add any new positions
                for trade in scalp_trades:
                    found_ids.add(trade.id)
                    if trade.id not in self.positions:
                        self.add_scalp_position(trade)

                # Remove positions that are no longer active
                with self._lock:
                    current_ids = set(self.positions.keys())
                    for trade_id in current_ids - found_ids:
                        self.remove_position(trade_id)

                self._stats["last_sync_time"] = datetime.utcnow().isoformat()
                logger.debug(f"[VSL] Synced {len(found_ids)} scalp positions from database")

        except Exception as e:
            logger.error(f"[VSL] Error syncing positions: {e}")

    def get_status(self) -> dict:
        """
        Get service status for monitoring/API.

        Returns:
            Status dictionary with service health info
        """
        with self._lock:
            positions_info = {
                tid: {
                    "epic": pos.epic,
                    "direction": pos.direction,
                    "entry_price": pos.entry_price,
                    "virtual_sl_price": pos.virtual_sl_price,
                    "virtual_sl_pips": pos.virtual_sl_pips,
                    "last_price": pos.last_price,
                    "last_price_time": pos.last_price_time.isoformat() if pos.last_price_time else None,
                    "close_in_progress": pos.close_in_progress,
                    # Dynamic VSL info
                    "dynamic_vsl_enabled": is_dynamic_vsl_enabled(),
                    "current_stage": pos.current_stage,
                    "dynamic_vsl_price": pos.dynamic_vsl_price,
                    "effective_vsl_price": pos.dynamic_vsl_price or pos.virtual_sl_price,
                    "breakeven_triggered": pos.breakeven_triggered,
                    "early_lock_triggered": pos.early_lock_triggered,
                    "stage1_triggered": pos.stage1_triggered,
                    "stage2_triggered": pos.stage2_triggered,
                    "peak_profit_pips": pos.peak_profit_pips,
                }
                for tid, pos in self.positions.items()
            }

        stream_status = self.stream_manager.get_status() if self.stream_manager else {}

        return {
            "enabled": VIRTUAL_STOP_LOSS_ENABLED,
            "dynamic_vsl_enabled": is_dynamic_vsl_enabled(),
            "running": self._running,
            "stream_connected": stream_status.get("connected", False),
            "positions_tracked": len(self.positions),
            "epics_subscribed": list(self.epic_to_trades.keys()),
            "positions": positions_info,
            "stats": self._stats,
            "stream_status": stream_status,
        }

    def get_position(self, trade_id: int) -> Optional[dict]:
        """
        Get details for a specific tracked position.

        Args:
            trade_id: Trade ID to lookup

        Returns:
            Position details or None
        """
        with self._lock:
            pos = self.positions.get(trade_id)
            if not pos:
                return None

            return {
                "trade_id": pos.trade_id,
                "deal_id": pos.deal_id,
                "epic": pos.epic,
                "direction": pos.direction,
                "entry_price": pos.entry_price,
                "virtual_sl_price": pos.virtual_sl_price,
                "virtual_sl_pips": pos.virtual_sl_pips,
                "position_size": pos.position_size,
                "last_price": pos.last_price,
                "opened_at": pos.opened_at.isoformat() if pos.opened_at else None,
                # Dynamic VSL info
                "current_stage": pos.current_stage,
                "dynamic_vsl_price": pos.dynamic_vsl_price,
                "effective_vsl_price": pos.dynamic_vsl_price or pos.virtual_sl_price,
                "breakeven_triggered": pos.breakeven_triggered,
                "early_lock_triggered": pos.early_lock_triggered,
                "stage1_triggered": pos.stage1_triggered,
                "stage2_triggered": pos.stage2_triggered,
                "peak_profit_pips": pos.peak_profit_pips,
            }


# Global service instance (set from main.py)
_vsl_service: Optional[VirtualStopLossService] = None


def get_vsl_service() -> Optional[VirtualStopLossService]:
    """Get the global VSL service instance."""
    return _vsl_service


def set_vsl_service(service: VirtualStopLossService):
    """Set the global VSL service instance."""
    global _vsl_service
    _vsl_service = service
