# core/trading/risk_manager.py
"""
Risk Manager - ENHANCED with Strategy Testing Mode
Handles risk calculations, position sizing, and safety checks

NEW FEATURES:
- Strategy Testing Mode bypass for signal development
- Configurable risk validation levels
- Simple validation for strategy tuning
- Easy switch between testing and live modes
- Database-driven configuration (NO FALLBACK to config.py)

CRITICAL: Database is the ONLY source of truth for configuration.
If database is unavailable, the system will FAIL rather than use fallback values.
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

# Import scanner config service for database-driven configuration
try:
    from forex_scanner.services.scanner_config_service import get_scanner_config
    SCANNER_CONFIG_AVAILABLE = True
except ImportError:
    try:
        from services.scanner_config_service import get_scanner_config
        SCANNER_CONFIG_AVAILABLE = True
    except ImportError:
        SCANNER_CONFIG_AVAILABLE = False


class RiskManager:
    """
    Manages risk calculations, position sizing, and safety checks
    ENHANCED: With Strategy Testing Mode for development

    CRITICAL: All configuration comes from database - no fallback to config.py
    """

    def __init__(self,
                 logger: Optional[logging.Logger] = None,
                 max_daily_loss: float = None,
                 max_positions: int = None,
                 default_risk_percent: float = None,
                 account_balance: float = None,
                 testing_mode: bool = None,
                 scanner_config = None):

        self.logger = logger or logging.getLogger(__name__)

        # =============================================================================
        # FAIL-FAST DATABASE CONFIGURATION - NO FALLBACK ALLOWED
        # =============================================================================
        if scanner_config:
            self._scanner_cfg = scanner_config
        elif SCANNER_CONFIG_AVAILABLE:
            try:
                self._scanner_cfg = get_scanner_config()
            except Exception as e:
                raise RuntimeError(
                    f"CRITICAL: Failed to load scanner config from database: {e} - no fallback allowed"
                )
        else:
            raise RuntimeError(
                "CRITICAL: Scanner config service not available - database is REQUIRED, no fallback allowed"
            )

        if not self._scanner_cfg:
            raise RuntimeError(
                "CRITICAL: Scanner config returned None - database is REQUIRED, no fallback allowed"
            )

        # =============================================================================
        # STRATEGY TESTING MODE configuration (from DATABASE)
        # =============================================================================
        self.testing_mode = testing_mode if testing_mode is not None else self._scanner_cfg.strategy_testing_mode
        self.disable_account_risk = self._scanner_cfg.disable_account_risk_validation if self.testing_mode else False
        self.disable_position_sizing = self._scanner_cfg.disable_position_sizing if self.testing_mode else False

        # =============================================================================
        # RISK PARAMETERS (from DATABASE, with passed params taking precedence)
        # =============================================================================
        # Calculate max_daily_loss from percentage and account balance
        db_account_balance = self._scanner_cfg.account_balance or 10000.0
        db_max_daily_loss_percent = self._scanner_cfg.max_daily_loss_percent or 5.0
        calculated_max_daily_loss = db_account_balance * (db_max_daily_loss_percent / 100.0)

        self.max_daily_loss = max_daily_loss or calculated_max_daily_loss
        self.max_positions = max_positions or self._scanner_cfg.max_open_positions or 5
        self.default_risk_percent = default_risk_percent or self._scanner_cfg.risk_per_trade_percent or 2.0
        self.account_balance = account_balance or db_account_balance

        # Risk tracking
        self.daily_loss = 0.0
        self.current_positions = 0
        self.daily_trades = []
        self.last_reset_date = datetime.now().date()

        # =============================================================================
        # RISK LIMITS (from DATABASE)
        # =============================================================================
        self.min_position_size = self._scanner_cfg.min_position_size or 0.01
        self.max_position_size = self._scanner_cfg.max_position_size or 1.0
        self.max_risk_per_trade = self._scanner_cfg.max_risk_per_trade or 5.0
        self.default_stop_distance = self._scanner_cfg.default_stop_distance or 20

        # TESTING MODE: Relaxed limits for strategy development (from DATABASE)
        if self.testing_mode:
            self.testing_max_stop_percent = self._scanner_cfg.testing_max_stop_percent or 20.0
            self.testing_min_confidence = self._scanner_cfg.testing_min_confidence or 0.0
        else:
            self.testing_max_stop_percent = 20.0
            self.testing_min_confidence = 0.0

        # Additional safety parameters (from DATABASE)
        self.max_daily_trades = self._scanner_cfg.max_daily_trades or 10
        self.max_concurrent_positions = self._scanner_cfg.max_open_positions or 3
        self.max_trades_per_pair = self._scanner_cfg.max_trades_per_pair or 3
        self.daily_profit_target_percent = self._scanner_cfg.daily_profit_target_percent or 3.0
        self.stop_on_daily_target = self._scanner_cfg.stop_on_daily_target or False
        self.min_account_balance = self._scanner_cfg.min_account_balance or 1000.0
        self.emergency_stop_enabled = self._scanner_cfg.emergency_stop_enabled and not self.testing_mode

        # PIP value configuration for different pairs (hardcoded, not config-dependent)
        pip_value_default = 1.0
        self.pip_values = {
            'CS.D.EURUSD.CEEM.IP': pip_value_default,
            'CS.D.GBPUSD.MINI.IP': pip_value_default,
            'CS.D.USDJPY.MINI.IP': 0.1,  # Different pip value for JPY pairs
            'CS.D.EURJPY.MINI.IP': 0.1,
            'CS.D.AUDJPY.MINI.IP': 0.1,
            'CS.D.GBPJPY.MINI.IP': 0.1,
            'default': pip_value_default
        }

        # Log initialization
        if self.testing_mode:
            self.logger.info("[DB] RiskManager initialized in STRATEGY TESTING MODE")
            self.logger.info("   Account risk validation: BYPASSED")
            self.logger.info("   Position sizing: SIMPLIFIED")
            self.logger.info("   Emergency stops: DISABLED")
            self.logger.info("   Remember to disable testing mode for live trading!")
        else:
            self.logger.info("[DB] RiskManager initialized in LIVE TRADING MODE")
            self.logger.info(f"   Max daily loss: ${self.max_daily_loss:.2f} ({db_max_daily_loss_percent}%)")
            self.logger.info(f"   Max positions: {self.max_positions}")
            self.logger.info(f"   Default risk: {self.default_risk_percent}%")
            self.logger.info(f"   Account balance: ${self.account_balance:.2f}")
            self.logger.info(f"   Config source: {self._scanner_cfg.source}")

    def validate_risk_parameters(self, signal: Dict) -> Tuple[bool, str]:
        """
        ENHANCED: Validate signal risk parameters with testing mode support
        """
        try:
            # STRATEGY TESTING MODE: Simple validation
            if self.testing_mode:
                return self._validate_risk_testing_mode(signal)

            # LIVE TRADING MODE: Full validation
            return self._validate_risk_live_mode(signal)

        except Exception as e:
            self.logger.error(f"Risk validation error: {e}")
            if self.testing_mode:
                return True, f"Testing mode - validation error ignored: {str(e)}"
            else:
                return False, f"Validation error: {str(e)}"

    def _validate_risk_testing_mode(self, signal: Dict) -> Tuple[bool, str]:
        """
        STRATEGY TESTING MODE: Simple validation for strategy development

        Only validates basic signal structure without complex risk calculations
        """
        try:
            # === BASIC SIGNAL VALIDATION ===

            # Check required fields exist
            epic = signal.get('epic')
            signal_type = signal.get('signal_type', '').upper()

            if not epic:
                return False, "Missing epic"

            if not signal_type:
                return False, "Missing signal type"

            # Validate signal type
            valid_types = ['BUY', 'SELL', 'BULL', 'BEAR', 'TEST_BULL', 'TEST_BEAR']
            if signal_type not in valid_types:
                return False, f"Invalid signal type: {signal_type}"

            # === PRICE VALIDATION (BASIC) ===

            # Extract entry price
            entry_price = self._extract_price(signal)
            if not entry_price or entry_price <= 0:
                return False, "Invalid or missing entry price"

            # Extract stop loss (optional for strategy testing)
            stop_loss = self._extract_stop_loss(signal)

            # If stop loss exists, do basic validation
            if stop_loss:
                if stop_loss <= 0:
                    return False, "Invalid stop loss price"

                # Basic direction check (warn but don't fail)
                if signal_type in ['BUY', 'BULL', 'TEST_BULL']:
                    if stop_loss >= entry_price:
                        self.logger.warning(f"Stop loss {stop_loss:.5f} should be below entry {entry_price:.5f} for BUY signal")
                elif signal_type in ['SELL', 'BEAR', 'TEST_BEAR']:
                    if stop_loss <= entry_price:
                        self.logger.warning(f"Stop loss {stop_loss:.5f} should be above entry {entry_price:.5f} for SELL signal")

                # Basic reasonableness check (very lenient for testing)
                price_diff_percent = abs(entry_price - stop_loss) / entry_price * 100
                if price_diff_percent > self.testing_max_stop_percent:
                    return False, f"Stop loss unreasonably wide: {price_diff_percent:.1f}% > {self.testing_max_stop_percent}% (epic: {epic})"

                # Log the price difference for debugging (but don't fail)
                self.logger.debug(f"Testing mode - Price difference: {price_diff_percent:.2f}% (entry: {entry_price:.5f}, stop: {stop_loss:.5f})")

            # === CONFIDENCE VALIDATION (OPTIONAL) ===

            confidence = signal.get('confidence_score', 0.5)
            if confidence < self.testing_min_confidence:
                self.logger.warning(f"Low confidence: {confidence:.1%} < {self.testing_min_confidence:.1%}")
                # Don't fail in testing mode - just warn

            # === SUCCESS ===

            self.logger.debug(f"Testing mode validation passed for {epic} {signal_type}")
            return True, "Strategy testing mode - basic validation passed"

        except Exception as e:
            self.logger.error(f"Testing mode validation error: {e}")
            return True, f"Testing mode - error ignored: {str(e)}"

    def _validate_risk_live_mode(self, signal: Dict) -> Tuple[bool, str]:
        """
        LIVE TRADING MODE: Full risk validation for live trading

        This is the original comprehensive validation logic
        """
        try:
            # Check epic presence
            epic = signal.get('epic')
            if not epic:
                return False, "Missing epic in signal"

            # Extract and validate prices
            entry_price = self._extract_price(signal)
            if not entry_price or entry_price <= 0:
                return False, "Invalid or missing entry price"

            stop_loss = self._extract_stop_loss(signal)
            if not stop_loss or stop_loss <= 0:
                # Try to calculate default stop loss
                stop_loss = self._calculate_default_stop_loss(signal)
                if not stop_loss:
                    return False, "Invalid or missing stop loss"

            # Validate signal type
            signal_type = signal.get('signal_type', '').upper()
            valid_types = ['BUY', 'SELL', 'BULL', 'BEAR', 'TEST_BULL', 'TEST_BEAR']
            if signal_type not in valid_types:
                return False, f"Invalid signal type: {signal_type}"

            # Use account risk calculation instead of price movement percentage
            if not self.disable_account_risk:
                # Calculate actual account risk percentage
                position_size = signal.get('position_size', 0.01)  # Default minimal position
                pip_value = self.pip_values.get(epic, self.pip_values['default'])
                pip_risk = abs(entry_price - stop_loss) / (0.01 if 'JPY' in epic else 0.0001)  # Convert to pips

                # Account risk = Position size x Pip risk x Pip value / Account balance
                account_risk_amount = position_size * pip_risk * pip_value
                account_risk_percent = (account_risk_amount / self.account_balance) * 100

                if account_risk_percent > self.max_risk_per_trade:
                    return False, f"Account risk too high: {account_risk_percent:.2f}% > {self.max_risk_per_trade}%"

                self.logger.debug(f"Account risk calculation: {account_risk_percent:.2f}% (amount: ${account_risk_amount:.2f})")
            else:
                # Fallback to price movement validation (less accurate but simpler)
                price_risk_percent = abs(entry_price - stop_loss) / entry_price * 100
                if price_risk_percent > 10.0:  # 10% maximum price movement
                    return False, f"Price movement too large: {price_risk_percent:.2f}% > 10%"

            # Validate signal direction vs stop loss
            if signal_type in ['BUY', 'BULL', 'TEST_BULL']:
                if stop_loss >= entry_price:
                    return False, "Stop loss must be below entry for BUY signal"
            elif signal_type in ['SELL', 'BEAR', 'TEST_BEAR']:
                if stop_loss <= entry_price:
                    return False, "Stop loss must be above entry for SELL signal"

            # Check take profit if provided
            take_profit_fields = ['take_profit', 'tp', 'target', 'take_profit_price']
            take_profit = None

            for field in take_profit_fields:
                if field in signal and signal[field] is not None:
                    try:
                        take_profit = float(signal[field])
                        break
                    except (ValueError, TypeError):
                        continue

            if take_profit:
                if take_profit <= 0:
                    return False, "Invalid take profit price"

                # Validate take profit direction
                if signal_type in ['BUY', 'BULL', 'TEST_BULL']:
                    if take_profit <= entry_price:
                        return False, "Take profit must be above entry for BUY signal"
                elif signal_type in ['SELL', 'BEAR', 'TEST_BEAR']:
                    if take_profit >= entry_price:
                        return False, "Take profit must be below entry for SELL signal"

            return True, "Live mode - full risk validation passed"

        except Exception as e:
            self.logger.error(f"Live mode risk validation error: {e}")
            return False, f"Live validation error: {str(e)}"

    def calculate_position_size(self,
                              signal: Dict,
                              account_balance: float = None) -> Tuple[float, str]:
        """
        ENHANCED: Calculate position size with testing mode support
        """
        try:
            # STRATEGY TESTING MODE: Return minimal position size
            if self.disable_position_sizing:
                return 0.01, "Testing mode - minimal position size"

            # LIVE TRADING MODE: Full position size calculation
            return self._calculate_position_size_live_mode(signal, account_balance)

        except Exception as e:
            self.logger.error(f"Position size calculation error: {e}")
            if self.testing_mode:
                return 0.01, f"Testing mode - error fallback: {str(e)}"
            else:
                return 0.0, f"Calculation error: {str(e)}"

    def _calculate_position_size_live_mode(self, signal: Dict, account_balance: float = None) -> Tuple[float, str]:
        """
        LIVE TRADING MODE: Full position size calculation

        This is the original comprehensive position sizing logic
        """
        # Use provided account balance or fallback to instance default
        balance = account_balance or self.account_balance

        # Handle different signal field names from TradingOrchestrator
        entry_price = self._extract_price(signal)
        stop_loss = self._extract_stop_loss(signal)
        epic = signal.get('epic', 'Unknown')

        if not entry_price:
            return 0.0, "Missing entry price in signal"

        if not stop_loss:
            # Try to calculate stop loss if not provided
            stop_loss = self._calculate_default_stop_loss(signal)
            if not stop_loss:
                return 0.0, "Missing stop loss and cannot calculate default"

        # Calculate risk per pip
        pip_risk = abs(entry_price - stop_loss)
        if pip_risk <= 0:
            return 0.0, "Invalid stop loss - no risk difference"

        # Get risk percentage (from signal or use default)
        risk_percent = signal.get('risk_percent', self.default_risk_percent)

        # Calculate risk amount
        risk_amount = balance * (risk_percent / 100)

        # Apply daily loss limit
        remaining_daily_risk = self.max_daily_loss - abs(self.daily_loss)
        if remaining_daily_risk <= 0:
            return 0.0, "Daily loss limit reached"

        # Use smaller of calculated risk or remaining daily risk
        effective_risk = min(risk_amount, remaining_daily_risk)

        # Get pip value for this epic
        pip_value = self.pip_values.get(epic, self.pip_values['default'])

        # Calculate position size
        # Position size = Risk amount / (Pip risk * Pip value)
        position_size = effective_risk / (pip_risk * pip_value)

        # Apply position size limits
        position_size = max(self.min_position_size,
                          min(position_size, self.max_position_size))

        # Validate against risk limits
        actual_risk_percent = (position_size * pip_risk * pip_value) / balance * 100
        if actual_risk_percent > self.max_risk_per_trade:
            position_size = (self.max_risk_per_trade / 100 * balance) / (pip_risk * pip_value)
            position_size = max(self.min_position_size, position_size)

        self.logger.debug(f"Position size calculated for {epic}: {position_size:.4f}")
        self.logger.debug(f"   Entry price: {entry_price}")
        self.logger.debug(f"   Stop loss: {stop_loss}")
        self.logger.debug(f"   Risk amount: ${effective_risk:.2f}")
        self.logger.debug(f"   Pip risk: {pip_risk:.5f}")
        self.logger.debug(f"   Risk percent: {actual_risk_percent:.2f}%")

        return round(position_size, 4), "Position size calculated successfully"

    def can_trade(self) -> Tuple[bool, str]:
        """
        ENHANCED: Check if trading is allowed with testing mode support
        """
        try:
            # STRATEGY TESTING MODE: Always allow trading
            if self.testing_mode:
                return True, "Strategy testing mode - trading always allowed"

            # LIVE TRADING MODE: Full checks
            return self._can_trade_live_mode()

        except Exception as e:
            self.logger.error(f"Error in can_trade check: {e}")
            if self.testing_mode:
                return True, f"Testing mode - error ignored: {str(e)}"
            else:
                return False, f"Risk check error: {str(e)}"

    def _can_trade_live_mode(self) -> Tuple[bool, str]:
        """
        LIVE TRADING MODE: Full trading permission checks
        """
        # Check emergency stop
        if self.emergency_stop_enabled and hasattr(self, 'emergency_stop_active'):
            if getattr(self, 'emergency_stop_active', False):
                return False, "Emergency stop is active"

        # Check daily loss limit
        loss_ok, loss_msg = self.check_daily_loss_limits()
        if not loss_ok:
            return False, loss_msg

        # Check position limit
        pos_ok, pos_msg = self.check_max_positions()
        if not pos_ok:
            return False, pos_msg

        # Check daily trade limit
        trades_ok, trades_msg = self.check_daily_trade_limits()
        if not trades_ok:
            return False, trades_msg

        # Check daily profit target if enabled
        if self.stop_on_daily_target:
            target_ok, target_msg = self.check_daily_profit_target()
            if not target_ok:
                return False, target_msg

        return True, "All risk checks passed"

    def check_daily_profit_target(self) -> Tuple[bool, str]:
        """Check if daily profit target has been reached"""
        if not self.stop_on_daily_target:
            return True, "Daily profit target check disabled"

        # Calculate profit target amount
        profit_target = self.account_balance * (self.daily_profit_target_percent / 100.0)

        # Check if we've reached the target (daily_loss is negative for losses, positive for profits)
        if self.daily_loss >= profit_target:
            return False, f"Daily profit target reached: ${self.daily_loss:.2f} >= ${profit_target:.2f}"

        return True, f"Daily profit: ${self.daily_loss:.2f} / ${profit_target:.2f} target"

    # =============================================================================
    # TESTING MODE CONTROL METHODS
    # =============================================================================

    def enable_testing_mode(self):
        """
        Enable strategy testing mode
        """
        self.testing_mode = True
        self.disable_account_risk = True
        self.disable_position_sizing = True
        self.emergency_stop_enabled = False

        self.logger.info("STRATEGY TESTING MODE ENABLED")
        self.logger.info("   Account risk validation: BYPASSED")
        self.logger.info("   Position sizing: SIMPLIFIED")
        self.logger.info("   Emergency stops: DISABLED")
        self.logger.info("   Remember to disable for live trading!")

    def disable_testing_mode(self):
        """
        Disable strategy testing mode and restore full validation
        """
        self.testing_mode = False
        self.disable_account_risk = False
        self.disable_position_sizing = False
        self.emergency_stop_enabled = self._scanner_cfg.emergency_stop_enabled

        self.logger.info("STRATEGY TESTING MODE DISABLED")
        self.logger.info("   Full risk validation: RESTORED")
        self.logger.info("   Account risk calculations: ACTIVE")
        self.logger.info("   Position sizing: ENABLED")
        self.logger.info("   Emergency stops: ENABLED")

    def is_testing_mode(self) -> bool:
        """Check if currently in testing mode"""
        return self.testing_mode

    def get_testing_status(self) -> Dict:
        """Get detailed testing mode status"""
        return {
            'testing_mode': self.testing_mode,
            'disable_account_risk': self.disable_account_risk,
            'disable_position_sizing': self.disable_position_sizing,
            'emergency_stop_enabled': self.emergency_stop_enabled,
            'testing_max_stop_percent': getattr(self, 'testing_max_stop_percent', 20.0),
            'testing_min_confidence': getattr(self, 'testing_min_confidence', 0.0),
            'mode_description': 'Strategy Testing Mode' if self.testing_mode else 'Live Trading Mode'
        }

    # =============================================================================
    # ORIGINAL METHODS (preserved for compatibility)
    # =============================================================================

    def _extract_price(self, signal: Dict) -> float:
        """Extract entry price from signal with multiple field name options"""
        # Try standard fields first
        price_fields = ['entry_price', 'price', 'current_price', 'signal_price']
        for field in price_fields:
            if field in signal and signal[field] is not None:
                try:
                    return float(signal[field])
                except (ValueError, TypeError):
                    continue

        # Check nested ema_data
        if 'ema_data' in signal:
            ema_data = signal['ema_data']
            for field in ['ema_1', 'ema_2', 'ema_5', 'current_price']:
                if field in ema_data and ema_data[field] is not None:
                    try:
                        return float(ema_data[field])
                    except (ValueError, TypeError):
                        continue

        return 0.0

    def _extract_stop_loss(self, signal: Dict) -> float:
        """Extract stop loss from signal with multiple field name options"""
        stop_fields = ['stop_loss', 'stop_price', 'sl', 'stop']

        for field in stop_fields:
            if field in signal and signal[field] is not None:
                try:
                    return float(signal[field])
                except (ValueError, TypeError):
                    continue

        return 0.0

    def _calculate_default_stop_loss(self, signal: Dict) -> float:
        """Calculate default stop loss if not provided"""
        try:
            entry_price = self._extract_price(signal)
            if not entry_price:
                return 0.0

            signal_type = signal.get('signal_type', '').upper()

            # Convert pips to price units (assuming 4-decimal currency pairs)
            pip_size = 0.0001
            stop_distance = self.default_stop_distance * pip_size

            if signal_type in ['BUY', 'BULL', 'TEST_BULL']:
                return entry_price - stop_distance
            elif signal_type in ['SELL', 'BEAR', 'TEST_BEAR']:
                return entry_price + stop_distance

            return 0.0

        except Exception as e:
            self.logger.error(f"Error calculating default stop loss: {e}")
            return 0.0

    def check_daily_loss_limits(self) -> Tuple[bool, str]:
        """Check if daily loss limits allow new trades"""
        # Reset daily tracking if new day
        current_date = datetime.now().date()
        if current_date != self.last_reset_date:
            self._reset_daily_tracking()

        remaining_loss = self.max_daily_loss - abs(self.daily_loss)

        if remaining_loss <= 0:
            return False, f"Daily loss limit reached: ${abs(self.daily_loss):.2f}"

        loss_percent = (abs(self.daily_loss) / self.max_daily_loss) * 100

        if loss_percent > 80:  # Warning at 80%
            return True, f"Warning: {loss_percent:.1f}% of daily loss limit used"

        return True, f"Daily loss limit OK: ${remaining_loss:.2f} remaining"

    def check_max_positions(self) -> Tuple[bool, str]:
        """Check if maximum position limit allows new trades"""
        if self.current_positions >= self.max_positions:
            return False, f"Maximum positions reached: {self.current_positions}/{self.max_positions}"

        # Also check concurrent positions limit
        if self.current_positions >= self.max_concurrent_positions:
            return False, f"Maximum concurrent positions reached: {self.current_positions}/{self.max_concurrent_positions}"

        return True, f"Position limit OK: {self.current_positions}/{min(self.max_positions, self.max_concurrent_positions)}"

    def check_daily_trade_limits(self) -> Tuple[bool, str]:
        """Check if daily trade limits allow new trades"""
        current_date = datetime.now().date()
        if current_date != self.last_reset_date:
            self._reset_daily_tracking()

        if len(self.daily_trades) >= self.max_daily_trades:
            return False, f"Daily trade limit reached: {len(self.daily_trades)}/{self.max_daily_trades}"

        return True, f"Daily trades OK: {len(self.daily_trades)}/{self.max_daily_trades}"

    def apply_risk_adjustments(self, position_size: float, signal: Dict) -> float:
        """Apply risk adjustments based on market conditions and signal quality"""
        try:
            # TESTING MODE: Skip adjustments
            if self.testing_mode:
                return position_size

            epic = signal.get('epic', '')

            # Start with original size
            adjusted_size = position_size

            # Adjust based on confidence
            confidence = signal.get('confidence_score', 0.75)
            if confidence < 0.70:
                confidence_adjustment = 0.7  # Reduce size for low confidence
            elif confidence > 0.90:
                confidence_adjustment = 1.2  # Increase size for high confidence
            else:
                confidence_adjustment = 1.0

            adjusted_size *= confidence_adjustment

            # Adjust based on strategy type
            strategy = signal.get('strategy', '').lower()
            if strategy == 'scalping':
                strategy_adjustment = 0.8  # Smaller sizes for scalping
            elif strategy == 'swing':
                strategy_adjustment = 1.1  # Larger sizes for swing trading
            elif strategy == 'combined':
                strategy_adjustment = 1.0  # Normal sizing for combined
            else:
                strategy_adjustment = 1.0

            adjusted_size *= strategy_adjustment

            # Assess market volatility (placeholder - would use real data in production)
            volatility = self._assess_market_volatility(epic)
            volatility_adjustment = volatility.get('risk_adjustment', 1.0)
            adjusted_size *= volatility_adjustment

            # Apply time-based adjustments
            time_adjustment = self._get_time_based_adjustment()
            adjusted_size *= time_adjustment

            # Ensure within limits
            adjusted_size = max(self.min_position_size,
                              min(adjusted_size, self.max_position_size))

            self.logger.debug(f"Risk adjustments applied to {epic}:")
            self.logger.debug(f"   Original size: {position_size:.4f}")
            self.logger.debug(f"   Confidence adjustment: {confidence_adjustment:.2f}")
            self.logger.debug(f"   Strategy adjustment: {strategy_adjustment:.2f}")
            self.logger.debug(f"   Volatility adjustment: {volatility_adjustment:.2f}")
            self.logger.debug(f"   Time adjustment: {time_adjustment:.2f}")
            self.logger.debug(f"   Final size: {adjusted_size:.4f}")

            return round(adjusted_size, 4)

        except Exception as e:
            self.logger.error(f"Error applying risk adjustments: {e}")
            return position_size

    def _assess_market_volatility(self, epic: str) -> Dict:
        """Assess current market volatility for risk adjustment"""
        try:
            current_hour = datetime.now().hour

            # Adjust based on market hours (more volatile during overlaps)
            if 8 <= current_hour <= 10 or 13 <= current_hour <= 15:  # Market overlaps
                volatility_level = 'HIGH'
                risk_adjustment = 0.9  # Reduce position size during high volatility
            elif 2 <= current_hour <= 6:  # Low volume hours
                volatility_level = 'LOW'
                risk_adjustment = 1.1  # Slightly increase size during low volatility
            else:
                volatility_level = 'MEDIUM'
                risk_adjustment = 1.0

            return {
                'volatility_level': volatility_level,
                'volatility_score': 0.5,
                'risk_adjustment': risk_adjustment,
                'recommendation': f'{volatility_level} volatility - position size adjusted'
            }

        except Exception as e:
            self.logger.error(f"Error assessing volatility: {e}")
            return {
                'volatility_level': 'UNKNOWN',
                'volatility_score': 0.5,
                'risk_adjustment': 1.0,
                'recommendation': 'Use default risk parameters'
            }

    def _get_time_based_adjustment(self) -> float:
        """Get position size adjustment based on time of day"""
        try:
            current_hour = datetime.now().hour

            # Reduce position sizes outside of main trading hours
            if 22 <= current_hour or current_hour <= 6:  # Late night/early morning
                return 0.8
            elif 6 <= current_hour <= 8:  # Pre-market
                return 0.9
            elif 8 <= current_hour <= 16:  # Main trading hours
                return 1.0
            elif 16 <= current_hour <= 18:  # After hours but active
                return 0.95
            else:  # Evening
                return 0.85

        except Exception:
            return 1.0

    def record_trade_result(self, trade_result: Dict):
        """Record trade result for risk tracking"""
        try:
            # TESTING MODE: Skip trade recording
            if self.testing_mode:
                self.logger.debug(f"Testing mode - trade result logged but not tracked: {trade_result.get('status', 'Unknown')}")
                return

            pnl = trade_result.get('pnl', 0.0)
            status = trade_result.get('status', 'UNKNOWN')
            epic = trade_result.get('epic', 'Unknown')

            # Update daily P&L
            self.daily_loss += pnl  # pnl is negative for losses

            # Update position count
            if status == 'OPENED':
                self.current_positions += 1
            elif status in ['CLOSED', 'FILLED', 'CANCELLED']:
                self.current_positions = max(0, self.current_positions - 1)

            # Add to daily trades
            trade_record = {
                'timestamp': datetime.now(),
                'epic': epic,
                'pnl': pnl,
                'status': status,
                'position_size': trade_result.get('position_size', 0),
                'signal_confidence': trade_result.get('confidence_score', 0)
            }
            self.daily_trades.append(trade_record)

            self.logger.info(f"Trade result recorded: {status} {epic}")
            self.logger.info(f"   PnL: ${pnl:.2f}")
            self.logger.info(f"   Daily total: ${self.daily_loss:.2f}")
            self.logger.info(f"   Open positions: {self.current_positions}")

            # Check for emergency stop conditions
            self._check_emergency_conditions()

        except Exception as e:
            self.logger.error(f"Error recording trade result: {e}")

    def _check_emergency_conditions(self):
        """Check if emergency stop should be activated"""
        try:
            # TESTING MODE: Skip emergency checks
            if self.testing_mode:
                return

            # Emergency stop if daily loss exceeds 90% of limit
            loss_percent = (abs(self.daily_loss) / self.max_daily_loss) * 100
            if loss_percent > 90:
                self.emergency_stop_active = True
                self.logger.error(f"EMERGENCY STOP ACTIVATED: {loss_percent:.1f}% of daily loss limit reached")

            # Emergency stop if too many consecutive losses
            recent_trades = self.daily_trades[-5:] if len(self.daily_trades) >= 5 else self.daily_trades
            if len(recent_trades) >= 5 and all(trade['pnl'] < 0 for trade in recent_trades):
                self.emergency_stop_active = True
                self.logger.error("EMERGENCY STOP ACTIVATED: 5 consecutive losses detected")

        except Exception as e:
            self.logger.error(f"Error checking emergency conditions: {e}")

    def _reset_daily_tracking(self):
        """Reset daily tracking for new trading day"""
        self.daily_loss = 0.0
        self.daily_trades = []
        self.last_reset_date = datetime.now().date()

        # Reset emergency stop at start of new day (only in live mode)
        if not self.testing_mode:
            self.emergency_stop_active = False

        self.logger.info("Daily risk tracking reset")

    def get_risk_status(self) -> Dict:
        """Get comprehensive risk status"""
        current_date = datetime.now().date()
        if current_date != self.last_reset_date:
            self._reset_daily_tracking()

        can_trade_result, can_trade_reason = self.can_trade()

        status = {
            'timestamp': datetime.now().isoformat(),
            'testing_mode': self.testing_mode,
            'config_source': self._scanner_cfg.source if self._scanner_cfg else 'unknown',
            'daily_loss': self.daily_loss,
            'max_daily_loss': self.max_daily_loss,
            'remaining_daily_risk': self.max_daily_loss - abs(self.daily_loss),
            'daily_loss_percent': (abs(self.daily_loss) / self.max_daily_loss) * 100,
            'current_positions': self.current_positions,
            'max_positions': self.max_positions,
            'max_concurrent_positions': self.max_concurrent_positions,
            'position_capacity': min(self.max_positions, self.max_concurrent_positions) - self.current_positions,
            'daily_trades': len(self.daily_trades),
            'max_daily_trades': self.max_daily_trades,
            'remaining_daily_trades': self.max_daily_trades - len(self.daily_trades),
            'can_trade': can_trade_result,
            'can_trade_reason': can_trade_reason,
            'emergency_stop_active': getattr(self, 'emergency_stop_active', False),
            'account_balance': self.account_balance,
            'risk_limits': {
                'min_position_size': self.min_position_size,
                'max_position_size': self.max_position_size,
                'max_risk_per_trade': self.max_risk_per_trade,
                'default_risk_percent': self.default_risk_percent,
                'max_trades_per_pair': self.max_trades_per_pair,
                'daily_profit_target_percent': self.daily_profit_target_percent,
                'stop_on_daily_target': self.stop_on_daily_target
            },
            'recent_trades': self.daily_trades[-5:] if self.daily_trades else []
        }

        # Add testing mode specific info
        if self.testing_mode:
            status['testing_status'] = self.get_testing_status()

        return status

    def update_risk_parameters(self, **kwargs):
        """Update risk parameters dynamically"""
        updated = []

        for key, value in kwargs.items():
            if hasattr(self, key):
                old_value = getattr(self, key)
                setattr(self, key, value)
                updated.append(f"{key}: {old_value} -> {value}")
                self.logger.info(f"Updated {key} from {old_value} to {value}")
            else:
                self.logger.warning(f"Unknown risk parameter: {key}")

        if updated:
            self.logger.info(f"Risk parameters updated: {'; '.join(updated)}")

        return len(updated) > 0

    def reset_emergency_stop(self):
        """Manually reset emergency stop"""
        self.emergency_stop_active = False
        self.logger.info("Emergency stop manually reset")

    def get_position_size_breakdown(self, signal: Dict, account_balance: float = None) -> Dict:
        """Get detailed breakdown of position size calculation"""
        try:
            balance = account_balance or self.account_balance
            entry_price = self._extract_price(signal)
            stop_loss = self._extract_stop_loss(signal) or self._calculate_default_stop_loss(signal)
            epic = signal.get('epic', 'Unknown')

            breakdown = {
                'signal': {
                    'epic': epic,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'signal_type': signal.get('signal_type', 'Unknown'),
                    'confidence': signal.get('confidence_score', 0),
                    'strategy': signal.get('strategy', 'Unknown')
                },
                'calculations': {},
                'adjustments': {},
                'final': {},
                'testing_mode': self.testing_mode,
                'config_source': self._scanner_cfg.source if self._scanner_cfg else 'unknown'
            }

            if self.testing_mode:
                breakdown['final'] = {
                    'calculated_position_size': 0.01,
                    'adjusted_position_size': 0.01,
                    'calculation_reason': 'Testing mode - minimal position',
                    'final_risk_amount': 0.0,
                    'final_risk_percent': 0.0
                }
            elif entry_price and stop_loss:
                pip_risk = abs(entry_price - stop_loss)
                risk_percent = signal.get('risk_percent', self.default_risk_percent)
                risk_amount = balance * (risk_percent / 100)
                pip_value = self.pip_values.get(epic, self.pip_values['default'])

                breakdown['calculations'] = {
                    'account_balance': balance,
                    'risk_percent': risk_percent,
                    'risk_amount': risk_amount,
                    'pip_risk': pip_risk,
                    'pip_value': pip_value,
                    'raw_position_size': risk_amount / (pip_risk * pip_value),
                    'daily_loss_remaining': self.max_daily_loss - abs(self.daily_loss)
                }

                # Calculate with all adjustments
                position_size, reason = self.calculate_position_size(signal, balance)
                adjusted_size = self.apply_risk_adjustments(position_size, signal)

                breakdown['final'] = {
                    'calculated_position_size': position_size,
                    'adjusted_position_size': adjusted_size,
                    'calculation_reason': reason,
                    'final_risk_amount': adjusted_size * pip_risk * pip_value,
                    'final_risk_percent': (adjusted_size * pip_risk * pip_value) / balance * 100
                }

            return breakdown

        except Exception as e:
            self.logger.error(f"Error creating position size breakdown: {e}")
            return {'error': str(e), 'testing_mode': self.testing_mode}


# =============================================================================
# INTEGRATION HELPER FUNCTIONS
# =============================================================================

def create_risk_manager(logger=None, testing_mode=None, **kwargs):
    """Factory function to create RiskManager with proper configuration"""
    return RiskManager(logger=logger, testing_mode=testing_mode, **kwargs)


def validate_signal_risk(signal: Dict, risk_manager: RiskManager = None) -> Tuple[bool, str]:
    """Standalone function to validate signal risk"""
    if not risk_manager:
        risk_manager = RiskManager()

    return risk_manager.validate_risk_parameters(signal)


if __name__ == "__main__":
    # Test the RiskManager with testing mode
    print("Testing RiskManager with Strategy Testing Mode...")

    # Test in testing mode
    print("\nTesting Mode:")
    risk_manager_testing = RiskManager(testing_mode=True)

    test_signal = {
        'epic': 'CS.D.GBPUSD.MINI.IP',
        'signal_type': 'BUY',
        'confidence_score': 0.45,  # Low confidence - would normally fail
        'strategy': 'EMA',
        'entry_price': 1.2500,
        'stop_loss': 1.2480,  # Normal stop
    }

    is_valid, reason = risk_manager_testing.validate_risk_parameters(test_signal)
    print(f"Testing mode validation: {'VALID' if is_valid else 'INVALID'} - {reason}")

    position_size, size_reason = risk_manager_testing.calculate_position_size(test_signal)
    print(f"Testing mode position size: {position_size} - {size_reason}")

    can_trade, trade_reason = risk_manager_testing.can_trade()
    print(f"Testing mode can trade: {'YES' if can_trade else 'NO'} - {trade_reason}")

    # Get risk status
    status = risk_manager_testing.get_risk_status()
    print(f"Config source: {status.get('config_source', 'unknown')}")

    print("\nRiskManager database-only configuration test completed!")
