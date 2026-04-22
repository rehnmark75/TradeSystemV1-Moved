"""
Enhanced Prompt Builder - Senior Technical Analyst Mode
Advanced prompt construction for institutional-grade forex analysis
Replaces the basic prompt builder with professional-grade prompts
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime


class PromptBuilder:
    """
    Enhanced prompt builder that makes Claude act as a senior technical analyst
    with professional-grade analysis and market context awareness
    
    ENHANCED: Now includes institutional-grade analysis capabilities
    while maintaining backwards compatibility with existing code
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Market sessions and their characteristics
        self.market_sessions = {
            'tokyo': {'volatility': 'low', 'pairs': ['USDJPY', 'AUDJPY'], 'hours': '23:00-08:00 UTC'},
            'london': {'volatility': 'high', 'pairs': ['GBPUSD', 'EURUSD'], 'hours': '07:00-16:00 UTC'},
            'new_york': {'volatility': 'high', 'pairs': ['USDCAD', 'EURUSD'], 'hours': '12:00-21:00 UTC'},
            'overlap': {'volatility': 'highest', 'pairs': ['EURUSD', 'GBPUSD'], 'hours': '12:00-16:00 UTC'}
        }
        
        # Professional analysis templates
        self.analysis_levels = {
            'institutional': 'Think like a bank trader with 20+ years experience',
            'hedge_fund': 'Analyze like a quantitative hedge fund analyst',
            'prop_trader': 'Approach like a seasoned proprietary trader',
            'risk_manager': 'Evaluate from a risk management perspective'
        }
    
    def build_minimal_prompt_with_complete_data(self, signal: Dict, technical_validation: Dict) -> str:
        """
        BACKWARDS COMPATIBLE: Original method enhanced with advanced capabilities
        Automatically chooses between simple and advanced prompts based on signal complexity
        """
        try:
            # Check if this should use advanced analysis
            if self._should_use_advanced_analysis(signal, technical_validation):
                return self.build_senior_analyst_prompt(signal, technical_validation, 'institutional')
            else:
                return self._build_simple_prompt(signal, technical_validation)
                
        except Exception as e:
            self.logger.error(f"Error building prompt: {e}")
            return self._build_fallback_prompt(signal, technical_validation)
    
    def build_senior_analyst_prompt(self, signal: Dict, technical_validation: Dict, 
                                  analysis_level: str = 'institutional') -> str:
        """
        NEW: Build advanced institutional-grade prompt
        Makes Claude act as a senior technical analyst with professional judgment
        """
        try:
            # Get market context
            market_context = self._analyze_market_context(signal)
            
            # Get professional analysis framework
            analysis_framework = self._get_analysis_framework(analysis_level)
            
            # Build comprehensive technical picture
            technical_picture = self._build_comprehensive_technical_analysis(signal)
            
            # Get risk assessment framework
            risk_framework = self._build_risk_assessment_framework(signal)
            
            # Get market regime analysis
            regime_analysis = self._analyze_market_regime(signal)
            
            prompt = f"""
You are a SENIOR FOREX TECHNICAL ANALYST with 20+ years of institutional trading experience.
{analysis_framework}

CRITICAL ANALYSIS TASK:
Analyze this forex signal with the depth and expertise of a seasoned professional trader.
Your analysis will directly impact real trading decisions and capital allocation.

═══════════════════════════════════════════════════════════════
📊 SIGNAL OVERVIEW
═══════════════════════════════════════════════════════════════
Instrument: {signal.get('epic', 'Unknown')}
Proposed Direction: {signal.get('signal_type', 'Unknown')}
Entry Price: {self._format_price(signal.get('price'))}
System Confidence: {signal.get('confidence_score', 0):.1%}
Detection Strategy: {signal.get('strategy', 'Unknown')}
Timestamp: {signal.get('timestamp', 'Unknown')}

{market_context}

═══════════════════════════════════════════════════════════════
🔬 COMPREHENSIVE TECHNICAL ANALYSIS
═══════════════════════════════════════════════════════════════
{technical_picture}

═══════════════════════════════════════════════════════════════
🌍 MARKET REGIME & CONTEXT
═══════════════════════════════════════════════════════════════
{regime_analysis}

═══════════════════════════════════════════════════════════════
⚠️ PROFESSIONAL RISK ASSESSMENT
═══════════════════════════════════════════════════════════════
{risk_framework}

═══════════════════════════════════════════════════════════════
📋 REQUIRED PROFESSIONAL ANALYSIS FORMAT
═══════════════════════════════════════════════════════════════

Since this is institutional analysis, provide ONLY these exact lines:

SCORE: [1-10]
DECISION: [APPROVE/REJECT]
REASON: [Your professional assessment in 2-3 sentences focusing on key technical factors, market context, and risk/reward]

Keep it concise but professional. Focus on the most critical factors that would matter to an institutional trader.
"""
            
            return prompt
            
        except Exception as e:
            self.logger.error(f"Error building senior analyst prompt: {e}")
            return self._build_fallback_prompt(signal, technical_validation)
    
    def _should_use_advanced_analysis(self, signal: Dict, technical_validation: Dict) -> bool:
        """Determine if signal warrants advanced institutional analysis"""
        try:
            # Use advanced analysis for:
            # 1. High confidence signals
            confidence = signal.get('confidence_score', 0)
            if confidence >= 0.8:
                return True
            
            # 2. Combined strategies
            strategy = signal.get('strategy', '').lower()
            if 'combined' in strategy:
                return True
            
            # 3. Signals with rich technical data
            technical_indicators = sum(1 for key in signal.keys() 
                                     if any(indicator in key.lower() 
                                           for indicator in ['ema', 'macd', 'rsi', 'atr', 'bb_', 'kama']))
            if technical_indicators >= 8:
                return True
            
            # 4. Major currency pairs during active sessions
            epic = signal.get('epic', '')
            if any(pair in epic for pair in ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF']):
                current_hour = datetime.now().hour
                if 7 <= current_hour <= 21:  # London or NY sessions
                    return True
            
            return False
            
        except Exception:
            return False  # Default to simple analysis on error
    
    def _build_simple_prompt(self, signal: Dict, technical_validation: Dict) -> str:
        """Build simple prompt for basic signals (backwards compatible)"""
        try:
            epic = str(signal.get('epic', 'Unknown'))
            signal_type = str(signal.get('signal_type', 'Unknown'))
            strategy = str(signal.get('strategy', 'Unknown'))
            
            try:
                price = float(signal.get('price', 0))
                confidence = float(signal.get('confidence_score', 0))
            except (ValueError, TypeError):
                price = 0.0
                confidence = 0.0
            
            prompt = f"""FOREX SIGNAL ANALYSIS - MINIMAL MODE

Signal: {epic} {signal_type}
Strategy: {strategy}
Price: {price:.5f}
Confidence: {confidence:.1%}

Technical Validation: ✅ PASSED
{technical_validation.get('summary', 'Technical analysis completed')}

Instructions: Provide only:
SCORE: [0-10]
DECISION: [APPROVE/REJECT]
REASON: [brief reason]

Be concise. Focus on signal quality."""

            return prompt
            
        except Exception as e:
            self.logger.error(f"Error building simple prompt: {e}")
            return self._build_fallback_prompt(signal, technical_validation)
    
    def _analyze_market_context(self, signal: Dict) -> str:
        """Analyze current market context and session"""
        try:
            epic = signal.get('epic', '')
            
            # Determine currency pair
            if 'EUR' in epic and 'USD' in epic:
                pair_type = "Major (EUR/USD)"
                session_relevance = "Prime for London/NY overlap"
            elif 'GBP' in epic and 'USD' in epic:
                pair_type = "Major (GBP/USD)"
                session_relevance = "Highly active during London session"
            elif 'USD' in epic and 'JPY' in epic:
                pair_type = "Major (USD/JPY)"
                session_relevance = "Active during Tokyo and NY sessions"
            else:
                pair_type = "Currency pair"
                session_relevance = "Session timing considerations apply"
            
            # Get current time context
            current_hour = datetime.now().hour
            if 7 <= current_hour <= 16:
                session = "London Session - High volatility expected"
            elif 12 <= current_hour <= 21:
                session = "New York Session - High liquidity"
            elif 23 <= current_hour or current_hour <= 8:
                session = "Tokyo Session - Lower volatility"
            else:
                session = "Transition period"
            
            return f"""
📍 Market Context:
• Instrument Type: {pair_type}
• Session Analysis: {session}
• Session Relevance: {session_relevance}
• Liquidity Expectation: {"High" if "High" in session else "Moderate"}
"""
        except Exception:
            return "• Market context analysis unavailable"
    
    def _get_analysis_framework(self, level: str) -> str:
        """Get the appropriate analysis framework for the specified level"""
        frameworks = {
            'institutional': """
You trade with institutional-grade discipline and risk management.
Focus on: Market structure, order flow, smart money concepts, and multi-timeframe analysis.
Your trades must withstand institutional scrutiny and align with professional standards.
""",
            'hedge_fund': """
You analyze with quantitative precision and systematic approach.
Focus on: Statistical edge, correlation analysis, volatility patterns, and systematic risk.
Every recommendation must have quantifiable reasoning and measurable risk parameters.
""",
            'prop_trader': """
You trade with performance-focused mentality and profit optimization.
Focus on: Execution quality, timing precision, risk/reward optimization, and market efficiency.
Balance aggressive profit-seeking with disciplined risk management.
""",
            'risk_manager': """
You evaluate from a capital preservation and risk control perspective.
Focus on: Downside protection, correlation risks, black swan events, and portfolio impact.
Every trade must justify its risk relative to potential portfolio damage.
"""
        }
        return frameworks.get(level, frameworks['institutional'])
    
    def _build_comprehensive_technical_analysis(self, signal: Dict) -> str:
        """Build comprehensive technical analysis section"""
        try:
            analysis_sections = []
            
            # 1. Price Action Analysis
            price_analysis = self._analyze_price_action(signal)
            if price_analysis:
                analysis_sections.append(f"💹 PRICE ACTION:\n{price_analysis}")
            
            # 2. Trend Analysis
            trend_analysis = self._analyze_trend_structure(signal)
            if trend_analysis:
                analysis_sections.append(f"📈 TREND STRUCTURE:\n{trend_analysis}")
            
            # 3. Momentum Analysis
            momentum_analysis = self._analyze_momentum_indicators(signal)
            if momentum_analysis:
                analysis_sections.append(f"⚡ MOMENTUM INDICATORS:\n{momentum_analysis}")
            
            # 4. Support/Resistance Analysis
            sr_analysis = self._analyze_support_resistance(signal)
            if sr_analysis:
                analysis_sections.append(f"🏗️ KEY LEVELS:\n{sr_analysis}")
            
            # 5. Volume Analysis
            volume_analysis = self._analyze_volume_patterns(signal)
            if volume_analysis:
                analysis_sections.append(f"📊 VOLUME ANALYSIS:\n{volume_analysis}")
            
            return "\n\n".join(analysis_sections) if analysis_sections else "Technical analysis data insufficient"
            
        except Exception as e:
            self.logger.error(f"Error building technical analysis: {e}")
            return "Technical analysis compilation failed"
    
    def _analyze_price_action(self, signal: Dict) -> str:
        """Analyze price action components"""
        elements = []
        
        current_price = signal.get('price', signal.get('close_price'))
        if current_price:
            elements.append(f"• Current Price: {self._format_price(current_price)}")
        
        # OHLC analysis
        if signal.get('open_price') and signal.get('high_price') and signal.get('low_price'):
            open_p = float(signal['open_price'])
            high_p = float(signal['high_price'])
            low_p = float(signal['low_price'])
            close_p = float(current_price or signal.get('close_price', open_p))
            
            candle_range = high_p - low_p
            body_size = abs(close_p - open_p)
            body_percentage = (body_size / candle_range * 100) if candle_range > 0 else 0
            
            candle_type = "Bullish" if close_p > open_p else "Bearish" if close_p < open_p else "Doji"
            elements.append(f"• Candle Type: {candle_type} ({body_percentage:.1f}% body)")
            
            if body_percentage < 20:
                elements.append("• Pattern: Indecision candle (small body)")
            elif body_percentage > 70:
                elements.append("• Pattern: Strong directional candle")
        
        return "\n".join(elements) if elements else None
    
    def _analyze_trend_structure(self, signal: Dict) -> str:
        """Analyze trend structure using EMAs"""
        elements = []
        
        # Get EMA values
        ema_9 = signal.get('ema_9', signal.get('ema_short'))
        ema_21 = signal.get('ema_21', signal.get('ema_long'))
        ema_200 = signal.get('ema_200', signal.get('ema_trend'))
        current_price = signal.get('price')
        
        if ema_9 and ema_21 and ema_200:
            ema_9, ema_21, ema_200 = float(ema_9), float(ema_21), float(ema_200)
            
            # Trend direction
            if ema_9 > ema_21 > ema_200:
                trend = "Strong Bullish (Perfect EMA alignment)"
            elif ema_9 < ema_21 < ema_200:
                trend = "Strong Bearish (Perfect EMA alignment)"
            elif ema_9 > ema_21:
                trend = "Short-term Bullish (EMA 9>21, watch EMA 200)"
            elif ema_9 < ema_21:
                trend = "Short-term Bearish (EMA 9<21, watch EMA 200)"
            else:
                trend = "Sideways/Consolidation"
            
            elements.append(f"• Trend Direction: {trend}")
            
            # EMA spacing analysis
            short_separation = abs(ema_9 - ema_21) / ema_21 * 10000  # in pips
            long_separation = abs(ema_21 - ema_200) / ema_200 * 10000
            
            elements.append(f"• EMA 9/21 Separation: {short_separation:.1f} pips")
            elements.append(f"• EMA 21/200 Separation: {long_separation:.1f} pips")
            
            # Price position relative to EMAs
            if current_price:
                price = float(current_price)
                if price > max(ema_9, ema_21, ema_200):
                    position = "Above all EMAs (Bullish territory)"
                elif price < min(ema_9, ema_21, ema_200):
                    position = "Below all EMAs (Bearish territory)"
                else:
                    position = "Inside EMA cloud (Consolidation zone)"
                elements.append(f"• Price Position: {position}")
        
        return "\n".join(elements) if elements else None
    
    def _analyze_momentum_indicators(self, signal: Dict) -> str:
        """Analyze momentum indicators (MACD, RSI, etc.)"""
        elements = []
        
        # MACD Analysis
        macd_line = signal.get('macd_line')
        macd_signal = signal.get('macd_signal')
        macd_histogram = signal.get('macd_histogram')
        
        if macd_histogram is not None:
            macd_hist = float(macd_histogram)
            if macd_hist > 0.0001:
                momentum = "Strong Bullish Momentum"
            elif macd_hist > 0:
                momentum = "Weak Bullish Momentum"
            elif macd_hist < -0.0001:
                momentum = "Strong Bearish Momentum"
            else:
                momentum = "Weak Bearish Momentum"
            
            elements.append(f"• MACD Histogram: {macd_hist:.6f} ({momentum})")
        
        if macd_line and macd_signal:
            macd_l, macd_s = float(macd_line), float(macd_signal)
            crossover = "Bullish crossover" if macd_l > macd_s else "Bearish crossover"
            elements.append(f"• MACD Line vs Signal: {crossover}")
        
        # RSI Analysis
        rsi = signal.get('rsi')
        if rsi:
            rsi_val = float(rsi)
            if rsi_val > 70:
                rsi_state = "Overbought (>70) - Caution for longs"
            elif rsi_val < 30:
                rsi_state = "Oversold (<30) - Caution for shorts"
            elif 40 <= rsi_val <= 60:
                rsi_state = "Neutral zone - No extreme bias"
            else:
                rsi_state = f"Moderate bias ({rsi_val:.1f})"
            elements.append(f"• RSI: {rsi_val:.1f} ({rsi_state})")
        
        return "\n".join(elements) if elements else None
    
    def _analyze_support_resistance(self, signal: Dict) -> str:
        """Analyze support and resistance levels"""
        elements = []
        
        support = signal.get('nearest_support')
        resistance = signal.get('nearest_resistance')
        current_price = signal.get('price')
        
        if support and current_price:
            support_dist = (float(current_price) - float(support)) * 10000
            elements.append(f"• Nearest Support: {self._format_price(support)} ({support_dist:.1f} pips away)")
        
        if resistance and current_price:
            resistance_dist = (float(resistance) - float(current_price)) * 10000
            elements.append(f"• Nearest Resistance: {self._format_price(resistance)} ({resistance_dist:.1f} pips away)")
        
        # Support/Resistance quality assessment
        if support and resistance and current_price:
            total_range = (float(resistance) - float(support)) * 10000
            current_position = ((float(current_price) - float(support)) / (float(resistance) - float(support))) * 100
            
            elements.append(f"• Range Size: {total_range:.1f} pips")
            elements.append(f"• Position in Range: {current_position:.1f}% from support")
            
            if current_position < 25:
                position_assessment = "Near support - Bullish bias"
            elif current_position > 75:
                position_assessment = "Near resistance - Bearish bias"
            else:
                position_assessment = "Mid-range - Direction unclear"
            
            elements.append(f"• Range Assessment: {position_assessment}")
        
        return "\n".join(elements) if elements else None
    
    def _analyze_volume_patterns(self, signal: Dict) -> str:
        """Analyze volume patterns"""
        elements = []
        
        volume = signal.get('volume')
        volume_ratio = signal.get('volume_ratio')
        
        if volume:
            elements.append(f"• Current Volume: {volume}")
        
        if volume_ratio:
            ratio = float(volume_ratio)
            if ratio > 1.5:
                vol_assessment = "High volume (>1.5x average) - Strong conviction"
            elif ratio > 1.2:
                vol_assessment = "Above average volume - Moderate conviction"
            elif ratio < 0.8:
                vol_assessment = "Below average volume - Weak conviction"
            else:
                vol_assessment = "Normal volume - Standard conviction"
            
            elements.append(f"• Volume Ratio: {ratio:.2f} ({vol_assessment})")
        
        return "\n".join(elements) if elements else None
    
    def _analyze_market_regime(self, signal: Dict) -> str:
        """Analyze current market regime"""
        elements = []
        
        # Volatility assessment
        atr = signal.get('atr')
        if atr:
            atr_val = float(atr)
            volatility = "High" if atr_val > 0.002 else "Moderate" if atr_val > 0.001 else "Low"
            elements.append(f"• Market Volatility: {volatility} (ATR: {atr_val:.5f})")
        
        # Trend vs Range assessment
        ema_9 = signal.get('ema_9', signal.get('ema_short'))
        ema_200 = signal.get('ema_200', signal.get('ema_trend'))
        
        if ema_9 and ema_200:
            ema_separation = abs(float(ema_9) - float(ema_200)) / float(ema_200) * 100
            if ema_separation > 0.5:
                regime = "Strong Trending Market"
            elif ema_separation > 0.2:
                regime = "Weak Trending Market"
            else:
                regime = "Range-bound Market"
            
            elements.append(f"• Market Regime: {regime} ({ema_separation:.2f}% EMA separation)")
        
        return "\n".join(elements) if elements else "Market regime analysis unavailable"
    
    def _build_risk_assessment_framework(self, signal: Dict) -> str:
        """Build comprehensive risk assessment"""
        elements = []
        
        # Position sizing recommendation
        confidence = signal.get('confidence_score', 0.5)
        if confidence > 0.8:
            position_size = "Standard (1-2% risk)"
        elif confidence > 0.6:
            position_size = "Reduced (0.5-1% risk)"
        else:
            position_size = "Minimal (0.25-0.5% risk)"
        
        elements.append(f"• Recommended Position Size: {position_size}")
        
        # Risk factors
        risk_factors = []
        
        # Check for counter-trend signals
        ema_9 = signal.get('ema_9')
        ema_200 = signal.get('ema_200')
        signal_type = signal.get('signal_type', '').upper()
        
        if ema_9 and ema_200:
            if signal_type == 'BULL' and float(ema_9) < float(ema_200):
                risk_factors.append("Counter-trend signal against EMA 200")
            elif signal_type == 'BEAR' and float(ema_9) > float(ema_200):
                risk_factors.append("Counter-trend signal against EMA 200")
        
        # MACD contradiction check
        macd_hist = signal.get('macd_histogram')
        if macd_hist:
            if signal_type == 'BULL' and float(macd_hist) < -0.0001:
                risk_factors.append("Strong MACD bearish divergence")
            elif signal_type == 'BEAR' and float(macd_hist) > 0.0001:
                risk_factors.append("Strong MACD bullish divergence")
        
        if risk_factors:
            elements.append(f"• Risk Factors: {'; '.join(risk_factors)}")
        else:
            elements.append("• Risk Factors: None identified")
        
        return "\n".join(elements)
    
    def _format_price(self, price) -> str:
        """Format price for display"""
        try:
            return f"{float(price):.5f}" if price else "N/A"
        except (ValueError, TypeError):
            return str(price) if price else "N/A"
    
    def _build_fallback_prompt(self, signal: Dict, technical_validation: Dict) -> str:
        """Fallback prompt if main prompt building fails"""
        try:
            epic = str(signal.get('epic', 'Unknown'))
            signal_type = str(signal.get('signal_type', 'Unknown'))
            
            return f"""FOREX SIGNAL ANALYSIS - FALLBACK MODE

Signal: {epic} {signal_type}

Instructions: Provide only:
SCORE: [0-10]
DECISION: [APPROVE/REJECT]
REASON: [brief reason]"""
        except:
            return """FOREX SIGNAL ANALYSIS - ERROR MODE

SCORE: 5
DECISION: NEUTRAL
REASON: Analysis error - neutral assessment"""

    def build_forex_vision_prompt(self, signal: Dict, has_chart: bool = True) -> str:
        """
        Build a vision-enabled prompt for strategy analysis.
        Routes to strategy-specific prompt builders.

        Args:
            signal: Signal dictionary with all trading data
            has_chart: Whether a chart image will be included

        Returns:
            Formatted prompt string for Claude vision analysis
        """
        strategy = signal.get('strategy', 'SMC_SIMPLE').upper()
        if strategy == 'FVG_RETEST':
            return self._build_fvg_retest_prompt(signal, has_chart)
        if strategy == 'MEAN_REVERSION':
            return self._build_mean_reversion_prompt(signal, has_chart)
        if strategy in {'RANGE_FADE', 'RANGE_FADE_5M', 'EURUSD_RANGE_FADE', 'EURUSD_RANGE_FADE_5M'}:
            return self._build_eurusd_range_fade_prompt(signal, has_chart)
        return self._build_smc_prompt(signal, has_chart)



    def _build_smc_prompt(self, signal: Dict, has_chart: bool = True) -> str:
        """
        Build a vision-enabled prompt for SMC Simple v2.3.0 strategy analysis.

        This prompt is designed to work with chart images for comprehensive
        multi-timeframe analysis of forex signals.

        Updated for v2.3.0 relaxed thresholds based on rejection analysis.
        Enhanced with rich strategy_indicators data for better AI analysis.
        """
        try:
            # Extract signal data
            epic = signal.get('epic', 'Unknown')
            pair = self._extract_pair(epic)
            direction = signal.get('signal_type', signal.get('signal', 'Unknown'))
            confidence = signal.get('confidence_score', 0)
            strategy = signal.get('strategy', 'SMC_SIMPLE')

            # Price levels
            entry_price = signal.get('entry_price', signal.get('price', 0))
            stop_loss = signal.get('stop_loss', 0)
            take_profit = signal.get('take_profit', 0)

            # Risk metrics
            risk_pips = signal.get('risk_pips', 0)
            reward_pips = signal.get('reward_pips', 0)
            rr_ratio = signal.get('rr_ratio', 0)

            # Extract rich data from strategy_indicators
            strategy_indicators = signal.get('strategy_indicators', {})

            # Tier 1 - EMA Bias data
            tier1_ema = strategy_indicators.get('tier1_ema', {})
            ema_value = tier1_ema.get('ema_value', signal.get('ema_value', 0))
            ema_distance = tier1_ema.get('distance_pips', signal.get('ema_distance_pips', 0))
            ema_direction = tier1_ema.get('direction', direction)

            # Actual strategy timeframes (populated by smc_simple_strategy._configure_scalp_mode
            # when scalp_mode_enabled=True; fall back to swing defaults otherwise).
            # Prefer the explicit scalp_mode flag; fall back to TF inference for
            # legacy signals that predate the flag.
            strategy_htf = tier1_ema.get('timeframe', '4h') or '4h'
            if 'scalp_mode' in signal:
                is_scalp_mode = bool(signal.get('scalp_mode'))
            else:
                is_scalp_mode = strategy_htf != '4h'

            # Tier 2 - Swing Break data
            tier2_swing = strategy_indicators.get('tier2_swing', {})
            swing_level = tier2_swing.get('swing_level', signal.get('swing_level', 0))
            body_close_confirmed = tier2_swing.get('body_close_confirmed', True)
            volume_confirmed = tier2_swing.get('volume_confirmed', signal.get('volume_confirmed', False))
            strategy_trigger = tier2_swing.get('timeframe', '15m') or '15m'
            # Raw volume ratio (break candle volume / 20-bar SMA). "volume_confirmed" is a spike
            # flag (ratio > 1.3x by default), not a "volume exists" flag — show the raw value
            # so Claude can tell "no spike but normal volume" apart from "genuinely quiet candle".
            volume_ratio_val = signal.get('volume_ratio')
            if volume_ratio_val is not None:
                try:
                    _vr = float(volume_ratio_val)
                    volume_spike_display = (
                        f"✅ Yes ({_vr:.2f}x avg)" if volume_confirmed
                        else f"❌ No ({_vr:.2f}x avg, threshold 1.30x)"
                    )
                except (TypeError, ValueError):
                    volume_spike_display = '✅ Yes' if volume_confirmed else '❌ No'
            else:
                volume_spike_display = '✅ Yes' if volume_confirmed else '❌ No'

            # Tier 3 - Entry data
            tier3_entry = strategy_indicators.get('tier3_entry', {})
            strategy_entry = tier3_entry.get('timeframe', '5m') or '5m'
            entry_type = tier3_entry.get('entry_type', signal.get('entry_type', 'PULLBACK'))
            pullback_depth = tier3_entry.get('pullback_depth', signal.get('pullback_depth', 0))
            fib_zone = tier3_entry.get('fib_zone', 'Unknown')
            in_optimal_zone = tier3_entry.get('in_optimal_zone', signal.get('in_optimal_zone', False))
            order_type = tier3_entry.get('order_type', 'market')

            # Risk management data
            risk_mgmt = strategy_indicators.get('risk_management', {})
            if risk_mgmt:
                stop_loss = risk_mgmt.get('stop_loss', stop_loss)
                take_profit = risk_mgmt.get('take_profit', take_profit)
                risk_pips = risk_mgmt.get('risk_pips', risk_pips)
                reward_pips = risk_mgmt.get('reward_pips', reward_pips)
                rr_ratio = risk_mgmt.get('rr_ratio', rr_ratio)

            # Check if fixed SL/TP override is enabled (from database config)
            fixed_sl_tp_enabled = False
            fixed_sl_note = ""
            try:
                from forex_scanner.services.smc_simple_config_service import get_smc_simple_config
                smc_config = get_smc_simple_config()
                epic = signal.get('epic', '')
                if smc_config.fixed_sl_tp_override_enabled:
                    fixed_sl_tp_enabled = True
                    # Get per-pair SL/TP (falls back to global if not set)
                    fixed_sl = smc_config.get_pair_fixed_stop_loss(epic)
                    fixed_tp = smc_config.get_pair_fixed_take_profit(epic)
                    if fixed_sl and fixed_tp:
                        risk_pips = fixed_sl
                        reward_pips = fixed_tp
                        rr_ratio = reward_pips / risk_pips if risk_pips > 0 else 0
                        fixed_sl_note = f"\n⚠️ **FIXED SL/TP MODE ACTIVE**: All trades use SL={fixed_sl} pips, TP={fixed_tp} pips (R:R={rr_ratio:.2f}:1) regardless of strategy calculation."
            except Exception:
                pass  # Database config not available, skip fixed SL/TP note

            # --- EXTENDED MARKET CONTEXT FIELDS ---
            # Market regime (from market_intelligence dict added by TradeValidator)
            market_intel = signal.get('market_intelligence', {})
            regime_analysis = market_intel.get('regime_analysis', {})
            market_regime = regime_analysis.get('dominant_regime', 'unknown')
            regime_confidence = regime_analysis.get('confidence', 0)
            current_session = market_intel.get('session_analysis', {}).get('current_session', 'unknown')
            volatility_level = market_intel.get('volatility_level', '')
            mi_confidence_modifier = signal.get('market_intelligence_confidence_modifier')

            # Technical indicators (added by _add_performance_metrics in smc_simple_strategy)
            rsi_value = signal.get('rsi')
            rsi_zone = signal.get('rsi_zone', '')
            adx_value = signal.get('adx_value')
            adx_trend_strength = signal.get('adx_trend_strength', '')
            mtf_confluence = signal.get('mtf_confluence_score')
            all_tfs_aligned = signal.get('all_timeframes_aligned')
            atr_percentile = signal.get('atr_percentile')
            entry_quality = signal.get('entry_quality_score')

            # LPF data (added by Loss Prevention Filter in TradeValidator)
            lpf_penalty = signal.get('lpf_penalty')
            lpf_would_block = signal.get('lpf_would_block', False)
            lpf_rules = signal.get('lpf_triggered_rules', [])

            # Day of week from signal timestamp
            from datetime import datetime as _datetime
            signal_ts = signal.get('timestamp') or signal.get('created_at')
            day_of_week = ''
            if signal_ts:
                try:
                    if isinstance(signal_ts, str):
                        signal_ts = _datetime.fromisoformat(signal_ts)
                    day_of_week = signal_ts.strftime('%A')
                except Exception:
                    pass

            # Opposite swing for SL reference
            opposite_swing = signal.get('opposite_swing', 0)

            # Dataframe analysis - S/R and additional indicators
            dataframe_analysis = strategy_indicators.get('dataframe_analysis', {})
            sr_data = dataframe_analysis.get('sr_data', {})
            ema_data = dataframe_analysis.get('ema_data', {})
            other_indicators = dataframe_analysis.get('other_indicators', {})

            # Confidence breakdown
            confidence_breakdown = strategy_indicators.get('confidence_breakdown', {})

            # Proportional S/R buffer: tight stops need tighter buffer rules.
            # max(0.6 × SL, 5) capped at 10 to preserve legacy behavior for swing setups.
            try:
                _risk = float(risk_pips) if risk_pips else 0
            except (TypeError, ValueError):
                _risk = 0
            sr_buffer_pips = int(round(max(0.6 * _risk, 5))) if _risk > 0 else 10
            sr_buffer_pips = min(sr_buffer_pips, 10)

            # HTF label: in scalp mode use the strategy's actual HTF (e.g. 1H),
            # not 4H. 4H remains as macro context only.
            primary_htf_label = strategy_htf.upper() if is_scalp_mode else '4H'

            # Build chart analysis instructions
            chart_instruction = ""
            if has_chart:
                # v2.3.0: Updated chart instructions for both entry types
                momentum_note = ""
                if entry_type == 'MOMENTUM':
                    momentum_note = """
**⚡ MOMENTUM ENTRY NOTE:**
This is a MOMENTUM continuation trade (price beyond swing break).
- Entry is AFTER the swing break, riding momentum
- Higher risk but captures strong directional moves
- Look for: Clean breakout, no immediate reversal signs, volume confirmation
"""

                # Build dynamic chart description based on actual strategy timeframes.
                # Scalp signals have an extra HTF layer (1h/15m) between the 4H macro
                # and the trigger/entry, so the chart and checklist must reflect that.
                if is_scalp_mode:
                    htf_upper = strategy_htf.upper()
                    trigger_upper = strategy_trigger.upper()
                    entry_upper = strategy_entry.upper()
                    tf_display_section = f"""**Timeframes Displayed (Scalp Mode):**
- 4H timeframe: Macro trend context — 50 EMA bias (purple line)
- {htf_upper} timeframe: Strategy HTF bias — EMA used for trade direction decisions
- {trigger_upper} timeframe: PRIMARY ANALYSIS — swing break, EMAs 9/21, S/R levels, entry/SL/TP
- {entry_upper} timeframe: Entry precision — entry zone, Fibonacci levels, entry type annotation"""
                    primary_markers_label = f"{trigger_upper} chart (PRIMARY)"
                    entry_markers_label = f"{entry_upper} chart"
                    checklist_items = f"""1. ✓ **{htf_upper} STRATEGY HTF (HIGHEST PRIORITY):** This signal uses {htf_upper} for directional bias — not 4H. Judge trend alignment against the {htf_upper} EMA/structure shown on the chart. Use 4H ONLY as macro backdrop: if {htf_upper} aligns with the signal, a conflicting 4H is a −1 penalty, not a rejection.
2. ✓ **RESISTANCE/SUPPORT PROXIMITY:** For BUY — is entry near a recent swing HIGH, resistance zone, or round number? For SELL — near a swing LOW, support zone, or round number? Within {sr_buffer_pips} pips = scoring penalty (−1 to −2). Entry AT a MAJOR multi-touch S/R AGAINST {htf_upper} trend = score ≤4. Entry near S/R WITH {htf_upper} trend = penalty only, NOT a cap (SMC reversal entries near S/R are the strategy's edge).
3. ✓ **POSITION IN RANGE:** For BUY: entry should be near the BOTTOM of a local range. For SELL: near the TOP. Entry at the TOP of a recovery in a downtrend = buying into supply. REJECT.
4. ✓ Is price clearly respecting the {htf_upper} EMA trend direction? (This is the EMA the strategy used for bias — CRITICAL for scalp quality.)
5. ✓ Is the swing break on the {trigger_upper} chart clean and confirmed (full candle close)?
6. ✓ Are EMA 9/21 aligned with the trade direction on the {trigger_upper} chart?
7. ✓ Is entry clear of nearby S/R obstacles shown on the {trigger_upper} chart?
8. ✓ For PULLBACK: Is entry within or near the optimal Fibonacci zone ({entry_upper})?
9. ✓ For MOMENTUM: Is breakout clean with strong directional candles?
10. ✓ Is stop loss placement below a valid structure low (for longs)?
11. ✓ Does the price action show clean trend structure?
12. ✓ Are there any concerning patterns (engulfing candles, dojis at entry)?
13. ✓ Does the entry type box ({entry_upper}) show favorable conditions?"""
                else:
                    tf_display_section = """**Timeframes Displayed:**
- 4H timeframe: Shows 50 EMA trend bias (purple line)
- 15m timeframe: PRIMARY ANALYSIS - Shows swing break, EMAs 9/21, S/R levels, entry/SL/TP
- 5m timeframe: Shows entry zone with Fibonacci levels and entry type annotation"""
                    primary_markers_label = "15m chart - PRIMARY"
                    entry_markers_label = "5m chart"
                    checklist_items = f"""1. ✓ **4H TREND STRUCTURE (HIGHEST PRIORITY):** Look at the 4H chart candles — is the trend making Higher Highs/Higher Lows (bullish) or Lower Highs/Lower Lows (bearish)? This is MORE important than EMA position alone. A BUY signal in a 4H downtrend (LH/LL) or SELL signal in a 4H uptrend (HH/HL) is COUNTER-TREND and must score ≤4.
2. ✓ **RESISTANCE/SUPPORT PROXIMITY:** For BUY — is entry near a recent swing HIGH, resistance zone, or round number (x.x000, x.x500)? For SELL — is entry near a recent swing LOW, support zone, or round number? Within {sr_buffer_pips} pips = scoring penalty (−1 to −2). Entry AT a MAJOR multi-touch S/R AGAINST the 4H trend = score ≤4. Entry near S/R WITH the trend = penalty only, NOT a cap.
3. ✓ **POSITION IN RANGE:** Is entry at a favorable location? For BUY: entry should be near the BOTTOM of a local range (at demand/support). For SELL: entry should be near the TOP of a local range (at supply/resistance). Entry at the TOP of a recovery in a downtrend = buying the worst location.
4. ✓ Is price clearly respecting the 4H EMA trend direction?
5. ✓ Is the swing break on 15m clean and confirmed (full candle close)?
6. ✓ Are EMA 9/21 aligned with the trade direction on 15m chart?
7. ✓ Is entry clear of nearby S/R obstacles shown on 15m?
8. ✓ For PULLBACK: Is entry within or near the optimal Fibonacci zone (5m)?
9. ✓ For MOMENTUM: Is breakout clean with strong directional candles?
10. ✓ Is stop loss placement below a valid structure low (for longs)?
11. ✓ Does the price action show clean trend structure?
12. ✓ Are there any concerning patterns (engulfing candles, dojis at entry)?
13. ✓ Does the entry type box (5m) show favorable conditions?"""

                # Authoritative 4H structure block — pre-computed, not inferred
                macro_structure = signal.get('_macro_structure') or {}
                macro_auth_block = ""
                if macro_structure and macro_structure.get('trend_label', 'UNKNOWN') != 'UNKNOWN':
                    seq_str = ' → '.join(macro_structure.get('swing_sequence', [])) or 'insufficient pivots'
                    ema_str = macro_structure.get('ema50_relation', 'unknown')
                    macro_auth_block = f"""
📊 [4H MACRO CONTEXT — PRE-COMPUTED FROM RAW PRICE DATA]
The following 4H structure values are accurate (computed, not inferred). Do not override these specific values with a visual impression of the 4H panel:
  • Trend: {macro_structure['trend_label']}
  • Recent swing sequence (oldest → newest): {seq_str}
  • Price vs 4H EMA50: {ema_str}

**How to use this data:** 4H is macro backdrop only — it is NOT the primary HTF for this signal and is NOT an automatic rejection trigger. The primary HTF (stated below) drives the verdict. A 4H trend that opposes the signal is at most a −1 penalty when the primary HTF aligns with the signal direction.
[END 4H MACRO CONTEXT]
"""

                chart_instruction = f"""
## CHART ANALYSIS (CRITICAL - EXAMINE CAREFULLY)
{macro_auth_block}
The attached chart shows multi-timeframe forex analysis with the following elements:

{tf_display_section}

**Key Visual Markers (on {primary_markers_label}):**
- GREEN dashed line: Entry price level
- RED dashed line: Stop loss level (below opposite swing)
- BLUE dashed line: Take profit target
- ORANGE line: EMA 9 (fast momentum)
- BLUE line: EMA 21 (trend confirmation)
- GREEN horizontal line: Support level with distance in pips
- RED horizontal line: Resistance level with distance in pips
- ORANGE horizontal lines: Swing high levels
- BLUE horizontal lines: Swing low levels

**Key Visual Markers (on {entry_markers_label}):**
- YELLOW shaded zone: Fibonacci optimal entry zone (38.2%-61.8%)
- Entry Type Box (top-right): Shows PULLBACK/MOMENTUM, depth %, zone status, volume ✓/✗
- LARGE ARROW MARKER (▲/▼ in circle): Points to the EXACT entry price level - this is where the trade will be entered
- TRADE SUMMARY BOX (lower-left): Contains key stats - Direction, Entry Type, SL pips, TP pips, R:R ratio, Confidence %
- GREEN dashed line with "ENTRY" label: Entry price level
- "NOW" marker: Shows where current price is relative to entry
{momentum_note}
**CRITICAL CHART ANALYSIS CHECKLIST (in priority order):**

{checklist_items}
"""

            # v2.3.0: Enhanced entry type explanation
            entry_type_detail = ""
            if entry_type == 'PULLBACK':
                zone_status = "✅ OPTIMAL (38.2%-61.8%)" if in_optimal_zone else "⚠️ OUTSIDE OPTIMAL"
                entry_type_detail = f"""
- Entry Style: PULLBACK (waiting for retracement)
- Pullback Depth: {pullback_depth:.1%} into swing range
- Fibonacci Zone: {zone_status}
- Risk Profile: Lower risk, better R:R potential"""
            else:  # MOMENTUM
                entry_type_detail = f"""
- Entry Style: MOMENTUM (riding continuation)
- Position: {abs(pullback_depth):.1%} beyond swing break point
- Momentum Quality: {'Strong' if abs(pullback_depth) < 0.35 else 'Extended'}
- Risk Profile: Higher risk, captures strong moves"""

            # v2.3.0: Add confidence breakdown to prompt
            confidence_detail = ""
            if confidence_breakdown:
                confidence_detail = f"""
**CONFIDENCE SCORE BREAKDOWN ({confidence:.1%} total):**
- EMA Alignment: {confidence_breakdown.get('ema_alignment', 0)*100:.1f}%
- Volume Bonus: {confidence_breakdown.get('volume_bonus', 0)*100:.1f}%
- Pullback Quality: {confidence_breakdown.get('pullback_quality', 0)*100:.1f}%
- R:R Quality: {confidence_breakdown.get('rr_quality', 0)*100:.1f}%
- Fib Accuracy: {confidence_breakdown.get('fib_accuracy', 0)*100:.1f}%
"""

            # Build S/R context section
            sr_context = ""
            if sr_data:
                nearest_support = sr_data.get('nearest_support')
                nearest_resistance = sr_data.get('nearest_resistance')
                dist_support = sr_data.get('distance_to_support_pips') or 0
                dist_resistance = sr_data.get('distance_to_resistance_pips') or 0

                support_str = f"{self._format_price(nearest_support)} ({dist_support:.1f} pips below)" if nearest_support else "N/A"
                resistance_str = f"{self._format_price(nearest_resistance)} ({dist_resistance:.1f} pips above)" if nearest_resistance else "N/A"

                # Check path to target (only if we have the relevant S/R data)
                path_clear = True
                if direction == 'BULL' and nearest_resistance and dist_resistance < reward_pips:
                    path_clear = False
                    path_note = '⚠️ Resistance in way'
                elif direction == 'BEAR' and nearest_support and dist_support < reward_pips:
                    path_clear = False
                    path_note = '⚠️ Support in way'
                else:
                    path_note = '✅ Clear path'

                # Build level flip context if available
                flip_context = ""
                flip_data = sr_data.get('flip_analysis', {})
                if flip_data:
                    flip_lines = []
                    for key, info in flip_data.items():
                        if key == 'warning':
                            continue
                        if isinstance(info, dict) and 'original_type' in info:
                            orig = info.get('original_type', 'unknown')
                            dist = info.get('distance_pips', 0)
                            recent = info.get('is_recent_flip', False)
                            strength = info.get('strength', 0)
                            # Extract price from key like "flipped_support_1.17230"
                            price_str = key.rsplit('_', 1)[-1] if '_' in key else '?'
                            recency = "RECENT" if recent else "older"
                            flip_lines.append(
                                f"  - {price_str}: former {orig} → now flipped ({recency}, strength {strength:.2f}, {dist:.1f} pips away)"
                            )
                    if flip_lines:
                        flip_context = "\n- **Level Flips (broken S/R):**\n" + "\n".join(flip_lines) + "\n  ⚠️ Flipped levels indicate a structural break — former resistance becomes support (and vice versa). Do NOT reject solely because price is near a level that has already been broken."

                sr_context = f"""
**SUPPORT/RESISTANCE CONTEXT:**
- Nearest Support: {support_str}
- Nearest Resistance: {resistance_str}
- Path to Target: {path_note}{flip_context}
"""

            # Build EMA stack context
            ema_stack_context = ""
            if ema_data:
                ema_9_val = ema_data.get('ema_9', 0)
                ema_21_val = ema_data.get('ema_21', 0)
                ema_50_val = ema_data.get('ema_50', 0)

                if ema_9_val and ema_21_val:
                    ema_alignment = "Bullish" if ema_9_val > ema_21_val else "Bearish"
                    ema_aligned_with_signal = (ema_alignment == "Bullish" and direction == "BULL") or \
                                              (ema_alignment == "Bearish" and direction == "BEAR")

                    ema_stack_context = f"""
**5M EMA MICRO-STRUCTURE:**
- EMA 9: {self._format_price(ema_9_val)}
- EMA 21: {self._format_price(ema_21_val)}
- EMA 50: {self._format_price(ema_50_val)}
- 5m Trend: {ema_alignment} {'✅ Aligned' if ema_aligned_with_signal else '⚠️ Conflict (scoring penalty only, NOT auto-reject — SMC reversal entries legitimately fire against 5m micro)'}
"""

            # Build Bollinger Band context if available
            bb_context = ""
            if other_indicators:
                bb_upper = other_indicators.get('bb_upper')
                bb_middle = other_indicators.get('bb_middle')
                bb_lower = other_indicators.get('bb_lower')

                if bb_upper and bb_lower and entry_price:
                    bb_width = (bb_upper - bb_lower) * 10000  # in pips
                    price_in_bb = "Upper band" if entry_price > bb_middle else "Lower band"

                    bb_context = f"""
**BOLLINGER BAND CONTEXT:**
- BB Width: {bb_width:.1f} pips ({'Wide/Volatile' if bb_width > 30 else 'Narrow/Consolidating'})
- Entry Position: {price_in_bb} region
"""

            # Build market context section
            rsi_warning = ''
            if rsi_value:
                if direction == 'BULL' and rsi_value > 75:
                    rsi_warning = ' ⚠️ Overbought for BUY (extended)'
                elif direction == 'BEAR' and rsi_value < 25:
                    rsi_warning = ' ⚠️ Oversold for SELL (extended)'
                elif direction == 'BULL' and rsi_value > 65:
                    rsi_warning = ' • Extended for BUY (score penalty only, not reject)'
                elif direction == 'BEAR' and rsi_value < 35:
                    rsi_warning = ' • Extended for SELL (score penalty only, not reject)'

            adx_warning = ' ⚠️ Weak trend' if adx_value and adx_value < 20 else ''

            mtf_status = ''
            if all_tfs_aligned:
                mtf_status = ' ✅ All TFs aligned'
            elif mtf_confluence is not None:
                mtf_status = ' ⚠️ Partial alignment'

            market_context_section = f"""
═══════════════════════════════════════════════════════════════
🌍 MARKET CONTEXT
═══════════════════════════════════════════════════════════════
**Regime:** {market_regime.upper()} (confidence: {regime_confidence:.0%})
**Session:** {current_session.upper()}{f'  |  Day: {day_of_week}' if day_of_week else ''}
**Volatility:** {volatility_level.upper() if volatility_level else 'N/A'}  |  ATR Percentile: {f'{atr_percentile:.0f}%' if atr_percentile is not None else 'N/A'}

**Technical Indicators:**
- RSI(14): {f'{rsi_value:.1f} ({rsi_zone})' if rsi_value else 'N/A'}{rsi_warning}
- ADX: {f'{adx_value:.1f} ({adx_trend_strength})' if adx_value else 'N/A'}{adx_warning}
- MTF Confluence: {f'{mtf_confluence:.2f}' if mtf_confluence is not None else 'N/A'}{mtf_status}
- Entry Quality Score: {f'{entry_quality:.2f}' if entry_quality is not None else 'N/A'}
- MI Confidence Modifier: {f'{mi_confidence_modifier:+.1%}' if mi_confidence_modifier is not None else 'N/A'}
"""

            # Build LPF section (only if LPF fired)
            lpf_section = ''
            if lpf_penalty or lpf_rules:
                rules_str = ', '.join(lpf_rules) if lpf_rules else 'none'
                lpf_section = f"""
⚠️ **LOSS PREVENTION FILTER ALERT:**
- Penalty: {f'{lpf_penalty:.2f}' if lpf_penalty else '0'}  |  Would Block: {'YES' if lpf_would_block else 'NO'}
- Rules Triggered: {rules_str}
(LPF uses historical win-rate data to flag high-risk conditions — weight this heavily in your assessment)
"""

            # Build SMC-specific analysis section
            smc_analysis = f"""
## SMC SIMPLE v2.3.0 STRATEGY DATA (3-TIER VALIDATION)

**TIER 1 - 4H Directional Bias:**
- 50 EMA Value: {self._format_price(ema_value)}
- Distance from EMA: {ema_distance:.1f} pips {'✅' if ema_distance >= 2.5 else '⚠️ Close to EMA'}
- Bias Direction: {ema_direction}

**TIER 2 - 15m Swing Break:**
- Swing Level Broken: {self._format_price(swing_level)}
- Opposite Swing (SL reference): {self._format_price(opposite_swing)}
- Body Close Confirmed: {'✅ Yes' if body_close_confirmed else '❌ No'}
- Volume Spike: {volume_spike_display} — normal volume is fine; this only flags above-average spikes

**TIER 3 - Entry Analysis:**
{entry_type_detail}
- Fib Zone: {fib_zone}
- Order Type: {order_type.upper()}
{confidence_detail}
{sr_context}
{ema_stack_context}
{bb_context}
"""

            # Build the complete prompt
            prompt = f"""You are a SENIOR FOREX TECHNICAL ANALYST with 20+ years of institutional trading experience specializing in Smart Money Concepts (SMC) analysis.

**YOUR ROLE:** Validate this SMC Simple v2.3.0 signal. This strategy uses relaxed entry thresholds to capture more opportunities while maintaining trend alignment discipline.

═══════════════════════════════════════════════════════════════
📊 SIGNAL OVERVIEW
═══════════════════════════════════════════════════════════════
• Pair: {pair}
• Direction: {direction}
• Strategy: {strategy} v2.3.0
• System Confidence: {confidence:.1%}
• Entry Type: {entry_type}

═══════════════════════════════════════════════════════════════
💰 TRADE LEVELS
═══════════════════════════════════════════════════════════════
• Entry Price: {self._format_price(entry_price)}
• Stop Loss: {risk_pips:.1f} pips
• Take Profit: {reward_pips:.1f} pips
• Risk:Reward Ratio: {rr_ratio:.2f}:1{fixed_sl_note}
{chart_instruction}
{market_context_section}
{lpf_section}
{smc_analysis}
═══════════════════════════════════════════════════════════════
📋 REQUIRED RESPONSE FORMAT — READ CAREFULLY
═══════════════════════════════════════════════════════════════

**CRITICAL:** Your response MUST begin with the literal token `SCORE:` as the very first characters. No preamble, no headers, no checklists, no markdown `##` sections, no "## SENIOR ANALYST ASSESSMENT" intros — previous outputs that led with a narrative checklist got truncated before reaching the verdict and were auto-rejected. Do the analysis silently, then emit only these three lines:

SCORE: [1-10]
DECISION: [APPROVE/REJECT]
REASON: [2-3 sentences explaining your professional assessment. Focus on: trend alignment, entry quality, and any visual concerns from the chart. Do NOT penalize for low R:R or small TP - we are testing small quick profits.]

**SCORING GUIDELINES FOR v2.9.0 (HTF-STRICT MODE):**
- Primary HTF for this signal: **{primary_htf_label}** (strategy bias TF). 4H is macro backdrop only.
- 8-10: Strong {primary_htf_label} trend alignment (HH/HL or LH/LL confirmed), clean swing break on trigger TF, volume confirmed, EMA 9/21 aligned, entry at favorable range location
- 6-7: Good {primary_htf_label} trend alignment with minor concerns (e.g., momentum slightly extended, volume not confirmed, minor S/R nearby)
- 4-5: Marginal setup - {primary_htf_label} trend unclear/choppy, entry near resistance/support level. NOTE: 5m EMA micro-structure conflict alone is NOT a marginal-setup trigger — apply as a −1 scoring penalty only. SMC reversal entries fire against 5m micro-structure by design.
- 1-3: Poor setup - counter-trend to {primary_htf_label} structure, entry AT resistance/support, or technical breakdown

**HARD SCORE CAPS (cannot exceed these regardless of other factors):**
- Counter-trend to **{primary_htf_label}** structure AND MTF Confluence < 0.6: MAX SCORE 4, REJECT. This cap is evaluated against the strategy's primary HTF ({primary_htf_label}), NOT 4H macro. If {primary_htf_label} aligns with the signal but 4H disagrees, apply only a −1 penalty.
  ⚠️ EXCEPTION: If MTF Confluence ≥ 0.6 OR "All TFs aligned" flag is present, the strategy's multi-timeframe system has already validated direction. Do NOT apply counter-trend cap — score on entry quality and S/R location instead.
- Entry within **{sr_buffer_pips} pips** of major resistance (for BUY) or support (for SELL): apply a −2 scoring penalty. This is NOT an automatic rejection — SMC reversal entries at S/R are the strategy's edge. Only cap at MAX SCORE 4 if the S/R level is confirmed MAJOR (multi-touch, HTF-significant) AND entry is within 2 pips AND {primary_htf_label} trend opposes the signal.
- Entry at round number psychological level (x.x000, x.x500) against HTF trend: MAX SCORE 3, REJECT
- Entry at TOP of local recovery in a downtrend (or BOTTOM of local selloff in uptrend): apply a −2 scoring penalty. Cap at MAX SCORE 4 REJECT only if ALL THREE are true: (a) {primary_htf_label} trend opposes signal, (b) no reversal confirmation candle visible on trigger TF, (c) a MAJOR S/R level is also within 2 pips blocking the path. Otherwise score penalty only — SMC often enters near local turning points by design.
- **4H macro structure (RANGING / LL / LH etc.) alone is NOT a cap or rejection trigger.** In scalp mode the primary HTF is {primary_htf_label}, not 4H. 4H is macro backdrop only — if {primary_htf_label} aligns with the signal, a conflicting 4H structure is at most a −1 penalty.

⚠️ TESTING MODE: R:R and TP minimums are DISABLED. We are testing small, quick profits. Do NOT reject based on low R:R or small TP. Focus on trend alignment, HTF structure, and entry location quality.

**ENTRY TYPE EVALUATION:**
- PULLBACK entries: Prefer entries in 38.2%-61.8% Fib zone (check 5m chart). Outside zone = lower score but not automatic rejection
- MOMENTUM entries: Accept up to 50% beyond break point. Look for strong directional candles on 15m. Exhaustion signs alone = −2 penalty, not auto-reject. Only REJECT a MOMENTUM entry when exhaustion is paired with counter-HTF bias AND a clear reversal candle on the trigger TF.
- Check entry type box on 5m chart for quick visual confirmation

**SUPPORT/RESISTANCE EVALUATION (check 15m chart):**
- Check if S/R levels shown on 15m chart obstruct the path to take profit
- For BULL: Resistance should be BEYOND take profit level
- For BEAR: Support should be BEYOND take profit level
- S/R within 50% of target distance = caution, within 25% = strong concern

**MARKET CONTEXT EVALUATION (use the 🌍 section above):**
- RANGING regime + trend-following signal: reduce score by 1-2 points
- RSI overbought (>65) for BUY or oversold (<35) for SELL: flag as concern, reduce score by 1
- ADX < 20 (weak trend): flag low trend strength, reduce score by 1 for trend-following entries
- MTF confluence < 0.5 or not all TFs aligned: treat as increased risk (informational — already captured by the HTF/MTF hard cap above, do NOT subtract again)
- Entry Quality Score < 0.4: strategy itself rated this entry poorly — reduce score by 1
- LPF "Would Block: YES": informational only — LPF runs as its own upstream filter. Do NOT apply a further score penalty here.
- MI Confidence Modifier strongly negative (< -10%): informational context only — do NOT apply a further score penalty here unless MI contradicts both HTF bias AND entry quality.

**IMPORTANT — penalty cap:** the combined penalty from the MARKET CONTEXT block above is capped at −2 total. Pick the single most severe signal and apply that; do NOT stack all items into a cumulative −4 to −6. HTF alignment, swing break cleanliness, major S/R obstruction, and exhaustion-at-momentum are the four factors that should drive the decision — everything else is shading, not compounding reject pressure.

**AUTOMATIC REJECTION CRITERIA (ANY ONE = REJECT):**
- **{primary_htf_label} trend structure opposes signal AND MTF Confluence < 0.6** — evaluate against the strategy's primary HTF ({primary_htf_label}). If MTF Confluence ≥ 0.6 or "All TFs aligned", skip this rejection. Do NOT reject based on 4H macro structure alone when {primary_htf_label} aligns with the signal.
- **Entry AT resistance (BUY) or AT support (SELL)** — REJECT ONLY if ALL THREE are true: (a) price within **2 pips** of a MAJOR (multi-touch, HTF-confirmed) resistance/support, (b) {primary_htf_label} trend opposes the signal, (c) no reversal confirmation candle visible. A near-S/R entry that's WITH the strategy's primary HTF bias is not an auto-reject — score penalty only.
- **Entry at top of local recovery / bottom of local selloff** — penalty only by default (see HARD SCORE CAPS above). REJECT here only if {primary_htf_label} opposes signal AND no reversal confirmation candle AND major S/R within 2 pips (same 3-condition AND-gate).
- MOMENTUM entry showing reversal candles on trigger TF (engulfing, pin bars against direction) AND {primary_htf_label} opposes the signal — reversal candle alone is a −2 penalty, not an auto-reject.
- S/R level on trigger TF blocking more than 75% of path to target AND {primary_htf_label} opposes the signal — partial obstruction with HTF bias alignment is a penalty, not a reject.

**NOT automatic rejection triggers (apply as score penalties instead):**
- EMA 9/21 crossed against signal direction on trigger TF → −1 penalty
- 5M EMA micro-structure conflict → −1 penalty
- Price close to {primary_htf_label} EMA → −1 penalty
- Counter-trend to 4H when primary HTF ({primary_htf_label}) aligns with signal → −1 penalty
- RSI 30–40 for SELL or 60–70 for BUY → −1 penalty (not "oversold/overbought" rejection)

Be concise but thorough. Your assessment determines if real money is risked."""

            return prompt

        except Exception as e:
            self.logger.error(f"Error building vision prompt: {e}")
            return self._build_fallback_prompt(signal, {})

    def _build_fvg_retest_prompt(self, signal: Dict, has_chart: bool = True) -> str:
        """
        Build vision-enabled prompt for FVG Retest strategy analysis.
        Covers both Type A (Deep Value / FVG Tap) and Type B (Institutional Initiation).
        """
        try:
            epic = signal.get('epic', 'Unknown')
            pair = self._extract_pair(epic)
            direction = signal.get('signal_type', signal.get('signal', 'Unknown'))
            confidence = signal.get('confidence_score', 0)
            entry_type = signal.get('entry_type', 'DEEP_VALUE')

            entry_price = signal.get('entry_price', signal.get('price', 0))
            risk_pips = signal.get('sl_pips', signal.get('risk_pips', 0))
            reward_pips = signal.get('tp_pips', signal.get('reward_pips', 0))
            rr_ratio = signal.get('rr_ratio', 0)

            # Entry-type specific data
            if entry_type == 'DEEP_VALUE':
                fvg_zone = signal.get('fvg_zone', 'N/A')
                fvg_size = signal.get('fvg_size_pips', 0)
                fvg_significance = signal.get('fvg_significance', 0)
                setup_age = signal.get('setup_age_minutes', 0)

                entry_detail = f"""**TYPE A - DEEP VALUE (FVG Tap) Entry:**
- FVG Zone: {fvg_zone}
- FVG Size: {fvg_size:.1f} pips
- FVG Significance: {fvg_significance:.3f}
- Setup Age: {setup_age:.0f} minutes since BOS
- Swing Level (invalidation): {self._format_price(signal.get('swing_level', 0))}
- Entry Logic: Price retraced into an unfilled Fair Value Gap created during a Break of Structure"""
            else:
                displacement = signal.get('displacement_ratio', 0)
                follow_through = signal.get('follow_through', False)
                volume_spike = signal.get('volume_spike', False)

                entry_detail = f"""**TYPE B - INSTITUTIONAL INITIATION Entry:**
- Displacement: {displacement:.2f}x ATR (break candle body vs average)
- Follow-Through: {'✅ Confirmed' if follow_through else '❌ Not confirmed'}
- Volume Spike: {'✅ Above average' if volume_spike else '❌ Below average'}
- Swing Level (invalidation): {self._format_price(signal.get('swing_level', 0))}
- Entry Logic: High-velocity BOS with institutional displacement — immediate entry without waiting for retest"""

            chart_instruction = ""
            if has_chart:
                chart_instruction = f"""
## CHART ANALYSIS (EXAMINE CAREFULLY)

The chart shows multi-timeframe analysis:

**Timeframes:**
- 1H: Shows 200 EMA trend bias (macro direction filter)
- 5m: Shows Break of Structure, FVG zones, and entry/SL/TP levels

**What to look for:**
- GREEN dashed line: Entry price
- RED dashed line: Stop loss
- BLUE dashed line: Take profit
- FVG zones (if visible): Semi-transparent shaded areas

**CHART ANALYSIS CHECKLIST:**
1. Is price clearly on the correct side of the 1H 200 EMA?
2. Is the Break of Structure clean (clear swing high/low violation)?
3. {'Is the FVG zone well-defined and price tapping into it cleanly?' if entry_type == 'DEEP_VALUE' else 'Are the displacement candles strong and impulsive?'}
4. Is there clean price structure (not choppy/ranging)?
5. Is stop loss behind a valid structural level?
6. Are there any reversal patterns at entry (engulfing, pin bars against direction)?
7. **CRITICAL - SWING PROXIMITY CHECK:**
   - For BUY signals: Is entry price dangerously close to a recent swing HIGH? Buying near swing highs means buying into resistance — HIGH REJECTION RISK
   - For SELL signals: Is entry price dangerously close to a recent swing LOW? Selling near swing lows means selling into support — HIGH REJECTION RISK
   - Look at the 5m chart for nearby swing highs/lows relative to entry price
   - If entry is within 5-10 pips of a swing high (for buys) or swing low (for sells), this is a MAJOR concern
"""

            prompt = f"""You are a SENIOR FOREX TECHNICAL ANALYST specializing in Smart Money Concepts (SMC) and Fair Value Gap analysis.

**YOUR ROLE:** Validate this FVG Retest strategy signal. This strategy detects Breaks of Structure on the 5m chart with 1H 200 EMA macro confirmation, then enters via FVG retest (Type A) or institutional displacement (Type B).

═══════════════════════════════════════════════════════════════
📊 SIGNAL OVERVIEW
═══════════════════════════════════════════════════════════════
• Pair: {pair}
• Direction: {direction}
• Strategy: FVG_RETEST
• Entry Mode: {entry_type}
• System Confidence: {confidence:.1%}

═══════════════════════════════════════════════════════════════
💰 TRADE LEVELS
═══════════════════════════════════════════════════════════════
• Entry Price: {self._format_price(entry_price)}
• Stop Loss: {risk_pips:.1f} pips
• Take Profit: {reward_pips:.1f} pips
• Risk:Reward Ratio: {rr_ratio:.2f}:1
{chart_instruction}
═══════════════════════════════════════════════════════════════
🔬 STRATEGY-SPECIFIC DATA
═══════════════════════════════════════════════════════════════

**MACRO FILTER (1H):**
- 200 EMA Direction: Price {'above' if direction in ('BUY', 'BULL') else 'below'} EMA → {direction} bias
- Last 1H candle confirmed direction alignment

**TRIGGER (5m):**
- Break of Structure detected on 5m chart
- BOS direction matches 1H macro bias

{entry_detail}

═══════════════════════════════════════════════════════════════
📋 REQUIRED RESPONSE FORMAT — READ CAREFULLY
═══════════════════════════════════════════════════════════════

**CRITICAL:** Your response MUST begin with the literal token `SCORE:` as the very first characters. No preamble, no headers, no checklists, no markdown `##` sections — previous outputs that led with a narrative checklist got truncated before reaching the verdict and were auto-rejected. Do the analysis silently, then emit only these three lines:

SCORE: [1-10]
DECISION: [APPROVE/REJECT]
REASON: [2-3 sentences. Focus on: trend alignment quality, BOS clarity, {'FVG zone quality and tap precision' if entry_type == 'DEEP_VALUE' else 'displacement strength and follow-through quality'}, and any visual concerns.]

**SCORING GUIDELINES:**
- 8-10: Clean BOS, strong 1H trend, {'well-defined FVG with clean tap' if entry_type == 'DEEP_VALUE' else 'strong displacement with confirmed follow-through'}, no reversal signs
- 6-7: Good setup with minor concerns (slightly choppy structure, {'FVG partially filled' if entry_type == 'DEEP_VALUE' else 'moderate displacement'})
- 4-5: Marginal — weak trend, unclear BOS, or entry quality issues
- 1-3: Poor — counter-trend risk, reversal patterns, or structural breakdown

**AUTOMATIC REJECTION CRITERIA:**
- Price on wrong side of 1H 200 EMA (counter-trend)
- BOS not clearly visible on 5m chart
- Reversal candlestick patterns at entry level
- Choppy, range-bound price action with no clear structure
- **BUY entry near a recent swing HIGH (buying into resistance)**
- **SELL entry near a recent swing LOW (selling into support)**

Be concise but thorough. Your assessment determines if real money is risked."""

            return prompt

        except Exception as e:
            self.logger.error(f"Error building FVG Retest prompt: {e}")
            return self._build_fallback_prompt(signal, {})

    def _build_mean_reversion_prompt(self, signal: Dict, has_chart: bool = True) -> str:
        """
        Vision-enabled prompt for the MEAN_REVERSION strategy.

        Mean-reversion is the OPPOSITE thesis to SMC_SIMPLE / FVG_RETEST:
            • SMC / FVG: trade WITH momentum, respect HTF EMA, use MACD as filter,
              "overextended momentum often continues" is a feature not a flaw.
            • MEAN_REVERSION: FADE the extreme, expect reversion to BB middle,
              overextension IS the setup, counter-MACD is the setup.

        This prompt therefore inverts the usual validation heuristics. Claude
        must evaluate whether the *exhaustion* is real (reject if still impulsing)
        and whether the HTF regime permits a mean-reversion trade at all (reject
        if 1h ADX breaks above 25 — a sign of a strong trend where fading is
        structurally unsafe).
        """
        try:
            epic = signal.get('epic', 'Unknown')
            pair = self._extract_pair(epic)
            direction = signal.get('signal_type', signal.get('signal', 'Unknown'))
            confidence = signal.get('confidence_score', signal.get('confidence', 0))

            entry_price = signal.get('entry_price', signal.get('price', 0))
            risk_pips = signal.get('stop_loss_pips', signal.get('risk_pips', 0))
            reward_pips = signal.get('take_profit_pips', signal.get('reward_pips', 0))
            try:
                rr_ratio = float(reward_pips) / float(risk_pips) if float(risk_pips) > 0 else 0
            except Exception:
                rr_ratio = 0

            monitor_only = bool(signal.get('monitor_only', False))

            indicators = signal.get('strategy_indicators', {}) or {}
            adx_15m = indicators.get('adx', signal.get('adx'))
            adx_1h = indicators.get('adx_htf', signal.get('adx_htf'))
            rsi_val = indicators.get('rsi', signal.get('rsi'))
            bb_upper = indicators.get('bb_upper')
            bb_lower = indicators.get('bb_lower')
            bb_mid = indicators.get('bb_mid')
            bb_mult = indicators.get('bb_mult')
            extremity = indicators.get('extremity')

            def fmt(v, prec=2, suffix=""):
                if v is None:
                    return "n/a"
                try:
                    return f"{float(v):.{prec}f}{suffix}"
                except Exception:
                    return str(v)

            # Which oscillator extremes triggered the signal
            if direction in ('BUY', 'BULL'):
                setup_side = "OVERSOLD"
                fade_what = "fade the drop"
                expected_move = "back up toward the BB middle"
                invalidator = "continued impulsive decline past the BB lower band"
                reversal_patterns_wanted = "bullish engulfing, bullish pin bar, hammer, doji rejection, double-bottom"
            else:
                setup_side = "OVERBOUGHT"
                fade_what = "fade the rally"
                expected_move = "back down toward the BB middle"
                invalidator = "continued impulsive rally past the BB upper band"
                reversal_patterns_wanted = "bearish engulfing, shooting star, inverted hammer, doji rejection, double-top"

            chart_instruction = ""
            if has_chart:
                chart_instruction = f"""
## CHART ANALYSIS (EXAMINE CAREFULLY — MEAN-REVERSION LENS)

The chart spans multiple timeframes. For mean-reversion you look for different things than trend-continuation:

**What to look for (POSITIVE):**
- Price has *just touched or poked through* the Bollinger band (not riding it for many candles — that's a trend)
- Decelerating candles approaching the extreme: smaller bodies, longer wicks, volume tapering
- {reversal_patterns_wanted} at or just past the band
- 1h chart: price is inside a range, NOT in a strong directional run (flat or flattening EMAs)
- Prior BB touches on the same side that reverted cleanly to mid-band (validates the regime)

**What to look for (NEGATIVE — REJECT if any):**
- Price is *still impulsing* past the band with a wide body and expanding volume — this is a breakout, not an exhaustion
- 1h chart shows clear directional structure (stair-step HH/HL or LH/LL) — we are fading a macro trend
- Price rode the BB band for many consecutive candles before the signal — band-walking means trend, not range
- Gap or news candle caused the extreme — fading news is catching knives

**Lines on chart:**
- GREEN dashed: Entry price
- RED dashed: Stop loss (structural — just beyond the band / recent wick)
- BLUE dashed: Take profit (aimed at BB middle, roughly)
"""

            mo_note = (
                "\n⚠️ **MONITOR-ONLY MODE:** This strategy is still in data-collection phase on this pair. "
                "Your score still matters for the historical dataset, but no capital is at risk on this alert.\n"
                if monitor_only else ""
            )

            prompt = f"""You are a SENIOR FOREX TECHNICAL ANALYST evaluating a MEAN-REVERSION signal.

**STRATEGY THESIS — READ BEFORE ANY ANALYSIS:**
MEAN_REVERSION fires when price has pushed to a statistical extreme (outside Bollinger
Band 20/2σ) with a matching RSI(14) extreme (≤ 30 or ≥ 70), in a low-ADX regime
(15m ADX ≤ {fmt(22, 0)}, 1h ADX ≤ {fmt(25, 0)}). The trade is to **{fade_what}** and
expect price to revert **{expected_move}**.

This is the OPPOSITE of a trend-following trade. Therefore:
- ❌ DO NOT reject because MACD is against the entry — mean-reversion is counter-MACD by construction
- ❌ DO NOT reject because price is on the "wrong" side of the HTF EMA — we are fading, not continuing
- ❌ DO NOT apply the "overextended momentum often continues" rule — overextension IS the setup here
- ✅ DO reject if price is STILL IMPULSING with wide-range candles and rising volume
- ✅ DO reject if the 1h chart shows a clear directional trend (band-walking, stair-step structure)
- ✅ DO reject if the extreme was caused by a news spike or gap
{mo_note}
═══════════════════════════════════════════════════════════════
📊 SIGNAL OVERVIEW
═══════════════════════════════════════════════════════════════
• Pair: {pair}
• Direction: {direction}  ({setup_side} fade)
• Strategy: MEAN_REVERSION
• System Confidence: {confidence:.1%}

═══════════════════════════════════════════════════════════════
💰 TRADE LEVELS
═══════════════════════════════════════════════════════════════
• Entry Price: {self._format_price(entry_price)}
• Stop Loss: {fmt(risk_pips, 1)} pips  (structural — just beyond the extreme)
• Take Profit: {fmt(reward_pips, 1)} pips (target: BB middle)
• Risk:Reward Ratio: {rr_ratio:.2f}:1
  ℹ️  MR R:R is structurally 1.5:1 not 2:1 — do NOT penalize the low R:R. The edge
      comes from high win-rate (target hits before wick to SL), not from pay-off skew.

═══════════════════════════════════════════════════════════════
🔬 STRATEGY-SPECIFIC DATA (populated by the strategy at signal time)
═══════════════════════════════════════════════════════════════
• 15m ADX: {fmt(adx_15m, 1)}  (hard ceiling: 22 — signal already passed)
• 1h  ADX: {fmt(adx_1h, 1)}   (hard ceiling: 25 — signal already passed)
• RSI(14) 15m: {fmt(rsi_val, 1)}  (threshold for {setup_side}: ≤ 30 BUY / ≥ 70 SELL)
• Bollinger Bands (20, {fmt(bb_mult, 1)}σ):
    upper {self._format_price(bb_upper)} / mid {self._format_price(bb_mid)} / lower {self._format_price(bb_lower)}
• Extremity score (0-1, how far past threshold): {fmt(extremity, 2)}
{chart_instruction}
═══════════════════════════════════════════════════════════════
📋 REQUIRED RESPONSE FORMAT — READ CAREFULLY
═══════════════════════════════════════════════════════════════

**CRITICAL:** Your response MUST begin with the literal token `SCORE:` as the very first characters. No preamble, no headers, no checklists, no markdown `##` sections — previous outputs that led with a narrative checklist got truncated before reaching the verdict and were auto-rejected. Do the analysis silently, then emit only these three lines:

SCORE: [1-10]
DECISION: [APPROVE/REJECT]
REASON: [2-3 sentences. Focus on: exhaustion quality at the band, 1h trend absence, and presence/absence of reversal patterns. Do NOT comment on MACD alignment or HTF EMA side — those are intentionally non-factors here.]

**SCORING GUIDELINES (mean-reversion lens):**
- 8-10: Clean exhaustion — smaller bodies + longer wicks into the band, reversal pattern visible at/just past the band, 1h flat/range, prior reverts on the same pair show the behavior
- 6-7: Acceptable — band touch with RSI extreme but exhaustion not yet confirmed, 1h slope mild
- 4-5: Marginal — price at the band but still impulsing (wide body current candle), or 1h showing directional hints
- 1-3: Poor — price band-walking for multiple candles, 1h in clear directional run, expanding volume/range on the break candle (this is a breakout, not a reversal)

**AUTOMATIC REJECTION CRITERIA (ANY ONE = REJECT):**
- Current candle is an impulsive wide-range expansion with above-average volume in the signal direction (catching a knife)
- 1h chart shows a clear, fresh directional structure (HH/HL for uptrend or LH/LL for downtrend) in the opposite direction of the fade
- Price has ridden the BB band on the same side for ≥ 5 consecutive candles before this signal (band-walking confirms trend)
- News / gap spike visible as the cause of the extreme (vertical candle, gap bar)
- No reversal structure whatsoever: price made the extreme on the CURRENT candle with wide body and no wick (no exhaustion yet — too early)

**NON-CRITERIA — DO NOT USE THESE TO REJECT:**
- MACD histogram against the entry (by design — ignore)
- Price on "wrong" side of HTF EMA (fade-of-trend inside a range is the whole point — ignore)
- Low R:R 1.5:1 (structural to the strategy — ignore)
- RSI extremity alone (extreme RSI IS the setup, not a concern)

Be concise but thorough. Remember: you are evaluating an *exhaustion fade*, not a trend entry."""

            return prompt

        except Exception as e:
            self.logger.error(f"Error building MEAN_REVERSION prompt: {e}")
            return self._build_fallback_prompt(signal, {})

    def _build_eurusd_range_fade_prompt(self, signal: Dict, has_chart: bool = True) -> str:
        """Vision prompt for EURUSD range-fade setups."""
        try:
            epic = signal.get('epic', 'Unknown')
            pair = self._extract_pair(epic)
            strategy = signal.get('strategy', 'RANGE_FADE')
            direction = signal.get('signal_type', signal.get('signal', 'Unknown'))
            confidence = signal.get('confidence_score', signal.get('confidence', 0))
            entry_price = signal.get('entry_price', signal.get('price', 0))
            risk_pips = signal.get('stop_loss_pips', signal.get('risk_pips', 0))
            reward_pips = signal.get('take_profit_pips', signal.get('reward_pips', 0))
            monitor_only = bool(signal.get('monitor_only', False))

            try:
                rr_ratio = float(reward_pips) / float(risk_pips) if float(risk_pips) > 0 else 0
            except Exception:
                rr_ratio = 0

            indicators = signal.get('strategy_indicators', {}) or {}
            rsi_val = indicators.get('rsi', signal.get('rsi'))
            bb_upper = indicators.get('bb_upper')
            bb_lower = indicators.get('bb_lower')
            bb_mid = indicators.get('bb_mid')
            band_width = indicators.get('band_width_pips')
            htf_bias = indicators.get('htf_bias', 'unknown')
            range_high = indicators.get('range_high')
            range_low = indicators.get('range_low')
            dist_low = indicators.get('distance_to_low_pips')
            dist_high = indicators.get('distance_to_high_pips')

            def fmt(v, prec=2, suffix=""):
                if v is None:
                    return "n/a"
                try:
                    return f"{float(v):.{prec}f}{suffix}"
                except Exception:
                    return str(v)

            if direction in ('BUY', 'BULL'):
                setup_side = "lower Bollinger Band fade (HTF-aligned long)"
                desired_confirmation = "stalling downside momentum at/near the lower band while the 1h bias remains bullish"
                reject_if = "the 1h has just flipped bearish or price is expanding lower through the band with strong continuation candles"
            else:
                setup_side = "upper Bollinger Band fade (HTF-aligned short)"
                desired_confirmation = "stalling upside momentum at/near the upper band while the 1h bias remains bearish"
                reject_if = "the 1h has just flipped bullish or price is expanding higher through the band with strong continuation candles"

            chart_instruction = ""
            if has_chart:
                chart_instruction = f"""
## CHART ANALYSIS (HTF-ALIGNED BB FADE LENS)

This is a **controlled mean-reversion** trade. Price has touched a Bollinger Band edge while the 1h HTF bias is {htf_bias}. We are fading the 5m stretch IN THE DIRECTION of the 1h bias, not calling a reversal of the 1h trend.

Positive signs:
- Wick rejection or stall at the outer band rather than a sustained band-walk
- 1h bias is clearly {htf_bias} and the entry direction aligns with it
- Current 5m bar is not an expansion candle — body size looks typical
- No recent breakout structure on the 1h that would invalidate the fade

Automatic concerns:
- 1h bias has visibly flipped or is in transition (e.g. HTF EMA being crossed right now)
- Multiple consecutive expansion candles pushing through the band (band-walking, not fading)
- News-driven spike or gap — this strategy has no macro filter
- Entry against an obvious intraday breakout on the 1h
"""

            mo_note = (
                "\n⚠️ **MONITOR-ONLY MODE:** This strategy is in live-signal validation. Score honestly; no capital is at risk.\n"
                if monitor_only else ""
            )

            return f"""You are a SENIOR FOREX TECHNICAL ANALYST evaluating a Bollinger Band fade with higher-timeframe alignment.

**STRATEGY THESIS — READ FIRST (v0.4.0 lean config)**
RANGE_FADE is a controlled mean-reversion trade on EURUSD 5m. It triggers when:
1. Price touches a Bollinger Band edge (upper for shorts, lower for longs)
2. RSI is past 40 (for longs) or 60 (for shorts) — moderate directional bias, NOT an extreme
3. The 1h HTF bias (EMA50 + slope) aligns with the fade direction — strict, no neutral
4. Current hour is within London/NY window (06–18 UTC)

It is NOT a strict range-extreme fade and NOT a momentum continuation trade. It fades a 5m stretch IN THE DIRECTION of the 1h trend.

Important:
- ❌ Do NOT reject because RSI isn't at 20/80 — the lean config explicitly uses 40/60 moderate thresholds (ablation showed tighter RSI killed sample size without adding edge).
- ❌ Do NOT demand price be within N pips of a horizontal range boundary — the proximity gate is intentionally disabled in v0.4.0.
- ❌ Do NOT apply a generic SMC "momentum continuation" or "wait for structure break" lens.
- ✅ DO reject if the 1h bias is actively flipping against the entry.
- ✅ DO reject if the current 5m bar is an expansion candle breaking clean out of the band (band-walk, not fade).
- ✅ DO approve if you see a normal pullback to the band in an HTF-aligned direction.
{mo_note}
═══════════════════════════════════════════════════════════════
📊 SIGNAL OVERVIEW
═══════════════════════════════════════════════════════════════
• Pair: {pair}
• Direction: {direction}
• Strategy: {strategy}
• Setup Type: {setup_side}
• System Confidence: {confidence:.1%}

═══════════════════════════════════════════════════════════════
💰 TRADE LEVELS
═══════════════════════════════════════════════════════════════
• Entry Price: {self._format_price(entry_price)}
• Stop Loss: {fmt(risk_pips, 1)} pips
• Take Profit: {fmt(reward_pips, 1)} pips
• Risk:Reward Ratio: {rr_ratio:.2f}:1

═══════════════════════════════════════════════════════════════
🔬 STRATEGY DATA
═══════════════════════════════════════════════════════════════
• RSI(14): {fmt(rsi_val, 1)}   (thresholds: 40 for longs / 60 for shorts — moderate, not extreme)
• Bollinger Bands:
    upper {self._format_price(bb_upper)} / mid {self._format_price(bb_mid)} / lower {self._format_price(bb_lower)}
• Band width: {fmt(band_width, 1)} pips   (informational — band-width gate is disabled)
• HTF bias (1h EMA50 + slope): {htf_bias}   ← **primary confluence, must align**
• Prior-range context: high {self._format_price(range_high)} / low {self._format_price(range_low)}
• Distance to range low: {fmt(dist_low, 1)} pips (context only; not a required proximity)
• Distance to range high: {fmt(dist_high, 1)} pips (context only; not a required proximity)
{chart_instruction}
═══════════════════════════════════════════════════════════════
📋 REQUIRED RESPONSE FORMAT
═══════════════════════════════════════════════════════════════

Your response MUST be exactly three lines:

SCORE: [1-10]
DECISION: [APPROVE/REJECT]
REASON: [2-3 sentences focused on (a) whether the 1h HTF bias truly aligns with the entry direction, (b) whether the current 5m bar looks like a normal pullback to the band vs. a breakout expansion, and (c) whether any obvious regime change on the 1h invalidates the fade.]

Reject if: {reject_if}.
Approve only if you see {desired_confirmation}.
"""
        except Exception as e:
            self.logger.error(f"Error building RANGE_FADE prompt: {e}")
            return self._build_fallback_prompt(signal, {})

    def _extract_pair(self, epic: str) -> str:
        """Extract currency pair from epic string"""
        try:
            # Format: CS.D.EURUSD.MINI.IP -> EURUSD
            parts = epic.split('.')
            if len(parts) >= 3:
                return parts[2]
            return epic
        except Exception:
            return epic


# Factory function for easy integration (backwards compatible)
def create_prompt_builder() -> PromptBuilder:
    """Create a prompt builder instance (backwards compatible)"""
    return PromptBuilder()

# Legacy support
def create_advanced_prompt_builder() -> PromptBuilder:
    """Legacy function name support"""
    return PromptBuilder()
