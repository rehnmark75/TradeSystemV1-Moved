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
        Build a vision-enabled prompt for forex strategy analysis.

        Routes to strategy-specific prompt builders based on the signal's strategy.

        Args:
            signal: Signal dictionary with all trading data
            has_chart: Whether a chart image will be included

        Returns:
            Formatted prompt string for Claude vision analysis
        """
        strategy = signal.get('strategy', '').upper()

        # Route to strategy-specific prompt builders
        if 'EMA_DOUBLE' in strategy:
            return self._build_ema_double_prompt(signal, has_chart)
        elif 'SILVER_BULLET' in strategy:
            return self._build_silver_bullet_prompt(signal, has_chart)
        elif 'SMC' in strategy:
            return self._build_smc_prompt(signal, has_chart)
        else:
            return self._build_generic_vision_prompt(signal, has_chart)

    def _build_ema_double_prompt(self, signal: Dict, has_chart: bool = True) -> str:
        """
        Build a prompt specifically for EMA_DOUBLE_CONFIRMATION strategy.

        This strategy uses:
        - EMA 9/21 crossovers on 15m timeframe
        - 4H EMA 21 HTF trend filter
        - FVG (Fair Value Gap) confirmation
        - ADX trend strength filter
        - Prior successful crossover requirement
        """
        try:
            # Extract signal data
            epic = signal.get('epic', 'Unknown')
            pair = self._extract_pair(epic)
            direction = signal.get('signal_type', signal.get('signal', 'Unknown'))
            confidence = signal.get('confidence_score', 0)

            # Price levels
            entry_price = signal.get('entry_price', signal.get('price', 0))
            stop_loss = signal.get('stop_loss', 0)
            take_profit = signal.get('take_profit', 0)

            # Risk metrics
            risk_pips = signal.get('risk_pips', 0)
            reward_pips = signal.get('reward_pips', 0)
            rr_ratio = signal.get('rr_ratio', 0)

            # EMA Double Confirmation specific data
            ema_9 = signal.get('ema_9', signal.get('ema_fast', 0))
            ema_21 = signal.get('ema_21', signal.get('ema_slow', 0))
            ema_200 = signal.get('ema_200', signal.get('ema_trend', 0))
            adx = signal.get('adx', 0)

            # Strategy metadata
            metadata = signal.get('metadata', {})
            prior_crossovers = metadata.get('prior_crossovers', 0)
            crossover_validation = metadata.get('crossover_validation', 'Unknown')
            htf_aligned = metadata.get('htf_aligned', False)
            fvg_confirmed = metadata.get('fvg_confirmed', False)

            # Build chart analysis instructions
            chart_instruction = ""
            if has_chart:
                chart_instruction = """
## CHART ANALYSIS (EXAMINE EMA CROSSOVER SETUP)

The attached chart shows EMA crossover analysis with the following elements:

**Key Visual Markers:**
- EMA 9 (fast): Orange/Yellow line
- EMA 21 (slow): Blue line
- EMA 200 (trend): Purple dashed line
- GREEN dashed line: Entry price level
- RED dashed line: Stop loss level
- BLUE dashed line: Take profit target

**CRITICAL CHART ANALYSIS CHECKLIST:**
1. ✓ Is the EMA 9/21 crossover clearly visible and confirmed?
2. ✓ Is price above/below EMA 200 aligned with signal direction?
3. ✓ Is there a Fair Value Gap (price imbalance) supporting the move?
4. ✓ Does price action show momentum continuation after crossover?
5. ✓ Is the crossover clean (not whipsawing back and forth)?
6. ✓ Are there any concerning reversal patterns near entry?
7. ✓ Is ADX showing trending conditions (ADX > 20)?
"""

            # Build EMA Double Confirmation specific analysis
            ema_analysis = f"""
## EMA DOUBLE CONFIRMATION STRATEGY DATA

**CROSSOVER VALIDATION:**
- Prior Successful Crossovers: {prior_crossovers}
- Crossover Status: {crossover_validation}
- Current Crossover: EMA 9 {'>' if direction == 'BULL' else '<'} EMA 21

**EMA STRUCTURE (15m):**
- EMA 9 (fast): {self._format_price(ema_9)}
- EMA 21 (slow): {self._format_price(ema_21)}
- EMA 200 (trend): {self._format_price(ema_200)}
- EMA 9/21 Alignment: {'Bullish (9>21)' if float(ema_9 or 0) > float(ema_21 or 0) else 'Bearish (9<21)'}

**FILTER CONFIRMATIONS:**
- 4H EMA 21 HTF Filter: {'✅ ALIGNED' if htf_aligned else '❌ NOT ALIGNED'}
- FVG Confirmation: {'✅ CONFIRMED' if fvg_confirmed else '❌ NOT CONFIRMED'}
- ADX Trend Strength: {f'{adx:.1f}' if adx else 'N/A'} {'✅ (Trending)' if adx and float(adx) >= 20 else '⚠️ (Weak/Ranging)' if adx else ''}

**STRATEGY LOGIC:**
This signal was generated because:
1. A prior EMA 9/21 crossover was validated as successful (price held direction)
2. A new crossover in the SAME direction occurred (2nd crossover = entry)
3. 4H EMA 21 confirms higher timeframe trend alignment
4. Fair Value Gap confirms institutional momentum
5. ADX confirms trending market conditions (not ranging)
"""

            # Build the complete prompt
            prompt = f"""You are a SENIOR FOREX TECHNICAL ANALYST specializing in EMA crossover strategies with confirmation filters.

**YOUR ROLE:** Validate this EMA DOUBLE CONFIRMATION strategy signal. This strategy requires a successful prior crossover before entering on the second crossover, with multiple filter confirmations.

═══════════════════════════════════════════════════════════════
📊 SIGNAL OVERVIEW
═══════════════════════════════════════════════════════════════
• Pair: {pair}
• Direction: {direction}
• Strategy: EMA_DOUBLE_CONFIRMATION
• System Confidence: {confidence:.1%}

═══════════════════════════════════════════════════════════════
💰 TRADE LEVELS
═══════════════════════════════════════════════════════════════
• Entry Price: {self._format_price(entry_price)}
• Stop Loss: {self._format_price(stop_loss)} ({risk_pips:.1f} pips risk)
• Take Profit: {self._format_price(take_profit)} ({reward_pips:.1f} pips reward)
• Risk:Reward Ratio: {rr_ratio:.2f}:1
{chart_instruction}
{ema_analysis}
═══════════════════════════════════════════════════════════════
📋 REQUIRED RESPONSE FORMAT
═══════════════════════════════════════════════════════════════

Analyze the EMA crossover setup then respond with EXACTLY these three lines:

SCORE: [1-10]
DECISION: [APPROVE/REJECT]
REASON: [2-3 sentences explaining your assessment. Focus on: EMA alignment quality, crossover confirmation strength, filter validation, and trend momentum]

**SCORING GUIDELINES:**
- 8-10: Clean crossover, all filters confirmed, strong trend momentum
- 6-7: Valid crossover with minor concerns (e.g., ADX borderline)
- 4-5: Crossover visible but filter confirmations weak
- 1-3: False crossover signal, counter-trend, or multiple filter failures

**AUTOMATIC REJECTION CRITERIA:**
- EMA crossover not clearly confirmed (whipsaw risk)
- Price on wrong side of EMA 200 (counter-trend)
- ADX below 15 (ranging/choppy market)
- No FVG support (weak institutional backing)
- R:R ratio below 1.5
- HTF trend filter not aligned

Be concise but thorough. This strategy relies on CONFIRMATION - reject if confirmations are missing."""

            return prompt

        except Exception as e:
            self.logger.error(f"Error building EMA double prompt: {e}")
            return self._build_fallback_prompt(signal, {})

    def _build_silver_bullet_prompt(self, signal: Dict, has_chart: bool = True) -> str:
        """
        Build a prompt specifically for ICT Silver Bullet strategy analysis.

        The Silver Bullet strategy uses:
        - Time-based entry windows (London Open, NY AM, NY PM)
        - Liquidity sweeps (BSL/SSL) for setup trigger
        - Market Structure Shift (MSS) for direction confirmation
        - Fair Value Gap (FVG) for precise entry
        - HTF trend alignment filter
        """
        try:
            # Extract signal data
            epic = signal.get('epic', 'Unknown')
            pair = self._extract_pair(epic)
            direction = signal.get('signal_type', signal.get('signal', 'Unknown'))
            confidence = signal.get('confidence_score', 0)

            # Price levels
            entry_price = signal.get('entry_price', signal.get('price', 0))
            stop_loss = signal.get('stop_loss', 0)
            take_profit = signal.get('take_profit', 0)

            # Risk metrics
            risk_pips = signal.get('risk_pips', 0)
            reward_pips = signal.get('reward_pips', 0)
            rr_ratio = signal.get('rr_ratio', 0)

            # Silver Bullet specific data - from metadata
            metadata = signal.get('metadata', {})

            # Session information
            session = metadata.get('session', 'Unknown')
            session_quality = metadata.get('session_quality', 0)

            # Liquidity sweep data
            sweep_type = metadata.get('sweep_type', 'Unknown')  # BSL or SSL
            sweep_status = metadata.get('sweep_status', 'Unknown')  # CLEAN, PARTIAL, PENDING
            sweep_pips = metadata.get('sweep_pips', 0)
            sweep_age_bars = metadata.get('sweep_age_bars', 0)
            liquidity_level = metadata.get('liquidity_level', 0)
            rejection_confirmed = metadata.get('rejection_confirmed', False)

            # Market Structure Shift data
            mss_confirmed = metadata.get('mss_confirmed', False)
            mss_break_pips = metadata.get('mss_break_pips', 0)
            mss_direction = metadata.get('mss_direction', 'Unknown')

            # FVG data
            fvg_type = metadata.get('fvg_type', 'Unknown')  # BULLISH_FVG or BEARISH_FVG
            fvg_size_pips = metadata.get('fvg_size_pips', 0)
            fvg_fill_pct = metadata.get('fvg_fill_percentage', 0)
            entry_type = metadata.get('entry_type', 'Unknown')  # IMMEDIATE or APPROACHING

            # HTF alignment
            htf_direction = metadata.get('htf_direction', 'Unknown')
            htf_strength = metadata.get('htf_strength', 0)
            htf_aligned = metadata.get('htf_aligned', False)

            # Build chart analysis instructions
            chart_instruction = ""
            if has_chart:
                chart_instruction = """
## CHART ANALYSIS (ICT SILVER BULLET SETUP)

The attached chart shows the Silver Bullet setup with the following elements:

**Key Visual Markers:**
- GREEN dashed line: Entry price level (FVG entry zone)
- RED dashed line: Stop loss level (beyond sweep/FVG)
- BLUE dashed line: Take profit target (opposite liquidity)
- ORANGE horizontal lines: Liquidity levels (swing highs)
- BLUE horizontal lines: Liquidity levels (swing lows)
- YELLOW shaded zone: Fair Value Gap (FVG) entry zone

**CRITICAL CHART ANALYSIS CHECKLIST:**
1. ✓ Is the liquidity sweep clearly visible (price wicked beyond level)?
2. ✓ Did price reject from the sweep level (not a breakout)?
3. ✓ Is the Market Structure Shift (MSS) confirmed by a clear break?
4. ✓ Is there a valid FVG formed after the MSS?
5. ✓ Is the entry at an optimal FVG level (not chasing)?
6. ✓ Is stop loss beyond the sweep low/high for protection?
7. ✓ Is there clear path to take profit target (opposite liquidity)?
"""

            # Session display with quality
            session_display = {
                'NY_AM': 'NY AM Session (10:00-11:00 NY) - BEST SESSION',
                'NY_PM': 'NY PM Session (14:00-15:00 NY) - Good Session',
                'LONDON_OPEN': 'London Open (03:00-04:00 NY) - Good for EUR/GBP'
            }.get(session, session)

            # Sweep status display with quality indicators
            sweep_status_display = {
                'CLEAN': '✅ CLEAN (Confirmed reversal)',
                'PARTIAL': '⚠️ PARTIAL (Possible reversal)',
                'PENDING': '❌ PENDING (Unconfirmed - high risk)',
                'BREAKOUT': '❌ BREAKOUT (Not a sweep)'
            }.get(sweep_status, sweep_status)

            # Build Silver Bullet specific analysis
            silver_bullet_analysis = f"""
## ICT SILVER BULLET STRATEGY DATA

**SESSION TIMING:**
- Active Session: {session_display}
- Session Quality Score: {session_quality:.0%}

**LIQUIDITY SWEEP (Setup Trigger):**
- Sweep Type: {sweep_type} ({'Buy-Side Liquidity' if sweep_type == 'BSL' else 'Sell-Side Liquidity' if sweep_type == 'SSL' else 'Unknown'})
- Sweep Status: {sweep_status_display}
- Sweep Depth: {sweep_pips:.1f} pips beyond level
- Liquidity Level: {self._format_price(liquidity_level)}
- Sweep Age: {sweep_age_bars} bars ago
- Rejection Confirmed: {'✅ Yes' if rejection_confirmed else '❌ No'}

**MARKET STRUCTURE SHIFT (MSS):**
- MSS Confirmed: {'✅ Yes' if mss_confirmed else '❌ No'}
- MSS Direction: {mss_direction}
- MSS Break Strength: {mss_break_pips:.1f} pips

**FAIR VALUE GAP (Entry Zone):**
- FVG Type: {fvg_type}
- FVG Size: {fvg_size_pips:.1f} pips
- FVG Fill: {fvg_fill_pct:.0%}
- Entry Type: {entry_type} {'(Price in FVG now)' if entry_type == 'IMMEDIATE' else '(Price approaching FVG)'}

**HIGHER TIMEFRAME ALIGNMENT:**
- HTF Direction: {htf_direction}
- HTF Trend Strength: {htf_strength:.0%}
- HTF Aligned: {'✅ Yes - WITH trend' if htf_aligned else '⚠️ No - COUNTER trend'}

**SILVER BULLET STRATEGY LOGIC:**
This signal was generated because:
1. Current time is within a Silver Bullet window ({session})
2. Liquidity sweep detected ({sweep_type}) with {sweep_status} status
3. Market Structure Shift confirmed direction change
4. Fair Value Gap provides optimal entry zone
5. HTF trend {'supports' if htf_aligned else 'OPPOSES'} trade direction
"""

            # Build the complete prompt
            prompt = f"""You are a SENIOR FOREX TECHNICAL ANALYST specializing in ICT (Inner Circle Trader) Smart Money Concepts with expertise in the Silver Bullet time-based strategy.

**YOUR ROLE:** Validate this ICT Silver Bullet signal. This strategy targets specific one-hour windows where institutional order flow creates high-probability setups through liquidity sweeps followed by FVG entries.

═══════════════════════════════════════════════════════════════
📊 SIGNAL OVERVIEW
═══════════════════════════════════════════════════════════════
• Pair: {pair}
• Direction: {direction}
• Strategy: SILVER_BULLET
• System Confidence: {confidence:.1%}

═══════════════════════════════════════════════════════════════
💰 TRADE LEVELS
═══════════════════════════════════════════════════════════════
• Entry Price: {self._format_price(entry_price)}
• Stop Loss: {self._format_price(stop_loss)} ({risk_pips:.1f} pips risk)
• Take Profit: {self._format_price(take_profit)} ({reward_pips:.1f} pips reward)
• Risk:Reward Ratio: {rr_ratio:.2f}:1
{chart_instruction}
{silver_bullet_analysis}
═══════════════════════════════════════════════════════════════
📋 REQUIRED RESPONSE FORMAT
═══════════════════════════════════════════════════════════════

Analyze the Silver Bullet setup then respond with EXACTLY these three lines:

SCORE: [1-10]
DECISION: [APPROVE/REJECT]
REASON: [2-3 sentences explaining your assessment. Focus on: sweep quality, MSS confirmation, FVG entry timing, and HTF alignment]

**SCORING GUIDELINES:**
- 8-10: CLEAN sweep with confirmed rejection, strong MSS, fresh FVG, HTF aligned
- 6-7: Valid setup with minor concerns (e.g., PARTIAL sweep, older FVG)
- 4-5: Setup present but quality issues (weak MSS, unfavorable session)
- 1-3: Setup failure (PENDING sweep, no MSS, counter-trend without confirmation)

**AUTOMATIC REJECTION CRITERIA:**
- PENDING or BREAKOUT sweep status (unconfirmed reversal)
- No clear Market Structure Shift after sweep
- FVG too small (<1 pip) or too filled (>90%)
- Counter-trend against strong HTF trend (>70% strength)
- R:R ratio below 1.5
- Outside valid Silver Bullet time window
- Sweep too old (>40 bars) - setup expired

**SILVER BULLET BEST PRACTICES:**
- NY AM session (10:00-11:00 NY) has highest win rate
- CLEAN sweeps with rejection are most reliable
- FVG should be entered at optimal edge (not middle)
- Stop loss should be beyond the sweep level for protection

Be concise but thorough. The Silver Bullet strategy is TIME-SENSITIVE - quality setups in the right window are key."""

            return prompt

        except Exception as e:
            self.logger.error(f"Error building Silver Bullet prompt: {e}")
            return self._build_fallback_prompt(signal, {})

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

            # Tier 2 - Swing Break data
            tier2_swing = strategy_indicators.get('tier2_swing', {})
            swing_level = tier2_swing.get('swing_level', signal.get('swing_level', 0))
            body_close_confirmed = tier2_swing.get('body_close_confirmed', True)
            volume_confirmed = tier2_swing.get('volume_confirmed', signal.get('volume_confirmed', False))

            # Tier 3 - Entry data
            tier3_entry = strategy_indicators.get('tier3_entry', {})
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

            # Check if fixed SL/TP override is enabled
            fixed_sl_tp_enabled = False
            fixed_sl_note = ""
            try:
                from config import (
                    FIXED_SL_TP_OVERRIDE_ENABLED,
                    FIXED_STOP_LOSS_PIPS,
                    FIXED_TAKE_PROFIT_PIPS
                )
                if FIXED_SL_TP_OVERRIDE_ENABLED:
                    fixed_sl_tp_enabled = True
                    # Override with fixed values for display
                    risk_pips = FIXED_STOP_LOSS_PIPS
                    reward_pips = FIXED_TAKE_PROFIT_PIPS
                    rr_ratio = reward_pips / risk_pips if risk_pips > 0 else 0
                    fixed_sl_note = f"\n⚠️ **FIXED SL/TP MODE ACTIVE**: All trades use SL={FIXED_STOP_LOSS_PIPS} pips, TP={FIXED_TAKE_PROFIT_PIPS} pips (R:R={rr_ratio:.2f}:1) regardless of strategy calculation."
            except ImportError:
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

                chart_instruction = f"""
## CHART ANALYSIS (CRITICAL - EXAMINE CAREFULLY)

The attached chart shows multi-timeframe forex analysis with the following elements:

**Timeframes Displayed:**
- 4H timeframe: Shows 50 EMA trend bias (purple line)
- 15m timeframe: PRIMARY ANALYSIS - Shows swing break, EMAs 9/21, S/R levels, entry/SL/TP
- 5m timeframe: Shows entry zone with Fibonacci levels and entry type annotation

**Key Visual Markers (on 15m chart - PRIMARY):**
- GREEN dashed line: Entry price level
- RED dashed line: Stop loss level (below opposite swing)
- BLUE dashed line: Take profit target
- ORANGE line: EMA 9 (fast momentum)
- BLUE line: EMA 21 (trend confirmation)
- GREEN horizontal line: Support level with distance in pips
- RED horizontal line: Resistance level with distance in pips
- ORANGE horizontal lines: Swing high levels
- BLUE horizontal lines: Swing low levels

**Key Visual Markers (on 5m chart):**
- YELLOW shaded zone: Fibonacci optimal entry zone (38.2%-61.8%)
- Entry Type Box (top-right): Shows PULLBACK/MOMENTUM, depth %, zone status, volume ✓/✗
{momentum_note}
**CRITICAL CHART ANALYSIS CHECKLIST:**
1. ✓ Is price clearly respecting the 4H EMA trend direction?
2. ✓ Is the swing break on 15m clean and confirmed (full candle close)?
3. ✓ Are EMA 9/21 aligned with the trade direction on 15m chart?
4. ✓ Is entry clear of nearby S/R obstacles shown on 15m?
5. ✓ For PULLBACK: Is entry within or near the optimal Fibonacci zone (5m)?
6. ✓ For MOMENTUM: Is breakout clean with strong directional candles?
7. ✓ Is stop loss placement below a valid structure low (for longs)?
8. ✓ Does the price action show clean trend structure?
9. ✓ Are there any concerning patterns (engulfing candles, dojis at entry)?
10. ✓ Does the entry type box (5m) show favorable conditions?
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

                sr_context = f"""
**SUPPORT/RESISTANCE CONTEXT:**
- Nearest Support: {support_str}
- Nearest Resistance: {resistance_str}
- Path to Target: {path_note}
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
- 5m Trend: {ema_alignment} {'✅ Aligned' if ema_aligned_with_signal else '⚠️ Conflict'}
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
- Volume Confirmed: {'✅ Yes' if volume_confirmed else '❌ No'}

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
{smc_analysis}
═══════════════════════════════════════════════════════════════
📋 REQUIRED RESPONSE FORMAT
═══════════════════════════════════════════════════════════════

Analyze the signal (and chart if provided) then respond with EXACTLY these three lines:

SCORE: [1-10]
DECISION: [APPROVE/REJECT]
REASON: [2-3 sentences explaining your professional assessment. Focus on: trend alignment, entry quality, R:R ratio, and any visual concerns from the chart]

**SCORING GUIDELINES FOR v2.8.1:**
- 8-10: Strong trend alignment, clean swing break on 15m, volume confirmed, EMA 9/21 aligned on 15m
- 6-7: Good setup with minor concerns (e.g., momentum slightly extended, volume not confirmed, S/R nearby but manageable)
- 4-5: Marginal setup - weak trend, entry quality issues, or EMA micro-structure conflict on 15m
- 1-3: Poor setup - counter-trend, S/R blocking target, or technical breakdown
NOTE: R:R is fixed at 1.67:1 (9 pip SL / 15 pip TP) - do not penalize for R:R.

**ENTRY TYPE EVALUATION:**
- PULLBACK entries: Prefer entries in 38.2%-61.8% Fib zone (check 5m chart). Outside zone = lower score but not automatic rejection
- MOMENTUM entries: Accept up to 50% beyond break point. Look for strong directional candles on 15m, reject if showing exhaustion
- Check entry type box on 5m chart for quick visual confirmation

**SUPPORT/RESISTANCE EVALUATION (check 15m chart):**
- Check if S/R levels shown on 15m chart obstruct the path to take profit
- For BULL: Resistance should be BEYOND take profit level
- For BEAR: Support should be BEYOND take profit level
- S/R within 50% of target distance = caution, within 25% = strong concern

**AUTOMATIC REJECTION CRITERIA:**
- Counter-trend trades (price on wrong side of 4H EMA)
- MOMENTUM entry showing reversal candles on 15m (engulfing, pin bars against direction)
- Price too close to 4H EMA (<2.5 pips) - buffer zone violation
- S/R level on 15m blocking more than 75% of path to target (15 pips)
- EMA 9/21 crossed against signal direction on 15m

Be concise but thorough. Your assessment determines if real money is risked."""

            return prompt

        except Exception as e:
            self.logger.error(f"Error building vision prompt: {e}")
            return self._build_fallback_prompt(signal, {})

    def _build_generic_vision_prompt(self, signal: Dict, has_chart: bool = True) -> str:
        """
        Build a generic vision prompt for strategies without specific prompt builders.

        This is the fallback for strategies like EMA, MACD, BB_SUPERTREND, etc.
        that don't have specialized prompt builders.
        """
        try:
            # Extract signal data
            epic = signal.get('epic', 'Unknown')
            pair = self._extract_pair(epic)
            direction = signal.get('signal_type', signal.get('signal', 'Unknown'))
            confidence = signal.get('confidence_score', 0)
            strategy = signal.get('strategy', 'Unknown')

            # Price levels
            entry_price = signal.get('entry_price', signal.get('price', 0))
            stop_loss = signal.get('stop_loss', 0)
            take_profit = signal.get('take_profit', 0)

            # Risk metrics
            risk_pips = signal.get('risk_pips', 0)
            reward_pips = signal.get('reward_pips', 0)
            rr_ratio = signal.get('rr_ratio', 0)

            # Technical indicators
            ema_9 = signal.get('ema_9', signal.get('ema_short', 0))
            ema_21 = signal.get('ema_21', signal.get('ema_long', 0))
            ema_200 = signal.get('ema_200', signal.get('ema_trend', 0))
            rsi = signal.get('rsi', 0)
            macd_hist = signal.get('macd_histogram', 0)
            atr = signal.get('atr', 0)

            # Build chart analysis instructions
            chart_instruction = ""
            if has_chart:
                chart_instruction = """
## CHART ANALYSIS

The attached chart shows the trading setup with the following elements:

**Key Visual Markers:**
- GREEN dashed line: Entry price level
- RED dashed line: Stop loss level
- BLUE dashed line: Take profit target
- Moving averages and technical indicators as applicable

**CHART ANALYSIS CHECKLIST:**
1. ✓ Is the trend direction clear and aligned with signal?
2. ✓ Is entry at a favorable price level (not chasing)?
3. ✓ Is stop loss placement at a logical structure level?
4. ✓ Does price action support the signal direction?
5. ✓ Are there any concerning reversal patterns near entry?
6. ✓ Is there sufficient room to target without major obstacles?
"""

            # Build technical analysis section
            technical_analysis = f"""
## TECHNICAL INDICATORS

**TREND ANALYSIS:**
- EMA 9: {self._format_price(ema_9)}
- EMA 21: {self._format_price(ema_21)}
- EMA 200: {self._format_price(ema_200)}
- Trend Alignment: {'Bullish' if float(ema_9 or 0) > float(ema_21 or 0) > float(ema_200 or 0) else 'Bearish' if float(ema_9 or 0) < float(ema_21 or 0) < float(ema_200 or 0) else 'Mixed'}

**MOMENTUM:**
- RSI: {rsi:.1f if rsi else 'N/A'} {'(Overbought)' if rsi and float(rsi) > 70 else '(Oversold)' if rsi and float(rsi) < 30 else ''}
- MACD Histogram: {macd_hist:.6f if macd_hist else 'N/A'}

**VOLATILITY:**
- ATR: {atr:.5f if atr else 'N/A'}
"""

            # Build the complete prompt
            prompt = f"""You are a SENIOR FOREX TECHNICAL ANALYST with institutional trading experience.

**YOUR ROLE:** Validate this forex trade signal with professional judgment. Your analysis will impact real trading decisions.

═══════════════════════════════════════════════════════════════
📊 SIGNAL OVERVIEW
═══════════════════════════════════════════════════════════════
• Pair: {pair}
• Direction: {direction}
• Strategy: {strategy}
• System Confidence: {confidence:.1%}

═══════════════════════════════════════════════════════════════
💰 TRADE LEVELS
═══════════════════════════════════════════════════════════════
• Entry Price: {self._format_price(entry_price)}
• Stop Loss: {self._format_price(stop_loss)} ({risk_pips:.1f} pips risk)
• Take Profit: {self._format_price(take_profit)} ({reward_pips:.1f} pips reward)
• Risk:Reward Ratio: {rr_ratio:.2f}:1
{chart_instruction}
{technical_analysis}
═══════════════════════════════════════════════════════════════
📋 REQUIRED RESPONSE FORMAT
═══════════════════════════════════════════════════════════════

Analyze the signal (and chart if provided) then respond with EXACTLY these three lines:

SCORE: [1-10]
DECISION: [APPROVE/REJECT]
REASON: [2-3 sentences explaining your professional assessment. Focus on: trend alignment, entry quality, R:R ratio, and any concerns]

**SCORING GUIDELINES:**
- 8-10: Strong setup, clear trend, optimal entry, good R:R
- 6-7: Acceptable setup with minor concerns
- 4-5: Marginal setup, consider passing
- 1-3: Poor setup, clear rejection reasons

**AUTOMATIC REJECTION CRITERIA:**
- Counter-trend trades without strong reversal confirmation
- R:R ratio below 1.5
- RSI extreme (>80 for longs, <20 for shorts)
- Price far from key EMAs (overextended)
- Obvious resistance/support blocking target

Be concise but thorough. Your assessment determines if real money is risked."""

            return prompt

        except Exception as e:
            self.logger.error(f"Error building generic vision prompt: {e}")
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