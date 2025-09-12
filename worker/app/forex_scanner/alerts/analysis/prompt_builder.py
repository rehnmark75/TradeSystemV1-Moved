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


# Factory function for easy integration (backwards compatible)
def create_prompt_builder() -> PromptBuilder:
    """Create a prompt builder instance (backwards compatible)"""
    return PromptBuilder()

# Legacy support
def create_advanced_prompt_builder() -> PromptBuilder:
    """Legacy function name support"""
    return PromptBuilder()