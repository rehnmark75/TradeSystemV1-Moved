# services/market_intelligence_service.py
"""
Market Intelligence Service for Streamlit
Provides market regime analysis and intelligence for the TradingView chart
"""

import sys
import os
import logging
import time
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd

# Add worker path to import MarketIntelligenceEngine
worker_path = os.path.join(os.path.dirname(__file__), '..', '..', 'worker', 'app')
if worker_path not in sys.path:
    sys.path.append(worker_path)

from services.data_fetcher_adapter import StreamlitDataFetcher


class MarketIntelligenceService:
    """Service for market intelligence analysis in Streamlit context"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.data_fetcher = None
        self.intelligence_engine = None
        self.cache = {}
        self.cache_duration = 300  # 5 minutes
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize intelligence components"""
        try:
            # Initialize data fetcher (engine will be passed when needed)
            self.data_fetcher = StreamlitDataFetcher()
            
            # Try to import and initialize MarketIntelligenceEngine
            try:
                from forex_scanner.core.intelligence.market_intelligence import MarketIntelligenceEngine
                self.intelligence_engine = MarketIntelligenceEngine(self.data_fetcher)
                self.logger.info("🧠 MarketIntelligenceEngine initialized successfully")
                
            except ImportError as e:
                self.logger.info(f"📊 Using simplified intelligence engine (forex_scanner not available)")
                self.intelligence_engine = None
                
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize intelligence components: {e}")
            self.intelligence_engine = None
    
    def is_available(self) -> bool:
        """Check if intelligence service is available"""
        # Return True if either full engine or simplified analysis is available
        return self.intelligence_engine is not None or self.data_fetcher is not None
    
    def _get_simplified_regime_analysis(self, epic: str, timeframe: str = '15m', lookback_hours: int = 24) -> Dict:
        """Simplified regime analysis using basic technical indicators"""
        try:
            if not self.data_fetcher or not self.data_fetcher.engine:
                return self._get_fallback_regime()
            
            df = self.data_fetcher.get_enhanced_data(epic, epic, timeframe, lookback_hours)
            if df is None or len(df) < 50:
                return self._get_fallback_regime()
            
            # Simple trend analysis using EMAs
            current_price = df['close'].iloc[-1]
            ema_21 = df.get('ema_21', df['close'].ewm(span=21).mean()).iloc[-1]
            ema_50 = df.get('ema_50', df['close'].ewm(span=50).mean()).iloc[-1]
            
            # Calculate volatility and percentile
            if 'atr' in df.columns:
                volatility = df['atr'].iloc[-1]
                volatility_series = df['atr'].dropna()
            else:
                volatility_series = df['close'].rolling(14).std().dropna()
                volatility = volatility_series.iloc[-1] if len(volatility_series) > 0 else 0
            
            avg_volatility = volatility_series.mean() if len(volatility_series) > 0 else volatility
            
            # Calculate volatility percentile
            if len(volatility_series) > 0:
                volatility_percentile = (volatility_series <= volatility).mean() * 100
            else:
                volatility_percentile = 50
            
            # Determine regime
            if current_price > ema_21 > ema_50:
                regime = "trending_up"
                confidence = 0.7
            elif current_price < ema_21 < ema_50:
                regime = "trending_down" 
                confidence = 0.7
            elif volatility > avg_volatility * 1.5:
                regime = "breakout"
                confidence = 0.6
            else:
                regime = "ranging"
                confidence = 0.6
            
            return {
                'regime': regime,
                'confidence': confidence,
                'description': f"Market appears to be {regime.replace('_', ' ')}",
                'volatility_level': 'high' if volatility > avg_volatility * 1.2 else 'medium',
                'volatility_percentile': volatility_percentile,
                'trend_strength': 'strong' if abs(current_price - ema_50) / ema_50 > 0.01 else 'weak',
                'analyzed_at': datetime.now().isoformat(),
                'data_points': len(df)
            }
            
        except Exception as e:
            self.logger.error(f"Error in simplified regime analysis: {e}")
            return self._get_fallback_regime()
    
    def _get_simplified_session_analysis(self) -> Dict:
        """Simplified session analysis based on UTC time"""
        try:
            import pytz
            utc_now = datetime.now(pytz.UTC)
            hour = utc_now.hour
            
            # Define session times (UTC)
            sessions = {
                'asian': {'start': 22, 'end': 8, 'name': 'Asian', 'active': False},
                'london': {'start': 8, 'end': 16, 'name': 'London', 'active': False}, 
                'new_york': {'start': 13, 'end': 22, 'name': 'New York', 'active': False}
            }
            
            # Check which sessions are active
            for session_key, session in sessions.items():
                if session['start'] <= session['end']:
                    # Normal case: start < end (within same day)
                    session['active'] = session['start'] <= hour < session['end']
                else:
                    # Crosses midnight (like Asian session)
                    session['active'] = hour >= session['start'] or hour < session['end']
            
            # Determine primary session and overlaps
            active_sessions = [s['name'] for s in sessions.values() if s['active']]
            
            if len(active_sessions) == 0:
                primary_session = "Off Hours"
                overlap_sessions = []
            elif len(active_sessions) == 1:
                primary_session = active_sessions[0]
                overlap_sessions = []
            else:
                primary_session = "/".join(active_sessions)
                overlap_sessions = active_sessions
            
            # Session characteristics
            characteristics = {
                'Asian': ['Lower volatility', 'JPY pairs active', 'Range-bound trading'],
                'London': ['High volatility', 'EUR/GBP pairs active', 'Trend following'],
                'New York': ['High volume', 'USD pairs active', 'Momentum trading'],
                'London/New York': ['Peak volatility', 'All majors active', 'Breakout opportunities']
            }
            
            session_chars = characteristics.get(primary_session, ['Market preparation', 'Low activity'])
            
            return {
                'current_session': primary_session,
                'active_sessions': active_sessions,
                'overlap_sessions': overlap_sessions,
                'session_characteristics': session_chars,
                'utc_time': utc_now.strftime('%H:%M UTC'),
                'market_hours': len(active_sessions) > 0,
                'analyzed_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error in simplified session analysis: {e}")
            return self._get_fallback_session()
    
    def _get_simplified_trade_context(self, trade_data: Dict, epic: str) -> Dict:
        """Simplified trade context analysis"""
        try:
            # Get basic regime and session info
            regime_info = self._get_simplified_regime_analysis(epic)
            session_info = self._get_simplified_session_analysis()
            
            # Extract trade details
            direction = trade_data.get('direction', 'BUY')
            pnl = trade_data.get('pnl', 0) if trade_data.get('pnl') is not None else trade_data.get('profit_loss', 0)
            entry_time = trade_data.get('entry_time', trade_data.get('timestamp', datetime.now()))
            entry_price = trade_data.get('entry_price', 0)
            strategy = trade_data.get('strategy', 'Unknown')
            
            # Simple trade assessment
            trade_outcome = 'profitable' if pnl > 0 else 'loss' if pnl < 0 else 'break_even'
            
            # Regime suitability assessment
            regime = regime_info.get('regime', 'unknown')
            regime_confidence = regime_info.get('confidence', 0.5)
            regime_suitable = False
            regime_score = 0.3  # Base score
            
            if direction == 'BUY' and 'trending_up' in regime:
                regime_suitable = True
                regime_score = 0.8 * regime_confidence
            elif direction == 'SELL' and 'trending_down' in regime:
                regime_suitable = True
                regime_score = 0.8 * regime_confidence
            elif 'breakout' in regime:
                regime_suitable = True
                regime_score = 0.7 * regime_confidence
            elif 'ranging' in regime:
                regime_score = 0.4  # Ranging is neutral
                
            # Session suitability
            session_suitable = session_info.get('market_hours', False)
            active_sessions = session_info.get('active_sessions', [])
            session_score = 0.3  # Base score
            
            if 'London' in active_sessions and 'New York' in active_sessions:
                session_score = 0.9  # Best overlap
            elif 'London' in active_sessions or 'New York' in active_sessions:
                session_score = 0.7  # Good single session
            elif 'Asian' in active_sessions:
                session_score = 0.5  # Moderate session
            else:
                session_score = 0.2  # Off hours
            
            # Calculate overall quality score
            overall_quality = (regime_score * 0.6 + session_score * 0.4)
            
            # Generate success factors and suggestions
            success_factors = []
            improvement_suggestions = []
            
            if regime_suitable:
                success_factors.append(f"Trade aligned with {regime.replace('_', ' ')} market")
            if session_suitable:
                success_factors.append(f"Executed during active {session_info.get('current_session', 'market')} session")
            if pnl > 0:
                success_factors.append(f"Trade was profitable (+{pnl:.2f})")
            if regime_confidence > 0.6:
                success_factors.append(f"High confidence regime ({regime_confidence:.0%})")
                
            if not regime_suitable:
                improvement_suggestions.append(f"Wait for better regime alignment for {direction} trades")
            if not session_suitable:
                improvement_suggestions.append("Consider trading during major sessions for better liquidity")
            if pnl < 0:
                improvement_suggestions.append("Review entry timing and stop loss placement")
            if regime_score < 0.5:
                improvement_suggestions.append("Look for stronger trend confirmation before entry")
            
            # Determine alignment status
            if regime_suitable and session_suitable:
                regime_alignment_status = "excellent"
            elif regime_suitable or session_suitable:
                regime_alignment_status = "moderate"
            else:
                regime_alignment_status = "poor"
            
            # Format for UI compatibility
            return {
                # Primary fields expected by UI
                'success': pnl > 0,
                'dominant_regime': regime,
                'trade_quality_score': overall_quality,
                'current_session': session_info.get('current_session', 'Unknown'),
                'volatility_percentile': regime_info.get('volatility_percentile', 50),
                'volume_relative': 1.0,  # Default normal volume
                'entry_price': entry_price,
                
                # Alignment info with scores
                'regime_alignment': {
                    'suitable': regime_suitable,
                    'alignment': regime_alignment_status,
                    'score': regime_score,
                    'reason': f"{'Good' if regime_suitable else 'Poor'} regime for {direction}"
                },
                'session_alignment': {
                    'suitable': session_suitable,
                    'alignment': 'active' if session_suitable else 'inactive',
                    'score': session_score,
                    'reason': f"{'Active' if session_suitable else 'Inactive'} market hours"
                },
                
                # Success factors and suggestions
                'success_factors': success_factors if success_factors else ["Trade executed successfully"],
                'improvement_suggestions': improvement_suggestions if improvement_suggestions else ["Continue monitoring market conditions"],
                
                # Additional context
                'trade_summary': {
                    'direction': direction,
                    'outcome': trade_outcome,
                    'pnl': pnl,
                    'entry_time': str(entry_time)
                },
                'market_context': {
                    'regime': regime_info,
                    'session': session_info,
                    'regime_suitable': regime_suitable,
                    'session_suitable': session_suitable
                },
                'analysis': {
                    'overall_score': 0.7 if regime_suitable and session_suitable else 0.5,
                    'notes': [
                        f"Trade direction ({direction}) {'aligned' if regime_suitable else 'not aligned'} with market regime ({regime})",
                        f"Trade occurred during {'active' if session_suitable else 'inactive'} market hours",
                        f"Trade resulted in {trade_outcome}"
                    ]
                },
                'analyzed_at': datetime.now().isoformat(),
                'analysis_type': 'simplified'
            }
            
        except Exception as e:
            self.logger.error(f"Error in simplified trade context: {e}")
            return self._get_fallback_trade_context()
    
    def _get_simplified_report(self, epic_list: List[str], lookback_hours: int = 24) -> Dict:
        """Generate simplified market intelligence report"""
        try:
            regimes = {}
            for epic in epic_list[:5]:  # Limit to 5 pairs for performance
                regime_info = self._get_simplified_regime_analysis(epic, '15m', lookback_hours)
                regimes[epic] = regime_info
            
            session_info = self._get_simplified_session_analysis()
            
            # Analyze overall market state
            trending_up = sum(1 for r in regimes.values() if 'trending_up' in r.get('regime', ''))
            trending_down = sum(1 for r in regimes.values() if 'trending_down' in r.get('regime', ''))
            ranging = sum(1 for r in regimes.values() if 'ranging' in r.get('regime', ''))
            
            if trending_up > trending_down and trending_up > ranging:
                overall_bias = 'bullish'
            elif trending_down > trending_up and trending_down > ranging:
                overall_bias = 'bearish'
            else:
                overall_bias = 'neutral'
            
            return {
                'overall_market_state': {
                    'bias': overall_bias,
                    'confidence': 0.6,
                    'trending_pairs': trending_up + trending_down,
                    'ranging_pairs': ranging
                },
                'session_analysis': session_info,
                'pair_regimes': regimes,
                'recommendations': [
                    f"Market bias is {overall_bias}",
                    f"{session_info.get('current_session', 'Unknown')} session active" if session_info.get('market_hours') else "Markets closed",
                    f"{trending_up + trending_down} pairs trending, {ranging} pairs ranging"
                ],
                'generated_at': datetime.now().isoformat(),
                'analysis_type': 'simplified'
            }
            
        except Exception as e:
            self.logger.error(f"Error in simplified report: {e}")
            return self._get_fallback_report()
    
    def get_market_intelligence_report(self, epic_list: List[str], 
                                     lookback_hours: int = 24) -> Optional[Dict]:
        """
        Generate comprehensive market intelligence report
        
        Args:
            epic_list: List of trading epics to analyze
            lookback_hours: Hours of data to analyze
            
        Returns:
            Intelligence report dict or None if unavailable
        """
        # Use simplified report if full engine not available
        if self.intelligence_engine is None:
            return self._get_simplified_report(epic_list, lookback_hours)
        
        try:
            # Check cache first
            cache_key = f"intelligence_{hash(str(sorted(epic_list)))}_{lookback_hours}"
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                return cached_result
            
            self.logger.info(f"🧠 Generating intelligence report for {len(epic_list)} pairs...")
            start_time = time.time()
            
            # Generate intelligence report
            report = self.intelligence_engine.generate_market_intelligence_report(epic_list)
            
            # Add metadata
            report['generated_at'] = datetime.now().isoformat()
            report['analysis_duration'] = time.time() - start_time
            report['epic_count'] = len(epic_list)
            report['lookback_hours'] = lookback_hours
            
            # Cache the result
            self._cache_result(cache_key, report)
            
            self.logger.info(f"✅ Intelligence report generated in {report['analysis_duration']:.2f}s")
            return report
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate intelligence report: {e}")
            return self._get_fallback_report()
    
    def get_regime_for_timeframe(self, epic: str, timeframe: str = '15m', 
                                lookback_hours: int = 24) -> Dict:
        """
        Get market regime analysis for specific timeframe
        
        Args:
            epic: Trading epic
            timeframe: Timeframe to analyze
            lookback_hours: Hours of data to analyze
            
        Returns:
            Regime analysis dict
        """
        # Use simplified analysis if full engine not available
        if self.intelligence_engine is None:
            return self._get_simplified_regime_analysis(epic, timeframe, lookback_hours)
        
        try:
            cache_key = f"regime_{epic}_{timeframe}_{lookback_hours}"
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                return cached_result
            
            # Analyze single pair regime
            regime = self.intelligence_engine._analyze_single_pair_regime(epic, lookback_hours)
            
            # Add metadata
            regime['epic'] = epic
            regime['timeframe'] = timeframe
            regime['analyzed_at'] = datetime.now().isoformat()
            
            self._cache_result(cache_key, regime)
            return regime
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get regime for {epic}: {e}")
            return self._get_fallback_regime()
    
    def get_session_analysis(self) -> Dict:
        """Get current trading session analysis"""
        # Use simplified analysis if full engine not available
        if self.intelligence_engine is None:
            return self._get_simplified_session_analysis()
        
        try:
            cache_key = "session_analysis"
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                return cached_result
            
            session_analysis = self.intelligence_engine.get_session_analysis()
            session_analysis['analyzed_at'] = datetime.now().isoformat()
            
            self._cache_result(cache_key, session_analysis)
            return session_analysis
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get session analysis: {e}")
            return self._get_fallback_session()
    
    def analyze_trade_context(self, trade_data: Dict, epic: str) -> Dict:
        """
        Analyze market context around a specific trade
        
        Args:
            trade_data: Trade information (entry_time, direction, etc.)
            epic: Trading epic
            
        Returns:
            Trade context analysis
        """
        try:
            # Use simplified analysis if full engine not available
            if self.intelligence_engine is None:
                return self._get_simplified_trade_context(trade_data, epic)
            
            entry_time = pd.to_datetime(trade_data.get('entry_time'))
            direction = trade_data.get('direction', 'BUY')
            pnl = trade_data.get('pnl', 0)
            
            # Get market data around trade time
            lookback_hours = 12
            df = self.data_fetcher.get_enhanced_data(
                epic, 
                epic.split('.')[-3] if '.' in epic else epic,
                timeframe='15m',
                lookback_hours=lookback_hours
            )
            
            if df is None or len(df) < 10:
                return self._get_fallback_trade_context()
            
            # Find closest candle to trade entry
            trade_index = df.index.get_loc(entry_time, method='nearest')
            trade_candle = df.iloc[trade_index]
            
            # Get regime at trade time
            regime_analysis = self.get_regime_for_timeframe(epic, '15m', lookback_hours)
            session_analysis = self.get_session_analysis()
            
            # Analyze trade context
            context = {
                'trade_time': entry_time.isoformat(),
                'direction': direction,
                'pnl': pnl,
                'success': pnl > 0,
                
                # Market conditions at trade
                'market_regime': regime_analysis.get('regime_scores', {}),
                'dominant_regime': max(regime_analysis.get('regime_scores', {}).items(), 
                                     key=lambda x: x[1])[0] if regime_analysis.get('regime_scores') else 'unknown',
                'volatility_percentile': regime_analysis.get('volatility_percentile', 50),
                'current_session': session_analysis.get('current_session', 'unknown'),
                
                # Price context
                'entry_price': float(trade_candle['close']),
                'atr': float(trade_candle.get('atr', 0)) if 'atr' in trade_candle else 0,
                'volume_relative': float(trade_candle.get('ltv', 0)) / df['ltv'].mean() if 'ltv' in df.columns else 1.0,
                
                # Support/Resistance context
                'sr_context': regime_analysis.get('support_resistance', {}),
                
                # Analysis
                'regime_alignment': self._assess_regime_alignment(direction, regime_analysis),
                'session_alignment': self._assess_session_alignment(direction, session_analysis),
                'trade_quality_score': self._calculate_trade_quality_score(direction, regime_analysis, session_analysis),
                'success_factors': self._identify_success_factors(pnl > 0, direction, regime_analysis, session_analysis),
                'improvement_suggestions': self._get_improvement_suggestions(pnl > 0, direction, regime_analysis, session_analysis)
            }
            
            return context
            
        except Exception as e:
            self.logger.error(f"❌ Failed to analyze trade context: {e}")
            return self._get_fallback_trade_context()
    
    def _assess_regime_alignment(self, direction: str, regime: Dict) -> Dict:
        """Assess how well trade direction aligns with market regime"""
        try:
            regime_scores = regime.get('regime_scores', {})
            
            if direction == 'BUY':
                # For BUY trades, trending up is good, ranging is neutral
                trending_score = regime_scores.get('trending', 0.5)
                breakout_score = regime_scores.get('breakout', 0.3)
                alignment_score = (trending_score + breakout_score) / 2
            else:  # SELL
                # For SELL trades, trending down or reversal is good
                trending_score = regime_scores.get('trending', 0.5) 
                reversal_score = regime_scores.get('reversal', 0.3)
                alignment_score = (trending_score + reversal_score) / 2
            
            if alignment_score > 0.7:
                alignment = 'excellent'
            elif alignment_score > 0.5:
                alignment = 'good'
            elif alignment_score > 0.3:
                alignment = 'fair'
            else:
                alignment = 'poor'
            
            return {
                'alignment': alignment,
                'score': alignment_score,
                'explanation': f"Trade direction {direction} has {alignment} alignment with market regime"
            }
            
        except:
            return {'alignment': 'unknown', 'score': 0.5, 'explanation': 'Could not assess regime alignment'}
    
    def _assess_session_alignment(self, direction: str, session: Dict) -> Dict:
        """Assess trade alignment with trading session"""
        try:
            current_session = session.get('current_session', 'unknown')
            session_config = session.get('session_config', {})
            volatility = session_config.get('volatility', 'medium')
            
            # Different sessions favor different trade types
            session_scores = {
                'asian': 0.3,    # Lower volatility, range-bound
                'london': 0.8,   # High volatility, trend potential  
                'new_york': 0.7, # High volume, momentum
                'overlap': 0.9   # Highest volatility and volume
            }
            
            score = session_scores.get(current_session, 0.5)
            
            if score > 0.8:
                alignment = 'excellent'
            elif score > 0.6:
                alignment = 'good'
            elif score > 0.4:
                alignment = 'fair'
            else:
                alignment = 'poor'
            
            return {
                'alignment': alignment,
                'score': score,
                'session': current_session,
                'explanation': f"{current_session.title()} session is {alignment} for active trading"
            }
            
        except:
            return {'alignment': 'unknown', 'score': 0.5, 'explanation': 'Could not assess session alignment'}
    
    def _calculate_trade_quality_score(self, direction: str, regime: Dict, session: Dict) -> float:
        """Calculate overall trade quality score (0-1)"""
        try:
            regime_alignment = self._assess_regime_alignment(direction, regime)
            session_alignment = self._assess_session_alignment(direction, session)
            
            # Weight regime more heavily than session
            quality_score = (regime_alignment['score'] * 0.7) + (session_alignment['score'] * 0.3)
            return min(1.0, max(0.0, quality_score))
            
        except:
            return 0.5
    
    def _identify_success_factors(self, was_successful: bool, direction: str, 
                                 regime: Dict, session: Dict) -> List[str]:
        """Identify factors that contributed to trade success/failure"""
        factors = []
        
        try:
            regime_scores = regime.get('regime_scores', {})
            session_name = session.get('current_session', 'unknown')
            
            if was_successful:
                # Success factors
                if regime_scores.get('trending', 0) > 0.6:
                    factors.append("Strong trending conditions supported the move")
                if regime_scores.get('breakout', 0) > 0.6:
                    factors.append("Breakout conditions provided momentum")
                if session_name in ['london', 'new_york', 'overlap']:
                    factors.append(f"Active {session_name} session provided good volatility")
                if regime.get('volatility_percentile', 50) > 70:
                    factors.append("High volatility supported larger moves")
            else:
                # Failure factors
                if regime_scores.get('ranging', 0) > 0.6:
                    factors.append("Ranging market conditions limited move potential")
                if regime_scores.get('reversal', 0) > 0.6:
                    factors.append("Potential reversal conditions worked against the trade")
                if session_name == 'asian':
                    factors.append("Low volatility Asian session limited movement")
                if regime.get('volatility_percentile', 50) < 30:
                    factors.append("Low volatility environment restricted price action")
            
            return factors if factors else ["Standard market conditions"]
            
        except:
            return ["Could not identify specific success factors"]
    
    def _get_improvement_suggestions(self, was_successful: bool, direction: str,
                                   regime: Dict, session: Dict) -> List[str]:
        """Get suggestions for improving future trades"""
        suggestions = []
        
        try:
            if not was_successful:
                regime_scores = regime.get('regime_scores', {})
                session_name = session.get('current_session', 'unknown')
                
                if regime_scores.get('ranging', 0) > 0.6:
                    suggestions.append("Avoid trend trades in ranging conditions - consider range trading")
                if session_name == 'asian':
                    suggestions.append("Consider smaller position sizes during Asian session")
                if regime.get('volatility_percentile', 50) < 30:
                    suggestions.append("Wait for higher volatility before entering trades")
                if regime_scores.get('reversal', 0) > 0.6:
                    suggestions.append("Watch for reversal signals before entering trend trades")
            else:
                suggestions.append("Good trade timing - continue monitoring these conditions")
            
            return suggestions if suggestions else ["Continue current approach"]
            
        except:
            return ["Monitor market conditions more closely before entry"]
    
    def _get_cached_result(self, key: str) -> Optional[Dict]:
        """Get cached result if still valid"""
        if key in self.cache:
            cached_data, timestamp = self.cache[key]
            if time.time() - timestamp < self.cache_duration:
                return cached_data
            else:
                del self.cache[key]  # Remove expired cache
        return None
    
    def _cache_result(self, key: str, data: Dict):
        """Cache result with timestamp"""
        self.cache[key] = (data, time.time())
        
        # Limit cache size
        if len(self.cache) > 50:
            # Remove oldest entries
            oldest_keys = sorted(self.cache.keys(), key=lambda k: self.cache[k][1])[:10]
            for old_key in oldest_keys:
                del self.cache[old_key]
    
    def _get_fallback_report(self) -> Dict:
        """Get fallback intelligence report when engine is unavailable"""
        return {
            'timestamp': datetime.now().isoformat(),
            'executive_summary': 'Market intelligence unavailable - using fallback analysis',
            'market_regime': {
                'dominant_regime': 'unknown',
                'confidence': 0.5,
                'regime_scores': {
                    'trending': 0.5,
                    'ranging': 0.5,
                    'breakout': 0.3,
                    'reversal': 0.3,
                    'high_volatility': 0.4,
                    'low_volatility': 0.6
                }
            },
            'session_analysis': self._get_fallback_session(),
            'confidence_score': 0.5,
            'intelligence_available': False
        }
    
    def _get_fallback_regime(self) -> Dict:
        """Get fallback regime analysis"""
        return {
            'regime_scores': {
                'trending': 0.5,
                'ranging': 0.5,
                'breakout': 0.3,
                'reversal': 0.3,
                'high_volatility': 0.4,
                'low_volatility': 0.6
            },
            'volatility_percentile': 50.0,
            'intelligence_available': False
        }
    
    def _get_fallback_session(self) -> Dict:
        """Get fallback session analysis"""
        current_hour = datetime.now().hour
        
        if 22 <= current_hour or current_hour < 6:
            session = 'asian'
        elif 6 <= current_hour < 14:
            session = 'london'
        elif 14 <= current_hour < 22:
            session = 'new_york'
        else:
            session = 'overlap'
        
        return {
            'current_session': session,
            'session_config': {
                'volatility': 'medium',
                'characteristics': 'Standard trading session',
                'risk_level': 'medium'
            },
            'intelligence_available': False
        }
    
    def _get_fallback_trade_context(self) -> Dict:
        """Get fallback trade context analysis"""
        return {
            'dominant_regime': 'unknown',
            'regime_alignment': {'alignment': 'unknown', 'score': 0.5},
            'session_alignment': {'alignment': 'unknown', 'score': 0.5},
            'trade_quality_score': 0.5,
            'success_factors': ['Standard market conditions'],
            'improvement_suggestions': ['Monitor market conditions more closely'],
            'intelligence_available': False
        }


# Global instance for use across Streamlit
_intelligence_service = None

def get_intelligence_service() -> MarketIntelligenceService:
    """Get global intelligence service instance"""
    global _intelligence_service
    if _intelligence_service is None:
        _intelligence_service = MarketIntelligenceService()
    return _intelligence_service