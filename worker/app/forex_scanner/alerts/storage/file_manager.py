"""
File Manager - File Operations Module
Handles saving of analysis results and batch summaries
Extracted from claude_api.py for better modularity
"""

import os
import logging
from typing import Dict, List
from datetime import datetime


class FileManager:
    """
    Handles file operations for Claude analysis results
    """
    
    def __init__(self, auto_save: bool = True, save_directory: str = "claude_analysis"):
        self.auto_save = auto_save
        self.save_directory = save_directory
        self.logger = logging.getLogger(__name__)
        
        if self.auto_save:
            self._ensure_save_directory()
    
    def _ensure_save_directory(self):
        """Ensure the claude_analysis directory exists"""
        try:
            os.makedirs(self.save_directory, exist_ok=True)
            self.logger.debug(f"📁 Analysis directory ready: {self.save_directory}")
        except Exception as e:
            self.logger.warning(f"⚠️ Could not create analysis directory: {e}")
            self.auto_save = False
    
    def _get_safe_timestamp_for_filename(self, signal: Dict) -> str:
        """
        FIXED: Safely extract timestamp for filename with comprehensive validation
        Fixes both 19700101_000533 and market_timestamp stale data issues
        """
        # Try multiple timestamp sources in order of preference
        timestamp_sources = [
            ('timestamp', signal.get('timestamp')),
            ('signal_timestamp', signal.get('signal_timestamp')),
            ('detection_time', signal.get('detection_time')),
            ('created_at', signal.get('created_at')),
            ('alert_timestamp', signal.get('alert_timestamp')),
            ('market_timestamp', signal.get('market_timestamp')),  # Try this but validate carefully
        ]
        
        for source_name, timestamp in timestamp_sources:
            if timestamp is None:
                continue
                
            try:
                # Handle datetime objects
                if hasattr(timestamp, 'strftime'):
                    # Check if it's a reasonable date (not epoch time)
                    if timestamp.year > 2020:
                        result = timestamp.strftime('%Y%m%d_%H%M%S')
                        self.logger.debug(f"✅ Using {source_name} timestamp: {result}")
                        return result
                    else:
                        self.logger.debug(f"⚠️ Rejecting {source_name} - year {timestamp.year} too old")
                        continue
                
                # Handle string timestamps
                if isinstance(timestamp, str):
                    try:
                        # Try pandas timestamp parsing first
                        parsed_dt = pd.to_datetime(timestamp)
                        if hasattr(parsed_dt, 'year') and parsed_dt.year > 2020:
                            result = parsed_dt.strftime('%Y%m%d_%H%M%S')
                            self.logger.debug(f"✅ Using parsed {source_name} timestamp: {result}")
                            return result
                        else:
                            self.logger.debug(f"⚠️ Rejecting parsed {source_name} - year {getattr(parsed_dt, 'year', 'unknown')} too old")
                            continue
                    except (ValueError, TypeError):
                        # Try direct string parsing for known formats
                        if '2024' in timestamp or '2025' in timestamp:
                            # Extract reasonable parts and create timestamp
                            clean_str = ''.join(c for c in timestamp if c.isdigit() or c in '-_: ')
                            if len(clean_str) >= 8:  # At least YYYYMMDD
                                try:
                                    parsed_dt = datetime.strptime(clean_str[:19].replace('-', '').replace('_', '').replace(':', '').replace(' ', ''), '%Y%m%d%H%M%S')
                                    if parsed_dt.year > 2020:
                                        result = parsed_dt.strftime('%Y%m%d_%H%M%S')
                                        self.logger.debug(f"✅ Using cleaned {source_name} timestamp: {result}")
                                        return result
                                except ValueError:
                                    continue
                
                # Handle numeric timestamps (potential Unix timestamps)
                if isinstance(timestamp, (int, float)):
                    # Only try if it looks like a reasonable Unix timestamp (after 2020)
                    if timestamp > 1577836800:  # Jan 1, 2020 in Unix time
                        try:
                            dt = datetime.fromtimestamp(timestamp)
                            if dt.year > 2020:
                                result = dt.strftime('%Y%m%d_%H%M%S')
                                self.logger.debug(f"✅ Using Unix {source_name} timestamp: {result}")
                                return result
                        except (ValueError, OSError):
                            continue
                        
            except Exception as e:
                self.logger.debug(f"⚠️ Failed to process {source_name} timestamp: {e}")
                continue
        
        # Ultimate fallback - use current time
        fallback_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.logger.warning(f"🚨 No valid timestamp found in signal, using current time: {fallback_timestamp}")
        return fallback_timestamp

    def save_minimal_analysis(self, signal: Dict, analysis: Dict):
        """Save minimal analysis to file with improved timestamp handling"""
        try:
            if not self.auto_save:
                self.logger.debug("📁 Auto-save disabled, skipping file save")
                return
                
            epic = signal.get('epic', 'unknown').replace('.', '_')
            timestamp = self._get_safe_timestamp_for_filename(signal)
            
            filename = f"{self.save_directory}/minimal_analysis_{epic}_{timestamp}.txt"
            
            # Ensure directory exists before writing
            os.makedirs(self.save_directory, exist_ok=True)
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"Claude Enhanced Analysis\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Epic: {signal.get('epic', 'N/A')}\n")
                f.write(f"Signal: {signal.get('signal_type', 'N/A')}\n")
                f.write(f"Price: {signal.get('price', 'N/A')}\n")
                f.write(f"Strategy: {self._identify_strategy(signal)}\n")
                f.write(f"Technical Validation: {'PASSED' if analysis.get('technical_validation_passed') else 'FAILED'}\n")
                f.write(f"\nCLAUDE DECISION:\n")
                f.write(f"Score: {analysis['score']}/10\n")
                f.write(f"Decision: {analysis['decision']}\n")
                f.write(f"Approved: {analysis['approved']}\n")
                f.write(f"Reason: {analysis['reason']}\n")
                f.write(f"\nRaw Response:\n{analysis['raw_response']}\n")
            
            self.logger.info(f"📁 Enhanced analysis saved to: {filename}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save minimal analysis: {e}")
            import traceback
            self.logger.debug(f"   Save error traceback: {traceback.format_exc()}")
    
    def save_complete_analysis(self, signal: Dict, analysis: Dict):
        """Save complete analysis to file with improved timestamp handling"""
        try:
            if not self.auto_save:
                self.logger.debug("📁 Auto-save disabled, skipping file save")
                return
                
            epic = signal.get('epic', 'unknown').replace('.', '_')
            timestamp = self._get_safe_timestamp_for_filename(signal)
            
            filename = f"{self.save_directory}/complete_analysis_{epic}_{timestamp}.txt"
            
            # Ensure directory exists before writing
            os.makedirs(self.save_directory, exist_ok=True)
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"Claude Complete DataFrame Analysis\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Epic: {signal.get('epic', 'N/A')}\n")
                f.write(f"System Signal: {signal.get('signal_type', 'N/A')}\n")
                f.write(f"Claude Determined: {analysis.get('claude_determined_type', 'N/A')}\n")
                f.write(f"Price: {signal.get('price', 'N/A')}\n")
                f.write(f"Triggering Strategy: {self._identify_strategy(signal)}\n")
                f.write(f"Technical Validation: {'PASSED' if analysis.get('technical_validation_passed') else 'FAILED'}\n")
                f.write(f"Analysis Type: {analysis.get('analysis_type', 'N/A')}\n")
                f.write(f"Indicators Analyzed: {', '.join(analysis.get('indicators_analyzed', []))}\n")
                f.write(f"\nCLAUDE COMPLETE ANALYSIS:\n")
                f.write(f"Score: {analysis['score']}/10\n")
                f.write(f"Correct Type: {analysis.get('correct_type', 'N/A')}\n")
                f.write(f"Decision: {analysis['decision']}\n")
                f.write(f"Approved: {analysis['approved']}\n")
                f.write(f"Reason: {analysis['reason']}\n")
                f.write(f"\nSignal Classification Match: {'✅' if analysis.get('claude_determined_type') in [signal.get('signal_type'), 'SYSTEM_CORRECT'] else '❌'}\n")
                f.write(f"\nRaw Response:\n{analysis['raw_response']}\n")
            
            self.logger.info(f"📁 Complete analysis saved to: {filename}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save complete analysis: {e}")
            import traceback
            self.logger.debug(f"   Save error traceback: {traceback.format_exc()}")
    
    def save_batch_summary_minimal(self, results: List[Dict]):
        """Save enhanced batch analysis summary"""
        try:
            if not self.auto_save:
                self.logger.debug("📁 Auto-save disabled, skipping batch summary")
                return
                
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{self.save_directory}/batch_enhanced_{timestamp}.txt"
            
            # Ensure directory exists before writing
            os.makedirs(self.save_directory, exist_ok=True)
            
            approved = len([r for r in results if r.get('approved')])
            tech_passed = len([r for r in results if r.get('technical_validation_passed')])
            total = len(results)
            avg_score = sum([r['score'] for r in results if r['score']]) / len([r for r in results if r['score']]) if results else 0
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"Claude Enhanced Batch Analysis\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Signals: {total}\n")
                f.write(f"Technical Validation Passed: {tech_passed} ({tech_passed/total*100:.1f}%)\n")
                f.write(f"Claude Approved: {approved} ({approved/total*100:.1f}%)\n")
                f.write(f"Average Score: {avg_score:.1f}/10\n\n")
                
                for i, result in enumerate(results, 1):
                    signal = result['signal']
                    tech_status = "✅" if result.get('technical_validation_passed') else "❌"
                    f.write(f"{i:2d}. {tech_status} {signal.get('epic', 'Unknown'):20s} {signal.get('signal_type', 'Unknown'):4s} ")
                    f.write(f"Score: {result['score'] or 'N/A':2s}/10 ")
                    f.write(f"Decision: {result['decision']:7s} ")
                    f.write(f"Reason: {result['reason'] or 'N/A'}\n")
            
            self.logger.info(f"📁 Enhanced batch summary saved: {tech_passed}/{total} tech valid, {approved}/{total} approved")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save enhanced batch summary: {e}")
            import traceback
            self.logger.debug(f"   Batch save error traceback: {traceback.format_exc()}")
    
    def save_minimal_analysis_enhanced(self, signal: Dict, analysis: Dict):
        """
        Enhanced save minimal analysis with better error handling and institutional features
        """
        try:
            if not self.auto_save:
                self.logger.debug("📁 Auto-save disabled, skipping file save")
                return None
            
            epic = signal.get('epic', 'unknown').replace('.', '_').replace(':', '_')
            timestamp = self._get_safe_timestamp_for_filename(signal)
            
            filename = f"{self.save_directory}/minimal_analysis_{epic}_{timestamp}.txt"
            
            # Ensure directory exists
            os.makedirs(self.save_directory, exist_ok=True)
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"Claude Enhanced Analysis - Institutional Grade\n")
                f.write(f"=" * 50 + "\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Analysis Level: {analysis.get('claude_analysis_level', 'institutional')}\n")
                f.write(f"Advanced Prompts: {analysis.get('claude_advanced_prompts', True)}\n")
                f.write(f"\nSIGNAL DETAILS:\n")
                f.write(f"Epic: {signal.get('epic', 'N/A')}\n")
                f.write(f"Signal Type: {signal.get('signal_type', 'N/A')}\n")
                f.write(f"Price: {signal.get('price', 'N/A')}\n")
                f.write(f"Confidence: {signal.get('confidence_score', 0):.1%}\n")
                f.write(f"Strategy: {self._identify_strategy(signal)}\n")
                f.write(f"Indicators: {', '.join(self._get_indicators_analyzed(signal))}\n")
                
                # Technical validation status
                tech_validation = analysis.get('technical_validation_passed', False)
                f.write(f"Technical Validation: {'✅ PASSED' if tech_validation else '❌ FAILED'}\n")
                
                f.write(f"\nCLAUDE ANALYSIS:\n")
                f.write(f"Score: {analysis.get('score', 'N/A')}/10\n")
                f.write(f"Decision: {analysis.get('decision', 'N/A')}\n")
                f.write(f"Approved: {'✅ YES' if analysis.get('approved') else '❌ NO'}\n")
                f.write(f"Reason: {analysis.get('reason', 'N/A')}\n")
                
                # Enhanced modular features
                if analysis.get('claude_analysis_level'):
                    f.write(f"Analysis Level Used: {analysis['claude_analysis_level']}\n")
                if analysis.get('claude_advanced_prompts'):
                    f.write(f"Advanced Prompts: {analysis['claude_advanced_prompts']}\n")
                
                f.write(f"\nRAW CLAUDE RESPONSE:\n")
                f.write(f"{'-' * 30}\n")
                f.write(f"{analysis.get('raw_response', 'N/A')}\n")
                f.write(f"{'-' * 30}\n")
                
                # Add metadata
                f.write(f"\nMETADATA:\n")
                f.write(f"File: {filename}\n")
                f.write(f"Component: FileManager (Modular Claude API)\n")
                f.write(f"Version: Enhanced with institutional analysis\n")
            
            self.logger.info(f"📁 Enhanced Claude analysis saved to: {filename}")
            return filename
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save enhanced minimal analysis: {e}")
            import traceback
            self.logger.debug(f"   Save error traceback: {traceback.format_exc()}")
            return None
    
    def save_batch_summary_minimal_enhanced(self, results: List[Dict]):
        """
        Enhanced batch summary with better error handling
        """
        try:
            if not self.auto_save:
                self.logger.debug("📁 Auto-save disabled, skipping batch summary")
                return None
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{self.save_directory}/batch_enhanced_{timestamp}.txt"
            
            # Ensure directory exists
            os.makedirs(self.save_directory, exist_ok=True)
            
            # Calculate statistics
            total = len(results)
            approved = len([r for r in results if r.get('approved')])
            tech_passed = len([r for r in results if r.get('technical_validation_passed')])
            
            # Calculate average score safely
            scores = [r.get('score') for r in results if r.get('score') is not None]
            avg_score = sum(scores) / len(scores) if scores else 0
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"Claude Enhanced Batch Analysis - Institutional Grade\n")
                f.write(f"=" * 60 + "\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Analysis Type: Modular Claude API with Advanced Prompts\n")
                f.write(f"\nBATCH STATISTICS:\n")
                f.write(f"Total Signals Analyzed: {total}\n")
                f.write(f"Technical Validation Passed: {tech_passed} ({tech_passed/total*100:.1f}%)\n")
                f.write(f"Claude Approved: {approved} ({approved/total*100:.1f}%)\n")
                f.write(f"Average Claude Score: {avg_score:.1f}/10\n")
                f.write(f"Success Rate: {approved/total*100:.1f}%\n")
                
                f.write(f"\nDETAILED RESULTS:\n")
                f.write(f"{'-' * 60}\n")
                
                for i, result in enumerate(results, 1):
                    signal = result.get('signal', {})
                    epic = signal.get('epic', 'Unknown')
                    signal_type = signal.get('signal_type', 'Unknown')
                    score = result.get('score', 'N/A')
                    decision = result.get('decision', 'N/A')
                    approved = '✅' if result.get('approved') else '❌'
                    tech_status = '✅' if result.get('technical_validation_passed') else '❌'
                    
                    f.write(f"{i:2d}. {epic} {signal_type} | Score: {score}/10 | {decision} {approved} | Tech: {tech_status}\n")
                
                f.write(f"{'-' * 60}\n")
                f.write(f"Component: FileManager (Enhanced Modular Claude API)\n")
                f.write(f"File: {filename}\n")
            
            self.logger.info(f"📁 Enhanced batch summary saved to: {filename}")
            return filename
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save enhanced batch summary: {e}")
            import traceback
            self.logger.debug(f"   Batch save error traceback: {traceback.format_exc()}")
            return None
    
    def _get_safe_timestamp_for_filename(self, signal: Dict) -> str:
        """
        Safely extract timestamp for filename with comprehensive validation
        Fixes both epoch time and stale data issues
        """
        # Try multiple timestamp sources in order of preference
        timestamp_sources = [
            ('timestamp', signal.get('timestamp')),
            ('signal_timestamp', signal.get('signal_timestamp')),
            ('detection_time', signal.get('detection_time')),
            ('created_at', signal.get('created_at')),
            ('alert_timestamp', signal.get('alert_timestamp')),
            ('market_timestamp', signal.get('market_timestamp')),
        ]
        
        for source_name, timestamp in timestamp_sources:
            if timestamp is None:
                continue
                
            try:
                # Handle datetime objects
                if hasattr(timestamp, 'strftime'):
                    # Check if it's a reasonable date (not epoch time)
                    if timestamp.year > 2020:
                        result = timestamp.strftime('%Y%m%d_%H%M%S')
                        self.logger.debug(f"✅ Using {source_name} timestamp: {result}")
                        return result
                    else:
                        self.logger.debug(f"⚠️ Rejecting {source_name} - year {timestamp.year} too old")
                        continue
                
                # Handle string timestamps
                if isinstance(timestamp, str):
                    # Skip obviously bad timestamps
                    if timestamp.startswith('1970') or timestamp.startswith('1969'):
                        self.logger.debug(f"⚠️ Rejecting {source_name} - starts with epoch year: {timestamp}")
                        continue
                    
                    # Try to parse various formats
                    timestamp_formats = [
                        '%Y-%m-%d %H:%M:%S.%f',
                        '%Y-%m-%d %H:%M:%S',
                        '%Y-%m-%dT%H:%M:%S.%fZ',
                        '%Y-%m-%dT%H:%M:%SZ',
                        '%Y-%m-%dT%H:%M:%S',
                        '%Y%m%d_%H%M%S',
                        '%Y-%m-%d',
                    ]
                    
                    for fmt in timestamp_formats:
                        try:
                            dt = datetime.strptime(timestamp, fmt)
                            if dt.year > 2020:
                                result = dt.strftime('%Y%m%d_%H%M%S')
                                self.logger.debug(f"✅ Parsed {source_name} timestamp: {result}")
                                return result
                        except ValueError:
                            continue
                    
                    # Try to clean string and extract parts
                    cleaned = str(timestamp).replace(':', '').replace(' ', '_').replace('-', '')
                    if len(cleaned) >= 8 and cleaned[:8].isdigit():
                        year_part = cleaned[:4]
                        try:
                            if int(year_part) > 2020:
                                if len(cleaned) < 15:
                                    cleaned += '0' * (15 - len(cleaned))
                                result = cleaned[:15]
                                self.logger.debug(f"✅ Cleaned {source_name} timestamp: {result}")
                                return result
                        except ValueError:
                            continue
                
                # Handle numeric timestamps (Unix time)
                if isinstance(timestamp, (int, float)):
                    if timestamp > 1600000000 and timestamp < 2000000000:  # Between 2020 and 2033
                        dt = datetime.fromtimestamp(timestamp)
                        result = dt.strftime('%Y%m%d_%H%M%S')
                        self.logger.debug(f"✅ Converted {source_name} Unix timestamp: {result}")
                        return result
                    else:
                        self.logger.debug(f"⚠️ Rejecting {source_name} - invalid Unix timestamp: {timestamp}")
                        continue
                    
            except Exception as e:
                self.logger.debug(f"❌ Error processing {source_name} timestamp {timestamp}: {e}")
                continue
        
        # Ultimate fallback - use current time
        current_time = datetime.now()
        result = current_time.strftime('%Y%m%d_%H%M%S')
        
        epic = signal.get('epic', 'unknown')
        self.logger.warning(f"⚠️ No valid timestamp found for {epic}, using current time: {result}")
        
        return result
    
    def _identify_strategy(self, signal: Dict) -> str:
        """Identify the strategy type from signal data"""
        strategy = signal.get('strategy', '').lower()
        
        if 'combined' in strategy:
            return 'COMBINED'
        elif 'macd' in strategy:
            return 'MACD'
        elif 'kama' in strategy:
            return 'KAMA'
        elif 'ema' in strategy:
            return 'EMA'
        else:
            # Try to identify from available indicators
            if signal.get('macd_line') is not None or signal.get('macd_histogram') is not None:
                return 'MACD'
            elif signal.get('kama_value') is not None or any(k.startswith('kama_') for k in signal.keys()):
                return 'KAMA'
            elif signal.get('ema_short') is not None or signal.get('ema_9') is not None:
                return 'EMA'
            else:
                return 'UNKNOWN'
    
    def _get_indicators_analyzed(self, signal: Dict) -> List[str]:
        """
        Get list of indicators that were analyzed
        """
        try:
            indicators = []
            
            # Check for various indicators in signal data
            if 'ema_9' in signal or 'ema_21' in signal or 'ema_200' in signal:
                indicators.append('EMA')
            
            if 'macd_line' in signal or 'macd_signal' in signal or 'macd_histogram' in signal:
                indicators.append('MACD')
            
            if 'kama_value' in signal:
                indicators.append('KAMA')
            
            if 'rsi' in signal:
                indicators.append('RSI')
            
            if 'bollinger' in signal:
                indicators.append('Bollinger Bands')
            
            # Check strategy-based indicators
            strategy = signal.get('strategy', '').lower()
            if 'ema' in strategy and 'EMA' not in indicators:
                indicators.append('EMA')
            if 'macd' in strategy and 'MACD' not in indicators:
                indicators.append('MACD')
            if 'kama' in strategy and 'KAMA' not in indicators:
                indicators.append('KAMA')
            
            return indicators if indicators else ['Price Action']
            
        except Exception as e:
            self.logger.debug(f"Error getting indicators: {e}")
            return ['Unknown']