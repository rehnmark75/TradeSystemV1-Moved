"""
Signal Validator - Signal-Specific Validation Module
Handles validation of signal data structure, completeness, and quality
Extracted from claude_api.py for better modularity

UPDATED: Enhanced to support nested price structures (ema_data, macd_data, etc.)
while maintaining all existing functionality and validation features.
"""

import logging
from typing import Dict, List, Optional, Set
from datetime import datetime
import re


class SignalValidator:
    """
    Validates signal data structure, completeness, and basic quality checks
    UPDATED: Now supports nested price structures like ema_data.current_price
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # UPDATED: Required fields for different signal types (price handled separately for nested structures)
        self.required_basic_fields = {
            'epic', 'signal_type', 'timestamp', 'strategy'
        }
        
        # ADDED: Price fields that can be found in flat or nested structures
        self.price_field_names = [
            'price', 'current_price', 'entry_price', 'signal_price', 'close_price',
            'last_price', 'market_price', 'bid_price', 'mid_price'
        ]
        
        # ADDED: Nested structures where price data might be found
        self.nested_price_structures = ['ema_data', 'macd_data', 'kama_data', 'other_indicators', 'technical_data']
        
        self.required_ema_fields = {
            'ema_short', 'ema_long', 'ema_trend'
        }
        
        self.required_macd_fields = {
            'macd_line', 'macd_signal', 'macd_histogram'
        }
        
        self.required_kama_fields = {
            'kama_value', 'efficiency_ratio'
        }
        
        # UPDATED: Valid values (added BUY/SELL for your signal format)
        self.valid_signal_types = {'BULL', 'BEAR', 'BUY', 'SELL'}
        self.valid_strategies = {'ema', 'macd', 'kama', 'combined', 'consensus', 'weighted'}
        self.valid_timeframes = {'1m', '5m', '15m', '30m', '1h', '4h', '1d'}
    
    def validate_signal_structure(self, signal: Dict) -> Dict:
        """
        UPDATED: Validate the basic structure and required fields with nested price support
        """
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'missing_fields': [],
            'invalid_fields': {},
            'completeness_score': 0.0
        }
        
        try:
            # Check basic required fields (excluding price - handled separately)
            missing_basic = self.required_basic_fields - set(signal.keys())
            if missing_basic:
                validation_result['missing_fields'].extend(missing_basic)
                validation_result['errors'].append(f"Missing basic fields: {missing_basic}")
                validation_result['valid'] = False
            
            # UPDATED: Enhanced price validation with nested structure support
            price_validation = self._validate_price_field(signal)
            if not price_validation['found']:
                # Only warn, don't fail validation (for backward compatibility)
                validation_result['warnings'].append(f"No price field found. {price_validation['message']}")
                # Add to missing fields for reporting but don't fail validation
                validation_result['missing_fields'].append('price')
            else:
                self.logger.debug(f"✅ Price found: {price_validation['message']}")
            
            # Validate signal type
            signal_type = signal.get('signal_type', '').upper()
            if signal_type not in self.valid_signal_types:
                validation_result['invalid_fields']['signal_type'] = signal_type
                validation_result['errors'].append(f"Invalid signal_type: {signal_type}. Must be one of {self.valid_signal_types}")
                validation_result['valid'] = False
            
            # Validate strategy
            strategy = signal.get('strategy', '').lower()
            if strategy and not any(valid_strat in strategy for valid_strat in self.valid_strategies):
                validation_result['warnings'].append(f"Unknown strategy: {strategy}")
            
            # Validate epic format
            epic = signal.get('epic', '')
            if epic and not self._is_valid_epic_format(epic):
                validation_result['warnings'].append(f"Epic format may be invalid: {epic}")
            
            # UPDATED: Enhanced price value validation (if found)
            if price_validation['found'] and price_validation['value'] is not None:
                try:
                    price_float = float(price_validation['value'])
                    if price_float <= 0:
                        validation_result['errors'].append(f"Price must be positive: {price_float}")
                        validation_result['valid'] = False
                except (ValueError, TypeError):
                    validation_result['invalid_fields']['price'] = price_validation['value']
                    validation_result['errors'].append(f"Price must be numeric: {price_validation['value']}")
                    validation_result['valid'] = False
            
            # Validate timestamp
            timestamp = signal.get('timestamp')
            if timestamp and not self._is_valid_timestamp(timestamp):
                validation_result['warnings'].append(f"Timestamp format may be invalid: {timestamp}")
            
            # Check strategy-specific fields
            strategy_validation = self._validate_strategy_fields(signal, strategy)
            validation_result['warnings'].extend(strategy_validation['warnings'])
            validation_result['missing_fields'].extend(strategy_validation['missing_fields'])
            
            # Calculate completeness score (UPDATED to handle nested structures)
            validation_result['completeness_score'] = self._calculate_completeness_score(signal)
            
            # Overall validation
            if validation_result['errors']:
                validation_result['valid'] = False
            
            self.logger.debug(f"Signal validation: {'✅ PASS' if validation_result['valid'] else '❌ FAIL'}")
            
        except Exception as e:
            validation_result['valid'] = False
            validation_result['errors'].append(f"Validation error: {str(e)}")
            self.logger.error(f"Signal validation failed: {e}")
        
        return validation_result
    
    def _validate_price_field(self, signal: Dict) -> Dict:
        """
        NEW: Enhanced price field validation with nested structure support
        
        Returns:
            Dict with 'found': bool, 'value': float/None, 'message': str
        """
        try:
            # Check flat price fields first
            for price_field in self.price_field_names:
                if price_field in signal and signal[price_field] is not None:
                    try:
                        price_value = float(signal[price_field])
                        return {
                            'found': True,
                            'value': price_value,
                            'message': f"Found in flat field '{price_field}': {price_value:.5f}"
                        }
                    except (ValueError, TypeError):
                        continue
            
            # Check nested structures
            for struct_name in self.nested_price_structures:
                if struct_name in signal and isinstance(signal[struct_name], dict):
                    struct_data = signal[struct_name]
                    # Check for price fields in nested structure
                    price_candidates = self.price_field_names + ['close', 'ema_5', 'ema_9']  # Additional candidates in nested data
                    for price_field in price_candidates:
                        if price_field in struct_data and struct_data[price_field] is not None:
                            try:
                                price_value = float(struct_data[price_field])
                                return {
                                    'found': True,
                                    'value': price_value,
                                    'message': f"Found in nested structure '{struct_name}.{price_field}': {price_value:.5f}"
                                }
                            except (ValueError, TypeError):
                                continue
            
            # Check for execution_price, mid_price as fallbacks
            fallback_fields = ['execution_price', 'mid_price', 'bid_price', 'ask_price']
            for price_field in fallback_fields:
                if price_field in signal and signal[price_field] is not None:
                    try:
                        price_value = float(signal[price_field])
                        return {
                            'found': True,
                            'value': price_value,
                            'message': f"Found in fallback field '{price_field}': {price_value:.5f}"
                        }
                    except (ValueError, TypeError):
                        continue
            
            # No price found
            available_fields = list(signal.keys())
            nested_info = []
            for struct_name in self.nested_price_structures:
                if struct_name in signal and isinstance(signal[struct_name], dict):
                    nested_fields = list(signal[struct_name].keys())
                    nested_info.append(f"{struct_name}: {nested_fields}")
            
            message = f"Expected price in: {self.price_field_names}. Available fields: {available_fields}"
            if nested_info:
                message += f". Nested structures: {'; '.join(nested_info)}"
            
            return {
                'found': False,
                'value': None,
                'message': message
            }
            
        except Exception as e:
            return {
                'found': False,
                'value': None,
                'message': f"Error validating price field: {str(e)}"
            }
    
    def validate_signal_quality(self, signal: Dict) -> Dict:
        """
        Validate signal quality and consistency
        """
        quality_result = {
            'quality_score': 0.0,
            'quality_issues': [],
            'quality_warnings': [],
            'data_consistency': True
        }
        
        try:
            quality_points = 0
            max_points = 0
            
            # 1. Confidence score validation (20 points)
            max_points += 20
            confidence = signal.get('confidence_score', 0)
            if confidence:
                try:
                    conf_float = float(confidence)
                    if 0 <= conf_float <= 1:
                        quality_points += int(conf_float * 20)  # 0-20 points based on confidence
                    else:
                        quality_result['quality_issues'].append(f"Confidence score out of range: {conf_float}")
                except (ValueError, TypeError):
                    quality_result['quality_issues'].append(f"Invalid confidence score: {confidence}")
            else:
                quality_result['quality_warnings'].append("No confidence score provided")
            
            # 2. Technical indicator consistency (30 points)
            max_points += 30
            consistency_score = self._check_technical_consistency(signal)
            quality_points += consistency_score
            
            # 3. Volume validation (20 points)
            max_points += 20
            volume_score = self._check_volume_quality(signal)
            quality_points += volume_score
            
            # 4. Market context validation (15 points)
            max_points += 15
            context_score = self._check_market_context(signal)
            quality_points += context_score
            
            # 5. Data completeness (15 points)
            max_points += 15
            completeness_score = signal.get('completeness_score', 0) * 15
            quality_points += completeness_score
            
            # Calculate final quality score
            quality_result['quality_score'] = (quality_points / max_points) * 100 if max_points > 0 else 0
            
            # Set data consistency flag
            quality_result['data_consistency'] = len(quality_result['quality_issues']) == 0
            
        except Exception as e:
            quality_result['quality_issues'].append(f"Quality validation error: {str(e)}")
            self.logger.error(f"Signal quality validation failed: {e}")
        
        return quality_result
    
    def validate_signal_batch(self, signals: List[Dict]) -> Dict:
        """
        Validate a batch of signals and provide summary statistics
        """
        batch_result = {
            'total_signals': len(signals),
            'valid_signals': 0,
            'invalid_signals': 0,
            'average_quality_score': 0.0,
            'common_issues': {},
            'batch_warnings': []
        }
        
        if not signals:
            batch_result['batch_warnings'].append("Empty signal batch")
            return batch_result
        
        quality_scores = []
        issue_counts = {}
        
        for i, signal in enumerate(signals):
            # Validate structure
            structure_result = self.validate_signal_structure(signal)
            
            if structure_result['valid']:
                batch_result['valid_signals'] += 1
                
                # Validate quality
                quality_result = self.validate_signal_quality(signal)
                quality_scores.append(quality_result['quality_score'])
                
                # Collect issues for statistics
                for issue in quality_result['quality_issues']:
                    issue_counts[issue] = issue_counts.get(issue, 0) + 1
                    
            else:
                batch_result['invalid_signals'] += 1
                
                # Collect validation errors
                for error in structure_result['errors']:
                    issue_counts[error] = issue_counts.get(error, 0) + 1
        
        # Calculate average quality score
        if quality_scores:
            batch_result['average_quality_score'] = sum(quality_scores) / len(quality_scores)
        
        # Find most common issues
        batch_result['common_issues'] = dict(sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:5])
        
        # Add batch-level warnings
        if batch_result['invalid_signals'] > batch_result['total_signals'] * 0.1:
            batch_result['batch_warnings'].append(f"High invalid signal rate: {batch_result['invalid_signals']}/{batch_result['total_signals']}")
        
        if batch_result['average_quality_score'] < 50:
            batch_result['batch_warnings'].append(f"Low average quality score: {batch_result['average_quality_score']:.1f}")
        
        return batch_result
    
    def _validate_strategy_fields(self, signal: Dict, strategy: str) -> Dict:
        """
        UPDATED: Validate strategy-specific required fields with nested structure support
        """
        result = {'warnings': [], 'missing_fields': []}
        
        strategy_lower = strategy.lower()
        
        if 'ema' in strategy_lower:
            missing_ema = self._get_missing_fields(signal, self.required_ema_fields, ['ema_9', 'ema_21', 'ema_200'])
            if missing_ema:
                result['missing_fields'].extend(missing_ema)
                result['warnings'].append(f"EMA strategy missing fields: {missing_ema}")
        
        if 'macd' in strategy_lower:
            missing_macd = self._get_missing_fields(signal, self.required_macd_fields)
            if missing_macd:
                result['missing_fields'].extend(missing_macd)
                result['warnings'].append(f"MACD strategy missing fields: {missing_macd}")
        
        if 'kama' in strategy_lower:
            missing_kama = self._get_missing_fields(signal, self.required_kama_fields)
            if missing_kama:
                result['missing_fields'].extend(missing_kama)
                result['warnings'].append(f"KAMA strategy missing fields: {missing_kama}")
        
        return result
    
    def _get_missing_fields(self, signal: Dict, required_fields: Set[str], alternative_fields: List[str] = None) -> List[str]:
        """
        UPDATED: Get missing fields, considering alternative field names and nested structures
        """
        missing = []
        
        for field in required_fields:
            found = False
            
            # Check flat fields first
            if field in signal and signal[field] is not None:
                found = True
            else:
                # Check alternative field names
                if alternative_fields:
                    for alt_field in alternative_fields:
                        if alt_field in signal and signal[alt_field] is not None:
                            found = True
                            break
                
                # ADDED: Check nested structures for the field
                if not found:
                    for struct_name in self.nested_price_structures:
                        if struct_name in signal and isinstance(signal[struct_name], dict):
                            struct_data = signal[struct_name]
                            if field in struct_data and struct_data[field] is not None:
                                found = True
                                break
                            # Also check alternative fields in nested structures
                            if alternative_fields:
                                for alt_field in alternative_fields:
                                    if alt_field in struct_data and struct_data[alt_field] is not None:
                                        found = True
                                        break
                            if found:
                                break
            
            if not found:
                missing.append(field)
        
        return missing
    
    def _is_valid_epic_format(self, epic: str) -> bool:
        """Validate epic format (e.g., CS.D.EURUSD.MINI.IP)"""
        if not epic:
            return False
        
        # Common IG epic patterns
        patterns = [
            r'^CS\.D\.[A-Z]{6}\.MINI\.IP$',  # Forex mini
            r'^CS\.D\.[A-Z]{6}\.IP$',       # Forex standard
            r'^IX\.D\.[A-Z0-9]+\.DAILY\.IP$', # Indices
            r'^CC\.D\.[A-Z]+\.USS\.IP$'     # Commodities
        ]
        
        return any(re.match(pattern, epic) for pattern in patterns)
    
    def _is_valid_timestamp(self, timestamp) -> bool:
        """Validate timestamp format and reasonableness"""
        try:
            if hasattr(timestamp, 'year'):
                # datetime object
                return 2020 <= timestamp.year <= 2030
            elif isinstance(timestamp, str):
                # String timestamp
                if timestamp.startswith('1970'):
                    return False  # Epoch time issue
                # Try to parse common formats
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                return 2020 <= dt.year <= 2030
            elif isinstance(timestamp, (int, float)):
                # Unix timestamp
                if 1600000000 <= timestamp <= 2000000000:  # 2020-2033
                    return True
            return False
        except:
            return False
    
    def _calculate_completeness_score(self, signal: Dict) -> float:
        """
        UPDATED: Calculate signal data completeness score (0-100) with nested structure support
        """
        try:
            # Define all possible fields with weights
            field_weights = {
                # Basic fields (high weight)
                'epic': 10, 'signal_type': 10, 'timestamp': 8, 'strategy': 8,
                
                # Technical indicators (medium-high weight)
                'ema_short': 6, 'ema_long': 6, 'ema_trend': 6,
                'macd_line': 6, 'macd_signal': 6, 'macd_histogram': 6,
                'kama_value': 5, 'efficiency_ratio': 5,
                
                # Market data (medium weight)
                'volume': 4, 'volume_ratio': 4, 'confidence_score': 5,
                'nearest_support': 3, 'nearest_resistance': 3,
                
                # Additional data (low weight)
                'timeframe': 2, 'market_session': 2, 'spread_pips': 2,
                'signal_trigger': 2, 'crossover_type': 2
            }
            
            # ADDED: Price fields (high weight)
            price_field_weight = 10
            
            total_weight = sum(field_weights.values()) + price_field_weight
            achieved_weight = 0
            
            # Check regular fields
            for field, weight in field_weights.items():
                field_found = self._check_field_in_signal(signal, field)
                if field_found:
                    achieved_weight += weight
            
            # ADDED: Check price fields (flat and nested)
            price_validation = self._validate_price_field(signal)
            if price_validation['found']:
                achieved_weight += price_field_weight
            
            return (achieved_weight / total_weight) * 100
            
        except Exception as e:
            self.logger.error(f"Error calculating completeness score: {e}")
            return 0.0
    
    def _check_field_in_signal(self, signal: Dict, field: str) -> bool:
        """
        NEW: Check if a field exists in signal (flat or nested structures) with meaningful value
        """
        try:
            # Check flat structure first
            if field in signal and signal[field] is not None:
                value = signal[field]
                if isinstance(value, str) and value.strip():
                    return True
                elif isinstance(value, (int, float)) and (field in ['price', 'volume'] and value > 0 or field not in ['price', 'volume']):
                    return True
                elif not isinstance(value, (str, int, float)):
                    return True  # Other types (datetime, etc.)
            
            # Check nested structures
            for struct_name in self.nested_price_structures:
                if struct_name in signal and isinstance(signal[struct_name], dict):
                    struct_data = signal[struct_name]
                    if field in struct_data and struct_data[field] is not None:
                        value = struct_data[field]
                        if isinstance(value, str) and value.strip():
                            return True
                        elif isinstance(value, (int, float)) and value != 0:
                            return True
                        elif not isinstance(value, (str, int, float)):
                            return True
            
            return False
            
        except Exception:
            return False
    
    def _check_technical_consistency(self, signal: Dict) -> int:
        """
        UPDATED: Check technical indicator consistency (returns 0-30 points) with nested structure support
        """
        try:
            points = 0
            
            # UPDATED: Enhanced indicator extraction with nested structure support
            ema_9 = self._get_indicator_value(signal, 'ema_9') or self._get_indicator_value(signal, 'ema_short')
            ema_21 = self._get_indicator_value(signal, 'ema_21') or self._get_indicator_value(signal, 'ema_long')
            ema_200 = self._get_indicator_value(signal, 'ema_200') or self._get_indicator_value(signal, 'ema_trend')
            
            # Get price using enhanced validation
            price_validation = self._validate_price_field(signal)
            price = price_validation['value'] if price_validation['found'] else None
            
            signal_type = signal.get('signal_type', '').upper()
            
            if all(x is not None for x in [ema_9, ema_21, ema_200, price]):
                try:
                    ema_9, ema_21, ema_200, price = float(ema_9), float(ema_21), float(ema_200), float(price)
                    
                    if signal_type in ['BULL', 'BUY']:
                        if price >= ema_9 >= ema_21 >= ema_200:
                            points += 15  # Perfect alignment
                        elif price > ema_200:
                            points += 10  # Basic trend filter
                        else:
                            points += 5   # Partial alignment
                    elif signal_type in ['BEAR', 'SELL']:
                        if price <= ema_9 <= ema_21 <= ema_200:
                            points += 15  # Perfect alignment
                        elif price < ema_200:
                            points += 10  # Basic trend filter
                        else:
                            points += 5   # Partial alignment
                except (ValueError, TypeError):
                    pass
            
            # UPDATED: Enhanced MACD consistency check
            macd_histogram = self._get_indicator_value(signal, 'macd_histogram')
            if macd_histogram is not None:
                try:
                    macd_histogram = float(macd_histogram)
                    
                    if signal_type in ['BULL', 'BUY'] and macd_histogram > 0:
                        points += 15  # MACD supports bullish signal
                    elif signal_type in ['BEAR', 'SELL'] and macd_histogram < 0:
                        points += 15  # MACD supports bearish signal
                    else:
                        points += 5   # MACD present but not aligned
                except (ValueError, TypeError):
                    pass
            
            return min(points, 30)  # Cap at 30 points
            
        except Exception as e:
            self.logger.error(f"Error checking technical consistency: {e}")
            return 0
    
    def _get_indicator_value(self, signal: Dict, field_name: str):
        """
        NEW: Get indicator value from flat or nested structures
        """
        try:
            # Check flat structure first
            if field_name in signal and signal[field_name] is not None:
                return signal[field_name]
            
            # Check nested structures
            for struct_name in self.nested_price_structures:
                if struct_name in signal and isinstance(signal[struct_name], dict):
                    struct_data = signal[struct_name]
                    if field_name in struct_data and struct_data[field_name] is not None:
                        return struct_data[field_name]
            
            return None
            
        except Exception:
            return None
    
    def _check_volume_quality(self, signal: Dict) -> int:
        """
        UPDATED: Check volume data quality (returns 0-20 points) with nested structure support
        """
        try:
            points = 0
            
            volume = self._get_indicator_value(signal, 'volume')
            volume_ratio = self._get_indicator_value(signal, 'volume_ratio')
            
            if volume is not None:
                try:
                    volume = float(volume)
                    if volume > 0:
                        points += 10  # Has volume data
                    
                    if volume_ratio is not None:
                        volume_ratio = float(volume_ratio)
                        if volume_ratio > 1.0:
                            points += 10  # Good volume confirmation
                        elif volume_ratio > 0.8:
                            points += 5   # Moderate volume
                except (ValueError, TypeError):
                    pass
            
            return points
            
        except Exception as e:
            self.logger.error(f"Error checking volume quality: {e}")
            return 0
    
    def _check_market_context(self, signal: Dict) -> int:
        """
        UPDATED: Check market context data (returns 0-15 points) with nested structure support
        """
        try:
            points = 0
            
            # Check for support/resistance levels
            if self._get_indicator_value(signal, 'nearest_support') is not None:
                points += 5
            if self._get_indicator_value(signal, 'nearest_resistance') is not None:
                points += 5
            
            # Check for market session info
            if self._get_indicator_value(signal, 'market_session'):
                points += 3
            
            # Check for timeframe
            timeframe = self._get_indicator_value(signal, 'timeframe')
            if timeframe in self.valid_timeframes:
                points += 2
            
            return points
            
        except Exception as e:
            self.logger.error(f"Error checking market context: {e}")
            return 0


# Usage example
if __name__ == "__main__":
    validator = SignalValidator()
    
    # Test signal with nested structure (your format)
    test_signal_nested = {
        'epic': 'CS.D.EURUSD.MINI.IP',
        'signal_type': 'BUY',
        'timestamp': datetime.now(),
        'strategy': 'ema',
        'confidence_score': 0.85,
        'ema_data': {
            'ema_5': 1.0845,
            'ema_9': 1.0845,
            'ema_21': 1.0840,
            'ema_200': 1.0820,
            'current_price': 1.0850,
            'close': 1.0850
        },
        'macd_data': {
            'macd_line': 0.0012,
            'macd_signal': 0.0008,
            'macd_histogram': 0.0004
        },
        'volume': 1500,
        'volume_ratio': 1.2
    }
    
    # Test signal with flat structure (legacy format)
    test_signal_flat = {
        'epic': 'CS.D.EURUSD.MINI.IP',
        'signal_type': 'BULL',
        'price': 1.0850,
        'timestamp': datetime.now(),
        'strategy': 'ema',
        'ema_short': 1.0845,
        'ema_long': 1.0840,
        'ema_trend': 1.0820,
        'confidence_score': 0.85,
        'volume': 1500,
        'volume_ratio': 1.2
    }
    
    print("=== Testing Nested Structure Signal ===")
    # Validate structure
    structure_result = validator.validate_signal_structure(test_signal_nested)
    print(f"📋 Structure validation: {'✅ PASS' if structure_result['valid'] else '❌ FAIL'}")
    print(f"   Completeness: {structure_result['completeness_score']:.1f}%")
    
    # Validate quality
    quality_result = validator.validate_signal_quality(test_signal_nested)
    print(f"⭐ Quality score: {quality_result['quality_score']:.1f}/100")
    
    if structure_result['errors']:
        print(f"❌ Errors: {structure_result['errors']}")
    
    if structure_result['warnings']:
        print(f"⚠️ Warnings: {structure_result['warnings']}")
    
    print("\n=== Testing Flat Structure Signal ===")
    # Validate structure
    structure_result_flat = validator.validate_signal_structure(test_signal_flat)
    print(f"📋 Structure validation: {'✅ PASS' if structure_result_flat['valid'] else '❌ FAIL'}")
    print(f"   Completeness: {structure_result_flat['completeness_score']:.1f}%")
    
    # Validate quality
    quality_result_flat = validator.validate_signal_quality(test_signal_flat)
    print(f"⭐ Quality score: {quality_result_flat['quality_score']:.1f}/100")
    
    if structure_result_flat['errors']:
        print(f"❌ Errors: {structure_result_flat['errors']}")
    
    if structure_result_flat['warnings']:
        print(f"⚠️ Warnings: {structure_result_flat['warnings']}")