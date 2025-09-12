"""
Timestamp Validator - Timestamp Validation and Processing Module
Handles validation, cleaning, and normalization of timestamps
Extracted from claude_api.py for better modularity
"""

import logging
from typing import Dict, Optional, Union, List
from datetime import datetime, timezone, timedelta
import pandas as pd
import re


class TimestampValidator:
    """
    Validates, cleans, and normalizes timestamps for signals
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Valid timestamp ranges
        self.min_year = 2020
        self.max_year = 2030
        self.epoch_threshold = datetime(1971, 1, 1)
        
        # Common timestamp formats to try
        self.timestamp_formats = [
            '%Y-%m-%d %H:%M:%S.%f',     # 2025-07-31 07:15:00.123456
            '%Y-%m-%d %H:%M:%S',        # 2025-07-31 07:15:00
            '%Y-%m-%dT%H:%M:%S.%fZ',    # 2025-07-31T07:15:00.123456Z
            '%Y-%m-%dT%H:%M:%S.%f',     # 2025-07-31T07:15:00.123456
            '%Y-%m-%dT%H:%M:%SZ',       # 2025-07-31T07:15:00Z
            '%Y-%m-%dT%H:%M:%S',        # 2025-07-31T07:15:00
            '%Y%m%d_%H%M%S',            # 20250731_071500
            '%Y-%m-%d',                 # 2025-07-31
            '%d/%m/%Y %H:%M:%S',        # 31/07/2025 07:15:00
            '%m/%d/%Y %H:%M:%S',        # 07/31/2025 07:15:00
        ]
    
    def validate_and_clean_timestamp(self, timestamp_data: Union[str, datetime, int, float], 
                                   field_name: str = "timestamp") -> Dict:
        """
        Validate and clean a single timestamp
        
        Returns:
            Dict with keys: 'valid', 'cleaned_timestamp', 'original', 'warnings', 'method_used'
        """
        result = {
            'valid': False,
            'cleaned_timestamp': None,
            'original': timestamp_data,
            'warnings': [],
            'method_used': None,
            'field_name': field_name
        }
        
        if timestamp_data is None:
            result['warnings'].append(f"{field_name} is None")
            return result
        
        try:
            # Method 1: Already a datetime object
            if isinstance(timestamp_data, datetime):
                return self._validate_datetime_object(timestamp_data, result)
            
            # Method 2: String timestamp
            elif isinstance(timestamp_data, str):
                return self._validate_string_timestamp(timestamp_data, result)
            
            # Method 3: Numeric timestamp (Unix time)
            elif isinstance(timestamp_data, (int, float)):
                return self._validate_numeric_timestamp(timestamp_data, result)
            
            # Method 4: Pandas timestamp
            elif hasattr(timestamp_data, 'to_pydatetime'):
                dt = timestamp_data.to_pydatetime()
                return self._validate_datetime_object(dt, result)
            
            else:
                result['warnings'].append(f"Unsupported timestamp type: {type(timestamp_data)}")
                return result
                
        except Exception as e:
            result['warnings'].append(f"Timestamp validation error: {str(e)}")
            self.logger.error(f"Error validating timestamp {timestamp_data}: {e}")
            return result
    
    def validate_signal_timestamps(self, signal: Dict) -> Dict:
        """
        Validate all timestamp fields in a signal
        """
        validation_result = {
            'valid': True,
            'cleaned_signal': signal.copy(),
            'timestamp_fields_processed': [],
            'warnings': [],
            'errors': []
        }
        
        # Common timestamp field names to check
        timestamp_fields = [
            'timestamp', 'signal_timestamp', 'detection_time', 'created_at',
            'alert_timestamp', 'market_timestamp', 'processed_at', 'updated_at'
        ]
        
        for field_name in timestamp_fields:
            if field_name in signal and signal[field_name] is not None:
                field_result = self.validate_and_clean_timestamp(signal[field_name], field_name)
                
                validation_result['timestamp_fields_processed'].append({
                    'field': field_name,
                    'original': field_result['original'],
                    'valid': field_result['valid'],
                    'method': field_result['method_used']
                })
                
                if field_result['valid']:
                    validation_result['cleaned_signal'][field_name] = field_result['cleaned_timestamp']
                    self.logger.debug(f"✅ Cleaned {field_name}: {field_result['cleaned_timestamp']}")
                else:
                    validation_result['errors'].append(f"Invalid {field_name}: {field_result['warnings']}")
                    validation_result['valid'] = False
                
                validation_result['warnings'].extend(field_result['warnings'])
        
        return validation_result
    
    def create_safe_filename_timestamp(self, timestamp_data: Union[str, datetime, int, float] = None,
                                     signal: Dict = None) -> str:
        """
        Create a safe timestamp string for filenames
        Tries multiple timestamp sources and formats
        """
        if signal is not None:
            # Try multiple timestamp sources from signal
            timestamp_sources = [
                ('timestamp', signal.get('timestamp')),
                ('signal_timestamp', signal.get('signal_timestamp')),
                ('detection_time', signal.get('detection_time')),
                ('created_at', signal.get('created_at')),
                ('alert_timestamp', signal.get('alert_timestamp')),
                ('market_timestamp', signal.get('market_timestamp')),
            ]
            
            for source_name, timestamp_value in timestamp_sources:
                if timestamp_value is not None:
                    result = self.validate_and_clean_timestamp(timestamp_value, source_name)
                    if result['valid']:
                        cleaned_ts = result['cleaned_timestamp']
                        filename_ts = cleaned_ts.strftime('%Y%m%d_%H%M%S')
                        self.logger.debug(f"✅ Using {source_name} for filename: {filename_ts}")
                        return filename_ts
        
        # Try provided timestamp_data
        if timestamp_data is not None:
            result = self.validate_and_clean_timestamp(timestamp_data, "provided")
            if result['valid']:
                cleaned_ts = result['cleaned_timestamp']
                return cleaned_ts.strftime('%Y%m%d_%H%M%S')
        
        # Fallback to current time
        current_time = datetime.now()
        result = current_time.strftime('%Y%m%d_%H%M%S')
        
        epic = signal.get('epic', 'unknown') if signal else 'unknown'
        self.logger.warning(f"⚠️ No valid timestamp found for {epic}, using current time: {result}")
        
        return result
    
    def detect_stale_timestamps(self, signals: List[Dict]) -> Dict:
        """
        Detect stale timestamps across multiple signals
        """
        analysis = {
            'total_signals': len(signals),
            'stale_timestamps': 0,
            'epoch_timestamps': 0,
            'future_timestamps': 0,
            'valid_timestamps': 0,
            'stale_fields': {},
            'recommendations': []
        }
        
        current_time = datetime.now()
        
        for signal in signals:
            timestamp_fields = ['timestamp', 'market_timestamp', 'signal_timestamp', 'detection_time']
            
            for field in timestamp_fields:
                if field in signal and signal[field] is not None:
                    result = self.validate_and_clean_timestamp(signal[field], field)
                    
                    if result['valid']:
                        ts = result['cleaned_timestamp']
                        
                        # Check for epoch timestamps (1970)
                        if ts < self.epoch_threshold:
                            analysis['epoch_timestamps'] += 1
                            if field not in analysis['stale_fields']:
                                analysis['stale_fields'][field] = 0
                            analysis['stale_fields'][field] += 1
                        
                        # Check for future timestamps (more than 1 hour ahead)
                        elif ts > current_time + timedelta(hours=1):
                            analysis['future_timestamps'] += 1
                        
                        # Check for very old timestamps (more than 1 week old)
                        elif ts < current_time - timedelta(weeks=1):
                            analysis['stale_timestamps'] += 1
                            if field not in analysis['stale_fields']:
                                analysis['stale_fields'][field] = 0
                            analysis['stale_fields'][field] += 1
                        
                        else:
                            analysis['valid_timestamps'] += 1
        
        # Generate recommendations
        if analysis['epoch_timestamps'] > 0:
            analysis['recommendations'].append(
                f"Fix {analysis['epoch_timestamps']} epoch timestamps (1970-01-01) - likely uninitialized data"
            )
        
        if analysis['stale_timestamps'] > 0:
            analysis['recommendations'].append(
                f"Review {analysis['stale_timestamps']} stale timestamps (older than 1 week)"
            )
        
        if analysis['future_timestamps'] > 0:
            analysis['recommendations'].append(
                f"Check {analysis['future_timestamps']} future timestamps - possible timezone issues"
            )
        
        if analysis['stale_fields']:
            most_problematic = max(analysis['stale_fields'].items(), key=lambda x: x[1])
            analysis['recommendations'].append(
                f"Field '{most_problematic[0]}' has the most timestamp issues ({most_problematic[1]} occurrences)"
            )
        
        return analysis
    
    def _validate_datetime_object(self, dt: datetime, result: Dict) -> Dict:
        """Validate a datetime object"""
        if self.min_year <= dt.year <= self.max_year:
            result['valid'] = True
            result['cleaned_timestamp'] = dt
            result['method_used'] = 'datetime_object'
        else:
            result['warnings'].append(f"Year {dt.year} out of valid range ({self.min_year}-{self.max_year})")
        
        return result
    
    def _validate_string_timestamp(self, timestamp_str: str, result: Dict) -> Dict:
        """Validate and parse a string timestamp"""
        timestamp_str = timestamp_str.strip()
        
        # Quick rejection of obviously bad timestamps
        if timestamp_str.startswith('1970') or timestamp_str.startswith('1969'):
            result['warnings'].append(f"Epoch timestamp detected: {timestamp_str}")
            return result
        
        # Try each format
        for fmt in self.timestamp_formats:
            try:
                dt = datetime.strptime(timestamp_str, fmt)
                
                # Add timezone if missing and format suggests UTC
                if timestamp_str.endswith('Z') and dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                
                if self.min_year <= dt.year <= self.max_year:
                    result['valid'] = True
                    result['cleaned_timestamp'] = dt
                    result['method_used'] = f'string_format_{fmt}'
                    return result
                else:
                    result['warnings'].append(f"Parsed year {dt.year} out of range with format {fmt}")
                    
            except ValueError:
                continue
        
        # Try ISO format parsing as fallback
        try:
            dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            if self.min_year <= dt.year <= self.max_year:
                result['valid'] = True
                result['cleaned_timestamp'] = dt
                result['method_used'] = 'iso_format'
                return result
        except ValueError:
            pass
        
        # Try to clean and extract date parts
        cleaned_result = self._try_clean_string_timestamp(timestamp_str)
        if cleaned_result:
            result.update(cleaned_result)
            return result
        
        result['warnings'].append(f"Could not parse string timestamp: {timestamp_str}")
        return result
    
    def _validate_numeric_timestamp(self, timestamp_num: Union[int, float], result: Dict) -> Dict:
        """Validate a numeric (Unix) timestamp"""
        try:
            # Check if it looks like a reasonable Unix timestamp
            if 1600000000 <= timestamp_num <= 2000000000:  # 2020 to 2033
                dt = datetime.fromtimestamp(timestamp_num)
                if self.min_year <= dt.year <= self.max_year:
                    result['valid'] = True
                    result['cleaned_timestamp'] = dt
                    result['method_used'] = 'unix_timestamp'
                else:
                    result['warnings'].append(f"Unix timestamp year {dt.year} out of range")
            else:
                result['warnings'].append(f"Numeric timestamp {timestamp_num} out of valid Unix range")
        except (ValueError, OSError) as e:
            result['warnings'].append(f"Could not convert numeric timestamp {timestamp_num}: {e}")
        
        return result
    
    def _try_clean_string_timestamp(self, timestamp_str: str) -> Optional[Dict]:
        """Try to clean and extract a timestamp from a malformed string"""
        try:
            # Remove common separators and try to extract date/time parts
            cleaned = re.sub(r'[^\d]', '', timestamp_str)
            
            if len(cleaned) >= 8:
                year_part = cleaned[:4]
                try:
                    year = int(year_part)
                    if self.min_year <= year <= self.max_year:
                        # Try to construct a reasonable timestamp
                        if len(cleaned) >= 14:  # YYYYMMDDHHMMSS
                            formatted = f"{cleaned[:4]}-{cleaned[4:6]}-{cleaned[6:8]} {cleaned[8:10]}:{cleaned[10:12]}:{cleaned[12:14]}"
                        elif len(cleaned) >= 12:  # YYYYMMDDHHMM
                            formatted = f"{cleaned[:4]}-{cleaned[4:6]}-{cleaned[6:8]} {cleaned[8:10]}:{cleaned[10:12]}:00"
                        elif len(cleaned) >= 8:   # YYYYMMDD
                            formatted = f"{cleaned[:4]}-{cleaned[4:6]}-{cleaned[6:8]} 00:00:00"
                        else:
                            return None
                        
                        dt = datetime.strptime(formatted, '%Y-%m-%d %H:%M:%S')
                        return {
                            'valid': True,
                            'cleaned_timestamp': dt,
                            'method_used': 'string_cleaning',
                            'warnings': [f"Cleaned malformed timestamp: {timestamp_str} -> {formatted}"]
                        }
                except ValueError:
                    pass
        except Exception:
            pass
        
        return None
    
    def is_timestamp_reasonable(self, timestamp: datetime) -> bool:
        """Check if a timestamp is reasonable for trading signals"""
        current_time = datetime.now()
        
        # Not too old (more than 30 days)
        if timestamp < current_time - timedelta(days=30):
            return False
        
        # Not too far in the future (more than 1 day)
        if timestamp > current_time + timedelta(days=1):
            return False
        
        # Not epoch time
        if timestamp < self.epoch_threshold:
            return False
        
        return True
    
    def normalize_timezone(self, timestamp: datetime, target_timezone: str = 'UTC') -> datetime:
        """Normalize timestamp to a specific timezone"""
        try:
            if timestamp.tzinfo is None:
                # Assume UTC if no timezone info
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            
            if target_timezone == 'UTC':
                return timestamp.astimezone(timezone.utc)
            else:
                # For other timezones, you'd need pytz or zoneinfo
                self.logger.warning(f"Timezone conversion to {target_timezone} not implemented")
                return timestamp
                
        except Exception as e:
            self.logger.error(f"Error normalizing timezone: {e}")
            return timestamp


# Factory function for creating timestamp validator
def create_timestamp_validator() -> TimestampValidator:
    """Create timestamp validator with default configuration"""
    return TimestampValidator()


# Usage example and testing
if __name__ == "__main__":
    validator = create_timestamp_validator()
    
    # Test various timestamp formats
    test_timestamps = [
        datetime.now(),                          # datetime object
        "2025-07-31 07:15:00",                  # standard format
        "2025-07-31T07:15:00Z",                 # ISO format
        "1970-01-01 00:00:00",                  # epoch (should be invalid)
        "20250731_071500",                      # filename format
        1722406500,                             # Unix timestamp
        "invalid_timestamp",                    # invalid
        None,                                   # None
    ]
    
    print("🕐 Testing Timestamp Validator")
    print("=" * 50)
    
    for i, ts in enumerate(test_timestamps):
        result = validator.validate_and_clean_timestamp(ts, f"test_{i}")
        status = "✅ VALID" if result['valid'] else "❌ INVALID"
        print(f"{status} | Original: {ts}")
        print(f"     | Cleaned: {result['cleaned_timestamp']}")
        print(f"     | Method: {result['method_used']}")
        if result['warnings']:
            print(f"     | Warnings: {result['warnings']}")
        print()
    
    # Test signal timestamp validation
    test_signal = {
        'epic': 'CS.D.EURUSD.MINI.IP',
        'timestamp': datetime.now(),
        'market_timestamp': "1970-01-01 00:00:00",  # Stale timestamp
        'signal_timestamp': "2025-07-31T07:15:00Z",
        'invalid_field': "not_a_timestamp"
    }
    
    print("📊 Testing Signal Timestamp Validation")
    print("=" * 50)
    
    signal_result = validator.validate_signal_timestamps(test_signal)
    print(f"Overall valid: {signal_result['valid']}")
    print(f"Fields processed: {len(signal_result['timestamp_fields_processed'])}")
    
    for field_info in signal_result['timestamp_fields_processed']:
        print(f"  {field_info['field']}: {'✅' if field_info['valid'] else '❌'} ({field_info['method']})")
    
    if signal_result['errors']:
        print(f"Errors: {signal_result['errors']}")
    
    # Test filename timestamp creation
    print("\n📁 Testing Filename Timestamp Creation")
    print("=" * 50)
    
    filename_ts = validator.create_safe_filename_timestamp(signal=test_signal)
    print(f"Safe filename timestamp: {filename_ts}")
    
    # Test stale timestamp detection
    print("\n🔍 Testing Stale Timestamp Detection")
    print("=" * 50)
    
    test_signals = [
        {'timestamp': datetime.now(), 'market_timestamp': "1970-01-01"},
        {'timestamp': datetime.now() - timedelta(days=10)},
        {'timestamp': datetime.now() + timedelta(hours=5)},
        {'timestamp': datetime.now()}
    ]
    
    stale_analysis = validator.detect_stale_timestamps(test_signals)
    print(f"Total signals: {stale_analysis['total_signals']}")
    print(f"Stale timestamps: {stale_analysis['stale_timestamps']}")
    print(f"Epoch timestamps: {stale_analysis['epoch_timestamps']}")
    print(f"Future timestamps: {stale_analysis['future_timestamps']}")
    print(f"Valid timestamps: {stale_analysis['valid_timestamps']}")
    
    if stale_analysis['recommendations']:
        print("\nRecommendations:")
        for rec in stale_analysis['recommendations']:
            print(f"  • {rec}")