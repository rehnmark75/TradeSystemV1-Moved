"""
Threshold Manager - Database-backed MACD threshold configuration
Purpose: Ensure thresholds are always valid and prevent weak signals from passing through
"""

import logging
from typing import Dict, Optional, Any
from datetime import datetime
import json
from decimal import Decimal

class ThresholdManager:
    """Manages MACD thresholds in database to prevent weak signal acceptance"""
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
        self._ensure_table_exists()
        
    def _ensure_table_exists(self):
        """Ensure the forex_thresholds table exists"""
        try:
            # Check if table exists
            query = """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'forex_thresholds'
                )
            """
            result = self.db_manager.fetch_one(query)
            
            if not result or not result[0]:
                self.logger.warning("forex_thresholds table doesn't exist, creating it...")
                self._create_table()
                self._populate_defaults()
                
        except Exception as e:
            self.logger.error(f"Error checking forex_thresholds table: {e}")
    
    def _create_table(self):
        """Create the forex_thresholds table"""
        create_query = """
            CREATE TABLE IF NOT EXISTS forex_thresholds (
                epic VARCHAR(50) PRIMARY KEY,
                base_threshold DECIMAL(10,8) NOT NULL,
                strength_thresholds JSONB NOT NULL DEFAULT '{}',
                session_multipliers JSONB DEFAULT '{"london": 1.0, "new_york": 1.1, "asian": 0.8}',
                pair_type VARCHAR(50),
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT true,
                notes TEXT
            )
        """
        self.db_manager.execute_query(create_query)
        self.logger.info("Created forex_thresholds table")
    
    def _populate_defaults(self):
        """Populate default CORRECTED thresholds"""
        defaults = {
            # Major USD pairs - CORRECTED values
            'CS.D.EURUSD.MINI.IP': {
                'base_threshold': 0.00005,
                'strength_thresholds': {'moderate': 0.0004, 'strong': 0.0008, 'very_strong': 0.0012},
                'pair_type': 'eur_major',
                'notes': 'Corrected from 0.000008 (6x increase)'
            },
            'CS.D.GBPUSD.MINI.IP': {
                'base_threshold': 0.00008,
                'strength_thresholds': {'moderate': 0.0004, 'strong': 0.0008, 'very_strong': 0.0012},
                'pair_type': 'gbp_volatile',
                'notes': 'Corrected from 0.000015 (5x increase)'
            },
            'CS.D.AUDUSD.MINI.IP': {
                'base_threshold': 0.00006,
                'strength_thresholds': {'moderate': 0.0004, 'strong': 0.0008, 'very_strong': 0.0012},
                'pair_type': 'aud_commodity',
                'notes': 'Corrected from 0.000012 (5x increase)'
            },
            # JPY pairs - CORRECTED values
            'CS.D.USDJPY.MINI.IP': {
                'base_threshold': 0.008,
                'strength_thresholds': {'moderate': 0.005, 'strong': 0.010, 'very_strong': 0.020},
                'pair_type': 'usdjpy_stable',
                'notes': 'Corrected from 0.0008 (10x increase)'
            },
            'CS.D.EURJPY.MINI.IP': {
                'base_threshold': 0.010,
                'strength_thresholds': {'moderate': 0.005, 'strong': 0.010, 'very_strong': 0.020},
                'pair_type': 'eurjpy_cross',
                'notes': 'Corrected from 0.0012 (8x increase)'
            },
            'CS.D.GBPJPY.MINI.IP': {
                'base_threshold': 0.012,
                'strength_thresholds': {'moderate': 0.005, 'strong': 0.010, 'very_strong': 0.020},
                'pair_type': 'gbpjpy_volatile',
                'notes': 'Corrected from 0.0015 (8x increase)'
            }
        }
        
        for epic, config in defaults.items():
            self.upsert_threshold(
                epic=epic,
                base_threshold=config['base_threshold'],
                strength_thresholds=config['strength_thresholds'],
                pair_type=config['pair_type'],
                notes=config['notes']
            )
        
        self.logger.info(f"Populated {len(defaults)} default thresholds")
    
    def get_threshold(self, epic: str) -> float:
        """
        Get MACD threshold for an epic, NEVER returns None or invalid values
        
        Args:
            epic: Trading pair epic code
            
        Returns:
            float: Valid threshold value (never None, never <= 0)
        """
        try:
            # Try database first
            query = """
                SELECT base_threshold 
                FROM forex_thresholds 
                WHERE epic = %s AND is_active = true
            """
            result = self.db_manager.fetch_one(query, (epic,))
            
            if result and result[0] and float(result[0]) > 0:
                return float(result[0])
            
            # Database failed or returned invalid value
            self.logger.warning(f"No valid threshold in database for {epic}")
            
            # Use CORRECTED fallback based on pair type
            if 'JPY' in epic.upper():
                threshold = 0.008  # Corrected JPY threshold
                self.logger.warning(f"Using corrected JPY fallback for {epic}: {threshold}")
            else:
                threshold = 0.00008  # Corrected non-JPY threshold (NOT 0.00001!)
                self.logger.warning(f"Using corrected non-JPY fallback for {epic}: {threshold}")
            
            # Try to insert this fallback into database for next time
            self.upsert_threshold(
                epic=epic,
                base_threshold=threshold,
                notes=f"Auto-generated corrected fallback on {datetime.now()}"
            )
            
            return threshold
            
        except Exception as e:
            self.logger.error(f"Critical error getting threshold for {epic}: {e}")
            
            # Emergency fallback - NEVER return None or permissive values
            if 'JPY' in epic.upper():
                return 0.008
            else:
                return 0.00008
    
    def get_strength_thresholds(self, epic: str) -> Dict[str, float]:
        """Get strength thresholds for histogram magnitude classification"""
        try:
            query = """
                SELECT strength_thresholds 
                FROM forex_thresholds 
                WHERE epic = %s AND is_active = true
            """
            result = self.db_manager.fetch_one(query, (epic,))
            
            if result and result[0]:
                return result[0]
            
            # Return corrected defaults
            if 'JPY' in epic.upper():
                return {'moderate': 0.005, 'strong': 0.010, 'very_strong': 0.020}
            else:
                return {'moderate': 0.0004, 'strong': 0.0008, 'very_strong': 0.0012}
                
        except Exception as e:
            self.logger.error(f"Error getting strength thresholds for {epic}: {e}")
            # Return safe defaults
            if 'JPY' in epic.upper():
                return {'moderate': 0.005, 'strong': 0.010, 'very_strong': 0.020}
            else:
                return {'moderate': 0.0004, 'strong': 0.0008, 'very_strong': 0.0012}
    
    def upsert_threshold(self, epic: str, base_threshold: float, 
                        strength_thresholds: Dict = None,
                        pair_type: str = None, notes: str = None):
        """Insert or update threshold configuration"""
        try:
            query = """
                INSERT INTO forex_thresholds 
                (epic, base_threshold, strength_thresholds, pair_type, notes, last_updated)
                VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (epic) DO UPDATE SET
                    base_threshold = EXCLUDED.base_threshold,
                    strength_thresholds = COALESCE(EXCLUDED.strength_thresholds, forex_thresholds.strength_thresholds),
                    pair_type = COALESCE(EXCLUDED.pair_type, forex_thresholds.pair_type),
                    notes = COALESCE(EXCLUDED.notes, forex_thresholds.notes),
                    last_updated = CURRENT_TIMESTAMP
            """
            
            strength_json = json.dumps(strength_thresholds) if strength_thresholds else '{}'
            
            self.db_manager.execute_query(query, (
                epic, base_threshold, strength_json, pair_type, notes
            ))
            
            self.logger.info(f"Updated threshold for {epic}: {base_threshold}")
            
        except Exception as e:
            self.logger.error(f"Error upserting threshold for {epic}: {e}")
    
    def validate_all_thresholds(self) -> Dict[str, Any]:
        """Validate all thresholds are within acceptable ranges"""
        try:
            query = """
                SELECT epic, base_threshold, pair_type
                FROM forex_thresholds
                WHERE is_active = true
            """
            results = self.db_manager.fetch_all(query)
            
            validation_report = {
                'valid': [],
                'invalid': [],
                'missing': [],
                'summary': {}
            }
            
            # Expected epics
            expected_epics = [
                'CS.D.EURUSD.MINI.IP', 'CS.D.GBPUSD.MINI.IP', 'CS.D.AUDUSD.MINI.IP',
                'CS.D.NZDUSD.MINI.IP', 'CS.D.USDCAD.MINI.IP', 'CS.D.USDCHF.MINI.IP',
                'CS.D.USDJPY.MINI.IP', 'CS.D.EURJPY.MINI.IP', 'CS.D.GBPJPY.MINI.IP'
            ]
            
            found_epics = []
            
            for row in results:
                epic, threshold, pair_type = row
                threshold = float(threshold)
                found_epics.append(epic)
                
                # Validate threshold is not too permissive
                if 'JPY' in epic.upper():
                    if threshold < 0.005:  # Too permissive for JPY
                        validation_report['invalid'].append({
                            'epic': epic,
                            'threshold': threshold,
                            'reason': f'JPY threshold too low (< 0.005)',
                            'recommended': 0.008
                        })
                    else:
                        validation_report['valid'].append(epic)
                else:
                    if threshold < 0.00005:  # Too permissive for non-JPY
                        validation_report['invalid'].append({
                            'epic': epic,
                            'threshold': threshold,
                            'reason': f'Non-JPY threshold too low (< 0.00005)',
                            'recommended': 0.00008
                        })
                    else:
                        validation_report['valid'].append(epic)
            
            # Check for missing epics
            for epic in expected_epics:
                if epic not in found_epics:
                    validation_report['missing'].append(epic)
            
            validation_report['summary'] = {
                'total_configured': len(found_epics),
                'valid_count': len(validation_report['valid']),
                'invalid_count': len(validation_report['invalid']),
                'missing_count': len(validation_report['missing'])
            }
            
            return validation_report
            
        except Exception as e:
            self.logger.error(f"Error validating thresholds: {e}")
            return {'error': str(e)}
    
    def get_all_thresholds(self) -> Dict[str, float]:
        """Get all active thresholds"""
        try:
            query = """
                SELECT epic, base_threshold
                FROM forex_thresholds
                WHERE is_active = true
                ORDER BY epic
            """
            results = self.db_manager.fetch_all(query)
            
            return {epic: float(threshold) for epic, threshold in results}
            
        except Exception as e:
            self.logger.error(f"Error getting all thresholds: {e}")
            return {}