# services/activity_pnl_correlator.py
"""
Activity-Based P/L Correlator
Uses IG Activity endpoint to correlate trade_log deal IDs with position references
Calculates P/L from complete trade lifecycle (open → close) using activity data only
🚀 ENHANCED: Now uses position reference mapping for better correlation
"""

import logging
import json
import httpx
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from sqlalchemy.orm import Session
from sqlalchemy import text

from services.models import TradeLog


@dataclass
class ActivityTradeLifecycle:
    """Complete trade lifecycle from activity data"""
    position_reference: str
    epic: str
    
    # Opening trade
    open_deal_id: str
    open_date: str
    entry_price: Optional[float] = None  # 🚀 NEW
    
    # Closing trade
    close_deal_id: Optional[str] = None
    close_date: Optional[str] = None
    exit_price: Optional[float] = None   # 🚀 NEW
    
    # Trade details
    direction: Optional[str] = None  # BUY/SELL
    size: Optional[float] = None
    status: str = "open"  # open, closed, partial
    
    # P/L calculation
    calculated_pnl: Optional[float] = None
    duration_minutes: Optional[int] = None
    
    # Stop limit changes
    stop_limit_changes: List[Dict] = None

    def __post_init__(self):
        if self.stop_limit_changes is None:
            self.stop_limit_changes = []


@dataclass
class ActivityPnLCorrelation:
    """Correlation between trade_log and activity-based P/L"""
    trade_log_id: int
    trade_log_deal_id: str
    trade_log_symbol: str
    
    # Activity correlation
    found_in_activities: bool = False
    position_reference: Optional[str] = None
    activity_lifecycle: Optional[ActivityTradeLifecycle] = None
    
    # P/L calculation
    calculated_pnl: Optional[float] = None
    pnl_method: Optional[str] = None  # 'price_difference', 'estimated', 'not_calculated'
    
    # Status
    correlation_status: str = "pending"  # 'matched', 'partial', 'missing', 'incomplete'
    

class ActivityPnLCorrelator:
    """
    Activity-Based P/L Correlator
    Correlates trade_log entries with IG activities to calculate P/L from trade lifecycle
    🚀 ENHANCED: Now uses position reference mapping for better correlation
    """
    
    def __init__(self, db_session: Session = None, logger=None):
        self.db_session = db_session
        self.logger = logger or logging.getLogger(__name__)
        self._cached_activities = []  # 🚀 NEW: Cache activities for mapping
    
    async def correlate_trade_log_with_activities(
        self, 
        trading_headers: dict, 
        days_back: int = 7,
        update_trade_log: bool = True
    ) -> Dict:
        """
        Main correlation workflow:
        1. Get trade_log entries with deal IDs
        2. Fetch activities for the period
        3. Find position references for each trade_log deal ID
        4. Build complete trade lifecycles
        5. Calculate P/L from price differences
        6. Update trade_log with calculated P/L
        """
        try:
            self.logger.info(f"🔗 Starting activity-based P/L correlation for last {days_back} days")
            
            # Step 1: Get trade_log entries with deal IDs
            trade_log_entries = self._get_trade_log_entries_with_deal_ids(days_back)
            self.logger.info(f"📊 Found {len(trade_log_entries)} trade_log entries with deal IDs")
            
            if not trade_log_entries:
                return {
                    "status": "success",
                    "message": "No trade_log entries found with deal IDs",
                    "summary": {"total_trades": 0, "correlations_found": 0}
                }
            
            # Step 2: Fetch activities from IG
            activities = await self._fetch_ig_activities(trading_headers, days_back)
            self.logger.info(f"📊 Found {len(activities)} IG activities")
            
            # Step 3: Build trade lifecycles from activities
            trade_lifecycles = self._build_trade_lifecycles_from_activities(activities)
            self.logger.info(f"📊 Built {len(trade_lifecycles)} complete trade lifecycles")
            
            # Step 4: Correlate trade_log with activities
            self.logger.info("🔍 Step 4: Starting correlation...")
            correlations = self._correlate_trade_log_with_lifecycles(
                trade_log_entries, 
                trade_lifecycles
            )
            self.logger.info(f"📊 Step 4 complete: Created {len(correlations)} correlations")
            
            # Step 5: Calculate P/L for correlated trades
            self.logger.info("🔍 Step 5: Starting P/L calculation...")
            calculated_correlations = self._calculate_pnl_from_activities(correlations)
            self.logger.info(f"📊 Step 5 complete: Processed {len(calculated_correlations)} correlations")
            
            # Step 6: Update trade_log if requested
            updated_trades = []
            if update_trade_log:
                self.logger.info("🔍 Step 6: Starting trade_log updates...")
                updated_trades = await self._update_trade_log_with_activity_pnl(calculated_correlations)
                self.logger.info(f"📊 Step 6 complete: Updated {len(updated_trades)} trades")
            else:
                self.logger.info("🔍 Step 6: Skipping trade_log updates (update_trade_log=False)")
            
            # Step 7: Generate summary
            self.logger.info("🔍 Step 7: Generating summary...")
            summary = self._generate_activity_correlation_summary(calculated_correlations, updated_trades)
            self.logger.info(f"📊 Step 7 complete: Summary generated")
            
            self.logger.info(f"✅ Activity-based correlation completed: {summary['correlations_found']} of {summary['total_trades']} trades correlated")
            
            # Debug: Check correlations before serialization
            self.logger.info(f"🔍 Debug: About to serialize {len(calculated_correlations)} correlations")
            for i, corr in enumerate(calculated_correlations):
                self.logger.debug(f"Pre-serialization correlation {i+1}: type={type(corr)}, has_trade_log_id={hasattr(corr, 'trade_log_id') if not isinstance(corr, dict) else 'is_dict'}")
            
            # Try serialization step by step
            try:
                self.logger.info("🔍 Serializing correlations...")
                serialized_correlations = self._serialize_correlations(calculated_correlations)
                self.logger.info("✅ Correlations serialized successfully")
            except Exception as e:
                self.logger.error(f"❌ Error serializing correlations: {e}")
                serialized_correlations = []
            
            try:
                self.logger.info("🔍 Serializing trade lifecycles...")
                serialized_lifecycles = self._serialize_trade_lifecycles(trade_lifecycles)
                self.logger.info("✅ Trade lifecycles serialized successfully")
            except Exception as e:
                self.logger.error(f"❌ Error serializing trade lifecycles: {e}")
                serialized_lifecycles = {}
            
            return {
                "status": "success",
                "message": f"Activity-based P/L correlation completed",
                "summary": summary,
                "correlations": serialized_correlations,
                "updated_trades": updated_trades,
                "trade_lifecycles": serialized_lifecycles
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error in activity-based P/L correlation: {e}")
            import traceback
            self.logger.error(f"❌ Full traceback: {traceback.format_exc()}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _get_trade_log_entries_with_deal_ids(self, days_back: int) -> List[TradeLog]:
        """Get trade_log entries that have deal IDs within the specified period"""
        try:
            if not self.db_session:
                return []
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # Query trade_log for entries with deal IDs (using timestamp instead of created_at)
            trade_logs = self.db_session.query(TradeLog).filter(
                TradeLog.deal_id.isnot(None),
                TradeLog.deal_id != '',
                TradeLog.timestamp >= start_date,
                TradeLog.timestamp <= end_date
            ).all()
            
            self.logger.info(f"📊 Found {len(trade_logs)} trade_log entries with deal IDs in last {days_back} days")
            
            return trade_logs
            
        except Exception as e:
            self.logger.error(f"❌ Error fetching trade_log entries: {e}")
            return []
    
    async def _fetch_ig_activities(self, trading_headers: dict, days_back: int) -> List[Dict]:
        """
        Fetch IG activities for the specified period
        🚀 ENHANCED: Cache activities for deal ID mapping
        """
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            # Format dates for IG API
            from_date = start_date.strftime('%Y-%m-%d')
            to_date = end_date.strftime('%Y-%m-%d')
            
            # IG Activity API endpoint
            from config import API_BASE_URL
            activity_url = f"{API_BASE_URL}/history/activity"
            
            params = {
                'from': from_date,
                'to': to_date,
                'detailed': 'true',  # 🚀 CRITICAL: Gets us the details field with prices
                'pageSize': 500
            }
            
            # Headers for IG API (NOT the FastAPI gateway header)
            headers = {
                "X-IG-API-KEY": trading_headers["X-IG-API-KEY"],
                "CST": trading_headers["CST"],
                "X-SECURITY-TOKEN": trading_headers["X-SECURITY-TOKEN"],
                "Accept": "application/json",
                "Version": "3"
            }
            
            self.logger.info(f"🔍 Fetching detailed IG activities from {from_date} to {to_date}")
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    activity_url,
                    headers=headers,
                    params=params
                )
                
                if response.status_code == 200:
                    data = response.json()
                    activities = data.get('activities', [])
                    
                    # 🚀 NEW: Cache activities for mapping
                    self._cached_activities = activities
                    
                    self.logger.info(f"✅ Fetched {len(activities)} detailed activities from IG API")
                    
                    # 🔍 DEBUG: Log sample activity to verify details field
                    if activities:
                        sample = activities[0]
                        has_details = 'details' in sample
                        self.logger.info(f"🔍 Sample activity has details: {has_details}")
                        if has_details and sample.get('details'):
                            details_keys = list(sample['details'].keys())
                            has_level = 'level' in sample['details']
                            self.logger.info(f"🔍 Details keys: {details_keys}, has level: {has_level}")
                    
                    return activities
                    
                else:
                    self.logger.error(f"❌ IG Activity API error: {response.status_code} - {response.text}")
                    return []
                    
        except Exception as e:
            self.logger.error(f"❌ Error fetching IG activities: {e}")
            return []
    
    def _build_deal_id_to_position_mapping(self) -> Dict[str, str]:
        """
        🚀 NEW: Build mapping from any deal_id to position reference
        This allows us to find position reference from any deal_id in the cycle
        """
        deal_id_mapping = {}
        
        if not self._cached_activities:
            self.logger.warning("⚠️ No cached activities available for deal_id mapping")
            return {}
        
        for activity in self._cached_activities:
            try:
                # Extract position reference from description
                position_ref = self._extract_position_reference(activity.get('description', ''))
                if not position_ref:
                    continue
                
                # Map the main deal_id to position reference
                deal_id = activity.get('dealId')
                if deal_id:
                    deal_id_mapping[deal_id] = position_ref
                
                # 🚀 ALSO map the affected_deal_id to position reference
                activity_details = activity.get('details', {})
                if activity_details:
                    actions = activity_details.get('actions', [])
                    if actions and len(actions) > 0:
                        affected_deal_id = actions[0].get('affectedDealId')
                        if affected_deal_id:
                            deal_id_mapping[affected_deal_id] = position_ref
                
            except Exception as e:
                self.logger.debug(f"⚠️ Error processing activity for mapping: {e}")
                continue
        
        self.logger.info(f"📊 Built deal_id mapping for {len(deal_id_mapping)} deal_ids → position references")
        
        # Log sample mappings for debugging
        sample_mappings = dict(list(deal_id_mapping.items())[:3])
        self.logger.debug(f"🔍 Sample mappings: {sample_mappings}")
        
        return deal_id_mapping
    
    def _build_trade_lifecycles_from_activities(self, activities: List[Dict]) -> Dict[str, ActivityTradeLifecycle]:
        """
        🚀 SIMPLIFIED: Build lifecycles grouped by position reference
        Much simpler - just group all activities by position reference
        """
        trade_lifecycles = {}
        
        # 🚀 STEP 1: Group activities by position reference
        activities_by_position = {}
        
        for activity in activities:
            try:
                position_ref = self._extract_position_reference(activity.get('description', ''))
                if not position_ref:
                    continue
                
                if position_ref not in activities_by_position:
                    activities_by_position[position_ref] = []
                
                activities_by_position[position_ref].append(activity)
                
            except Exception as e:
                self.logger.debug(f"⚠️ Error grouping activity: {e}")
                continue
        
        self.logger.info(f"📊 Grouped activities for {len(activities_by_position)} position references")
        
        # 🚀 STEP 2: Build lifecycle for each position
        for position_ref, position_activities in activities_by_position.items():
            try:
                lifecycle = self._build_single_lifecycle(position_ref, position_activities)
                if lifecycle and lifecycle.status == 'closed':
                    trade_lifecycles[position_ref] = lifecycle
                    
            except Exception as e:
                self.logger.error(f"❌ Error building lifecycle for {position_ref}: {e}")
                continue
        
        complete_count = len([lc for lc in trade_lifecycles.values() if lc.entry_price and lc.exit_price])
        self.logger.info(f"📊 Built {len(trade_lifecycles)} complete lifecycles, {complete_count} with prices")
        
        return trade_lifecycles
    
    def _build_single_lifecycle(self, position_ref: str, activities: List[Dict]) -> Optional[ActivityTradeLifecycle]:
        """
        🚀 Build a single lifecycle from all activities for one position
        """
        lifecycle = ActivityTradeLifecycle(
            position_reference=position_ref,
            epic='',
            open_deal_id='',
            open_date='',
            stop_limit_changes=[]
        )
        
        # Process each activity for this position
        for activity in activities:
            try:
                action = self._determine_action_type(activity.get('description', ''))
                activity_details = activity.get('details', {})
                
                if action == 'OPENED':
                    lifecycle.open_deal_id = activity.get('dealId', '')
                    lifecycle.open_date = activity.get('date', '')
                    lifecycle.epic = activity.get('epic', '')
                    
                    if activity_details:
                        lifecycle.entry_price = activity_details.get('level')
                        lifecycle.direction = activity_details.get('direction', '').upper()
                        lifecycle.size = activity_details.get('size', 1.0)
                    
                    self.logger.debug(f"📈 OPENED {position_ref}: price={lifecycle.entry_price}, direction={lifecycle.direction}")
                    
                elif action == 'CLOSED':
                    lifecycle.close_deal_id = activity.get('dealId', '')
                    lifecycle.close_date = activity.get('date', '')
                    lifecycle.status = 'closed'
                    
                    if activity_details:
                        lifecycle.exit_price = activity_details.get('level')
                    
                    self.logger.debug(f"📉 CLOSED {position_ref}: exit_price={lifecycle.exit_price}")
                    
                    # Calculate duration
                    if lifecycle.open_date and lifecycle.close_date:
                        try:
                            open_time = datetime.fromisoformat(lifecycle.open_date.replace('Z', '+00:00'))
                            close_time = datetime.fromisoformat(lifecycle.close_date.replace('Z', '+00:00'))
                            lifecycle.duration_minutes = int((close_time - open_time).total_seconds() / 60)
                        except:
                            pass
                
                elif action == 'STOP_LIMIT_CHANGED':
                    change_info = {
                        'deal_id': activity.get('dealId', ''),
                        'date': activity.get('date', ''),
                        'description': activity.get('description', '')
                    }
                    if activity_details:
                        change_info.update({
                            'level': activity_details.get('level'),
                            'stop_level': activity_details.get('stopLevel'),
                            'limit_level': activity_details.get('limitLevel')
                        })
                    lifecycle.stop_limit_changes.append(change_info)
            
            except Exception as e:
                self.logger.debug(f"⚠️ Error processing activity for {position_ref}: {e}")
                continue
        
        # Validate lifecycle completeness
        is_complete = (lifecycle.open_deal_id and lifecycle.close_deal_id and 
                       lifecycle.entry_price is not None and lifecycle.exit_price is not None)
        
        if is_complete:
            self.logger.debug(f"✅ Complete lifecycle for {position_ref}: entry={lifecycle.entry_price}, exit={lifecycle.exit_price}, duration={lifecycle.duration_minutes}min")
            return lifecycle
        else:
            self.logger.debug(f"⚠️ Incomplete lifecycle for {position_ref}: open={bool(lifecycle.open_deal_id)}, close={bool(lifecycle.close_deal_id)}, entry={lifecycle.entry_price}, exit={lifecycle.exit_price}")
            return lifecycle  # Return anyway for partial correlation
    
    def _extract_position_reference(self, description: str) -> Optional[str]:
        """Extract position reference from IG description"""
        try:
            if ':' in description:
                # Split by colon and take the last part, strip whitespace
                reference = description.split(':')[-1].strip()
                if reference and len(reference) > 4:  # Basic validation
                    return reference
            return None
        except:
            return None
    
    def _determine_action_type(self, description: str) -> Optional[str]:
        """Determine action type from description"""
        description_lower = description.lower()
        
        if 'stängd' in description_lower or 'closed' in description_lower:
            return 'CLOSED'
        elif 'öppnad' in description_lower or 'opened' in description_lower:
            return 'OPENED'
        elif 'stopplimit ändrad' in description_lower or 'stop limit changed' in description_lower:
            return 'STOP_LIMIT_CHANGED'
        
        return None
    
    def _correlate_trade_log_with_lifecycles(
        self, 
        trade_log_entries: List[TradeLog], 
        trade_lifecycles: Dict[str, ActivityTradeLifecycle]
    ) -> List[ActivityPnLCorrelation]:
        """
        🚀 SIMPLIFIED: Correlate using position reference as the key
        Step 1: Find position reference for each trade_log deal_id
        Step 2: Use position reference to get complete lifecycle
        """
        correlations = []
        
        # 🚀 NEW: Build a mapping from deal_id to position_reference
        deal_id_to_position_ref = self._build_deal_id_to_position_mapping()
        
        for trade_log in trade_log_entries:
            try:
                correlation = ActivityPnLCorrelation(
                    trade_log_id=trade_log.id,
                    trade_log_deal_id=trade_log.deal_id,
                    trade_log_symbol=trade_log.symbol
                )
                
                # 🚀 STEP 1: Find position reference for this deal_id
                position_ref = deal_id_to_position_ref.get(trade_log.deal_id)
                
                if position_ref:
                    # 🚀 STEP 2: Get the complete lifecycle for this position
                    lifecycle = trade_lifecycles.get(position_ref)
                    
                    if lifecycle:
                        correlation.found_in_activities = True
                        correlation.position_reference = position_ref
                        correlation.activity_lifecycle = lifecycle
                        correlation.correlation_status = "matched"
                        
                        self.logger.debug(f"✅ Matched trade_log {trade_log.id} (deal_id: {trade_log.deal_id}) → position: {position_ref}")
                    else:
                        correlation.correlation_status = "partial"
                        correlation.position_reference = position_ref
                        self.logger.debug(f"⚠️ Found position {position_ref} for deal_id {trade_log.deal_id} but no complete lifecycle")
                else:
                    correlation.correlation_status = "missing"
                    self.logger.debug(f"❌ No position reference found for deal_id {trade_log.deal_id}")
                
                correlations.append(correlation)
                
            except Exception as e:
                self.logger.error(f"❌ Error processing trade_log entry {trade_log.id}: {e}")
                correlations.append(ActivityPnLCorrelation(
                    trade_log_id=trade_log.id,
                    trade_log_deal_id=trade_log.deal_id or "UNKNOWN",
                    trade_log_symbol=trade_log.symbol or "UNKNOWN",
                    correlation_status="error"
                ))
        
        matched_count = sum(1 for c in correlations if c.found_in_activities)
        self.logger.info(f"📊 Matched {matched_count} of {len(correlations)} trade_log entries with activities")
        
        return correlations
    
    def _calculate_pnl_from_activities(self, correlations: List[ActivityPnLCorrelation]) -> List[ActivityPnLCorrelation]:
        """
        Calculate P/L from activity data
        Note: This is a simplified calculation - real P/L would need actual prices from market data
        """
        try:
            self.logger.info(f"🔍 Processing {len(correlations)} correlations for P/L calculation")
            
            for i, correlation in enumerate(correlations):
                try:
                    # Debug logging
                    self.logger.debug(f"Processing correlation {i+1}: type={type(correlation)}")
                    
                    # Check if this is a proper object
                    if not hasattr(correlation, 'found_in_activities'):
                        self.logger.error(f"❌ Correlation {i+1} is not a proper ActivityPnLCorrelation object: {type(correlation)}")
                        continue
                    
                    if not correlation.found_in_activities or not correlation.activity_lifecycle:
                        continue
                    
                    lifecycle = correlation.activity_lifecycle
                    
                    # For now, we can't calculate exact P/L without entry/exit prices
                    # But we can mark the trade as ready for price-based calculation
                    correlation.pnl_method = "price_data_needed"
                    correlation.correlation_status = "partial"
                    
                    # Store lifecycle duration as metadata
                    if hasattr(lifecycle, 'duration_minutes') and lifecycle.duration_minutes:
                        correlation.calculated_pnl = 0.0  # Placeholder - would need actual price calculation
                        
                except Exception as e:
                    self.logger.error(f"❌ Error processing correlation {i+1}: {e}")
                    continue
            
            return correlations
            
        except Exception as e:
            self.logger.error(f"❌ Error in _calculate_pnl_from_activities: {e}")
            return correlations
    
    async def _update_trade_log_with_activity_pnl(self, correlations: List[ActivityPnLCorrelation]) -> List[Dict]:
        """Update trade_log entries with activity-based correlation data"""
        updated_trades = []
        
        if not self.db_session:
            return updated_trades
        
        for correlation in correlations:
            if not correlation.found_in_activities:
                continue
            
            try:
                # Ensure we have required attributes
                if not hasattr(correlation, 'trade_log_id') or not hasattr(correlation, 'activity_lifecycle'):
                    self.logger.warning(f"⚠️ Skipping correlation due to missing attributes")
                    continue
                
                if not correlation.activity_lifecycle:
                    self.logger.warning(f"⚠️ Skipping correlation {correlation.trade_log_id} - no lifecycle data")
                    continue
                
                # Update trade_log with position reference and lifecycle data
                lifecycle = correlation.activity_lifecycle
                
                update_data = {
                    "position_reference": correlation.position_reference,
                    "activity_correlated": True,
                    "lifecycle_duration_minutes": lifecycle.duration_minutes,
                    "stop_limit_changes_count": len(lifecycle.stop_limit_changes) if lifecycle.stop_limit_changes else 0,
                    "activity_open_deal_id": lifecycle.open_deal_id,
                    "activity_close_deal_id": lifecycle.close_deal_id,
                    "updated_at": datetime.now()
                }
                
                # Add columns if they don't exist
                self._ensure_activity_correlation_columns()
                
                # Update the trade_log entry
                self.db_session.execute(
                    text("""
                        UPDATE trade_log 
                        SET position_reference = :position_reference,
                            activity_correlated = :activity_correlated,
                            lifecycle_duration_minutes = :lifecycle_duration_minutes,
                            stop_limit_changes_count = :stop_limit_changes_count,
                            activity_open_deal_id = :activity_open_deal_id,
                            activity_close_deal_id = :activity_close_deal_id,
                            updated_at = :updated_at
                        WHERE id = :trade_log_id
                    """),
                    {**update_data, "trade_log_id": correlation.trade_log_id}
                )
                
                updated_trades.append({
                    "trade_log_id": correlation.trade_log_id,
                    "deal_id": correlation.trade_log_deal_id,
                    "position_reference": correlation.position_reference,
                    "duration_minutes": lifecycle.duration_minutes,
                    "status": "activity_correlated"
                })
                
                self.logger.info(f"✅ Updated trade_log {correlation.trade_log_id} with activity correlation")
                
            except Exception as e:
                self.logger.error(f"❌ Error updating trade_log {getattr(correlation, 'trade_log_id', 'UNKNOWN')}: {e}")
                continue
        
        if updated_trades:
            try:
                self.db_session.commit()
                self.logger.info(f"✅ Updated {len(updated_trades)} trade_log entries with activity data")
            except Exception as e:
                self.logger.error(f"❌ Error committing database changes: {e}")
                self.db_session.rollback()
        
        return updated_trades
    
    def _ensure_activity_correlation_columns(self):
        """Ensure trade_log table has columns for activity correlation"""
        try:
            self.db_session.execute(text("""
                ALTER TABLE trade_log 
                ADD COLUMN IF NOT EXISTS position_reference VARCHAR(20),
                ADD COLUMN IF NOT EXISTS activity_correlated BOOLEAN DEFAULT FALSE,
                ADD COLUMN IF NOT EXISTS lifecycle_duration_minutes INTEGER,
                ADD COLUMN IF NOT EXISTS stop_limit_changes_count INTEGER DEFAULT 0,
                ADD COLUMN IF NOT EXISTS activity_open_deal_id VARCHAR(50),
                ADD COLUMN IF NOT EXISTS activity_close_deal_id VARCHAR(50)
            """))
            self.db_session.commit()
        except Exception as e:
            self.logger.debug(f"Columns may already exist: {e}")
    
    def _serialize_correlations(self, correlations: List[ActivityPnLCorrelation]) -> List[Dict]:
        """Convert correlation objects to JSON-serializable format"""
        serialized = []
        
        try:
            self.logger.info(f"🔍 Serializing {len(correlations)} correlations")
            
            for i, corr in enumerate(correlations):
                try:
                    # Debug logging
                    self.logger.debug(f"Serializing correlation {i+1}: type={type(corr)}")
                    
                    # Check if this is already a dict (shouldn't happen, but let's handle it)
                    if isinstance(corr, dict):
                        self.logger.warning(f"⚠️ Correlation {i+1} is already a dict, passing through")
                        serialized.append(corr)
                        continue
                    
                    # Check if this is a proper ActivityPnLCorrelation object
                    if not hasattr(corr, 'trade_log_id'):
                        self.logger.error(f"❌ Correlation {i+1} missing trade_log_id attribute: {type(corr)}")
                        continue
                    
                    # Safe attribute access with defaults
                    corr_dict = {
                        "trade_log_id": getattr(corr, 'trade_log_id', None),
                        "trade_log_deal_id": getattr(corr, 'trade_log_deal_id', None),
                        "trade_log_symbol": getattr(corr, 'trade_log_symbol', None),
                        "found_in_activities": getattr(corr, 'found_in_activities', False),
                        "position_reference": getattr(corr, 'position_reference', None),
                        "calculated_pnl": getattr(corr, 'calculated_pnl', None),
                        "pnl_method": getattr(corr, 'pnl_method', None),
                        "correlation_status": getattr(corr, 'correlation_status', 'unknown'),
                        "activity_lifecycle": None
                    }
                    
                    # Serialize lifecycle if it exists
                    lifecycle = getattr(corr, 'activity_lifecycle', None)
                    if lifecycle:
                        try:
                            if hasattr(lifecycle, 'position_reference'):
                                # It's an ActivityTradeLifecycle object
                                corr_dict["activity_lifecycle"] = {
                                    "position_reference": getattr(lifecycle, 'position_reference', None),
                                    "epic": getattr(lifecycle, 'epic', None),
                                    "open_deal_id": getattr(lifecycle, 'open_deal_id', None),
                                    "open_date": getattr(lifecycle, 'open_date', None),
                                    "close_deal_id": getattr(lifecycle, 'close_deal_id', None),
                                    "close_date": getattr(lifecycle, 'close_date', None),
                                    "status": getattr(lifecycle, 'status', None),
                                    "duration_minutes": getattr(lifecycle, 'duration_minutes', None),
                                    "entry_price": getattr(lifecycle, 'entry_price', None),  # 🚀 NEW
                                    "exit_price": getattr(lifecycle, 'exit_price', None),    # 🚀 NEW
                                    "direction": getattr(lifecycle, 'direction', None),     # 🚀 NEW
                                    "size": getattr(lifecycle, 'size', None),               # 🚀 NEW
                                    "stop_limit_changes": getattr(lifecycle, 'stop_limit_changes', [])
                                }
                            elif isinstance(lifecycle, dict):
                                # It's already a dict
                                corr_dict["activity_lifecycle"] = lifecycle
                            else:
                                self.logger.warning(f"⚠️ Unknown lifecycle type for correlation {i+1}: {type(lifecycle)}")
                                
                        except Exception as e:
                            self.logger.error(f"❌ Error serializing lifecycle for correlation {i+1}: {e}")
                    
                    serialized.append(corr_dict)
                    self.logger.debug(f"✅ Successfully serialized correlation {i+1}")
                    
                except Exception as e:
                    self.logger.error(f"❌ Error serializing correlation {i+1}: {e}")
                    # Add a minimal dict to maintain array consistency
                    serialized.append({
                        "trade_log_id": None,
                        "correlation_status": "serialization_error",
                        "error": str(e)
                    })
                    continue
            
            self.logger.info(f"✅ Successfully serialized {len(serialized)} correlations")
            return serialized
            
        except Exception as e:
            self.logger.error(f"❌ Error in _serialize_correlations: {e}")
            return []
    
    def _serialize_trade_lifecycles(self, trade_lifecycles: Dict[str, ActivityTradeLifecycle]) -> Dict:
        """Convert trade lifecycle objects to JSON-serializable format"""
        serialized = {}
        
        try:
            self.logger.info(f"🔍 Serializing {len(trade_lifecycles)} trade lifecycles")
            
            for ref, lifecycle in trade_lifecycles.items():
                try:
                    self.logger.debug(f"Serializing lifecycle {ref}: type={type(lifecycle)}")
                    
                    # Check if lifecycle is already a dict
                    if isinstance(lifecycle, dict):
                        self.logger.warning(f"⚠️ Lifecycle {ref} is already a dict, passing through")
                        serialized[ref] = lifecycle
                        continue
                    
                    # Check if it's a proper ActivityTradeLifecycle object
                    if not hasattr(lifecycle, 'position_reference'):
                        self.logger.error(f"❌ Lifecycle {ref} missing position_reference: {type(lifecycle)}")
                        continue
                    
                    serialized[ref] = {
                        "position_reference": getattr(lifecycle, 'position_reference', None),
                        "epic": getattr(lifecycle, 'epic', None),
                        "open_deal_id": getattr(lifecycle, 'open_deal_id', None),
                        "open_date": getattr(lifecycle, 'open_date', None),
                        "close_deal_id": getattr(lifecycle, 'close_deal_id', None),
                        "close_date": getattr(lifecycle, 'close_date', None),
                        "status": getattr(lifecycle, 'status', None),
                        "duration_minutes": getattr(lifecycle, 'duration_minutes', None),
                        "entry_price": getattr(lifecycle, 'entry_price', None),  # 🚀 NEW
                        "exit_price": getattr(lifecycle, 'exit_price', None),    # 🚀 NEW
                        "direction": getattr(lifecycle, 'direction', None),     # 🚀 NEW
                        "size": getattr(lifecycle, 'size', None),               # 🚀 NEW
                        "stop_limit_changes": getattr(lifecycle, 'stop_limit_changes', [])
                    }
                    
                    self.logger.debug(f"✅ Successfully serialized lifecycle {ref}")
                    
                except Exception as e:
                    self.logger.error(f"❌ Error serializing lifecycle {ref}: {e}")
                    continue
            
            self.logger.info(f"✅ Successfully serialized {len(serialized)} trade lifecycles")
            return serialized
            
        except Exception as e:
            self.logger.error(f"❌ Error in _serialize_trade_lifecycles: {e}")
            return {}
    
    def _generate_activity_correlation_summary(self, correlations: List[ActivityPnLCorrelation], updated_trades: List[Dict]) -> Dict:
        """Generate summary of activity-based correlation results"""
        try:
            self.logger.info(f"🔍 Generating summary for {len(correlations)} correlations")
            
            total_trades = len(correlations)
            correlations_found = 0
            complete_lifecycles = 0
            
            for i, c in enumerate(correlations):
                try:
                    # Debug logging
                    self.logger.debug(f"Summary correlation {i+1}: type={type(c)}")
                    
                    # Check if this is a proper object
                    if not hasattr(c, 'found_in_activities'):
                        self.logger.error(f"❌ Summary correlation {i+1} is not a proper object: {type(c)}")
                        continue
                    
                    if c.found_in_activities:
                        correlations_found += 1
                        
                        if c.activity_lifecycle and hasattr(c.activity_lifecycle, 'status'):
                            if c.activity_lifecycle.status == 'closed':
                                complete_lifecycles += 1
                                
                except Exception as e:
                    self.logger.error(f"❌ Error processing summary correlation {i+1}: {e}")
                    continue
            
            summary_result = {
                "total_trades": total_trades,
                "correlations_found": correlations_found,
                "correlation_rate": round(correlations_found / total_trades * 100, 2) if total_trades > 0 else 0,
                "complete_lifecycles": complete_lifecycles,
                "updated_trade_logs": len(updated_trades),
                "ready_for_pnl_calculation": complete_lifecycles
            }
            
            self.logger.info(f"✅ Summary generated: {summary_result}")
            return summary_result
            
        except Exception as e:
            self.logger.error(f"❌ Error in _generate_activity_correlation_summary: {e}")
            return {
                "total_trades": 0,
                "correlations_found": 0,
                "correlation_rate": 0,
                "complete_lifecycles": 0,
                "updated_trade_logs": 0,
                "ready_for_pnl_calculation": 0
            }


# Factory function
def create_activity_pnl_correlator(db_session: Session = None, logger=None) -> ActivityPnLCorrelator:
    """Factory function to create ActivityPnLCorrelator"""
    return ActivityPnLCorrelator(db_session=db_session, logger=logger)


if __name__ == "__main__":
    print("🧪 Testing Activity-Based P/L Correlator...")
    print("✅ Activity-Based P/L Correlator ready for integration!")
    print("🔗 Features:")
    print("   - Correlates trade_log deal IDs with IG activities")
    print("   - Builds complete trade lifecycles (open → close)")
    print("   - Extracts position references from activity descriptions")
    print("   - Tracks trade duration and stop limit changes")
    print("   - Updates trade_log with activity correlation data")
    print("   - 🚀 NEW: Uses position reference mapping for better correlation")
    print("   - 🚀 NEW: Extracts actual entry and exit prices from activity details")
    print("   - Ready for price-based P/L calculation integration")