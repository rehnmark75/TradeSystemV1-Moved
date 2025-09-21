"""
Chart-Table Synchronization Component
Handles synchronization between chart markers and table selections
"""

import streamlit as st
from typing import Dict, List, Optional, Any, Callable
import pandas as pd


class ChartTableSync:
    """Synchronization manager for chart markers and table selections"""

    def __init__(self):
        self.selected_signals = []
        self.highlighted_signals = []
        self._chart_callback = None
        self._table_callback = None

    def register_chart_callback(self, callback: Callable):
        """Register callback for chart updates"""
        self._chart_callback = callback

    def register_table_callback(self, callback: Callable):
        """Register callback for table updates"""
        self._table_callback = callback

    def select_signals_from_table(self, selected_signals: List[Dict[str, Any]]):
        """Handle signal selection from table"""
        self.selected_signals = selected_signals

        # Update chart highlighting
        if self._chart_callback:
            self._chart_callback(selected_signals)

        # Store in session state
        st.session_state.sync_selected_signals = selected_signals

    def select_signals_from_chart(self, selected_timestamps: List[str]):
        """Handle signal selection from chart (if chart supports it)"""
        # This would need to be implemented if the chart library supports interaction
        # For now, this is a placeholder for future enhancement
        pass

    def highlight_signal_on_chart(self, signal: Dict[str, Any]):
        """Highlight a specific signal on the chart"""
        self.highlighted_signals = [signal]

        if self._chart_callback:
            self._chart_callback([signal], highlight=True)

    def clear_selections(self):
        """Clear all selections"""
        self.selected_signals = []
        self.highlighted_signals = []

        if 'sync_selected_signals' in st.session_state:
            del st.session_state.sync_selected_signals

        # Notify callbacks
        if self._chart_callback:
            self._chart_callback([])
        if self._table_callback:
            self._table_callback([])

    def get_selected_signals(self) -> List[Dict[str, Any]]:
        """Get currently selected signals"""
        return getattr(st.session_state, 'sync_selected_signals', [])

    def create_enhanced_markers(self, base_markers: List[Dict[str, Any]],
                              signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create enhanced markers with selection highlighting"""
        if not self.selected_signals:
            return base_markers

        # Create a set of selected timestamps for quick lookup
        selected_timestamps = {
            pd.to_datetime(signal.get('timestamp')).timestamp()
            for signal in self.selected_signals
        }

        enhanced_markers = []
        for marker in base_markers:
            enhanced_marker = marker.copy()

            # Check if this marker corresponds to a selected signal
            marker_time = marker.get('time')
            if marker_time in selected_timestamps:
                # Enhance selected markers
                enhanced_marker['size'] = 2  # Make it larger
                enhanced_marker['shape'] = 'square'  # Change shape
                enhanced_marker['color'] = '#ff6b6b'  # Highlight color

            enhanced_markers.append(enhanced_marker)

        return enhanced_markers


# Global instance for the session
def get_chart_table_sync() -> ChartTableSync:
    """Get the global chart-table synchronization instance"""
    if 'chart_table_sync' not in st.session_state:
        st.session_state.chart_table_sync = ChartTableSync()
    return st.session_state.chart_table_sync


def create_signal_detail_view(signal: Dict[str, Any]):
    """Create a detailed view of a selected signal"""
    if not signal:
        return

    st.subheader(f"📊 Signal Details")

    # Basic information
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Strategy", signal.get('strategy', 'Unknown'))
        st.metric("Signal Type", signal.get('signal_type', 'Unknown'))

    with col2:
        st.metric("Direction", signal.get('direction', 'Unknown'))
        st.metric("Entry Price", f"{signal.get('entry_price', 0):.5f}")

    with col3:
        confidence = signal.get('confidence', 0) * 100
        st.metric("Confidence", f"{confidence:.1f}%")
        st.metric("Timeframe", signal.get('timeframe', 'Unknown'))

    # Performance metrics
    st.subheader("💰 Performance")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        max_profit = signal.get('max_profit_pips', 0)
        st.metric("Max Profit", f"{max_profit:.1f} pips")

    with col2:
        max_loss = signal.get('max_loss_pips', 0)
        st.metric("Max Loss", f"{max_loss:.1f} pips")

    with col3:
        net_profit = max_profit - max_loss
        st.metric("Net Profit", f"{net_profit:+.1f} pips")

    with col4:
        risk_reward = signal.get('profit_loss_ratio', 0)
        st.metric("Risk/Reward", f"{risk_reward:.2f}")

    # Additional details
    st.subheader("ℹ️ Additional Information")

    detail_data = {}
    exclude_keys = {
        'strategy', 'signal_type', 'direction', 'entry_price', 'confidence',
        'timeframe', 'max_profit_pips', 'max_loss_pips', 'profit_loss_ratio'
    }

    for key, value in signal.items():
        if key not in exclude_keys and value is not None:
            detail_data[key.replace('_', ' ').title()] = value

    if detail_data:
        st.json(detail_data)


def create_comparison_view(signals: List[Dict[str, Any]]):
    """Create a comparison view for multiple selected signals"""
    if len(signals) < 2:
        return

    st.subheader(f"🔄 Comparing {len(signals)} Signals")

    # Create comparison DataFrame
    comparison_data = []
    for i, signal in enumerate(signals):
        comparison_data.append({
            'Signal': f"Signal {i+1}",
            'Strategy': signal.get('strategy', 'Unknown'),
            'Direction': signal.get('direction', 'Unknown'),
            'Entry Price': signal.get('entry_price', 0),
            'Confidence': signal.get('confidence', 0) * 100,
            'Max Profit (pips)': signal.get('max_profit_pips', 0),
            'Max Loss (pips)': signal.get('max_loss_pips', 0),
            'Net Profit (pips)': signal.get('max_profit_pips', 0) - signal.get('max_loss_pips', 0),
            'Risk/Reward': signal.get('profit_loss_ratio', 0)
        })

    comparison_df = pd.DataFrame(comparison_data)

    # Display comparison table
    st.dataframe(comparison_df, use_container_width=True)

    # Comparison statistics
    st.subheader("📊 Comparison Statistics")

    col1, col2, col3 = st.columns(3)

    with col1:
        avg_confidence = comparison_df['Confidence'].mean()
        st.metric("Average Confidence", f"{avg_confidence:.1f}%")

    with col2:
        total_profit = comparison_df['Net Profit (pips)'].sum()
        st.metric("Total Net Profit", f"{total_profit:+.1f} pips")

    with col3:
        best_signal_idx = comparison_df['Net Profit (pips)'].idxmax()
        best_profit = comparison_df.loc[best_signal_idx, 'Net Profit (pips)']
        st.metric("Best Signal Profit", f"{best_profit:+.1f} pips")