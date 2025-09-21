"""
Results Table Component
Interactive table for displaying and filtering backtest signals
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime


class BacktestResultsTable:
    """Interactive results table for backtest signals"""

    def __init__(self, signals: List[Dict[str, Any]], on_signal_select: Optional[Callable] = None):
        self.signals = signals
        self.on_signal_select = on_signal_select
        self.df = self._prepare_dataframe()

    def _prepare_dataframe(self) -> pd.DataFrame:
        """Prepare signals DataFrame for display"""
        if not self.signals:
            return pd.DataFrame()

        df = pd.DataFrame(self.signals)

        # Ensure required columns exist
        required_columns = {
            'timestamp': datetime.now(),
            'signal_type': 'UNKNOWN',
            'direction': 'UNKNOWN',
            'entry_price': 0.0,
            'confidence': 0.0,
            'max_profit_pips': 0.0,
            'max_loss_pips': 0.0,
            'profit_loss_ratio': 0.0,
            'strategy': 'Unknown',
            'epic': '',
            'timeframe': ''
        }

        for col, default_val in required_columns.items():
            if col not in df.columns:
                df[col] = default_val

        # Convert data types
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['entry_price'] = pd.to_numeric(df['entry_price'], errors='coerce')
        df['confidence'] = pd.to_numeric(df['confidence'], errors='coerce')
        df['max_profit_pips'] = pd.to_numeric(df['max_profit_pips'], errors='coerce')
        df['max_loss_pips'] = pd.to_numeric(df['max_loss_pips'], errors='coerce')
        df['profit_loss_ratio'] = pd.to_numeric(df['profit_loss_ratio'], errors='coerce')

        # Add derived columns
        df['profit_potential'] = df['max_profit_pips'] - df['max_loss_pips']
        df['is_profitable'] = df['profit_potential'] > 0
        df['formatted_timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

        return df

    def render(self, container=None) -> Optional[Dict[str, Any]]:
        """
        Render the interactive results table

        Args:
            container: Streamlit container to render in (optional)

        Returns:
            Selected signal data if any, None otherwise
        """
        if container is None:
            container = st

        if self.df.empty:
            container.info("📊 No signals to display")
            return None

        # Render filters
        selected_signal = self._render_filters(container)

        # Render table
        self._render_table(container)

        # Render summary statistics
        self._render_summary(container)

        return selected_signal

    def _render_filters(self, container) -> Optional[Dict[str, Any]]:
        """Render filter controls"""
        container.subheader("🔍 Filters & Search")

        col1, col2, col3, col4 = container.columns(4)

        with col1:
            # Direction filter
            direction_options = ['All'] + sorted(self.df['direction'].unique().tolist())
            direction_filter = st.selectbox(
                "Direction",
                direction_options,
                key="direction_filter"
            )

        with col2:
            # Signal type filter
            signal_type_options = ['All'] + sorted(self.df['signal_type'].unique().tolist())
            signal_type_filter = st.selectbox(
                "Signal Type",
                signal_type_options,
                key="signal_type_filter"
            )

        with col3:
            # Profitability filter
            profitability_options = ['All', 'Profitable', 'Unprofitable', 'Breakeven']
            profitability_filter = st.selectbox(
                "Profitability",
                profitability_options,
                key="profitability_filter"
            )

        with col4:
            # Strategy filter
            strategy_options = ['All'] + sorted(self.df['strategy'].unique().tolist())
            strategy_filter = st.selectbox(
                "Strategy",
                strategy_options,
                key="strategy_filter"
            )

        # Numeric filters
        col1, col2, col3 = container.columns(3)

        with col1:
            confidence_range = st.slider(
                "Confidence Range",
                min_value=0.0,
                max_value=1.0,
                value=(0.0, 1.0),
                step=0.05,
                key="confidence_range"
            )

        with col2:
            profit_range = st.slider(
                "Profit Range (pips)",
                min_value=float(self.df['max_profit_pips'].min()),
                max_value=float(self.df['max_profit_pips'].max()),
                value=(
                    float(self.df['max_profit_pips'].min()),
                    float(self.df['max_profit_pips'].max())
                ),
                step=0.1,
                key="profit_range"
            )

        with col3:
            # Date filter
            date_range = st.date_input(
                "Date Range",
                value=(
                    self.df['timestamp'].min().date(),
                    self.df['timestamp'].max().date()
                ),
                key="date_range"
            )

        # Apply filters
        self.filtered_df = self._apply_filters(
            direction_filter,
            signal_type_filter,
            profitability_filter,
            strategy_filter,
            confidence_range,
            profit_range,
            date_range
        )

        # Search functionality
        search_term = container.text_input(
            "🔍 Search signals",
            placeholder="Search by epic, strategy, or any field...",
            key="signal_search"
        )

        if search_term:
            self.filtered_df = self._apply_search(self.filtered_df, search_term)

        return None

    def _apply_filters(self, direction_filter, signal_type_filter, profitability_filter,
                      strategy_filter, confidence_range, profit_range, date_range) -> pd.DataFrame:
        """Apply all filters to the DataFrame"""
        filtered_df = self.df.copy()

        # Direction filter
        if direction_filter != 'All':
            filtered_df = filtered_df[filtered_df['direction'] == direction_filter]

        # Signal type filter
        if signal_type_filter != 'All':
            filtered_df = filtered_df[filtered_df['signal_type'] == signal_type_filter]

        # Profitability filter
        if profitability_filter == 'Profitable':
            filtered_df = filtered_df[filtered_df['is_profitable']]
        elif profitability_filter == 'Unprofitable':
            filtered_df = filtered_df[~filtered_df['is_profitable'] & (filtered_df['profit_potential'] != 0)]
        elif profitability_filter == 'Breakeven':
            filtered_df = filtered_df[filtered_df['profit_potential'] == 0]

        # Strategy filter
        if strategy_filter != 'All':
            filtered_df = filtered_df[filtered_df['strategy'] == strategy_filter]

        # Confidence range
        filtered_df = filtered_df[
            (filtered_df['confidence'] >= confidence_range[0]) &
            (filtered_df['confidence'] <= confidence_range[1])
        ]

        # Profit range
        filtered_df = filtered_df[
            (filtered_df['max_profit_pips'] >= profit_range[0]) &
            (filtered_df['max_profit_pips'] <= profit_range[1])
        ]

        # Date range
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date = pd.to_datetime(date_range[0])
            end_date = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)  # Include end date
            filtered_df = filtered_df[
                (filtered_df['timestamp'] >= start_date) &
                (filtered_df['timestamp'] < end_date)
            ]

        return filtered_df

    def _apply_search(self, df: pd.DataFrame, search_term: str) -> pd.DataFrame:
        """Apply text search across all columns"""
        search_term = search_term.lower()
        mask = pd.Series([False] * len(df))

        # Search in text columns
        text_columns = ['epic', 'strategy', 'signal_type', 'direction']
        for col in text_columns:
            if col in df.columns:
                mask |= df[col].astype(str).str.lower().str.contains(search_term, na=False)

        # Search in formatted timestamp
        mask |= df['formatted_timestamp'].str.lower().str.contains(search_term, na=False)

        return df[mask]

    def _render_table(self, container):
        """Render the main data table"""
        container.subheader("📊 Signals Table")

        if self.filtered_df.empty:
            container.warning("No signals match the current filters")
            return

        # Prepare display DataFrame
        display_df = self._prepare_display_dataframe()

        # Column configuration
        column_config = self._get_column_config()

        # Render dataframe with selection
        selected_indices = container.dataframe(
            display_df,
            use_container_width=True,
            height=400,
            column_config=column_config,
            hide_index=True,
            on_select="rerun",
            selection_mode="multi-row"
        )

        # Handle selection
        if hasattr(selected_indices, 'selection') and selected_indices.selection.rows:
            selected_rows = selected_indices.selection.rows
            st.session_state.selected_signal_indices = selected_rows

            if self.on_signal_select:
                selected_signals = [self.filtered_df.iloc[i].to_dict() for i in selected_rows]
                self.on_signal_select(selected_signals)

    def _prepare_display_dataframe(self) -> pd.DataFrame:
        """Prepare DataFrame for display with proper formatting"""
        display_df = self.filtered_df.copy()

        # Select and reorder columns for display
        display_columns = [
            'formatted_timestamp',
            'signal_type',
            'direction',
            'entry_price',
            'confidence',
            'max_profit_pips',
            'max_loss_pips',
            'profit_potential',
            'strategy',
            'epic'
        ]

        # Add any additional columns that exist
        additional_columns = [col for col in display_df.columns if col not in display_columns and col not in [
            'timestamp', 'is_profitable', 'profit_loss_ratio'
        ]]

        all_columns = display_columns + additional_columns
        available_columns = [col for col in all_columns if col in display_df.columns]

        display_df = display_df[available_columns]

        # Format columns
        if 'confidence' in display_df.columns:
            display_df['confidence'] = (display_df['confidence'] * 100).round(1).astype(str) + '%'

        for col in ['max_profit_pips', 'max_loss_pips', 'profit_potential']:
            if col in display_df.columns:
                display_df[col] = display_df[col].round(1).astype(str) + ' pips'

        if 'entry_price' in display_df.columns:
            display_df['entry_price'] = display_df['entry_price'].round(5)

        # Rename columns for better display
        column_renames = {
            'formatted_timestamp': 'Timestamp',
            'signal_type': 'Type',
            'direction': 'Direction',
            'entry_price': 'Entry Price',
            'confidence': 'Confidence',
            'max_profit_pips': 'Max Profit',
            'max_loss_pips': 'Max Loss',
            'profit_potential': 'Net Profit',
            'strategy': 'Strategy',
            'epic': 'Epic'
        }

        display_df.rename(columns=column_renames, inplace=True)

        return display_df

    def _get_column_config(self) -> Dict[str, Any]:
        """Get column configuration for the dataframe"""
        return {
            "Timestamp": st.column_config.DatetimeColumn(
                "Timestamp",
                width="medium",
                format="YYYY-MM-DD HH:mm:ss"
            ),
            "Type": st.column_config.TextColumn(
                "Signal Type",
                width="small"
            ),
            "Direction": st.column_config.TextColumn(
                "Direction",
                width="small"
            ),
            "Entry Price": st.column_config.NumberColumn(
                "Entry Price",
                width="small",
                format="%.5f"
            ),
            "Confidence": st.column_config.TextColumn(
                "Confidence",
                width="small"
            ),
            "Max Profit": st.column_config.TextColumn(
                "Max Profit",
                width="small"
            ),
            "Max Loss": st.column_config.TextColumn(
                "Max Loss",
                width="small"
            ),
            "Net Profit": st.column_config.TextColumn(
                "Net Profit",
                width="small"
            ),
            "Strategy": st.column_config.TextColumn(
                "Strategy",
                width="small"
            ),
            "Epic": st.column_config.TextColumn(
                "Epic",
                width="medium"
            )
        }

    def _render_summary(self, container):
        """Render summary statistics"""
        container.subheader("📈 Summary Statistics")

        if self.filtered_df.empty:
            return

        col1, col2, col3, col4 = container.columns(4)

        with col1:
            total_signals = len(self.filtered_df)
            container.metric("Total Signals", total_signals)

        with col2:
            profitable_signals = len(self.filtered_df[self.filtered_df['is_profitable']])
            win_rate = (profitable_signals / total_signals * 100) if total_signals > 0 else 0
            container.metric("Win Rate", f"{win_rate:.1f}%")

        with col3:
            avg_confidence = self.filtered_df['confidence'].mean() * 100
            container.metric("Avg Confidence", f"{avg_confidence:.1f}%")

        with col4:
            avg_profit = self.filtered_df['profit_potential'].mean()
            container.metric("Avg Net Profit", f"{avg_profit:.1f} pips")

        # Additional statistics
        if total_signals > 0:
            col1, col2, col3, col4 = container.columns(4)

            with col1:
                avg_profit_win = self.filtered_df[self.filtered_df['is_profitable']]['max_profit_pips'].mean()
                if not pd.isna(avg_profit_win):
                    container.metric("Avg Profit (Winners)", f"{avg_profit_win:.1f} pips")

            with col2:
                avg_loss = self.filtered_df[~self.filtered_df['is_profitable']]['max_loss_pips'].mean()
                if not pd.isna(avg_loss):
                    container.metric("Avg Loss (Losers)", f"{avg_loss:.1f} pips")

            with col3:
                best_signal = self.filtered_df.loc[self.filtered_df['profit_potential'].idxmax()]
                container.metric("Best Signal", f"{best_signal['profit_potential']:.1f} pips")

            with col4:
                worst_signal = self.filtered_df.loc[self.filtered_df['profit_potential'].idxmin()]
                container.metric("Worst Signal", f"{worst_signal['profit_potential']:.1f} pips")

    def get_filtered_data(self) -> pd.DataFrame:
        """Get the currently filtered DataFrame"""
        return getattr(self, 'filtered_df', self.df)

    def get_selected_signals(self) -> List[Dict[str, Any]]:
        """Get currently selected signals"""
        selected_indices = getattr(st.session_state, 'selected_signal_indices', [])
        filtered_df = self.get_filtered_data()

        if not selected_indices or filtered_df.empty:
            return []

        return [filtered_df.iloc[i].to_dict() for i in selected_indices if i < len(filtered_df)]

    def export_to_csv(self) -> str:
        """Export filtered data to CSV string"""
        return self.get_filtered_data().to_csv(index=False)

    def export_to_json(self) -> str:
        """Export filtered data to JSON string"""
        return self.get_filtered_data().to_json(orient='records', date_format='iso', indent=2)