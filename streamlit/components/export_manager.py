"""
Export Manager Component
Handles exporting backtest results in various formats
"""

import streamlit as st
import pandas as pd
import json
import io
import zipfile
from typing import Dict, List, Optional, Any
from datetime import datetime
import base64

from ..services.backtest_service import BacktestResult


class BacktestExportManager:
    """Manager for exporting backtest results in various formats"""

    def __init__(self, result: BacktestResult):
        self.result = result

    def render_export_section(self):
        """Render the complete export section"""
        if not self.result.success:
            st.warning("⚠️ Cannot export failed backtest results")
            return

        st.subheader("📁 Export Results")

        # Export options in tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Signals", "📈 Performance", "🗂️ Complete", "⚙️ Config"
        ])

        with tab1:
            self._render_signals_export()

        with tab2:
            self._render_performance_export()

        with tab3:
            self._render_complete_export()

        with tab4:
            self._render_config_export()

    def _render_signals_export(self):
        """Render signals export options"""
        st.write("**Export Trading Signals**")

        if not self.result.signals:
            st.info("No signals to export")
            return

        col1, col2 = st.columns(2)

        with col1:
            # CSV Export
            if st.button("📊 Export as CSV", key="signals_csv"):
                csv_data = self._create_signals_csv()
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{self.result.strategy_name}_signals_{timestamp}.csv"

                st.download_button(
                    label="💾 Download CSV",
                    data=csv_data,
                    file_name=filename,
                    mime="text/csv",
                    key="download_signals_csv"
                )

        with col2:
            # Excel Export
            if st.button("📈 Export as Excel", key="signals_excel"):
                excel_data = self._create_signals_excel()
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{self.result.strategy_name}_signals_{timestamp}.xlsx"

                st.download_button(
                    label="💾 Download Excel",
                    data=excel_data,
                    file_name=filename,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="download_signals_excel"
                )

        # Preview
        with st.expander("👁️ Preview Signals Data"):
            signals_df = pd.DataFrame(self.result.signals)
            st.dataframe(signals_df.head(10), use_container_width=True)
            st.caption(f"Showing first 10 of {len(signals_df)} signals")

    def _render_performance_export(self):
        """Render performance metrics export options"""
        st.write("**Export Performance Metrics**")

        col1, col2 = st.columns(2)

        with col1:
            # JSON Export
            if st.button("📊 Export as JSON", key="performance_json"):
                json_data = self._create_performance_json()
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{self.result.strategy_name}_performance_{timestamp}.json"

                st.download_button(
                    label="💾 Download JSON",
                    data=json_data,
                    file_name=filename,
                    mime="application/json",
                    key="download_performance_json"
                )

        with col2:
            # Report Export
            if st.button("📄 Export Report", key="performance_report"):
                report_data = self._create_performance_report()
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{self.result.strategy_name}_report_{timestamp}.txt"

                st.download_button(
                    label="💾 Download Report",
                    data=report_data,
                    file_name=filename,
                    mime="text/plain",
                    key="download_performance_report"
                )

        # Preview
        with st.expander("👁️ Preview Performance Data"):
            st.json(self.result.performance_metrics)

    def _render_complete_export(self):
        """Render complete export options"""
        st.write("**Export Complete Backtest Package**")

        col1, col2 = st.columns(2)

        with col1:
            # ZIP Package
            if st.button("🗂️ Create ZIP Package", key="complete_zip"):
                zip_data = self._create_complete_package()
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{self.result.strategy_name}_backtest_{timestamp}.zip"

                st.download_button(
                    label="💾 Download ZIP",
                    data=zip_data,
                    file_name=filename,
                    mime="application/zip",
                    key="download_complete_zip"
                )

        with col2:
            # Full JSON Export
            if st.button("📦 Export Full JSON", key="complete_json"):
                full_json = self._create_complete_json()
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{self.result.strategy_name}_complete_{timestamp}.json"

                st.download_button(
                    label="💾 Download JSON",
                    data=full_json,
                    file_name=filename,
                    mime="application/json",
                    key="download_complete_json"
                )

        st.info("📦 **ZIP Package includes:**\n- Signals CSV\n- Performance metrics JSON\n- Configuration file\n- Summary report")

    def _render_config_export(self):
        """Render configuration export options"""
        st.write("**Export Configuration for Reproduction**")

        # Create config data
        config_data = {
            'strategy_name': self.result.strategy_name,
            'epic': self.result.epic,
            'timeframe': self.result.timeframe,
            'export_timestamp': datetime.now().isoformat(),
            'backtest_execution_time': self.result.execution_time,
            'total_signals': self.result.total_signals
        }

        col1, col2 = st.columns(2)

        with col1:
            # Config JSON
            if st.button("⚙️ Export Config JSON", key="config_json"):
                config_json = json.dumps(config_data, indent=2)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{self.result.strategy_name}_config_{timestamp}.json"

                st.download_button(
                    label="💾 Download Config",
                    data=config_json,
                    file_name=filename,
                    mime="application/json",
                    key="download_config_json"
                )

        with col2:
            # Reproduce Script
            if st.button("🔄 Generate Reproduce Script", key="reproduce_script"):
                script_data = self._create_reproduce_script()
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{self.result.strategy_name}_reproduce_{timestamp}.py"

                st.download_button(
                    label="💾 Download Script",
                    data=script_data,
                    file_name=filename,
                    mime="text/plain",
                    key="download_reproduce_script"
                )

        # Preview config
        with st.expander("👁️ Preview Configuration"):
            st.json(config_data)

    def _create_signals_csv(self) -> str:
        """Create CSV data for signals"""
        if not self.result.signals:
            return "No signals available"

        df = pd.DataFrame(self.result.signals)

        # Format columns for CSV
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')

        if 'confidence' in df.columns:
            df['confidence_percent'] = (df['confidence'] * 100).round(1)

        return df.to_csv(index=False)

    def _create_signals_excel(self) -> bytes:
        """Create Excel data for signals"""
        if not self.result.signals:
            return b""

        df = pd.DataFrame(self.result.signals)

        # Format columns for Excel
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Create Excel buffer
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Signals', index=False)

            # Add performance metrics sheet
            if self.result.performance_metrics:
                perf_df = pd.DataFrame([self.result.performance_metrics])
                perf_df.to_excel(writer, sheet_name='Performance', index=False)

        return buffer.getvalue()

    def _create_performance_json(self) -> str:
        """Create JSON data for performance metrics"""
        performance_data = {
            'strategy': self.result.strategy_name,
            'epic': self.result.epic,
            'timeframe': self.result.timeframe,
            'execution_time': self.result.execution_time,
            'total_signals': self.result.total_signals,
            'performance_metrics': self.result.performance_metrics,
            'export_timestamp': datetime.now().isoformat()
        }

        return json.dumps(performance_data, indent=2)

    def _create_performance_report(self) -> str:
        """Create a text report of performance"""
        report_lines = [
            f"Backtest Performance Report",
            f"=" * 50,
            f"",
            f"Strategy: {self.result.strategy_name}",
            f"Epic: {self.result.epic}",
            f"Timeframe: {self.result.timeframe}",
            f"Execution Time: {self.result.execution_time:.2f} seconds",
            f"Total Signals: {self.result.total_signals}",
            f"",
            f"Performance Metrics:",
            f"-" * 20
        ]

        if self.result.performance_metrics:
            for key, value in self.result.performance_metrics.items():
                if isinstance(value, float):
                    if 'rate' in key.lower() or 'confidence' in key.lower():
                        report_lines.append(f"{key.replace('_', ' ').title()}: {value:.1%}")
                    else:
                        report_lines.append(f"{key.replace('_', ' ').title()}: {value:.2f}")
                else:
                    report_lines.append(f"{key.replace('_', ' ').title()}: {value}")

        report_lines.extend([
            f"",
            f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Generated by: Modular Backtest System"
        ])

        return "\n".join(report_lines)

    def _create_complete_package(self) -> bytes:
        """Create a complete ZIP package"""
        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add signals CSV
            if self.result.signals:
                signals_csv = self._create_signals_csv()
                zip_file.writestr("signals.csv", signals_csv)

            # Add performance JSON
            performance_json = self._create_performance_json()
            zip_file.writestr("performance.json", performance_json)

            # Add report
            report = self._create_performance_report()
            zip_file.writestr("report.txt", report)

            # Add configuration
            config_data = {
                'strategy_name': self.result.strategy_name,
                'epic': self.result.epic,
                'timeframe': self.result.timeframe,
                'package_created': datetime.now().isoformat()
            }
            zip_file.writestr("config.json", json.dumps(config_data, indent=2))

            # Add README
            readme_content = self._create_readme()
            zip_file.writestr("README.txt", readme_content)

        return zip_buffer.getvalue()

    def _create_complete_json(self) -> str:
        """Create complete JSON export"""
        complete_data = {
            'strategy_name': self.result.strategy_name,
            'epic': self.result.epic,
            'timeframe': self.result.timeframe,
            'execution_time': self.result.execution_time,
            'total_signals': self.result.total_signals,
            'success': self.result.success,
            'signals': self.result.signals,
            'performance_metrics': self.result.performance_metrics,
            'export_metadata': {
                'export_timestamp': datetime.now().isoformat(),
                'export_version': '1.0',
                'exported_by': 'Modular Backtest System'
            }
        }

        return json.dumps(complete_data, indent=2, default=str)

    def _create_reproduce_script(self) -> str:
        """Create a Python script to reproduce the backtest"""
        script_lines = [
            f"#!/usr/bin/env python3",
            f'"""',
            f'Reproduce backtest for {self.result.strategy_name}',
            f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
            f'"""',
            f'',
            f'from services.backtest_service import get_backtest_runner, BacktestConfig',
            f'',
            f'def reproduce_backtest():',
            f'    """Reproduce the backtest with the same configuration"""',
            f'    ',
            f'    config = BacktestConfig(',
            f'        strategy_name="{self.result.strategy_name}",',
            f'        epic="{self.result.epic}",',
            f'        timeframe="{self.result.timeframe}",',
            f'        parameters={{',
            f'            "show_signals": True',
            f'        }}',
            f'    )',
            f'    ',
            f'    runner = get_backtest_runner()',
            f'    result = runner.run_backtest(config)',
            f'    ',
            f'    if result.success:',
            f'        print(f"✅ Backtest completed: {{result.total_signals}} signals")',
            f'        return result',
            f'    else:',
            f'        print(f"❌ Backtest failed: {{result.error_message}}")',
            f'        return None',
            f'',
            f'if __name__ == "__main__":',
            f'    reproduce_backtest()',
        ]

        return "\n".join(script_lines)

    def _create_readme(self) -> str:
        """Create README for the export package"""
        readme_lines = [
            f"Backtest Results Package",
            f"=" * 30,
            f"",
            f"Strategy: {self.result.strategy_name}",
            f"Epic: {self.result.epic}",
            f"Timeframe: {self.result.timeframe}",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"",
            f"Files Included:",
            f"- signals.csv: Trading signals data",
            f"- performance.json: Performance metrics",
            f"- report.txt: Human-readable performance report",
            f"- config.json: Backtest configuration",
            f"- README.txt: This file",
            f"",
            f"Usage:",
            f"1. Review the performance report for quick insights",
            f"2. Analyze signals.csv for detailed signal data",
            f"3. Use config.json to reproduce the backtest",
            f"",
            f"Generated by: Modular Backtest System v1.0"
        ]

        return "\n".join(readme_lines)


def render_export_manager(result: BacktestResult):
    """Convenience function to render the export manager"""
    export_manager = BacktestExportManager(result)
    export_manager.render_export_section()