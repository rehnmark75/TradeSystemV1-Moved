"""
Advanced Log Search & Explorer - Powerful search capabilities for trading logs
Real-time log exploration with regex support, filtering, and export features
"""

import streamlit as st
import re
import os
from datetime import datetime, timedelta
import sys
import json

# Add services to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'services'))

try:
    from simple_log_intelligence import SimpleLogParser
except ImportError as e:
    st.error(f"Failed to import simple log intelligence: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Log Search Hub",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for search styling
st.markdown("""
<style>
    .search-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        font-size: 1.8rem;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }

    .search-result {
        background: #fff;
        padding: 1rem;
        border-left: 4px solid #007bff;
        border-radius: 4px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
    }

    .search-result-error {
        border-left-color: #dc3545;
        background: #fff5f5;
    }

    .search-result-warning {
        border-left-color: #ffc107;
        background: #fffcf0;
    }

    .search-result-signal {
        border-left-color: #28a745;
        background: #f0fff4;
    }

    .search-meta {
        color: #6c757d;
        font-size: 0.75rem;
        margin-bottom: 0.5rem;
    }

    .search-content {
        color: #333;
        white-space: pre-wrap;
        word-break: break-word;
    }

    .highlight {
        background-color: #ffeb3b;
        padding: 2px 4px;
        border-radius: 3px;
        font-weight: bold;
    }

    .filter-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }

    .stats-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }

    .export-section {
        background: #e9ecef;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_log_parser():
    """Get cached log parser instance"""
    return SimpleLogParser()

def search_logs(parser, search_term, log_types, start_date, end_date, regex_mode=False, case_sensitive=False, max_results=500):
    """Search through log files with advanced filtering"""
    results = []

    # Prepare search pattern
    if regex_mode:
        try:
            flags = 0 if case_sensitive else re.IGNORECASE
            pattern = re.compile(search_term, flags)
        except re.error as e:
            st.error(f"Invalid regex pattern: {e}")
            return []
    else:
        if case_sensitive:
            search_func = lambda text: search_term in text
        else:
            search_func = lambda text: search_term.lower() in text.lower()

    # Define log file mappings
    log_files_to_search = []
    if 'forex_scanner' in log_types:
        log_files_to_search.extend(parser.log_files['forex_scanner'])
    if 'stream_service' in log_types:
        log_files_to_search.extend(parser.log_files['stream_service'])
    if 'trade_monitor' in log_types:
        log_files_to_search.extend(parser.log_files['trade_monitor'])

    for log_file in log_files_to_search:
        if parser.base_log_dir == "":
            file_path = log_file
        else:
            file_path = os.path.join(parser.base_log_dir, log_file)

        if not os.path.exists(file_path):
            continue

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                line_number = 0
                for line in f:
                    line_number += 1

                    # Parse timestamp for filtering
                    timestamp_match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                    if timestamp_match:
                        try:
                            log_time = datetime.strptime(timestamp_match.group(1), '%Y-%m-%d %H:%M:%S')

                            # Filter by date range
                            if log_time.date() < start_date or log_time.date() > end_date:
                                continue
                        except ValueError:
                            continue
                    else:
                        continue

                    # Search in line
                    match_found = False
                    if regex_mode:
                        match = pattern.search(line)
                        if match:
                            match_found = True
                    else:
                        if search_func(line):
                            match_found = True

                    if match_found:
                        # Determine log type
                        log_type = 'info'
                        if ' - ERROR - ' in line:
                            log_type = 'error'
                        elif ' - WARNING - ' in line:
                            log_type = 'warning'
                        elif '📊' in line or 'signal' in line.lower():
                            log_type = 'signal'

                        results.append({
                            'file': os.path.basename(file_path),
                            'line_number': line_number,
                            'timestamp': log_time if timestamp_match else None,
                            'content': line.strip(),
                            'log_type': log_type
                        })

                        if len(results) >= max_results:
                            break

        except Exception as e:
            st.warning(f"Error reading {file_path}: {e}")
            continue

        if len(results) >= max_results:
            break

    return results

def highlight_search_term(text, search_term, regex_mode=False, case_sensitive=False):
    """Highlight search term in text"""
    if not search_term:
        return text

    if regex_mode:
        try:
            flags = 0 if case_sensitive else re.IGNORECASE
            pattern = re.compile(f'({search_term})', flags)
            return pattern.sub(r'<span class="highlight">\1</span>', text)
        except re.error:
            return text
    else:
        if case_sensitive:
            return text.replace(search_term, f'<span class="highlight">{search_term}</span>')
        else:
            # Case insensitive replacement
            import re
            pattern = re.compile(re.escape(search_term), re.IGNORECASE)
            return pattern.sub(lambda m: f'<span class="highlight">{m.group()}</span>', text)

def export_results(results, search_term):
    """Export search results to JSON"""
    export_data = {
        'search_term': search_term,
        'timestamp': datetime.now().isoformat(),
        'total_results': len(results),
        'results': []
    }

    for result in results:
        export_data['results'].append({
            'file': result['file'],
            'line_number': result['line_number'],
            'timestamp': result['timestamp'].isoformat() if result['timestamp'] else None,
            'content': result['content'],
            'log_type': result['log_type']
        })

    return json.dumps(export_data, indent=2)

def render_search_interface():
    """Render main search interface"""
    st.markdown('<div class="search-header">🔍 Advanced Log Search & Explorer</div>', unsafe_allow_html=True)

    # Initialize parser
    try:
        parser = get_log_parser()
    except Exception as e:
        st.error(f"Failed to initialize log parser: {e}")
        return

    # Search Controls
    st.subheader("🎯 Search Configuration")

    col1, col2 = st.columns([3, 1])

    with col1:
        search_term = st.text_input(
            "🔍 Search Term",
            placeholder="Enter search term or regex pattern...",
            help="Enter text to search for, or enable regex mode for pattern matching"
        )

    with col2:
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)

    # Advanced Filters
    st.markdown('<div class="filter-section">', unsafe_allow_html=True)
    st.write("**🔧 Advanced Filters**")

    filter_cols = st.columns(4)

    with filter_cols[0]:
        log_types = st.multiselect(
            "📁 Log Sources",
            options=['forex_scanner', 'stream_service', 'trade_monitor'],
            default=['forex_scanner'],
            help="Select which log sources to search"
        )

    with filter_cols[1]:
        regex_mode = st.checkbox("🔧 Regex Mode", help="Enable regular expression patterns")
        case_sensitive = st.checkbox("🔤 Case Sensitive", help="Case sensitive search")

    with filter_cols[2]:
        start_date = st.date_input(
            "📅 Start Date",
            value=datetime.now().date() - timedelta(days=7),
            help="Search from this date"
        )

    with filter_cols[3]:
        end_date = st.date_input(
            "📅 End Date",
            value=datetime.now().date(),
            help="Search until this date"
        )

    max_results = st.slider("📊 Max Results", min_value=50, max_value=1000, value=200, step=50)

    st.markdown('</div>', unsafe_allow_html=True)

    # Quick Search Buttons
    st.write("**⚡ Quick Searches**")
    quick_cols = st.columns(6)

    quick_searches = [
        ("🚀 Signals", "📊.*CS\\.D\\.[A-Z]{6}\\.MINI\\.IP", True),
        ("❌ Errors", "ERROR", False),
        ("⚠️ Warnings", "WARNING", False),
        ("🚫 Rejected", "REJECTED", False),
        ("🎯 High Confidence", "\\(9[0-9]\\.[0-9]%\\)", True),
        ("💰 Orders", "Order", False)
    ]

    for idx, (label, term, is_regex) in enumerate(quick_searches):
        with quick_cols[idx]:
            if st.button(label, use_container_width=True):
                st.session_state.search_term = term
                st.session_state.regex_mode = is_regex
                st.rerun()

    # Update search term from session state
    if 'search_term' in st.session_state:
        search_term = st.session_state.search_term
        regex_mode = st.session_state.get('regex_mode', False)

    # Perform search
    results = []
    if search_button and search_term:
        with st.spinner("🔍 Searching logs..."):
            results = search_logs(
                parser, search_term, log_types, start_date, end_date,
                regex_mode, case_sensitive, max_results
            )

        # Display search statistics
        if results:
            st.markdown(f"""
            <div class="stats-box">
                <strong>📊 Search Results</strong><br>
                Found {len(results)} matches for "{search_term}"<br>
                Searched from {start_date} to {end_date}
            </div>
            """, unsafe_allow_html=True)

            # Export functionality
            st.markdown('<div class="export-section">', unsafe_allow_html=True)
            col1, col2 = st.columns([3, 1])

            with col1:
                st.write("**📤 Export Results**")

            with col2:
                if st.button("📥 Download JSON"):
                    export_data = export_results(results, search_term)
                    st.download_button(
                        label="💾 Save Results",
                        data=export_data,
                        file_name=f"search_results_{search_term}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
            st.markdown('</div>', unsafe_allow_html=True)

            # Filter results by type
            st.subheader("🔍 Filter Results")

            type_filter = st.selectbox(
                "Filter by log type:",
                options=["all", "signal", "error", "warning", "info"],
                index=0
            )

            if type_filter != "all":
                results = [r for r in results if r['log_type'] == type_filter]

            # Display results
            st.subheader(f"📋 Search Results ({len(results)} items)")

            for idx, result in enumerate(results):
                # Determine CSS class based on log type
                css_class = f"search-result-{result['log_type']}"

                # Highlight search term
                highlighted_content = highlight_search_term(
                    result['content'], search_term, regex_mode, case_sensitive
                )

                timestamp_str = result['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if result['timestamp'] else 'Unknown'

                st.markdown(f"""
                <div class="search-result {css_class}">
                    <div class="search-meta">
                        📁 {result['file']} | 📍 Line {result['line_number']} |
                        🕒 {timestamp_str} | 🏷️ {result['log_type'].upper()}
                    </div>
                    <div class="search-content">{highlighted_content}</div>
                </div>
                """, unsafe_allow_html=True)

                # Add separator every 10 results for better readability
                if (idx + 1) % 10 == 0 and idx + 1 < len(results):
                    st.markdown("---")

        else:
            st.info(f"No results found for '{search_term}' in the selected date range and log sources.")

    elif search_term and not search_button:
        st.info("👆 Click the Search button to start searching")

    # Search Tips
    with st.expander("💡 Search Tips & Examples"):
        st.markdown("""
        **🔍 Search Examples:**

        **Basic Text Search:**
        - `AUDJPY` - Find all mentions of AUD/JPY pair
        - `ERROR` - Find all error messages
        - `signal detected` - Find signal detection messages

        **Regex Patterns (enable Regex Mode):**
        - `CS\\.D\\.[A-Z]{6}\\.MINI\\.IP` - Find all forex instruments
        - `\\(9[0-9]\\.[0-9]%\\)` - Find high confidence signals (90%+)
        - `\\b(BULL|BEAR)\\b` - Find bull or bear signals
        - `Order.*failed` - Find failed orders
        - `\\d{4}-\\d{2}-\\d{2}` - Find date patterns

        **Tips:**
        - Use Quick Search buttons for common patterns
        - Enable Regex Mode for powerful pattern matching
        - Filter by date range to narrow results
        - Export results for further analysis
        - Use Case Sensitive for exact matches
        """)

def main():
    """Main application"""
    render_search_interface()

if __name__ == "__main__":
    main()