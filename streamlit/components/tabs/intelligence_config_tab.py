"""
Intelligence System Configuration Tab Component

Renders configuration management UI for Market Intelligence system with:
- Preset Selection and Overview
- Global Settings by Category
- Regime-Strategy Modifiers
- Smart Money Configuration
- Audit Trail
"""

import streamlit as st
import json
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional, List

from services.intelligence_config_service import (
    get_intelligence_config,
    get_config_summary,
    get_parameters_by_category,
    get_audit_history,
    save_parameter,
    save_multiple_parameters,
    save_preset,
    save_regime_modifier,
)


def render_intelligence_config_tab():
    """Main entry point for Intelligence Config tab"""
    st.header("Market Intelligence Configuration")
    st.markdown("*Database-driven configuration for the Market Intelligence system*")

    # Load current config
    config = get_intelligence_config()

    if not config or not config['parameters']:
        st.error("Failed to load intelligence configuration from database.")
        st.info("Make sure the intelligence config tables are created in strategy_config database.")
        st.code("""
# Run this migration to create the tables:
cat worker/app/forex_scanner/migrations/create_intelligence_config_tables.sql | \\
    docker exec -i postgres psql -U postgres -d strategy_config
        """)
        return

    # Display summary metrics
    summary = get_config_summary()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Active Preset", summary['preset'])
    with col2:
        st.metric("Threshold", f"{summary['threshold']:.1%}")
    with col3:
        engine_status = "Running" if summary['use_engine'] else "Disabled"
        st.metric("Intelligence Engine", engine_status)
    with col4:
        st.metric("Cleanup Days", summary['cleanup_days'])

    st.divider()

    # Sub-tabs
    sub_tabs = st.tabs([
        "Presets",
        "Core Settings",
        "Smart Money",
        "Regime Detection",
        "Regime Modifiers",
        "Audit Trail"
    ])

    with sub_tabs[0]:
        render_presets_section(config, summary)

    with sub_tabs[1]:
        render_core_settings(config)

    with sub_tabs[2]:
        render_smart_money_settings(config)

    with sub_tabs[3]:
        render_regime_detection_settings(config)

    with sub_tabs[4]:
        render_regime_modifiers_section(config)

    with sub_tabs[5]:
        render_audit_trail()


def render_presets_section(config: Dict[str, Any], summary: Dict[str, Any]):
    """Render preset selection and configuration"""
    st.subheader("Intelligence Presets")

    st.markdown("""
    Presets provide quick configuration switching for different trading scenarios.
    The **collect_only** preset is recommended for data collection without signal filtering.
    """)

    # Current preset info
    st.info(f"**Current Preset:** {summary['preset']} - {summary['preset_description']}")

    # Preset selector
    col1, col2 = st.columns([2, 1])
    with col1:
        preset_names = list(config['presets'].keys())
        current_preset = summary['preset']
        current_index = preset_names.index(current_preset) if current_preset in preset_names else 0

        new_preset = st.selectbox(
            "Select Preset",
            options=preset_names,
            index=current_index,
            key="intel_preset_selector"
        )

    with col2:
        if st.button("Apply Preset", key="apply_preset"):
            if save_parameter('intelligence_preset', new_preset, 'streamlit_user', f'Changed preset to {new_preset}'):
                st.success(f"Preset changed to: {new_preset}")
                st.rerun()

    # Show preset comparison table
    st.markdown("### Preset Comparison")
    preset_data = []
    for name, details in config['presets'].items():
        components = config['preset_components'].get(name, {})
        enabled = [k.replace('_filter', '') for k, v in components.items() if v]
        preset_data.append({
            'Preset': name,
            'Threshold': f"{details['threshold']:.0%}",
            'Engine': 'Yes' if details['use_intelligence_engine'] else 'No',
            'Enabled Filters': ', '.join(enabled) if enabled else 'None',
            'Description': details['description'],
        })

    df = pd.DataFrame(preset_data)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_core_settings(config: Dict[str, Any]):
    """Render core intelligence settings"""
    st.subheader("Core Settings")

    params_by_category = get_parameters_by_category()

    # Track pending changes
    if 'intel_pending_changes' not in st.session_state:
        st.session_state.intel_pending_changes = {}

    # User identifier for audit
    updated_by = st.text_input(
        "Your Name (for audit trail)",
        value=st.session_state.get('intel_config_user', 'streamlit_user'),
        key="intel_core_user"
    )
    st.session_state['intel_config_user'] = updated_by

    # Core settings
    if 'core' in params_by_category:
        with st.expander("Core Intelligence", expanded=True):
            render_parameter_group(params_by_category['core'], config)

    # Component weights
    if 'weights' in params_by_category:
        with st.expander("Component Weights", expanded=False):
            st.markdown("Weights determine how much each component contributes to the intelligence score.")
            render_parameter_group(params_by_category['weights'], config)

    # Component enablement
    if 'components' in params_by_category:
        with st.expander("Component Filters", expanded=False):
            st.markdown("Enable/disable individual filtering components. When disabled, signals bypass that filter.")
            render_parameter_group(params_by_category['components'], config)

    # Filtering settings
    if 'filtering' in params_by_category:
        with st.expander("Filtering Settings", expanded=False):
            render_parameter_group(params_by_category['filtering'], config)

    # Analysis settings
    if 'analysis' in params_by_category:
        with st.expander("Analysis Settings", expanded=False):
            render_parameter_group(params_by_category['analysis'], config)

    # Scanner settings
    if 'scanner' in params_by_category:
        with st.expander("Scanner Settings", expanded=False):
            render_parameter_group(params_by_category['scanner'], config)

    # Storage settings
    if 'storage' in params_by_category:
        with st.expander("Storage Settings", expanded=False):
            render_parameter_group(params_by_category['storage'], config)

    # Debug settings
    if 'debug' in params_by_category:
        with st.expander("Debug Settings", expanded=False):
            render_parameter_group(params_by_category['debug'], config)

    # Claude AI integration
    if 'claude' in params_by_category:
        with st.expander("Claude AI Integration", expanded=False):
            render_parameter_group(params_by_category['claude'], config)

    # Save button
    if st.session_state.intel_pending_changes:
        st.divider()
        st.warning(f"You have {len(st.session_state.intel_pending_changes)} unsaved changes")

        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("Save Changes", type="primary", key="save_intel_core"):
                if save_multiple_parameters(
                    st.session_state.intel_pending_changes,
                    updated_by,
                    "Bulk update from Streamlit"
                ):
                    st.success("Changes saved successfully!")
                    st.session_state.intel_pending_changes = {}
                    st.rerun()
        with col2:
            if st.button("Discard Changes", key="discard_intel_core"):
                st.session_state.intel_pending_changes = {}
                st.rerun()


def render_parameter_group(params: List[Dict[str, Any]], config: Dict[str, Any]):
    """Render a group of parameters with appropriate input widgets"""
    cols = st.columns(2)

    for i, param in enumerate(params):
        name = param['name']
        value = param['value']
        param_type = param['type']
        description = param.get('description', '')
        min_val = param.get('min_value')
        max_val = param.get('max_value')
        valid_options = param.get('valid_options')
        is_editable = param.get('is_editable', True)

        with cols[i % 2]:
            if param_type == 'bool':
                new_val = st.checkbox(
                    name.replace('_', ' ').title(),
                    value=value,
                    help=description,
                    key=f"intel_{name}",
                    disabled=not is_editable
                )
            elif param_type == 'int':
                new_val = st.number_input(
                    name.replace('_', ' ').title(),
                    value=int(value),
                    min_value=int(min_val) if min_val is not None else None,
                    max_value=int(max_val) if max_val is not None else None,
                    help=description,
                    key=f"intel_{name}",
                    disabled=not is_editable
                )
            elif param_type == 'float':
                new_val = st.number_input(
                    name.replace('_', ' ').title(),
                    value=float(value),
                    min_value=float(min_val) if min_val is not None else None,
                    max_value=float(max_val) if max_val is not None else None,
                    step=0.01,
                    format="%.3f",
                    help=description,
                    key=f"intel_{name}",
                    disabled=not is_editable
                )
            elif valid_options:
                # Dropdown for string with valid options
                options = json.loads(valid_options) if isinstance(valid_options, str) else valid_options
                current_index = options.index(value) if value in options else 0
                new_val = st.selectbox(
                    name.replace('_', ' ').title(),
                    options=options,
                    index=current_index,
                    help=description,
                    key=f"intel_{name}",
                    disabled=not is_editable
                )
            else:
                new_val = st.text_input(
                    name.replace('_', ' ').title(),
                    value=str(value),
                    help=description,
                    key=f"intel_{name}",
                    disabled=not is_editable
                )

            # Track changes
            if new_val != value and is_editable:
                st.session_state.intel_pending_changes[name] = new_val
            elif name in st.session_state.intel_pending_changes and new_val == value:
                del st.session_state.intel_pending_changes[name]


def render_smart_money_settings(config: Dict[str, Any]):
    """Render Smart Money configuration section"""
    st.subheader("Smart Money Configuration")

    st.markdown("""
    Smart Money analysis includes:
    - **Market Structure Analysis**: Break of Structure (BOS), Change of Character (ChoCh), Swing detection
    - **Order Flow Analysis**: Order Blocks, Fair Value Gaps (FVGs), Supply/Demand zones
    """)

    params_by_category = get_parameters_by_category()

    if 'smart_money' in params_by_category:
        smart_money_params = params_by_category['smart_money']

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Data Collection")
            for param in smart_money_params:
                if 'collection' in param['name']:
                    name = param['name']
                    new_val = st.checkbox(
                        param['description'],
                        value=param['value'],
                        key=f"sm_{name}"
                    )
                    if new_val != param['value']:
                        st.session_state.intel_pending_changes[name] = new_val

        with col2:
            st.markdown("### Signal Validation")
            for param in smart_money_params:
                if 'validation' in param['name']:
                    name = param['name']
                    new_val = st.checkbox(
                        param['description'],
                        value=param['value'],
                        key=f"sm_{name}"
                    )
                    if new_val != param['value']:
                        st.session_state.intel_pending_changes[name] = new_val

    st.divider()

    # Show current status
    collection_enabled = config['parameters'].get('enable_smart_money_collection', {}).get('value', False)
    order_flow_enabled = config['parameters'].get('enable_order_flow_collection', {}).get('value', False)

    if collection_enabled or order_flow_enabled:
        st.success("Smart Money data collection is **ENABLED**")
        st.markdown(f"""
        - Market Structure (BOS/ChoCh/Swing): {'Enabled' if collection_enabled else 'Disabled'}
        - Order Flow (Order Blocks/FVGs): {'Enabled' if order_flow_enabled else 'Disabled'}
        """)
    else:
        st.warning("Smart Money data collection is **DISABLED**")


def render_regime_detection_settings(config: Dict[str, Any]):
    """Render regime detection configuration"""
    st.subheader("Regime Detection Settings")

    st.markdown("""
    Enhanced regime detection uses ADX-based analysis for more accurate market classification.
    This replaces the legacy volatility-based regime detection.
    """)

    params_by_category = get_parameters_by_category()

    if 'regime' in params_by_category:
        with st.expander("ADX-Based Regime Detection", expanded=True):
            render_parameter_group(params_by_category['regime'], config)

    if 'modifiers' in params_by_category:
        with st.expander("Confidence Modifier Settings", expanded=False):
            render_parameter_group(params_by_category['modifiers'], config)

    # Show regime thresholds visualization
    st.markdown("### Regime Classification Thresholds")

    adx_trending = config['parameters'].get('adx_trending_threshold', {}).get('value', 25)
    adx_strong = config['parameters'].get('adx_strong_trend_threshold', {}).get('value', 40)
    adx_weak = config['parameters'].get('adx_weak_trend_threshold', {}).get('value', 20)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Weak/Ranging", f"ADX < {adx_weak}")
    with col2:
        st.metric("Trending", f"ADX > {adx_trending}")
    with col3:
        st.metric("Strong Trend", f"ADX > {adx_strong}")


def render_regime_modifiers_section(config: Dict[str, Any]):
    """Render regime-strategy confidence modifiers"""
    st.subheader("Regime-Strategy Confidence Modifiers")

    st.markdown("""
    These modifiers adjust signal confidence based on how well a strategy performs in each market regime.
    - **1.0** = Perfect compatibility (no adjustment)
    - **0.7-0.9** = Good compatibility (slight reduction)
    - **0.4-0.6** = Moderate compatibility (medium reduction)
    - **< 0.4** = Poor compatibility (significant reduction or blocked)
    """)

    regime_modifiers = config['regime_modifiers']

    if not regime_modifiers:
        st.warning("No regime modifiers found in database.")
        return

    # Regime selector
    regimes = list(regime_modifiers.keys())
    selected_regime = st.selectbox("Select Regime", regimes, key="modifier_regime")

    if selected_regime:
        st.markdown(f"### {selected_regime.replace('_', ' ').title()} Regime")

        modifiers = regime_modifiers[selected_regime]

        # Create DataFrame for display
        modifier_data = []
        for strategy, details in modifiers.items():
            modifier = details['modifier']
            # Determine compatibility level
            if modifier >= 0.9:
                level = "Excellent"
                color = "green"
            elif modifier >= 0.7:
                level = "Good"
                color = "blue"
            elif modifier >= 0.5:
                level = "Moderate"
                color = "orange"
            else:
                level = "Poor"
                color = "red"

            modifier_data.append({
                'Strategy': strategy,
                'Modifier': f"{modifier:.2f}",
                'Compatibility': level,
                'Description': details.get('description', ''),
            })

        df = pd.DataFrame(modifier_data)
        df = df.sort_values('Modifier', ascending=False)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Edit section
        st.divider()
        st.markdown("### Edit Modifier")

        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            strategies = list(modifiers.keys())
            edit_strategy = st.selectbox("Strategy", strategies, key="edit_strategy")
        with col2:
            current_modifier = modifiers.get(edit_strategy, {}).get('modifier', 1.0)
            new_modifier = st.number_input(
                "Modifier",
                value=current_modifier,
                min_value=0.0,
                max_value=1.0,
                step=0.05,
                key="new_modifier"
            )
        with col3:
            if st.button("Save Modifier", key="save_modifier"):
                if save_regime_modifier(
                    selected_regime,
                    edit_strategy,
                    new_modifier,
                    None,
                    st.session_state.get('intel_config_user', 'streamlit_user')
                ):
                    st.success(f"Updated {edit_strategy} modifier to {new_modifier}")
                    st.rerun()


def render_audit_trail():
    """Render audit trail for intelligence config changes"""
    st.subheader("Configuration Audit Trail")

    # Load audit history
    audit_history = get_audit_history(limit=100)

    if not audit_history:
        st.info("No configuration changes recorded yet.")
        return

    # Convert to DataFrame
    audit_data = []
    for record in audit_history:
        audit_data.append({
            'Timestamp': record['changed_at'].strftime('%Y-%m-%d %H:%M:%S') if record['changed_at'] else '',
            'Table': record['table_name'],
            'Change Type': record['change_type'],
            'Changed By': record['changed_by'],
            'Reason': record.get('change_reason', ''),
        })

    df = pd.DataFrame(audit_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Show details for selected record
    if audit_history:
        st.divider()
        st.markdown("### Change Details")

        record_index = st.selectbox(
            "Select record to view details",
            options=range(len(audit_history)),
            format_func=lambda x: f"{audit_data[x]['Timestamp']} - {audit_data[x]['Change Type']} by {audit_data[x]['Changed By']}",
            key="audit_record_selector"
        )

        selected = audit_history[record_index]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Previous Values:**")
            if selected.get('previous_values'):
                st.json(selected['previous_values'])
            else:
                st.text("N/A")

        with col2:
            st.markdown("**New Values:**")
            if selected.get('new_values'):
                st.json(selected['new_values'])
            else:
                st.text("N/A")
