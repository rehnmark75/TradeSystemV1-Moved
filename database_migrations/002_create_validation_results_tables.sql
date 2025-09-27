-- =============================================================================
-- STATISTICAL VALIDATION FRAMEWORK DATABASE SCHEMA
-- Migration: 002_create_validation_results_tables.sql
-- Description: Create database tables for statistical validation framework results
-- =============================================================================

-- =============================================================================
-- VALIDATION RESULTS STORAGE
-- =============================================================================

CREATE TABLE IF NOT EXISTS backtest_validation_results (
    id SERIAL PRIMARY KEY,
    execution_id INTEGER REFERENCES backtest_executions(id) ON DELETE CASCADE,

    -- Validation metadata
    validation_timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    validation_level VARCHAR(20) NOT NULL, -- MEDIUM, HIGH, etc.
    validation_framework_version VARCHAR(20) DEFAULT '1.0.0',

    -- Overall validation scores
    composite_score DECIMAL(5,4) NOT NULL,
    validation_result VARCHAR(20) NOT NULL, -- PASS, FAIL, WARNING
    confidence_level DECIMAL(5,4),

    -- Component scores
    data_quality_score DECIMAL(5,4),
    statistical_significance_score DECIMAL(5,4),
    overfitting_detection_score DECIMAL(5,4),
    realtime_correlation_score DECIMAL(5,4),
    pipeline_consistency_score DECIMAL(5,4),

    -- Detailed validation data (JSON)
    validation_data JSONB,

    -- Status and timestamps
    status VARCHAR(20) DEFAULT 'completed',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),

    CONSTRAINT unique_validation_per_execution UNIQUE(execution_id, validation_timestamp)
);

-- =============================================================================
-- VALIDATION ALERTS AND MONITORING
-- =============================================================================

CREATE TABLE IF NOT EXISTS validation_alerts (
    id SERIAL PRIMARY KEY,
    alert_id VARCHAR(100) UNIQUE NOT NULL,
    execution_id INTEGER REFERENCES backtest_executions(id) ON DELETE CASCADE,

    -- Alert details
    severity VARCHAR(20) NOT NULL, -- INFO, WARNING, CRITICAL, EMERGENCY
    metric VARCHAR(50) NOT NULL, -- performance_correlation, data_quality_degradation, etc.
    message TEXT NOT NULL,

    -- Threshold information
    threshold_value DECIMAL(12,6),
    actual_value DECIMAL(12,6),

    -- Alert metadata
    alert_details JSONB,
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),

    -- Alert lifecycle
    acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_at TIMESTAMP,
    acknowledged_by VARCHAR(100),
    resolved BOOLEAN DEFAULT FALSE,
    resolved_at TIMESTAMP,
    resolved_by VARCHAR(100),

    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),

    CONSTRAINT valid_alert_severity CHECK (severity IN ('INFO', 'WARNING', 'CRITICAL', 'EMERGENCY'))
);

-- =============================================================================
-- CORRELATION ANALYSIS RESULTS
-- =============================================================================

CREATE TABLE IF NOT EXISTS backtest_correlation_analysis (
    id SERIAL PRIMARY KEY,
    execution_id INTEGER REFERENCES backtest_executions(id) ON DELETE CASCADE,
    strategy_name VARCHAR(50) NOT NULL,
    epic VARCHAR(50),

    -- Analysis metadata
    analysis_timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    sample_size INTEGER NOT NULL,
    correlation_possible BOOLEAN DEFAULT FALSE,

    -- Correlation metrics
    pearson_correlation DECIMAL(8,6),
    spearman_correlation DECIMAL(8,6),
    kendall_tau DECIMAL(8,6),
    regression_r2 DECIMAL(8,6),

    -- Risk metrics
    tracking_error DECIMAL(12,6),
    information_ratio DECIMAL(8,4),
    information_coefficient DECIMAL(8,6),

    -- Stability metrics
    correlation_stability DECIMAL(5,4),
    hit_rate DECIMAL(5,4),

    -- Detailed analysis data
    correlation_data JSONB,

    created_at TIMESTAMP DEFAULT NOW(),

    CONSTRAINT unique_correlation_analysis UNIQUE(execution_id, strategy_name, epic, analysis_timestamp)
);

-- =============================================================================
-- OVERFITTING DETECTION RESULTS
-- =============================================================================

CREATE TABLE IF NOT EXISTS backtest_overfitting_analysis (
    id SERIAL PRIMARY KEY,
    execution_id INTEGER REFERENCES backtest_executions(id) ON DELETE CASCADE,

    -- Analysis metadata
    analysis_timestamp TIMESTAMP NOT NULL DEFAULT NOW(),

    -- Overfitting assessment
    is_overfitted BOOLEAN NOT NULL DEFAULT FALSE,
    risk_level VARCHAR(20) NOT NULL, -- LOW, MODERATE, HIGH, SEVERE
    confidence_score DECIMAL(5,4) NOT NULL,

    -- Walk-forward analysis results
    wfa_periods_analyzed INTEGER,
    wfa_degradation_ratio DECIMAL(8,4),
    wfa_is_significant_degradation BOOLEAN DEFAULT FALSE,

    -- Cross-validation results
    cv_folds INTEGER,
    cv_score_mean DECIMAL(8,6),
    cv_overfitting_ratio DECIMAL(8,4),
    cv_stability_score DECIMAL(5,4),

    -- Parameter sensitivity
    parameters_tested INTEGER,
    robustness_score DECIMAL(5,4),
    parameter_stability DECIMAL(5,4),

    -- Deflated Sharpe ratio
    original_sharpe_ratio DECIMAL(8,4),
    deflated_sharpe_ratio DECIMAL(8,4),
    n_trials_estimated INTEGER,

    -- Detailed analysis data
    overfitting_data JSONB,

    created_at TIMESTAMP DEFAULT NOW(),

    CONSTRAINT valid_overfitting_risk_level CHECK (risk_level IN ('LOW', 'MODERATE', 'HIGH', 'SEVERE'))
);

-- =============================================================================
-- DATA QUALITY ASSESSMENT RESULTS
-- =============================================================================

CREATE TABLE IF NOT EXISTS backtest_data_quality_assessment (
    id SERIAL PRIMARY KEY,
    execution_id INTEGER REFERENCES backtest_executions(id) ON DELETE CASCADE,

    -- Assessment metadata
    assessment_timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    total_records INTEGER NOT NULL,

    -- Overall quality scores
    overall_quality_score DECIMAL(5,4) NOT NULL,
    quality_level VARCHAR(20) NOT NULL, -- EXCELLENT, GOOD, ACCEPTABLE, POOR, UNACCEPTABLE

    -- Component quality scores
    completeness_score DECIMAL(5,4),
    consistency_score DECIMAL(5,4),
    validity_score DECIMAL(5,4),
    accuracy_score DECIMAL(5,4),
    timeliness_score DECIMAL(5,4),

    -- Quality metrics
    missing_data_percentage DECIMAL(5,4),
    outlier_percentage DECIMAL(5,4),
    temporal_gaps_count INTEGER,

    -- Detailed quality data
    quality_data JSONB,

    created_at TIMESTAMP DEFAULT NOW(),

    CONSTRAINT valid_data_quality_level CHECK (quality_level IN ('EXCELLENT', 'GOOD', 'ACCEPTABLE', 'POOR', 'UNACCEPTABLE'))
);

-- =============================================================================
-- PIPELINE CONSISTENCY RESULTS
-- =============================================================================

CREATE TABLE IF NOT EXISTS backtest_pipeline_consistency (
    id SERIAL PRIMARY KEY,
    execution_id INTEGER REFERENCES backtest_executions(id) ON DELETE CASCADE,

    -- Consistency assessment
    assessment_timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    overall_consistency_score DECIMAL(5,4) NOT NULL,
    consistency_level VARCHAR(20) NOT NULL, -- PERFECT, EXCELLENT, GOOD, ACCEPTABLE, POOR, CRITICAL

    -- Component consistency scores
    signal_generation_score DECIMAL(5,4),
    feature_calculation_score DECIMAL(5,4),
    trade_validation_score DECIMAL(5,4),
    configuration_score DECIMAL(5,4),
    data_processing_score DECIMAL(5,4),

    -- Configuration analysis
    configuration_hash_match BOOLEAN DEFAULT FALSE,
    parameter_differences_count INTEGER,
    version_compatible BOOLEAN DEFAULT TRUE,

    -- Test results summary
    total_consistency_tests INTEGER,
    passed_consistency_tests INTEGER,

    -- Detailed consistency data
    consistency_data JSONB,

    created_at TIMESTAMP DEFAULT NOW(),

    CONSTRAINT valid_consistency_level CHECK (consistency_level IN ('PERFECT', 'EXCELLENT', 'GOOD', 'ACCEPTABLE', 'POOR', 'CRITICAL'))
);

-- =============================================================================
-- VALIDATION REPORTS STORAGE
-- =============================================================================

CREATE TABLE IF NOT EXISTS validation_reports (
    id SERIAL PRIMARY KEY,
    execution_id INTEGER REFERENCES backtest_executions(id) ON DELETE CASCADE,

    -- Report metadata
    report_timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    report_type VARCHAR(30) NOT NULL, -- executive_summary, technical_detailed, comprehensive, etc.
    report_version VARCHAR(20) DEFAULT '1.0.0',

    -- Report content
    executive_summary TEXT,
    validation_overview TEXT,
    risk_assessment JSONB,
    recommendations TEXT[],

    -- Full report data
    report_data JSONB,

    -- Report status
    status VARCHAR(20) DEFAULT 'generated',
    generated_by VARCHAR(100) DEFAULT 'statistical_validation_framework',

    created_at TIMESTAMP DEFAULT NOW(),

    CONSTRAINT valid_report_type CHECK (report_type IN ('executive_summary', 'technical_detailed', 'compliance_audit', 'risk_assessment', 'performance_benchmark', 'comprehensive'))
);

-- =============================================================================
-- MONITORING METRICS HISTORY
-- =============================================================================

CREATE TABLE IF NOT EXISTS validation_monitoring_history (
    id BIGSERIAL PRIMARY KEY,
    execution_id INTEGER REFERENCES backtest_executions(id) ON DELETE CASCADE,

    -- Monitoring metadata
    timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
    metric_type VARCHAR(50) NOT NULL,

    -- Metric values
    metric_value DECIMAL(12,6),
    threshold_value DECIMAL(12,6),
    is_within_threshold BOOLEAN DEFAULT TRUE,

    -- Monitoring context
    monitoring_window_minutes INTEGER,
    sample_size INTEGER,

    -- Additional metric data
    metric_data JSONB,

    created_at TIMESTAMP DEFAULT NOW()
);

-- =============================================================================
-- OPTIMIZED INDEXES FOR VALIDATION QUERIES
-- =============================================================================

-- Validation results lookup
CREATE INDEX IF NOT EXISTS idx_validation_results_execution ON backtest_validation_results(execution_id, validation_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_validation_results_score ON backtest_validation_results(composite_score DESC, validation_result);

-- Alert management
CREATE INDEX IF NOT EXISTS idx_validation_alerts_execution ON validation_alerts(execution_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_validation_alerts_severity ON validation_alerts(severity, acknowledged, resolved);
CREATE INDEX IF NOT EXISTS idx_validation_alerts_metric ON validation_alerts(metric, timestamp DESC);

-- Correlation analysis
CREATE INDEX IF NOT EXISTS idx_correlation_analysis_execution ON backtest_correlation_analysis(execution_id, analysis_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_correlation_analysis_strategy ON backtest_correlation_analysis(strategy_name, epic, analysis_timestamp DESC);

-- Overfitting analysis
CREATE INDEX IF NOT EXISTS idx_overfitting_analysis_execution ON backtest_overfitting_analysis(execution_id, analysis_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_overfitting_analysis_risk ON backtest_overfitting_analysis(risk_level, confidence_score DESC);

-- Data quality assessment
CREATE INDEX IF NOT EXISTS idx_data_quality_execution ON backtest_data_quality_assessment(execution_id, assessment_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_data_quality_score ON backtest_data_quality_assessment(overall_quality_score DESC, quality_level);

-- Pipeline consistency
CREATE INDEX IF NOT EXISTS idx_pipeline_consistency_execution ON backtest_pipeline_consistency(execution_id, assessment_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_pipeline_consistency_score ON backtest_pipeline_consistency(overall_consistency_score DESC, consistency_level);

-- Report lookup
CREATE INDEX IF NOT EXISTS idx_validation_reports_execution ON validation_reports(execution_id, report_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_validation_reports_type ON validation_reports(report_type, status, report_timestamp DESC);

-- Monitoring history (partitioned by time for performance)
CREATE INDEX IF NOT EXISTS idx_monitoring_history_execution_time ON validation_monitoring_history(execution_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_monitoring_history_metric ON validation_monitoring_history(metric_type, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_monitoring_history_threshold ON validation_monitoring_history(is_within_threshold, timestamp DESC)
    WHERE is_within_threshold = FALSE;

-- =============================================================================
-- HELPER FUNCTIONS FOR VALIDATION
-- =============================================================================

-- Function to get latest validation results
CREATE OR REPLACE FUNCTION get_latest_validation_results(p_execution_id INTEGER)
RETURNS TABLE (
    validation_timestamp TIMESTAMP,
    composite_score DECIMAL,
    validation_result VARCHAR,
    component_scores JSONB
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        vr.validation_timestamp,
        vr.composite_score,
        vr.validation_result,
        jsonb_build_object(
            'data_quality', vr.data_quality_score,
            'statistical_significance', vr.statistical_significance_score,
            'overfitting_detection', vr.overfitting_detection_score,
            'realtime_correlation', vr.realtime_correlation_score,
            'pipeline_consistency', vr.pipeline_consistency_score
        ) as component_scores
    FROM backtest_validation_results vr
    WHERE vr.execution_id = p_execution_id
    ORDER BY vr.validation_timestamp DESC
    LIMIT 1;
END;
$$ LANGUAGE plpgsql;

-- Function to get active alerts for execution
CREATE OR REPLACE FUNCTION get_active_validation_alerts(p_execution_id INTEGER)
RETURNS TABLE (
    alert_id VARCHAR,
    severity VARCHAR,
    metric VARCHAR,
    message TEXT,
    timestamp TIMESTAMP
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        va.alert_id,
        va.severity,
        va.metric,
        va.message,
        va.timestamp
    FROM validation_alerts va
    WHERE va.execution_id = p_execution_id
      AND va.resolved = FALSE
      AND va.timestamp > NOW() - INTERVAL '24 hours'
    ORDER BY va.timestamp DESC;
END;
$$ LANGUAGE plpgsql;

-- Function to calculate validation trends
CREATE OR REPLACE FUNCTION get_validation_trends(p_execution_id INTEGER, p_days INTEGER DEFAULT 7)
RETURNS TABLE (
    date DATE,
    avg_composite_score DECIMAL,
    validation_count INTEGER
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        vr.validation_timestamp::DATE as date,
        AVG(vr.composite_score) as avg_composite_score,
        COUNT(*) as validation_count
    FROM backtest_validation_results vr
    WHERE vr.execution_id = p_execution_id
      AND vr.validation_timestamp > NOW() - INTERVAL '%d days'
    GROUP BY vr.validation_timestamp::DATE
    ORDER BY date DESC;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- TRIGGERS FOR AUTOMATIC UPDATES
-- =============================================================================

-- Update validation_results.updated_at on changes
CREATE OR REPLACE FUNCTION update_validation_results_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_validation_results_updated_at
    BEFORE UPDATE ON backtest_validation_results
    FOR EACH ROW EXECUTE FUNCTION update_validation_results_updated_at();

-- Update validation_alerts.updated_at on changes
CREATE OR REPLACE FUNCTION update_validation_alerts_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    -- Set acknowledged_at when acknowledged changes to true
    IF OLD.acknowledged = FALSE AND NEW.acknowledged = TRUE THEN
        NEW.acknowledged_at = NOW();
    END IF;
    -- Set resolved_at when resolved changes to true
    IF OLD.resolved = FALSE AND NEW.resolved = TRUE THEN
        NEW.resolved_at = NOW();
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_validation_alerts_updated_at
    BEFORE UPDATE ON validation_alerts
    FOR EACH ROW EXECUTE FUNCTION update_validation_alerts_updated_at();

-- =============================================================================
-- VIEWS FOR COMMON VALIDATION QUERIES
-- =============================================================================

-- Comprehensive validation overview
CREATE OR REPLACE VIEW validation_overview AS
SELECT
    be.id as execution_id,
    be.execution_name,
    be.strategy_name,
    be.status as execution_status,
    vr.validation_timestamp,
    vr.composite_score,
    vr.validation_result,
    vr.confidence_level,
    dqa.overall_quality_score as data_quality_score,
    oa.risk_level as overfitting_risk,
    pc.consistency_level as pipeline_consistency,
    COUNT(va.id) as active_alerts_count
FROM backtest_executions be
LEFT JOIN backtest_validation_results vr ON be.id = vr.execution_id
LEFT JOIN backtest_data_quality_assessment dqa ON be.id = dqa.execution_id
LEFT JOIN backtest_overfitting_analysis oa ON be.id = oa.execution_id
LEFT JOIN backtest_pipeline_consistency pc ON be.id = pc.execution_id
LEFT JOIN validation_alerts va ON be.id = va.execution_id AND va.resolved = FALSE
GROUP BY be.id, be.execution_name, be.strategy_name, be.status,
         vr.validation_timestamp, vr.composite_score, vr.validation_result, vr.confidence_level,
         dqa.overall_quality_score, oa.risk_level, pc.consistency_level;

-- Alert summary view
CREATE OR REPLACE VIEW validation_alert_summary AS
SELECT
    execution_id,
    severity,
    COUNT(*) as alert_count,
    MAX(timestamp) as latest_alert,
    COUNT(CASE WHEN acknowledged = FALSE THEN 1 END) as unacknowledged_count,
    COUNT(CASE WHEN resolved = FALSE THEN 1 END) as unresolved_count
FROM validation_alerts
WHERE timestamp > NOW() - INTERVAL '24 hours'
GROUP BY execution_id, severity;

-- Grant permissions (adjust as needed for your environment)
-- GRANT SELECT, INSERT, UPDATE, DELETE ON backtest_validation_results TO trading_user;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON validation_alerts TO trading_user;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON backtest_correlation_analysis TO trading_user;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON backtest_overfitting_analysis TO trading_user;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON backtest_data_quality_assessment TO trading_user;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON backtest_pipeline_consistency TO trading_user;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON validation_reports TO trading_user;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON validation_monitoring_history TO trading_user;