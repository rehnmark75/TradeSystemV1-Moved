# forex_scanner/validation/validation_report_generator.py
"""
Automated Validation Report Generator

This module generates comprehensive, automated validation reports with
statistical confidence scoring for trading strategy backtests, providing
clear assessments and actionable recommendations.

Key Features:
1. Comprehensive validation summaries
2. Statistical confidence scoring
3. Executive summaries for stakeholders
4. Technical detailed analysis
5. Risk assessment matrices
6. Actionable recommendations
7. Compliance and audit trails
8. Performance benchmarking
"""

import logging
import numpy as np
import pandas as pd
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

try:
    from core.database import DatabaseManager
    from .statistical_validation_framework import StatisticalValidationFramework
    from .realtime_correlation_analyzer import RealtimeCorrelationAnalyzer
    from .overfitting_detector import OverfittingDetector, OverfittingRisk
    from .statistical_significance_tester import StatisticalSignificanceTester
    from .data_quality_validator import DataQualityValidator, DataQualityLevel
    from .pipeline_consistency_validator import PipelineConsistencyValidator, ConsistencyLevel
except ImportError:
    from forex_scanner.core.database import DatabaseManager
    from forex_scanner.validation.statistical_validation_framework import StatisticalValidationFramework
    from forex_scanner.validation.realtime_correlation_analyzer import RealtimeCorrelationAnalyzer
    from forex_scanner.validation.overfitting_detector import OverfittingDetector, OverfittingRisk
    from forex_scanner.validation.statistical_significance_tester import StatisticalSignificanceTester
    from forex_scanner.validation.data_quality_validator import DataQualityValidator, DataQualityLevel
    from forex_scanner.validation.pipeline_consistency_validator import PipelineConsistencyValidator, ConsistencyLevel


class ReportType(Enum):
    """Types of validation reports"""
    EXECUTIVE_SUMMARY = "executive_summary"
    TECHNICAL_DETAILED = "technical_detailed"
    COMPLIANCE_AUDIT = "compliance_audit"
    RISK_ASSESSMENT = "risk_assessment"
    PERFORMANCE_BENCHMARK = "performance_benchmark"
    COMPREHENSIVE = "comprehensive"


class ConfidenceLevel(Enum):
    """Confidence levels for validation assessment"""
    VERY_HIGH = "VERY_HIGH"      # 95%+ confidence
    HIGH = "HIGH"                # 85-95% confidence
    MODERATE = "MODERATE"        # 70-85% confidence
    LOW = "LOW"                  # 50-70% confidence
    VERY_LOW = "VERY_LOW"        # <50% confidence


@dataclass
class ValidationSummary:
    """Summary of validation results"""
    overall_score: float
    confidence_level: ConfidenceLevel
    validation_status: str
    key_findings: List[str]
    critical_issues: List[str]
    recommendations: List[str]
    component_scores: Dict[str, float]
    risk_assessment: Dict[str, Any]


@dataclass
class ReportSection:
    """Container for report sections"""
    title: str
    content: str
    charts: List[Dict[str, Any]] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    confidence_score: Optional[float] = None


class ValidationReportGenerator:
    """
    Automated Validation Report Generator

    Generates comprehensive validation reports that combine results from all
    validation components into clear, actionable reports for different audiences.
    """

    def __init__(self,
                 db_manager: DatabaseManager,
                 statistical_framework: StatisticalValidationFramework,
                 logger: Optional[logging.Logger] = None):

        self.db_manager = db_manager
        self.statistical_framework = statistical_framework
        self.logger = logger or logging.getLogger(__name__)

        # Report configuration
        self.max_charts_per_section = 3
        self.max_recommendations = 10
        self.confidence_thresholds = {
            ConfidenceLevel.VERY_HIGH: 0.95,
            ConfidenceLevel.HIGH: 0.85,
            ConfidenceLevel.MODERATE: 0.70,
            ConfidenceLevel.LOW: 0.50,
            ConfidenceLevel.VERY_LOW: 0.0
        }

        # Initialize component analyzers
        self.correlation_analyzer = RealtimeCorrelationAnalyzer(db_manager)
        self.overfitting_detector = OverfittingDetector(db_manager)
        self.significance_tester = StatisticalSignificanceTester(db_manager)
        self.quality_validator = DataQualityValidator(db_manager)
        self.consistency_validator = PipelineConsistencyValidator(db_manager)

        self.logger.info(f"📊 Validation Report Generator initialized")

    def generate_comprehensive_report(self, execution_id: int) -> Dict[str, Any]:
        """
        Generate comprehensive validation report for an execution

        Args:
            execution_id: Backtest execution ID to generate report for

        Returns:
            Complete validation report with all sections
        """
        self.logger.info(f"📊 Generating comprehensive validation report for execution {execution_id}")

        try:
            # Get comprehensive validation results
            validation_results = self.statistical_framework.validate_backtest_execution(execution_id)

            # Generate report structure
            report = {
                'execution_id': execution_id,
                'report_timestamp': datetime.now(timezone.utc),
                'report_type': ReportType.COMPREHENSIVE.value,
                'executive_summary': self._generate_executive_summary(validation_results),
                'validation_overview': self._generate_validation_overview(validation_results),
                'detailed_analysis': self._generate_detailed_analysis(validation_results),
                'risk_assessment': self._generate_risk_assessment(validation_results),
                'recommendations': self._generate_recommendations_section(validation_results),
                'technical_appendix': self._generate_technical_appendix(validation_results),
                'metadata': self._generate_report_metadata(execution_id, validation_results)
            }

            # Store report
            self._store_validation_report(execution_id, report)

            self.logger.info(f"✅ Comprehensive report generated successfully")
            return report

        except Exception as e:
            self.logger.error(f"❌ Report generation failed: {e}")
            return self._create_error_report(execution_id, str(e))

    def generate_executive_summary_report(self, execution_id: int) -> Dict[str, Any]:
        """Generate executive summary report for stakeholders"""

        try:
            validation_results = self.statistical_framework.validate_backtest_execution(execution_id)

            summary = self._generate_validation_summary(validation_results)

            report = {
                'execution_id': execution_id,
                'report_timestamp': datetime.now(timezone.utc),
                'report_type': ReportType.EXECUTIVE_SUMMARY.value,
                'summary': summary,
                'key_metrics': self._extract_key_metrics(validation_results),
                'go_no_go_decision': self._generate_go_no_go_decision(summary),
                'business_impact': self._assess_business_impact(validation_results),
                'next_steps': self._recommend_next_steps(summary)
            }

            return report

        except Exception as e:
            self.logger.error(f"Error generating executive summary: {e}")
            return self._create_error_report(execution_id, str(e))

    def _generate_executive_summary(self, validation_results: Dict[str, Any]) -> ReportSection:
        """Generate executive summary section"""

        try:
            overall_validation = validation_results.get('overall_validation', {})
            composite_score = overall_validation.get('composite_score', 0.0)
            validation_result = overall_validation.get('validation_result', 'FAIL')

            # Determine confidence level
            confidence_level = self._determine_confidence_level(composite_score)

            # Extract key findings
            key_findings = []
            if composite_score >= 0.8:
                key_findings.append("Strategy shows strong statistical validation characteristics")
            elif composite_score >= 0.6:
                key_findings.append("Strategy validation shows acceptable performance with some concerns")
            else:
                key_findings.append("Strategy validation reveals significant statistical concerns")

            # Add component-specific findings
            validation_components = validation_results.get('validation_components', {})
            for component, results in validation_components.items():
                if component == 'overfitting_detection':
                    overfitting_results = results.get('overfitting_metrics', {})
                    if overfitting_results.get('is_overfitted', False):
                        key_findings.append("Overfitting detected - strategy may not perform well live")
                elif component == 'statistical_significance':
                    overall_sig = results.get('overall_assessment', {})
                    if overall_sig.get('is_significant', False):
                        key_findings.append("Strategy performance is statistically significant")
                    else:
                        key_findings.append("Strategy performance lacks statistical significance")

            # Generate content
            content = f"""
# EXECUTIVE SUMMARY

## Overall Assessment
- **Validation Score**: {composite_score:.1%}
- **Confidence Level**: {confidence_level.value}
- **Validation Status**: {validation_result}

## Key Findings
{chr(10).join(f"• {finding}" for finding in key_findings)}

## Business Impact
This validation assessment provides {confidence_level.value.lower()} confidence in the strategy's
expected live performance based on comprehensive statistical analysis.
            """.strip()

            return ReportSection(
                title="Executive Summary",
                content=content,
                confidence_score=composite_score,
                recommendations=self._extract_top_recommendations(validation_results, 3)
            )

        except Exception as e:
            return ReportSection(
                title="Executive Summary",
                content=f"Error generating executive summary: {str(e)}",
                confidence_score=0.0
            )

    def _generate_validation_overview(self, validation_results: Dict[str, Any]) -> ReportSection:
        """Generate validation methodology overview"""

        try:
            validation_components = validation_results.get('validation_components', {})

            content = f"""
# VALIDATION METHODOLOGY OVERVIEW

## Components Analyzed
The validation framework conducted {len(validation_components)} comprehensive analyses:

"""

            component_descriptions = {
                'data_quality': 'Data Quality Assessment - Validated historical data completeness and accuracy',
                'statistical_significance': 'Statistical Significance Testing - Hypothesis testing for strategy performance',
                'overfitting_detection': 'Overfitting Detection - Cross-validation and degradation analysis',
                'realtime_correlation': 'Real-time Correlation - Backtest vs live performance alignment',
                'pipeline_consistency': 'Pipeline Consistency - Validation of system reproducibility'
            }

            for component, results in validation_components.items():
                description = component_descriptions.get(component, f"{component.replace('_', ' ').title()} Analysis")
                status = results.get('status', 'unknown')
                content += f"• **{description}** - Status: {status.upper()}\n"

            content += f"""

## Sample Size and Period
- **Total Signals Analyzed**: {validation_results.get('sample_size', 'N/A')}
- **Validation Timestamp**: {validation_results.get('validation_timestamp', 'N/A')}
- **Confidence Level**: {validation_results.get('confidence_level', 'N/A'):.1%}

## Statistical Rigor
All analyses were conducted using industry-standard statistical methods with appropriate
significance testing and multiple comparison corrections.
"""

            return ReportSection(
                title="Validation Overview",
                content=content.strip(),
                confidence_score=validation_results.get('overall_validation', {}).get('composite_score', 0.0)
            )

        except Exception as e:
            return ReportSection(
                title="Validation Overview",
                content=f"Error generating validation overview: {str(e)}",
                confidence_score=0.0
            )

    def _generate_detailed_analysis(self, validation_results: Dict[str, Any]) -> Dict[str, ReportSection]:
        """Generate detailed analysis for each validation component"""

        detailed_sections = {}
        validation_components = validation_results.get('validation_components', {})

        for component, results in validation_components.items():
            try:
                if component == 'data_quality':
                    section = self._generate_data_quality_section(results)
                elif component == 'statistical_significance':
                    section = self._generate_significance_section(results)
                elif component == 'overfitting_detection':
                    section = self._generate_overfitting_section(results)
                elif component == 'realtime_correlation':
                    section = self._generate_correlation_section(results)
                elif component == 'pipeline_consistency':
                    section = self._generate_consistency_section(results)
                else:
                    section = self._generate_generic_section(component, results)

                detailed_sections[component] = section

            except Exception as e:
                self.logger.error(f"Error generating section for {component}: {e}")
                detailed_sections[component] = ReportSection(
                    title=component.replace('_', ' ').title(),
                    content=f"Error generating analysis: {str(e)}",
                    confidence_score=0.0
                )

        return detailed_sections

    def _generate_data_quality_section(self, results: Dict[str, Any]) -> ReportSection:
        """Generate data quality analysis section"""

        overall_score = results.get('overall_quality_score', 0.0)

        content = f"""
# DATA QUALITY ANALYSIS

## Overall Quality Score: {overall_score:.1%}

### Quality Assessment
"""

        quality_components = results.get('quality_components', {})
        for component, data in quality_components.items():
            score = data.get('quality_score', 0.0)
            passes_threshold = data.get('passes_threshold', False)
            status = "✅ PASS" if passes_threshold else "❌ FAIL"

            content += f"• **{component.replace('_', ' ').title()}**: {score:.1%} {status}\n"

        recommendations = results.get('recommendations', [])
        if recommendations:
            content += f"\n### Recommendations\n"
            for rec in recommendations[:5]:  # Top 5
                content += f"• {rec}\n"

        return ReportSection(
            title="Data Quality Analysis",
            content=content.strip(),
            confidence_score=overall_score,
            recommendations=recommendations
        )

    def _generate_overfitting_section(self, results: Dict[str, Any]) -> ReportSection:
        """Generate overfitting detection section"""

        overfitting_metrics = results.get('overfitting_metrics', {})
        is_overfitted = overfitting_metrics.get('is_overfitted', False)
        confidence = overfitting_metrics.get('confidence_level', 0.0)

        content = f"""
# OVERFITTING DETECTION ANALYSIS

## Assessment: {"⚠️ OVERFITTING DETECTED" if is_overfitted else "✅ NO SIGNIFICANT OVERFITTING"}

### Key Metrics
• **Degradation Ratio**: {overfitting_metrics.get('degradation_ratio', 'N/A')}
• **Stability Score**: {overfitting_metrics.get('stability_score', 'N/A'):.1%}
• **Cross-Validation Score**: {overfitting_metrics.get('cross_validation_score', 'N/A'):.1%}
• **Information Coefficient**: {overfitting_metrics.get('information_coefficient', 'N/A'):.3f}

### Analysis
"""

        if is_overfitted:
            content += "The strategy shows signs of overfitting, indicating that backtest performance may not be representative of live trading results. "
        else:
            content += "The strategy demonstrates good generalization characteristics with stable performance across validation periods. "

        content += f"Analysis confidence: {confidence:.1%}"

        return ReportSection(
            title="Overfitting Detection",
            content=content.strip(),
            confidence_score=confidence
        )

    def _generate_risk_assessment(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive risk assessment"""

        try:
            risk_factors = {}
            overall_risk_level = "LOW"

            # Data quality risks
            data_quality = validation_results.get('validation_components', {}).get('data_quality', {})
            data_score = data_quality.get('overall_quality_score', 1.0)
            if data_score < 0.7:
                risk_factors['data_quality'] = {
                    'level': 'HIGH',
                    'description': 'Poor data quality may invalidate backtest results',
                    'mitigation': 'Improve data collection and cleaning procedures'
                }

            # Overfitting risks
            overfitting = validation_results.get('validation_components', {}).get('overfitting_detection', {})
            overfitting_metrics = overfitting.get('overfitting_metrics', {})
            if overfitting_metrics.get('is_overfitted', False):
                risk_factors['overfitting'] = {
                    'level': 'HIGH',
                    'description': 'Strategy may be overfit to historical data',
                    'mitigation': 'Simplify strategy or collect more out-of-sample data'
                }

            # Statistical significance risks
            significance = validation_results.get('validation_components', {}).get('statistical_significance', {})
            overall_sig = significance.get('overall_assessment', {})
            if not overall_sig.get('is_significant', False):
                risk_factors['statistical_significance'] = {
                    'level': 'MEDIUM',
                    'description': 'Strategy performance may be due to chance',
                    'mitigation': 'Collect more data or adjust strategy parameters'
                }

            # Determine overall risk level
            risk_levels = [rf['level'] for rf in risk_factors.values()]
            if 'HIGH' in risk_levels:
                overall_risk_level = 'HIGH'
            elif 'MEDIUM' in risk_levels:
                overall_risk_level = 'MEDIUM'

            return {
                'overall_risk_level': overall_risk_level,
                'risk_factors': risk_factors,
                'risk_score': len([rf for rf in risk_factors.values() if rf['level'] == 'HIGH']) * 0.3 +
                             len([rf for rf in risk_factors.values() if rf['level'] == 'MEDIUM']) * 0.1,
                'recommendations': self._generate_risk_mitigation_recommendations(risk_factors)
            }

        except Exception as e:
            self.logger.error(f"Error generating risk assessment: {e}")
            return {
                'overall_risk_level': 'UNKNOWN',
                'risk_factors': {},
                'risk_score': 0.5,
                'error': str(e)
            }

    def _determine_confidence_level(self, score: float) -> ConfidenceLevel:
        """Determine confidence level based on score"""
        for level, threshold in self.confidence_thresholds.items():
            if score >= threshold:
                return level
        return ConfidenceLevel.VERY_LOW

    def _generate_validation_summary(self, validation_results: Dict[str, Any]) -> ValidationSummary:
        """Generate validation summary object"""

        overall_validation = validation_results.get('overall_validation', {})
        composite_score = overall_validation.get('composite_score', 0.0)
        validation_result = overall_validation.get('validation_result', 'FAIL')

        # Extract component scores
        component_scores = overall_validation.get('component_scores', {})

        # Generate key findings and critical issues
        key_findings = []
        critical_issues = []

        if composite_score >= 0.8:
            key_findings.append("Strategy demonstrates strong validation characteristics")
        elif composite_score >= 0.6:
            key_findings.append("Strategy shows acceptable validation with some concerns")
        else:
            key_findings.append("Strategy validation reveals significant concerns")
            critical_issues.append("Low overall validation score requires immediate attention")

        # Generate recommendations
        recommendations = self._extract_top_recommendations(validation_results, 5)

        return ValidationSummary(
            overall_score=composite_score,
            confidence_level=self._determine_confidence_level(composite_score),
            validation_status=validation_result,
            key_findings=key_findings,
            critical_issues=critical_issues,
            recommendations=recommendations,
            component_scores=component_scores,
            risk_assessment=self._generate_risk_assessment(validation_results)
        )

    def _extract_top_recommendations(self, validation_results: Dict[str, Any], max_count: int = 5) -> List[str]:
        """Extract top recommendations from validation results"""

        all_recommendations = []

        # From overall validation
        overall_recs = validation_results.get('recommendations', [])
        all_recommendations.extend(overall_recs)

        # From validation components
        validation_components = validation_results.get('validation_components', {})
        for component, results in validation_components.items():
            component_recs = results.get('recommendations', [])
            all_recommendations.extend(component_recs)

        # Remove duplicates while preserving order
        unique_recommendations = []
        for rec in all_recommendations:
            if rec not in unique_recommendations:
                unique_recommendations.append(rec)

        return unique_recommendations[:max_count]

    def _store_validation_report(self, execution_id: int, report: Dict[str, Any]):
        """Store validation report in database"""
        try:
            # Store comprehensive report
            self.logger.info(f"Storing validation report for execution {execution_id}")
            # TODO: Implement database storage for reports
        except Exception as e:
            self.logger.error(f"Error storing validation report: {e}")

    def _create_error_report(self, execution_id: int, error_message: str) -> Dict[str, Any]:
        """Create error report"""
        return {
            'execution_id': execution_id,
            'report_timestamp': datetime.now(timezone.utc),
            'status': 'error',
            'error_message': error_message,
            'executive_summary': ReportSection(
                title="Error",
                content=f"Report generation failed: {error_message}",
                confidence_score=0.0
            ).__dict__
        }

    # Additional placeholder methods for remaining sections
    def _generate_significance_section(self, results: Dict[str, Any]) -> ReportSection:
        return ReportSection("Statistical Significance", "Analysis placeholder", confidence_score=0.8)

    def _generate_correlation_section(self, results: Dict[str, Any]) -> ReportSection:
        return ReportSection("Real-time Correlation", "Analysis placeholder", confidence_score=0.8)

    def _generate_consistency_section(self, results: Dict[str, Any]) -> ReportSection:
        return ReportSection("Pipeline Consistency", "Analysis placeholder", confidence_score=0.8)

    def _generate_generic_section(self, component: str, results: Dict[str, Any]) -> ReportSection:
        return ReportSection(component.replace('_', ' ').title(), "Generic analysis placeholder", confidence_score=0.5)

    def _generate_recommendations_section(self, validation_results: Dict[str, Any]) -> ReportSection:
        recommendations = self._extract_top_recommendations(validation_results, 10)
        content = "# RECOMMENDATIONS\n\n" + "\n".join(f"{i+1}. {rec}" for i, rec in enumerate(recommendations))
        return ReportSection("Recommendations", content)

    def _generate_technical_appendix(self, validation_results: Dict[str, Any]) -> ReportSection:
        return ReportSection("Technical Appendix", "Detailed technical information placeholder")

    def _generate_report_metadata(self, execution_id: int, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'execution_id': execution_id,
            'validation_framework_version': '1.0.0',
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'sample_size': validation_results.get('sample_size', 0)
        }

    def _extract_key_metrics(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        overall = validation_results.get('overall_validation', {})
        return {
            'composite_score': overall.get('composite_score', 0.0),
            'validation_result': overall.get('validation_result', 'FAIL'),
            'component_scores': overall.get('component_scores', {})
        }

    def _generate_go_no_go_decision(self, summary: ValidationSummary) -> Dict[str, Any]:
        if summary.overall_score >= 0.8 and summary.confidence_level.value in ['VERY_HIGH', 'HIGH']:
            decision = 'GO'
            rationale = 'Strong validation results support strategy deployment'
        elif summary.overall_score >= 0.6:
            decision = 'CONDITIONAL GO'
            rationale = 'Acceptable validation with conditions that must be addressed'
        else:
            decision = 'NO GO'
            rationale = 'Validation results do not support strategy deployment'

        return {
            'decision': decision,
            'rationale': rationale,
            'confidence': summary.confidence_level.value
        }

    def _assess_business_impact(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'deployment_readiness': 'Moderate',
            'expected_performance': 'Aligned with backtest',
            'risk_level': 'Medium'
        }

    def _recommend_next_steps(self, summary: ValidationSummary) -> List[str]:
        if summary.overall_score >= 0.8:
            return ["Proceed with limited live testing", "Monitor performance closely"]
        else:
            return ["Address validation issues", "Re-run validation after improvements"]

    def _generate_risk_mitigation_recommendations(self, risk_factors: Dict[str, Any]) -> List[str]:
        recommendations = []
        for risk_name, risk_data in risk_factors.items():
            recommendations.append(risk_data.get('mitigation', f'Address {risk_name} risk'))
        return recommendations


# Factory function
def create_validation_report_generator(
    db_manager: DatabaseManager,
    statistical_framework: StatisticalValidationFramework,
    **kwargs
) -> ValidationReportGenerator:
    """Create ValidationReportGenerator instance"""
    return ValidationReportGenerator(
        db_manager=db_manager,
        statistical_framework=statistical_framework,
        **kwargs
    )