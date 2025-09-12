"""
Result Formatter - Analysis Result Formatting Module
Handles formatting of Claude analysis results for different output formats
Extracted from claude_api.py for better modularity
"""

import logging
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum


class OutputFormat(Enum):
    """Supported output formats"""
    TEXT = "text"
    JSON = "json"
    CSV = "csv"
    HTML = "html"
    MARKDOWN = "markdown"


@dataclass
class FormattedResult:
    """Container for formatted analysis results"""
    content: str
    format: OutputFormat
    metadata: Dict[str, Any]
    generated_at: datetime


class ResultFormatter:
    """
    Formats Claude analysis results into various output formats
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def format_analysis_result(self, 
                             signal: Dict, 
                             analysis: Dict, 
                             format_type: OutputFormat = OutputFormat.TEXT) -> FormattedResult:
        """
        Format a single analysis result
        """
        try:
            if format_type == OutputFormat.TEXT:
                content = self._format_text(signal, analysis)
            elif format_type == OutputFormat.JSON:
                content = self._format_json(signal, analysis)
            elif format_type == OutputFormat.MARKDOWN:
                content = self._format_markdown(signal, analysis)
            elif format_type == OutputFormat.HTML:
                content = self._format_html(signal, analysis)
            elif format_type == OutputFormat.CSV:
                content = self._format_csv(signal, analysis)
            else:
                raise ValueError(f"Unsupported format: {format_type}")
            
            metadata = self._extract_metadata(signal, analysis)
            
            return FormattedResult(
                content=content,
                format=format_type,
                metadata=metadata,
                generated_at=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error formatting result: {e}")
            # Return error format
            return FormattedResult(
                content=f"Formatting error: {str(e)}",
                format=format_type,
                metadata={'error': True},
                generated_at=datetime.now()
            )
    
    def format_batch_results(self, 
                           results: List[Dict], 
                           format_type: OutputFormat = OutputFormat.TEXT) -> FormattedResult:
        """
        Format multiple analysis results
        """
        try:
            if format_type == OutputFormat.TEXT:
                content = self._format_batch_text(results)
            elif format_type == OutputFormat.JSON:
                content = self._format_batch_json(results)
            elif format_type == OutputFormat.MARKDOWN:
                content = self._format_batch_markdown(results)
            elif format_type == OutputFormat.HTML:
                content = self._format_batch_html(results)
            elif format_type == OutputFormat.CSV:
                content = self._format_batch_csv(results)
            else:
                raise ValueError(f"Unsupported format: {format_type}")
            
            metadata = self._extract_batch_metadata(results)
            
            return FormattedResult(
                content=content,
                format=format_type,
                metadata=metadata,
                generated_at=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error formatting batch results: {e}")
            return FormattedResult(
                content=f"Batch formatting error: {str(e)}",
                format=format_type,
                metadata={'error': True, 'batch_size': len(results)},
                generated_at=datetime.now()
            )
    
    def format_summary_report(self, 
                            results: List[Dict], 
                            format_type: OutputFormat = OutputFormat.MARKDOWN) -> FormattedResult:
        """
        Format a comprehensive summary report
        """
        try:
            # Calculate summary statistics
            total = len(results)
            approved = len([r for r in results if r.get('approved', False)])
            tech_passed = len([r for r in results if r.get('technical_validation_passed', False)])
            
            scores = [r['score'] for r in results if r.get('score') is not None]
            avg_score = sum(scores) / len(scores) if scores else 0
            
            # Generate content based on format
            if format_type == OutputFormat.MARKDOWN:
                content = self._format_summary_markdown(results, total, approved, tech_passed, avg_score)
            elif format_type == OutputFormat.HTML:
                content = self._format_summary_html(results, total, approved, tech_passed, avg_score)
            elif format_type == OutputFormat.TEXT:
                content = self._format_summary_text(results, total, approved, tech_passed, avg_score)
            elif format_type == OutputFormat.JSON:
                content = self._format_summary_json(results, total, approved, tech_passed, avg_score)
            else:
                raise ValueError(f"Unsupported summary format: {format_type}")
            
            metadata = {
                'report_type': 'summary',
                'total_signals': total,
                'approved_signals': approved,
                'average_score': avg_score,
                'approval_rate': (approved / total * 100) if total > 0 else 0
            }
            
            return FormattedResult(
                content=content,
                format=format_type,
                metadata=metadata,
                generated_at=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error formatting summary report: {e}")
            return FormattedResult(
                content=f"Summary formatting error: {str(e)}",
                format=format_type,
                metadata={'error': True},
                generated_at=datetime.now()
            )
    
    def _format_text(self, signal: Dict, analysis: Dict) -> str:
        """Format single result as plain text"""
        tech_status = "✅ PASSED" if analysis.get('technical_validation_passed') else "❌ FAILED"
        
        text = f"""Claude Analysis for {signal.get('epic', 'Unknown')} {signal.get('signal_type', 'Unknown')} Signal

TECHNICAL VALIDATION: {tech_status}
Signal Quality Score: {analysis.get('score', 'N/A')}/10
Decision: {analysis.get('decision', 'Unknown')}
Approved: {analysis.get('approved', False)}
Reason: {analysis.get('reason', 'No reason provided')}

Strategy: {self._identify_strategy(signal)}
Price: {signal.get('price', 'N/A')}
Confidence: {signal.get('confidence_score', 0):.1%}
Analysis Mode: {analysis.get('mode', 'standard')}
Timestamp: {analysis.get('analysis_timestamp', 'N/A')}
"""
        return text.strip()
    
    def _format_json(self, signal: Dict, analysis: Dict) -> str:
        """Format single result as JSON"""
        result = {
            'signal_info': {
                'epic': signal.get('epic'),
                'signal_type': signal.get('signal_type'),
                'price': signal.get('price'),
                'strategy': self._identify_strategy(signal),
                'confidence_score': signal.get('confidence_score'),
                'timestamp': signal.get('timestamp').isoformat() if signal.get('timestamp') else None
            },
            'claude_analysis': {
                'score': analysis.get('score'),
                'decision': analysis.get('decision'),
                'approved': analysis.get('approved'),
                'reason': analysis.get('reason'),
                'mode': analysis.get('mode'),
                'technical_validation_passed': analysis.get('technical_validation_passed'),
                'analysis_timestamp': analysis.get('analysis_timestamp')
            },
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'format_version': '1.0'
            }
        }
        
        return json.dumps(result, indent=2, default=str)
    
    def _format_markdown(self, signal: Dict, analysis: Dict) -> str:
        """Format single result as Markdown"""
        tech_status = "✅ PASSED" if analysis.get('technical_validation_passed') else "❌ FAILED"
        
        markdown = f"""# Claude Analysis Report

## Signal Information
- **Pair**: {signal.get('epic', 'Unknown')}
- **Type**: {signal.get('signal_type', 'Unknown')}
- **Price**: {signal.get('price', 'N/A')}
- **Strategy**: {self._identify_strategy(signal)}
- **Confidence**: {signal.get('confidence_score', 0):.1%}

## Claude Analysis Results
- **Technical Validation**: {tech_status}
- **Score**: {analysis.get('score', 'N/A')}/10
- **Decision**: {analysis.get('decision', 'Unknown')}
- **Approved**: {'✅ Yes' if analysis.get('approved') else '❌ No'}
- **Analysis Mode**: {analysis.get('mode', 'standard')}

## Reasoning
{analysis.get('reason', 'No reason provided')}

---
*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        return markdown.strip()
    
    def _format_html(self, signal: Dict, analysis: Dict) -> str:
        """Format single result as HTML"""
        tech_status = "✅ PASSED" if analysis.get('technical_validation_passed') else "❌ FAILED"
        approved_color = "green" if analysis.get('approved') else "red"
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Claude Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 10px; border-radius: 5px; }}
        .section {{ margin: 15px 0; }}
        .approved {{ color: {approved_color}; font-weight: bold; }}
        .score {{ font-size: 1.2em; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Claude Analysis Report</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="section">
        <h2>Signal Information</h2>
        <ul>
            <li><strong>Pair:</strong> {signal.get('epic', 'Unknown')}</li>
            <li><strong>Type:</strong> {signal.get('signal_type', 'Unknown')}</li>
            <li><strong>Price:</strong> {signal.get('price', 'N/A')}</li>
            <li><strong>Strategy:</strong> {self._identify_strategy(signal)}</li>
            <li><strong>Confidence:</strong> {signal.get('confidence_score', 0):.1%}</li>
        </ul>
    </div>
    
    <div class="section">
        <h2>Analysis Results</h2>
        <ul>
            <li><strong>Technical Validation:</strong> {tech_status}</li>
            <li><strong>Score:</strong> <span class="score">{analysis.get('score', 'N/A')}/10</span></li>
            <li><strong>Decision:</strong> {analysis.get('decision', 'Unknown')}</li>
            <li><strong>Approved:</strong> <span class="approved">{'Yes' if analysis.get('approved') else 'No'}</span></li>
        </ul>
    </div>
    
    <div class="section">
        <h2>Reasoning</h2>
        <p>{analysis.get('reason', 'No reason provided')}</p>
    </div>
</body>
</html>"""
        return html
    
    def _format_csv(self, signal: Dict, analysis: Dict) -> str:
        """Format single result as CSV"""
        # CSV header and data
        headers = [
            "epic", "signal_type", "price", "strategy", "confidence_score",
            "score", "decision", "approved", "technical_validation_passed",
            "mode", "reason", "analysis_timestamp"
        ]
        
        values = [
            signal.get('epic', ''),
            signal.get('signal_type', ''),
            signal.get('price', ''),
            self._identify_strategy(signal),
            signal.get('confidence_score', ''),
            analysis.get('score', ''),
            analysis.get('decision', ''),
            analysis.get('approved', ''),
            analysis.get('technical_validation_passed', ''),
            analysis.get('mode', ''),
            analysis.get('reason', '').replace('\n', ' ').replace(',', ';'),  # Clean reason for CSV
            analysis.get('analysis_timestamp', '')
        ]
        
        return ','.join(headers) + '\n' + ','.join(str(v) for v in values)
    
    def _format_batch_text(self, results: List[Dict]) -> str:
        """Format batch results as plain text"""
        total = len(results)
        approved = len([r for r in results if r.get('approved', False)])
        tech_passed = len([r for r in results if r.get('technical_validation_passed', False)])
        
        scores = [r['score'] for r in results if r.get('score') is not None]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        text = f"""Claude Enhanced Batch Analysis
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SUMMARY:
Total Signals: {total}
Technical Validation Passed: {tech_passed} ({tech_passed/total*100:.1f}%)
Claude Approved: {approved} ({approved/total*100:.1f}%)
Average Score: {avg_score:.1f}/10

INDIVIDUAL RESULTS:
"""
        
        for i, result in enumerate(results, 1):
            signal = result['signal']
            tech_status = "✅" if result.get('technical_validation_passed') else "❌"
            text += f"{i:2d}. {tech_status} {signal.get('epic', 'Unknown'):20s} {signal.get('signal_type', 'Unknown'):4s} "
            text += f"Score: {result['score'] or 'N/A':2s}/10 "
            text += f"Decision: {result['decision']:7s} "
            text += f"Reason: {result['reason'] or 'N/A'}\n"
        
        return text
    
    def _format_batch_json(self, results: List[Dict]) -> str:
        """Format batch results as JSON"""
        batch_data = {
            'summary': self._extract_batch_metadata(results),
            'results': results,
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'format_version': '1.0'
            }
        }
        
        return json.dumps(batch_data, indent=2, default=str)
    
    def _format_batch_csv(self, results: List[Dict]) -> str:
        """Format batch results as CSV"""
        if not results:
            return "No results to format"
        
        # Headers
        headers = [
            "epic", "signal_type", "price", "strategy", "confidence_score",
            "score", "decision", "approved", "technical_validation_passed",
            "mode", "reason", "analysis_timestamp"
        ]
        
        csv_lines = [','.join(headers)]
        
        for result in results:
            signal = result['signal']
            values = [
                signal.get('epic', ''),
                signal.get('signal_type', ''),
                signal.get('price', ''),
                self._identify_strategy(signal),
                signal.get('confidence_score', ''),
                result.get('score', ''),
                result.get('decision', ''),
                result.get('approved', ''),
                result.get('technical_validation_passed', ''),
                result.get('mode', ''),
                (result.get('reason', '') or '').replace('\n', ' ').replace(',', ';'),
                result.get('analysis_timestamp', '')
            ]
            csv_lines.append(','.join(str(v) for v in values))
        
        return '\n'.join(csv_lines)
    
    def _format_summary_markdown(self, results: List[Dict], total: int, approved: int, tech_passed: int, avg_score: float) -> str:
        """Format summary report as Markdown"""
        markdown = f"""# Claude Analysis Summary Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 Overview Statistics

| Metric | Value | Percentage |
|--------|-------|------------|
| Total Signals | {total} | 100% |
| Technical Validation Passed | {tech_passed} | {tech_passed/total*100:.1f}% |
| Claude Approved | {approved} | {approved/total*100:.1f}% |
| Average Score | {avg_score:.1f}/10 | {avg_score*10:.1f}% |

## 📈 Performance Analysis

### Signal Quality Distribution
"""
        
        # Add score distribution
        score_ranges = {'0-3': 0, '4-5': 0, '6-7': 0, '8-10': 0}
        for result in results:
            score = result.get('score', 0)
            if score <= 3:
                score_ranges['0-3'] += 1
            elif score <= 5:
                score_ranges['4-5'] += 1
            elif score <= 7:
                score_ranges['6-7'] += 1
            else:
                score_ranges['8-10'] += 1
        
        for range_name, count in score_ranges.items():
            percentage = count / total * 100 if total > 0 else 0
            markdown += f"- **{range_name} points:** {count} signals ({percentage:.1f}%)\n"
        
        # Add strategy breakdown
        strategies = {}
        for result in results:
            strategy = self._identify_strategy(result['signal'])
            strategies[strategy] = strategies.get(strategy, 0) + 1
        
        markdown += "\n### Strategy Breakdown\n"
        for strategy, count in sorted(strategies.items()):
            percentage = count / total * 100 if total > 0 else 0
            markdown += f"- **{strategy}:** {count} signals ({percentage:.1f}%)\n"
        
        return markdown
    
    def _extract_metadata(self, signal: Dict, analysis: Dict) -> Dict:
        """Extract metadata from single analysis"""
        return {
            'epic': signal.get('epic'),
            'signal_type': signal.get('signal_type'),
            'strategy': self._identify_strategy(signal),
            'score': analysis.get('score'),
            'approved': analysis.get('approved'),
            'technical_validation_passed': analysis.get('technical_validation_passed'),
            'analysis_mode': analysis.get('mode')
        }
    
    def _extract_batch_metadata(self, results: List[Dict]) -> Dict:
        """Extract metadata from batch results"""
        total = len(results)
        approved = len([r for r in results if r.get('approved', False)])
        tech_passed = len([r for r in results if r.get('technical_validation_passed', False)])
        
        scores = [r['score'] for r in results if r.get('score') is not None]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        return {
            'total_signals': total,
            'approved_signals': approved,
            'technical_passed': tech_passed,
            'approval_rate': (approved / total * 100) if total > 0 else 0,
            'technical_pass_rate': (tech_passed / total * 100) if total > 0 else 0,
            'average_score': avg_score,
            'score_count': len(scores)
        }
    
    def _identify_strategy(self, signal: Dict) -> str:
        """Identify the strategy type from signal data"""
        strategy = signal.get('strategy', '').lower()
        
        if 'combined' in strategy:
            return 'COMBINED'
        elif 'macd' in strategy:
            return 'MACD'
        elif 'kama' in strategy:
            return 'KAMA'
        elif 'ema' in strategy:
            return 'EMA'
        else:
            return 'UNKNOWN'


# Factory function
def create_result_formatter() -> ResultFormatter:
    """Create result formatter with default configuration"""
    return ResultFormatter()


# Usage example
if __name__ == "__main__":
    formatter = create_result_formatter()
    
    # Test data
    test_signal = {
        'epic': 'CS.D.EURUSD.MINI.IP',
        'signal_type': 'BULL',
        'price': 1.0850,
        'strategy': 'ema',
        'confidence_score': 0.85,
        'timestamp': datetime.now()
    }
    
    test_analysis = {
        'score': 8,
        'decision': 'APPROVE',
        'approved': True,
        'reason': 'Strong technical indicators with good EMA alignment',
        'technical_validation_passed': True,
        'mode': 'minimal',
        'analysis_timestamp': datetime.now().isoformat()
    }
    
    # Test different formats
    formats = [OutputFormat.TEXT, OutputFormat.JSON, OutputFormat.MARKDOWN, OutputFormat.HTML]
    
    for fmt in formats:
        print(f"\n{'='*20} {fmt.value.upper()} FORMAT {'='*20}")
        result = formatter.format_analysis_result(test_signal, test_analysis, fmt)
        print(result.content[:500] + "..." if len(result.content) > 500 else result.content)