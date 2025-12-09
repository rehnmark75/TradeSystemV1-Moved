"""
Stock Response Parser

Parses Claude API responses into structured ClaudeAnalysis objects.
Handles JSON extraction, validation, and fallback parsing.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ClaudeAnalysis:
    """Structured Claude analysis result"""

    # Core assessment
    grade: str = 'C'  # A+, A, B, C, D
    score: int = 5  # 1-10
    conviction: str = 'MEDIUM'  # HIGH, MEDIUM, LOW
    action: str = 'HOLD'  # STRONG BUY, BUY, HOLD, AVOID

    # Reasoning
    key_strengths: List[str] = field(default_factory=list)
    key_risks: List[str] = field(default_factory=list)
    thesis: str = ''

    # Recommendations
    position_recommendation: str = 'Quarter'  # Full, Half, Quarter, Skip
    stop_adjustment: str = 'Keep'  # Tighten, Keep, Widen
    time_horizon: str = 'Swing'  # Intraday, Swing, Position

    # Optional extended fields
    catalyst_watch: Optional[str] = None
    alternative_entry: Optional[str] = None

    # Metadata
    raw_response: str = ''
    analysis_timestamp: datetime = field(default_factory=datetime.now)
    tokens_used: int = 0
    latency_ms: int = 0
    model: str = ''
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage"""
        return {
            'claude_grade': self.grade,
            'claude_score': self.score,
            'claude_conviction': self.conviction,
            'claude_action': self.action,
            'claude_thesis': self.thesis,
            'claude_key_strengths': self.key_strengths,
            'claude_key_risks': self.key_risks,
            'claude_position_rec': self.position_recommendation,
            'claude_stop_adjustment': self.stop_adjustment,
            'claude_time_horizon': self.time_horizon,
            'claude_raw_response': self.raw_response,
            'claude_analyzed_at': self.analysis_timestamp,
            'claude_tokens_used': self.tokens_used,
            'claude_latency_ms': self.latency_ms,
            'claude_model': self.model,
        }

    @property
    def is_bullish(self) -> bool:
        """Check if analysis is bullish"""
        return self.action in ['STRONG BUY', 'BUY']

    @property
    def is_high_conviction(self) -> bool:
        """Check if high conviction"""
        return self.conviction == 'HIGH' and self.grade in ['A+', 'A']


class StockResponseParser:
    """
    Parses Claude API responses for stock analysis.

    Handles:
    - JSON extraction from markdown code blocks
    - Field validation and normalization
    - Fallback parsing for malformed responses
    """

    # Valid values for enum fields
    VALID_GRADES = ['A+', 'A', 'B', 'C', 'D']
    VALID_CONVICTIONS = ['HIGH', 'MEDIUM', 'LOW']
    VALID_ACTIONS = ['STRONG BUY', 'BUY', 'HOLD', 'AVOID']
    VALID_POSITIONS = ['Full', 'Half', 'Quarter', 'Skip']
    VALID_STOP_ADJ = ['Tighten', 'Keep', 'Widen']
    VALID_HORIZONS = ['Intraday', 'Swing', 'Position']

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def parse_response(
        self,
        response_text: str,
        tokens_used: int = 0,
        latency_ms: int = 0,
        model: str = ''
    ) -> ClaudeAnalysis:
        """
        Parse Claude API response into structured ClaudeAnalysis.

        Args:
            response_text: Raw response from Claude API
            tokens_used: Number of tokens used
            latency_ms: API latency in milliseconds
            model: Model used for analysis

        Returns:
            ClaudeAnalysis object with parsed data
        """
        analysis = ClaudeAnalysis(
            raw_response=response_text,
            tokens_used=tokens_used,
            latency_ms=latency_ms,
            model=model,
            analysis_timestamp=datetime.now()
        )

        if not response_text:
            analysis.error = 'Empty response'
            return analysis

        # Try to extract JSON from response
        json_data = self._extract_json(response_text)

        if json_data:
            self._populate_from_json(analysis, json_data)
        else:
            # Fallback to text parsing
            self._populate_from_text(analysis, response_text)
            analysis.error = 'JSON parsing failed, used text fallback'

        return analysis

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON object from response text"""

        # Try direct JSON parse first
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        # Try to find JSON in markdown code block
        patterns = [
            r'```json\s*([\s\S]*?)\s*```',  # ```json ... ```
            r'```\s*([\s\S]*?)\s*```',  # ``` ... ```
            r'\{[\s\S]*\}',  # Raw JSON object
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                json_str = match.group(1) if '```' in pattern else match.group(0)
                try:
                    return json.loads(json_str.strip())
                except json.JSONDecodeError:
                    continue

        return None

    def _populate_from_json(
        self,
        analysis: ClaudeAnalysis,
        data: Dict[str, Any]
    ) -> None:
        """Populate analysis from parsed JSON data"""

        # Grade
        grade = str(data.get('grade', 'C')).upper().strip()
        if grade in self.VALID_GRADES:
            analysis.grade = grade
        else:
            # Try to normalize
            if grade.startswith('A') and '+' in grade:
                analysis.grade = 'A+'
            elif grade.startswith('A'):
                analysis.grade = 'A'
            elif grade.startswith('B'):
                analysis.grade = 'B'
            elif grade.startswith('D'):
                analysis.grade = 'D'
            else:
                analysis.grade = 'C'

        # Score (1-10)
        try:
            score = int(data.get('score', 5))
            analysis.score = max(1, min(10, score))
        except (ValueError, TypeError):
            analysis.score = 5

        # Conviction
        conviction = str(data.get('conviction', 'MEDIUM')).upper().strip()
        if conviction in self.VALID_CONVICTIONS:
            analysis.conviction = conviction
        else:
            analysis.conviction = 'MEDIUM'

        # Action
        action = str(data.get('action', 'HOLD')).upper().strip()
        if action in self.VALID_ACTIONS:
            analysis.action = action
        else:
            # Normalize common variations
            if 'STRONG' in action and 'BUY' in action:
                analysis.action = 'STRONG BUY'
            elif 'BUY' in action:
                analysis.action = 'BUY'
            elif 'AVOID' in action or 'SELL' in action:
                analysis.action = 'AVOID'
            else:
                analysis.action = 'HOLD'

        # Key strengths
        strengths = data.get('key_strengths', [])
        if isinstance(strengths, list):
            analysis.key_strengths = [str(s) for s in strengths[:5]]
        elif isinstance(strengths, str):
            analysis.key_strengths = [strengths]

        # Key risks
        risks = data.get('key_risks', [])
        if isinstance(risks, list):
            analysis.key_risks = [str(r) for r in risks[:5]]
        elif isinstance(risks, str):
            analysis.key_risks = [risks]

        # Thesis
        analysis.thesis = str(data.get('thesis', ''))[:500]

        # Position recommendation
        pos_rec = str(data.get('position_recommendation', 'Quarter')).title().strip()
        if pos_rec in self.VALID_POSITIONS:
            analysis.position_recommendation = pos_rec
        else:
            # Map based on grade/conviction
            if analysis.grade in ['A+', 'A'] and analysis.conviction == 'HIGH':
                analysis.position_recommendation = 'Full'
            elif analysis.grade in ['A+', 'A']:
                analysis.position_recommendation = 'Half'
            elif analysis.grade == 'B':
                analysis.position_recommendation = 'Quarter'
            else:
                analysis.position_recommendation = 'Skip'

        # Stop adjustment
        stop_adj = str(data.get('stop_adjustment', 'Keep')).title().strip()
        if stop_adj in self.VALID_STOP_ADJ:
            analysis.stop_adjustment = stop_adj
        else:
            analysis.stop_adjustment = 'Keep'

        # Time horizon
        horizon = str(data.get('time_horizon', 'Swing')).title().strip()
        if horizon in self.VALID_HORIZONS:
            analysis.time_horizon = horizon
        else:
            analysis.time_horizon = 'Swing'

        # Optional fields
        if 'catalyst_watch' in data:
            analysis.catalyst_watch = str(data['catalyst_watch'])[:200]
        if 'alternative_entry' in data:
            analysis.alternative_entry = str(data['alternative_entry'])[:100]

    def _populate_from_text(
        self,
        analysis: ClaudeAnalysis,
        text: str
    ) -> None:
        """Fallback parsing from plain text response"""

        text_upper = text.upper()

        # Try to extract grade
        for grade in self.VALID_GRADES:
            if f'GRADE: {grade}' in text_upper or f'GRADE:{grade}' in text_upper:
                analysis.grade = grade
                break
            if f'"{grade}"' in text or f"'{grade}'" in text:
                analysis.grade = grade
                break

        # Try to extract action
        for action in self.VALID_ACTIONS:
            if action in text_upper:
                analysis.action = action
                break

        # Try to extract conviction
        for conv in self.VALID_CONVICTIONS:
            if f'CONVICTION: {conv}' in text_upper or conv in text_upper:
                analysis.conviction = conv
                break

        # Try to extract score
        score_match = re.search(r'SCORE[:\s]*(\d+)', text_upper)
        if score_match:
            try:
                analysis.score = max(1, min(10, int(score_match.group(1))))
            except ValueError:
                pass

        # Use the text as thesis if short enough
        if len(text) < 500:
            analysis.thesis = text.strip()
        else:
            # Try to find a sentence that looks like a thesis
            sentences = re.split(r'[.!?]', text)
            for sentence in sentences:
                if len(sentence) > 50 and any(word in sentence.lower() for word in ['recommend', 'suggest', 'because', 'due to', 'given']):
                    analysis.thesis = sentence.strip()[:500]
                    break

        self.logger.warning(f"Used text fallback parsing, grade={analysis.grade}, action={analysis.action}")

    def validate_analysis(self, analysis: ClaudeAnalysis) -> bool:
        """Validate that analysis has required fields"""

        if analysis.grade not in self.VALID_GRADES:
            return False
        if analysis.score < 1 or analysis.score > 10:
            return False
        if analysis.conviction not in self.VALID_CONVICTIONS:
            return False
        if analysis.action not in self.VALID_ACTIONS:
            return False

        return True

    def get_default_analysis(
        self,
        error_message: str = 'Analysis failed'
    ) -> ClaudeAnalysis:
        """Get a default/error analysis object"""
        return ClaudeAnalysis(
            grade='C',
            score=5,
            conviction='LOW',
            action='HOLD',
            thesis='Analysis could not be completed',
            key_risks=['Analysis error - manual review recommended'],
            position_recommendation='Skip',
            error=error_message
        )
