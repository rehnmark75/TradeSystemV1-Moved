"""
Response Parser - Claude Response Processing Module
Handles parsing of Claude responses into structured data
Extracted from claude_api.py for better modularity
"""

import logging
import os
import re
import time
from typing import Dict, Optional

# Claude approval threshold resolver.
#
# Historically the parser hard-coded `score >= 6 -> APPROVE`, which shadowed
# `scanner_global_config.min_claude_quality_score` (the DB flag that
# trade_validator.py already reads). These two gates are now unified:
# the parser also reads the DB value, with an env-var override for
# fast debug/A-B without a DB write.
#
# Resolution order:
#   1. CLAUDE_MIN_APPROVAL_SCORE env var (if set & parseable)
#   2. scanner_global_config.min_claude_quality_score
#   3. Hard default 6 (previous behavior)
#
# Cached for _APPROVAL_TTL_S so DB edits are picked up without restart.
_APPROVAL_TTL_S = 60
_approval_cache: Dict[str, float] = {"value": 6.0, "ts": 0.0}


def get_min_approval_score(default: int = 6) -> int:
    env_val = os.environ.get("CLAUDE_MIN_APPROVAL_SCORE")
    if env_val is not None:
        try:
            return int(float(env_val))
        except ValueError:
            pass

    now = time.time()
    if now - _approval_cache["ts"] < _APPROVAL_TTL_S:
        return int(_approval_cache["value"])

    try:
        from services.scanner_config_service import get_scanner_config
        cfg = get_scanner_config()
        val = int(getattr(cfg, "min_claude_quality_score", default) or default)
        _approval_cache["value"] = float(val)
        _approval_cache["ts"] = now
        return val
    except Exception:
        return default


class ResponseParser:
    """
    Handles parsing of Claude responses with fallback mechanisms
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    # Headers can arrive bolded (**SCORE:**), lower-case, or with extra
    # whitespace. The header regex tolerates all three and an optional
    # leading "- "/"* " bullet that Claude sometimes emits.
    _HEADER_RE = re.compile(
        r'^\s*(?:[-*]\s+)?\**\s*(SCORE|DECISION|REASON|CORRECT_TYPE)\s*\**\s*[:\-]\s*(.*)$',
        re.IGNORECASE,
    )

    # Last-resort regexes that scan the whole blob if line-based parsing fails.
    _SCORE_FALLBACK_RE = re.compile(r'score\s*[:\-]?\s*\**\s*(\d+)', re.IGNORECASE)
    _DECISION_FALLBACK_RE = re.compile(
        r'decision\s*[:\-]?\s*\**\s*(APPROVE|REJECT|NEUTRAL)', re.IGNORECASE
    )

    @staticmethod
    def _strip_code_fence(response: str) -> str:
        """Remove ```...``` or ```lang ...``` wrappers Claude sometimes uses."""
        stripped = response.strip()
        if stripped.startswith('```'):
            stripped = re.sub(r'^```[^\n]*\n?', '', stripped)
            if stripped.endswith('```'):
                stripped = stripped[:-3]
        return stripped.strip()

    def parse_minimal_response(self, response: str) -> Dict:
        """
        Parse the minimal Claude response into structured data.

        Tolerant of:
          - markdown-bolded headers (**SCORE:**)
          - case variations (score:, Score:)
          - leading bullets ("- SCORE: 7")
          - ``` code fences
          - trailing extras on the score line (e.g. "SCORE: 7/10")

        Always returns a usable dict — on parse failure, score defaults to 0
        and decision to 'REJECT' so downstream numeric comparisons are safe.
        """
        result = {
            'score': 0,
            'decision': 'REJECT',
            'reason': None,
            'approved': False,
            'parse_ok': False,
        }

        if not response or not response.strip():
            result['reason'] = 'Empty Claude response'
            return result

        try:
            cleaned = self._strip_code_fence(response)
            lines = [ln for ln in cleaned.split('\n')]

            found_score = False
            found_decision = False

            i = 0
            while i < len(lines):
                line = lines[i].strip()
                if not line:
                    i += 1
                    continue

                header_match = self._HEADER_RE.match(line)
                if not header_match:
                    i += 1
                    continue

                field = header_match.group(1).upper()
                value = header_match.group(2).strip()
                # Strip surrounding markdown bold/italic artifacts (e.g. **REASON:** X)
                value = re.sub(r'^\**\s*', '', value)
                value = re.sub(r'\s*\**$', '', value).strip()

                if field == 'SCORE':
                    numbers = re.findall(r'\d+', value)
                    if numbers:
                        try:
                            result['score'] = max(0, min(10, int(numbers[0])))
                            found_score = True
                        except (ValueError, IndexError):
                            pass

                elif field == 'DECISION':
                    decision = value.upper()
                    decision = re.sub(r'[^A-Z]', '', decision)  # strip brackets
                    if decision in ('APPROVE', 'REJECT', 'NEUTRAL'):
                        result['decision'] = decision
                        result['approved'] = decision == 'APPROVE'
                        found_decision = True

                elif field == 'REASON':
                    reason_parts = [value] if value else []
                    j = i + 1
                    while j < len(lines):
                        next_line = lines[j].strip()
                        if self._HEADER_RE.match(next_line):
                            break
                        if next_line:
                            reason_parts.append(next_line)
                        j += 1
                    joined = ' '.join(p for p in reason_parts if p).strip()
                    if joined:
                        result['reason'] = joined
                    i = j
                    continue

                i += 1

            # Fallback: scan entire blob if headers missing
            if not found_score:
                m = self._SCORE_FALLBACK_RE.search(cleaned)
                if m:
                    try:
                        result['score'] = max(0, min(10, int(m.group(1))))
                        found_score = True
                    except ValueError:
                        pass

            if not found_decision:
                m = self._DECISION_FALLBACK_RE.search(cleaned)
                if m:
                    decision = m.group(1).upper()
                    result['decision'] = decision
                    result['approved'] = decision == 'APPROVE'
                    found_decision = True

            result['parse_ok'] = found_score and found_decision

            if not result['parse_ok']:
                # Preserve a slice of raw response so the operator can diagnose
                preview = cleaned[:200].replace('\n', ' ⏎ ')
                result['reason'] = (
                    result.get('reason')
                    or f"Parse failed (missing SCORE/DECISION). Raw: {preview}"
                )
                self.logger.warning(
                    "⚠️ Claude response parse failed — score_found=%s decision_found=%s preview=%r",
                    found_score, found_decision, preview,
                )

            return result

        except Exception as e:
            self.logger.error(f"Error parsing minimal response: {e}", exc_info=True)
            return {
                'score': 0,
                'decision': 'REJECT',
                'reason': f'Parse error: {e}',
                'approved': False,
                'parse_ok': False,
            }
    
    def parse_enhanced_response(self, response: str) -> Dict:
        """
        Enhanced response parser with better error handling
        Addresses incomplete Claude response parsing
        """
        result = {
            'score': 5,  # Default to middle score
            'correct_type': 'SYSTEM_CORRECT',  # Default to accepting system classification
            'decision': 'APPROVE',  # Default to approving
            'reason': 'Analysis completed',
            'approved': True
        }
        
        try:
            if not response or not response.strip():
                self.logger.warning("⚠️ Empty Claude response received")
                result.update({
                    'score': 0,
                    'decision': 'REJECT',
                    'reason': 'Empty response from Claude',
                    'approved': False
                })
                return result
            
            response_text = response.strip()
            
            # Try structured format first
            parsed_structured = self._try_parse_structured_format(response_text)
            if parsed_structured:
                self.logger.debug("✅ Successfully parsed structured format")
                return parsed_structured
            
            # Try natural language parsing
            parsed_text = self._try_parse_natural_language(response_text)
            if parsed_text:
                self.logger.debug("✅ Successfully parsed natural language format")
                return parsed_text
            
            # Enhanced fallback parsing
            fallback_result = self._enhanced_fallback_parse(response_text)
            if fallback_result.get('score') is not None:
                self.logger.info("✅ Successfully used enhanced fallback parsing")
                return fallback_result
            
            # If all parsing fails, log warning but don't fail completely
            self.logger.warning("⚠️ Incomplete Claude response parsing - using safe defaults")
            return {
                'score': 5,
                'correct_type': 'SYSTEM_CORRECT',
                'decision': 'APPROVE',
                'reason': 'Parsing incomplete - using safe defaults',
                'approved': True,
                'parsing_failed': True
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error parsing Claude response: {e}")
            return {
                'score': 0,
                'correct_type': 'UNKNOWN',
                'decision': 'REJECT',
                'reason': f'Parse error: {str(e)}',
                'approved': False,
                'parsing_error': True
            }
    
    def _try_parse_structured_format(self, response: str) -> Optional[Dict]:
        """Try to parse the expected structured format with enhanced debugging"""
        result = {}
        
        try:
            lines = response.strip().split('\n')
            self.logger.debug(f"🔍 Structured parsing: Found {len(lines)} lines")
            
            found_fields = []
            section_markers = ('SCORE:', 'DECISION:', 'REASON:', 'CORRECT_TYPE:')

            i = 0
            while i < len(lines):
                line = lines[i].strip()
                if not line:
                    i += 1
                    continue

                self.logger.debug(f"  Processing line {i}: '{line}'")

                if line.startswith('SCORE:'):
                    score_text = line.replace('SCORE:', '').strip()
                    self.logger.debug(f"    Found SCORE field: '{score_text}'")
                    try:
                        score_match = re.search(r'(\d+)', score_text)
                        if score_match:
                            score = int(score_match.group(1))
                            result['score'] = max(0, min(10, score))
                            found_fields.append('score')
                            self.logger.debug(f"    Parsed score: {result['score']}")
                    except (ValueError, AttributeError) as e:
                        self.logger.debug(f"    Score parsing failed: {e}")
                        continue
                        
                elif line.startswith('CORRECT_TYPE:'):
                    correct_type = line.replace('CORRECT_TYPE:', '').strip().upper()
                    self.logger.debug(f"    Found CORRECT_TYPE field: '{correct_type}'")
                    if correct_type in ['BULL', 'BEAR', 'NONE', 'SYSTEM_CORRECT', 'UNKNOWN']:
                        result['correct_type'] = correct_type
                        found_fields.append('correct_type')
                        self.logger.debug(f"    Parsed correct_type: {result['correct_type']}")
                        
                elif line.startswith('DECISION:'):
                    decision = line.replace('DECISION:', '').strip().upper()
                    self.logger.debug(f"    Found DECISION field: '{decision}'")
                    if decision in ['APPROVE', 'REJECT', 'NEUTRAL']:
                        result['decision'] = decision
                        result['approved'] = decision == 'APPROVE'
                        found_fields.append('decision')
                        self.logger.debug(f"    Parsed decision: {result['decision']}")
                        
                elif line.startswith('REASON:'):
                    reason_parts = [line.replace('REASON:', '').strip()]
                    j = i + 1
                    while j < len(lines) and not lines[j].strip().startswith(section_markers):
                        if lines[j].strip():
                            reason_parts.append(lines[j].strip())
                        j += 1
                    reason = ' '.join(p for p in reason_parts if p).strip()
                    if reason:
                        result['reason'] = reason
                        found_fields.append('reason')
                        self.logger.debug(f"    Parsed reason: '{reason[:50]}...'")
                    i = j
                    continue
                i += 1

            self.logger.debug(f"🔍 Structured parsing found fields: {found_fields}")
            
            # Check if we got the minimum required fields
            if 'score' in result and 'decision' in result:
                # Fill in missing fields with defaults
                if 'correct_type' not in result:
                    result['correct_type'] = 'SYSTEM_CORRECT'
                    self.logger.debug("    Added default correct_type: SYSTEM_CORRECT")
                if 'reason' not in result:
                    result['reason'] = 'Structured analysis completed'
                    self.logger.debug("    Added default reason")
                
                self.logger.debug(f"✅ Structured parsing successful: {result}")
                return result
            else:
                self.logger.debug(f"❌ Structured parsing failed - missing required fields")
                return None
            
        except Exception as e:
            self.logger.debug(f"❌ Exception in structured parsing: {e}")
            return None
    
    def _try_parse_natural_language(self, response: str) -> Optional[Dict]:
        """Parse natural language responses from Claude with enhanced pattern matching"""
        result = {}
        
        try:
            response_lower = response.lower()
            self.logger.debug(f"🔍 Natural language parsing on {len(response)} chars")
            
            # Extract score using various patterns
            score_patterns = [
                r'score[:\s]+(\d+)(?:/10)?',
                r'(\d+)(?:/10)?\s*(?:score|points?|rating)',
                r'rate[sd]?\s*(?:this|it|the\s+signal)?\s*(?:at\s+)?(\d+)(?:/10)?',
                r'give[s]?\s*(?:this|it)?\s*(?:a\s+)?(\d+)(?:/10)?',
                r'(\d+)\s*out\s*of\s*10',
                r'quality[:\s]+(\d+)',
                r'strength[:\s]+(\d+)'
            ]
            
            score = None
            for i, pattern in enumerate(score_patterns):
                match = re.search(pattern, response_lower)
                if match:
                    try:
                        score = int(match.group(1))
                        if 0 <= score <= 10:
                            result['score'] = score
                            self.logger.debug(f"    Found score {score} using pattern {i}")
                            break
                    except (ValueError, IndexError):
                        continue
            
            if 'score' not in result:
                self.logger.debug("    No score found in natural language")
            
            # Determine decision based on keywords and score
            approve_keywords = ['approve', 'accept', 'good', 'strong', 'valid', 'buy', 'sell', 'trade', 'recommend', 'favorable']
            reject_keywords = ['reject', 'decline', 'weak', 'poor', 'invalid', 'avoid', 'skip', 'unfavorable', 'bad']
            
            approval_count = sum(1 for word in approve_keywords if word in response_lower)
            rejection_count = sum(1 for word in reject_keywords if word in response_lower)
            
            self.logger.debug(f"    Approval keywords: {approval_count}, Rejection keywords: {rejection_count}")
            
            if approval_count > rejection_count and approval_count > 0:
                decision = 'APPROVE'
            elif rejection_count > approval_count and rejection_count > 0:
                decision = 'REJECT'
            elif score is not None:
                decision = 'APPROVE' if score >= get_min_approval_score() else 'REJECT'
                self.logger.debug(f"    Decision based on score {score}: {decision}")
            else:
                decision = 'APPROVE'  # Conservative default
            
            if decision:
                result['decision'] = decision
                result['approved'] = decision == 'APPROVE'
                self.logger.debug(f"    Determined decision: {decision}")
            
            # Determine signal type
            if 'bull' in response_lower or 'buy' in response_lower or 'bullish' in response_lower:
                result['correct_type'] = 'BULL'
            elif 'bear' in response_lower or 'sell' in response_lower or 'bearish' in response_lower:
                result['correct_type'] = 'BEAR'
            else:
                result['correct_type'] = 'SYSTEM_CORRECT'
            
            # Extract reason
            sentences = re.split(r'[.!?]+', response)
            reason_found = False
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 10 and any(word in sentence.lower() for word in 
                    ['because', 'due to', 'analysis', 'signal', 'trend', 'support', 'resistance', 'indicator', 'momentum']):
                    result['reason'] = sentence
                    reason_found = True
                    break
            
            if not reason_found and sentences and sentences[0].strip():
                result['reason'] = sentences[0].strip()
            
            # Ensure minimum required fields
            if 'score' not in result:
                result['score'] = 6 if result.get('decision') == 'APPROVE' else 4
                
            if 'decision' not in result:
                result['decision'] = 'APPROVE' if result.get('score', 5) >= 6 else 'REJECT'
                result['approved'] = result['decision'] == 'APPROVE'
                
            if 'reason' not in result:
                result['reason'] = 'Natural language analysis completed'
            
            self.logger.debug(f"✅ Natural language parsing result: {result}")
            return result if result else None
            
        except Exception as e:
            self.logger.debug(f"❌ Exception in natural language parsing: {e}")
            return None
    
    def _enhanced_fallback_parse(self, response: str) -> Dict:
        """Enhanced fallback parsing with better intelligence"""
        try:
            response_lower = response.lower()
            
            # Look for any numerical scores
            score_patterns = [
                r'score[:\s]*(\d+)(?:/10)?',
                r'(\d+)(?:/10)?\s*(?:score|points?|rating)',
                r'rate[sd]?\s*(?:this|it|the\s+signal)?\s*(?:at\s+)?(\d+)(?:/10)?',
                r'give[s]?\s*(?:this|it)?\s*(?:a\s+)?(\d+)(?:/10)?',
                r'(\d+)\s*out\s*of\s*10',
                r'quality[:\s]*(\d+)',
                r'strength[:\s]*(\d+)',
                r'confidence[:\s]*(\d+)',
                r'rating[:\s]*(\d+)'
            ]
            
            score = None
            for pattern in score_patterns:
                matches = re.findall(pattern, response_lower)
                for match in matches:
                    try:
                        potential_score = int(match)
                        if 1 <= potential_score <= 10:
                            score = potential_score
                            break
                    except ValueError:
                        continue
                if score:
                    break
            
            # Decision keywords
            strong_approve = ['excellent', 'strong', 'good', 'recommend', 'buy', 'sell', 'take', 'enter']
            weak_approve = ['acceptable', 'okay', 'fair', 'moderate', 'reasonable']
            weak_reject = ['questionable', 'uncertain', 'risky', 'caution']
            strong_reject = ['reject', 'avoid', 'poor', 'weak', 'bad', 'dangerous']
            
            # Count keyword occurrences
            strong_approve_count = sum(1 for word in strong_approve if word in response_lower)
            weak_approve_count = sum(1 for word in weak_approve if word in response_lower)
            weak_reject_count = sum(1 for word in weak_reject if word in response_lower)
            strong_reject_count = sum(1 for word in strong_reject if word in response_lower)
            
            # Decision logic
            total_approve = strong_approve_count + (weak_approve_count * 0.5)
            total_reject = strong_reject_count + (weak_reject_count * 0.5)
            
            if total_reject > total_approve:
                decision = 'REJECT'
            elif total_approve > total_reject:
                decision = 'APPROVE'
            elif score is not None:
                decision = 'APPROVE' if score >= get_min_approval_score() else 'REJECT'
            else:
                decision = 'APPROVE'  # Conservative default
            
            # If no score found, estimate from keywords
            if score is None:
                if strong_approve_count > 0:
                    score = 8
                elif weak_approve_count > 0:
                    score = 6  
                elif weak_reject_count > 0:
                    score = 4
                elif strong_reject_count > 0:
                    score = 2
                else:
                    score = 5  # Neutral
            
            # Signal type detection
            signal_type = 'SYSTEM_CORRECT'  # Default
            if 'bull' in response_lower or 'bullish' in response_lower or 'buy' in response_lower:
                signal_type = 'BULL'
            elif 'bear' in response_lower or 'bearish' in response_lower or 'sell' in response_lower:
                signal_type = 'BEAR'
            
            # Extract reasoning
            sentences = re.split(r'[.!?]+', response)
            reason = 'Enhanced fallback analysis'
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 20:  # Meaningful length
                    reason = sentence[:200]  # Limit length
                    break
            
            return {
                'score': score,
                'correct_type': signal_type,
                'decision': decision,
                'reason': reason,
                'approved': decision == 'APPROVE',
                'parsing_method': 'enhanced_fallback'
            }
            
        except Exception as e:
            self.logger.error(f"Enhanced fallback parsing failed: {e}")
            return {
                'score': 5,
                'correct_type': 'SYSTEM_CORRECT',
                'decision': 'APPROVE',
                'reason': 'Fallback parsing error - using safe defaults',
                'approved': True,
                'parsing_method': 'safe_default'
            }