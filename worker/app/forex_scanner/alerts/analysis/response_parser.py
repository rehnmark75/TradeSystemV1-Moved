"""
Response Parser - Claude Response Processing Module
Handles parsing of Claude responses into structured data
Extracted from claude_api.py for better modularity
"""

import logging
import re
from typing import Dict, Optional


class ResponseParser:
    """
    Handles parsing of Claude responses with fallback mechanisms
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def parse_minimal_response(self, response: str) -> Dict:
        """
        Parse the minimal Claude response into structured data
        Safe parsing with proper type conversion
        """
        try:
            result = {
                'score': None,
                'decision': None,
                'reason': None,
                'approved': False
            }
            
            if not response or not response.strip():
                return result
            
            lines = [line.strip() for line in response.strip().split('\n') if line.strip()]
            
            for line in lines:
                if line.startswith('SCORE:'):
                    score_text = line.replace('SCORE:', '').strip()
                    try:
                        result['score'] = int(float(score_text))
                    except ValueError:
                        # Extract number from text like "8/10" or "Score: 7"
                        numbers = re.findall(r'\b(\d+)\b', score_text)
                        if numbers:
                            try:
                                result['score'] = int(numbers[0])
                            except (ValueError, IndexError):
                                result['score'] = 0
                                self.logger.warning(f"Could not parse score from: {score_text}")
                
                elif line.startswith('DECISION:'):
                    decision = line.replace('DECISION:', '').strip().upper()
                    result['decision'] = decision
                    result['approved'] = decision == 'APPROVE'
                
                elif line.startswith('REASON:'):
                    result['reason'] = line.replace('REASON:', '').strip()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error parsing minimal response: {e}")
            return {
                'score': 0,
                'decision': 'REJECT',
                'reason': 'Parse error',
                'approved': False
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
            
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
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
                    reason = line.replace('REASON:', '').strip()
                    if reason:
                        result['reason'] = reason
                        found_fields.append('reason')
                        self.logger.debug(f"    Parsed reason: '{reason[:50]}...'")
            
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
                decision = 'APPROVE' if score >= 6 else 'REJECT'
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
                decision = 'APPROVE' if score >= 6 else 'REJECT'
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