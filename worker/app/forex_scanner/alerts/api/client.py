"""
Claude API Client - Core Communication Module
Handles all Claude API communication with retry logic and rate limiting
Extracted from the large claude_api.py file for better modularity
"""

import requests
import json
import logging
import time
import random
from typing import Optional, Dict
from datetime import datetime


class APIClient:
    """
    Handles all Claude API communication with enhanced error handling
    """
    
    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022"):
        self.api_key = api_key
        self.base_url = "https://api.anthropic.com/v1/messages"
        self.model = model
        self.max_tokens = 150
        self.timeout = 30
        
        # Enhanced retry configuration
        self.max_retries = 3
        self.base_delay = 2.0
        self.max_delay = 30.0
        self.exponential_base = 2.0
        
        # Rate limiting protection
        self.last_api_call = 0
        self.min_call_interval = 1.2  # 50 calls/minute
        
        self.logger = logging.getLogger(__name__)
        
        if not api_key:
            self.logger.warning("⚠️ No Claude API key provided")
    
    def call_api(self, prompt: str, max_tokens: int = None) -> Optional[str]:
        """
        Enhanced API call with retry logic and rate limiting
        """
        if not self.api_key:
            self.logger.warning("No API key available")
            return None
        
        # Rate limiting protection
        current_time = time.time()
        time_since_last_call = current_time - self.last_api_call
        if time_since_last_call < self.min_call_interval:
            sleep_time = self.min_call_interval - time_since_last_call
            self.logger.debug(f"⏱️ Rate limiting: sleeping {sleep_time:.1f}s")
            time.sleep(sleep_time)
        
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
        
        data = {
            "model": self.model,
            "max_tokens": max_tokens or self.max_tokens,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                self.last_api_call = time.time()
                
                response = requests.post(
                    self.base_url,
                    headers=headers,
                    json=data,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    content = result.get('content', [])
                    if content and len(content) > 0:
                        return content[0].get('text', '')
                    else:
                        self.logger.warning("Empty content in successful response")
                        return None
                
                elif response.status_code == 529:
                    # API overload handling
                    error_msg = f"Claude API overloaded (attempt {attempt + 1}/{self.max_retries + 1})"
                    self.logger.warning(f"⚠️ {error_msg}")
                    
                    if attempt < self.max_retries:
                        delay = min(
                            self.base_delay * (self.exponential_base ** attempt) + random.uniform(0, 2),
                            self.max_delay
                        )
                        self.logger.info(f"🔄 Retrying in {delay:.1f}s due to API overload...")
                        time.sleep(delay)
                        continue
                    else:
                        self.logger.error(f"❌ Claude API overloaded after {self.max_retries} retries")
                        return None
                
                elif response.status_code == 429:
                    # Rate limit handling
                    error_msg = f"Rate limit exceeded (attempt {attempt + 1}/{self.max_retries + 1})"
                    self.logger.warning(f"⚠️ {error_msg}")
                    
                    if attempt < self.max_retries:
                        delay = min(60 + random.uniform(0, 30), self.max_delay)
                        self.logger.info(f"🔄 Retrying in {delay:.1f}s due to rate limit...")
                        time.sleep(delay)
                        continue
                    else:
                        self.logger.error(f"❌ Rate limit exceeded after {self.max_retries} retries")
                        return None
                
                else:
                    # Other HTTP errors
                    error_text = response.text
                    self.logger.error(f"❌ Claude API error: {response.status_code} - {error_text}")
                    
                    # Don't retry for client errors (4xx) except 429
                    if 400 <= response.status_code < 500 and response.status_code != 429:
                        return None
                    
                    if attempt < self.max_retries:
                        delay = self.base_delay * (self.exponential_base ** attempt)
                        self.logger.info(f"🔄 Retrying in {delay:.1f}s...")
                        time.sleep(delay)
                        continue
                    else:
                        return None
                
            except requests.exceptions.Timeout as e:
                last_exception = e
                error_msg = f"Request timeout (attempt {attempt + 1}/{self.max_retries + 1})"
                self.logger.warning(f"⚠️ {error_msg}")
                
                if attempt < self.max_retries:
                    delay = self.base_delay * (self.exponential_base ** attempt)
                    self.logger.info(f"🔄 Retrying in {delay:.1f}s due to timeout...")
                    time.sleep(delay)
                    continue
                else:
                    self.logger.error(f"❌ Claude API timeout after {self.max_retries} retries")
                    return None
                    
            except requests.exceptions.RequestException as e:
                last_exception = e
                error_msg = f"Request failed: {str(e)} (attempt {attempt + 1}/{self.max_retries + 1})"
                self.logger.warning(f"⚠️ {error_msg}")
                
                if attempt < self.max_retries:
                    delay = self.base_delay * (self.exponential_base ** attempt)
                    self.logger.info(f"🔄 Retrying in {delay:.1f}s due to request error...")
                    time.sleep(delay)
                    continue
                else:
                    self.logger.error(f"❌ Claude API request failed after {self.max_retries} retries: {e}")
                    return None
                    
            except Exception as e:
                last_exception = e
                self.logger.error(f"❌ Unexpected error in Claude API call: {e}")
                return None
        
        self.logger.error(f"❌ All retry attempts exhausted. Last exception: {last_exception}")
        return None
    
    def test_connection(self) -> bool:
        """Test Claude API connection"""
        if not self.api_key:
            return False
        
        try:
            test_prompt = "Please respond with 'Connection successful' if you can read this message."
            response = self.call_api(test_prompt)
            return response is not None and "successful" in response.lower()
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False
    
    def get_health_status(self) -> Dict:
        """
        Check Claude API health and return status
        """
        if not self.api_key:
            return {
                'status': 'unavailable',
                'reason': 'No API key configured',
                'recommendation': 'Configure CLAUDE_API_KEY'
            }
        
        try:
            test_response = self.call_api("Health check. Respond with 'OK' if working.", max_tokens=10)
            
            if test_response and 'OK' in test_response.upper():
                return {
                    'status': 'healthy',
                    'reason': 'API responding normally',
                    'recommendation': 'Continue normal operations'
                }
            elif test_response:
                return {
                    'status': 'degraded',
                    'reason': 'API responding but with unexpected response',
                    'recommendation': 'Monitor closely, consider fallback mode'
                }
            else:
                return {
                    'status': 'unhealthy',
                    'reason': 'API not responding or returning errors',
                    'recommendation': 'Use fallback analysis mode'
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'reason': f'Health check failed: {str(e)}',
                'recommendation': 'Use fallback analysis mode'
            }