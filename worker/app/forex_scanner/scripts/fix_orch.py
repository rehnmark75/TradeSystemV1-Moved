#!/usr/bin/env python3
"""
Fix TradingOrchestrator to properly use strategic Claude analysis
This script updates the orchestrator to handle your strategic_minimal mode with learning focus
"""

import os
import re
import shutil
from datetime import datetime

def backup_file(file_path):
    """Create backup of the file"""
    if os.path.exists(file_path):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = f"{file_path}.backup_{timestamp}"
        shutil.copy2(file_path, backup_path)
        print(f"✅ Backup created: {backup_path}")
        return backup_path
    return None

def update_claude_initialization():
    """Update the Claude analyzer initialization to handle strategic mode"""
    file_path = '/app/forex_scanner/core/trading/trading_orchestrator.py'
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False
    
    backup_file(file_path)
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # 1. Update the Claude initialization to handle strategic modes
        old_init_pattern = r'(analysis_mode = getattr\(config, \'CLAUDE_ANALYSIS_MODE\', \'minimal\'\).*?)(                        except ImportError as e:)'
        
        new_init_code = '''analysis_mode = getattr(config, 'CLAUDE_ANALYSIS_MODE', 'minimal')
                        strategic_focus = getattr(config, 'CLAUDE_STRATEGIC_FOCUS', 'comprehensive')
                        
                        # Initialize Claude analyzer
                        self.claude_analyzer = ClaudeAnalyzer(claude_api_key, auto_save=True)
                        
                        # Set parameters based on analysis mode
                        if 'strategic' in analysis_mode:
                            self.claude_analyzer.max_tokens = 200  # More tokens for strategic analysis
                            self.analysis_mode = 'strategic'
                            self.strategic_focus = strategic_focus
                            self.logger.info("✅ Claude strategic analyzer initialized")
                            self.logger.info(f"   Mode: {analysis_mode}")
                            self.logger.info(f"   Strategic focus: {strategic_focus}")
                            self.logger.info(f"   Max tokens: {self.claude_analyzer.max_tokens}")
                        else:
                            self.claude_analyzer.max_tokens = 100  # Standard for minimal
                            self.analysis_mode = 'minimal'
                            self.strategic_focus = None
                            self.logger.info("✅ Claude minimal analyzer initialized")
                            self.logger.info(f"   Mode: {analysis_mode}")
                        
                        self.logger.info(f"   API Key: {claude_api_key[:20]}...{claude_api_key[-4:]}")
                        
                    '''
        
        if re.search(old_init_pattern, content, re.DOTALL):
            content = re.sub(old_init_pattern, new_init_code + r'\2', content, flags=re.DOTALL)
            print("✅ Updated Claude initialization for strategic mode")
        else:
            print("⚠️ Could not find Claude initialization pattern, trying fallback...")
            # Fallback: simpler replacement
            if 'analysis_mode = getattr(config' in content:
                content = content.replace(
                    "analysis_mode = getattr(config, 'CLAUDE_ANALYSIS_MODE', 'minimal')",
                    '''analysis_mode = getattr(config, 'CLAUDE_ANALYSIS_MODE', 'minimal')
                        strategic_focus = getattr(config, 'CLAUDE_STRATEGIC_FOCUS', 'comprehensive')
                        
                        # Initialize Claude analyzer
                        self.claude_analyzer = ClaudeAnalyzer(claude_api_key, auto_save=True)
                        
                        # Set parameters based on analysis mode
                        if 'strategic' in analysis_mode:
                            self.claude_analyzer.max_tokens = 200
                            self.analysis_mode = 'strategic'
                            self.strategic_focus = strategic_focus
                            self.logger.info(f"✅ Claude strategic analyzer initialized (focus: {strategic_focus})")
                        else:
                            self.analysis_mode = 'minimal'
                            self.strategic_focus = None
                            self.logger.info("✅ Claude minimal analyzer initialized")'''
                )
                print("✅ Applied fallback update for Claude initialization")
        
        # 2. Update the _run_claude_analysis method
        if '_run_claude_analysis' in content:
            # Find and replace the _run_claude_analysis method
            claude_analysis_pattern = r'(def _run_claude_analysis\(self, signal: Dict\) -> Optional\[Dict\]:.*?)(    def \w+|$)'
            
            new_claude_method = '''def _run_claude_analysis(self, signal: Dict) -> Optional[Dict]:
        """
        Run Claude analysis on signal - UPDATED to support strategic analysis
        """
        if not self.claude_analyzer:
            return None
        
        try:
            epic = signal.get('epic', 'Unknown')
            
            # Use strategic analysis if configured
            if hasattr(self, 'analysis_mode') and self.analysis_mode == 'strategic':
                strategic_focus = getattr(self, 'strategic_focus', 'comprehensive')
                
                self.logger.info(f"🧠 Running strategic Claude analysis: {epic} (focus: {strategic_focus})")
                
                # Call the strategic analysis method
                result = self.claude_analyzer.analyze_signal_strategic(signal, strategic_focus)
                
                if result:
                    self.logger.info(f"✅ Strategic analysis complete: {epic}")
                    self.logger.info(f"   Score: {result.get('score', 'N/A')}/10")
                    self.logger.info(f"   Decision: {result.get('decision', 'N/A')}")
                    self.logger.info(f"   Risk Level: {result.get('risk_level', 'N/A')}")
                    self.logger.info(f"   Key Insight: {result.get('key_insight', 'N/A')[:50]}...")
                    return result
                else:
                    self.logger.warning(f"⚠️ Strategic analysis returned no result for {epic}")
                    return None
            
            # Fallback to minimal analysis
            elif hasattr(self.claude_analyzer, 'analyze_signal_minimal'):
                self.logger.info(f"🤖 Running minimal Claude analysis: {epic}")
                result = self.claude_analyzer.analyze_signal_minimal(signal)
                
                if result:
                    self.logger.info(f"✅ Minimal analysis complete: {epic}")
                    self.logger.info(f"   Score: {result.get('score', 'N/A')}/10")
                    self.logger.info(f"   Decision: {result.get('decision', 'N/A')}")
                    return result
                else:
                    self.logger.warning(f"⚠️ Minimal analysis returned no result for {epic}")
                    return None
            
            # Final fallback to full analysis
            else:
                self.logger.info(f"🔍 Running full Claude analysis: {epic}")
                analysis = self.claude_analyzer.analyze_signal(signal)
                
                if analysis:
                    self.logger.info(f"✅ Full analysis complete: {epic}")
                    return {'analysis': analysis, 'mode': 'full'}
                else:
                    self.logger.warning(f"⚠️ Full analysis returned no result for {epic}")
                    return None
                    
        except Exception as e:
            self.logger.error(f"❌ Claude analysis failed for {epic}: {e}")
            return None

    '''
            
            if re.search(claude_analysis_pattern, content, re.DOTALL):
                content = re.sub(claude_analysis_pattern, new_claude_method + r'\2', content, flags=re.DOTALL)
                print("✅ Updated _run_claude_analysis method for strategic analysis")
            else:
                print("⚠️ Could not find _run_claude_analysis method")
        else:
            print("⚠️ _run_claude_analysis method not found in file")
        
        # 3. Ensure else clause for disabled Claude has proper attributes
        if 'self.claude_analyzer = None' in content and 'self.analysis_mode = ' not in content:
            content = content.replace(
                'self.claude_analyzer = None',
                '''self.claude_analyzer = None
            self.analysis_mode = 'disabled'
            self.strategic_focus = None'''
            )
            print("✅ Added missing attributes for disabled Claude mode")
        
        # Write the updated content
        with open(file_path, 'w') as f:
            f.write(content)
        
        print("✅ TradingOrchestrator updated for strategic Claude analysis!")
        return True
        
    except Exception as e:
        print(f"❌ Error updating file: {e}")
        return False

def test_strategic_configuration():
    """Test that the strategic configuration is working"""
    print("\n🧪 Testing strategic configuration...")
    
    try:
        import sys
        sys.path.insert(0, '/app/forex_scanner')
        
        import config
        
        # Check config values
        claude_mode = getattr(config, 'CLAUDE_ANALYSIS_MODE', 'NOT_SET')
        strategic_focus = getattr(config, 'CLAUDE_STRATEGIC_FOCUS', 'NOT_SET')
        claude_api_key = getattr(config, 'CLAUDE_API_KEY', 'NOT_SET')
        
        print(f"✅ CLAUDE_ANALYSIS_MODE: {claude_mode}")
        print(f"✅ CLAUDE_STRATEGIC_FOCUS: {strategic_focus}")
        print(f"✅ CLAUDE_API_KEY: {'SET' if claude_api_key != 'NOT_SET' else 'NOT_SET'}")
        
        # Test that strategic analysis method exists
        try:
            from alerts.claude_api import ClaudeAnalyzer
            
            if hasattr(ClaudeAnalyzer, 'analyze_signal_strategic'):
                print("✅ ClaudeAnalyzer.analyze_signal_strategic method exists")
            else:
                print("❌ ClaudeAnalyzer.analyze_signal_strategic method missing")
            
        except ImportError as e:
            print(f"❌ Could not import ClaudeAnalyzer: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

if __name__ == "__main__":
    print("🔧 ENABLING STRATEGIC CLAUDE ANALYSIS")
    print("=" * 50)
    
    # Test current configuration
    config_ok = test_strategic_configuration()
    
    if config_ok:
        # Update the orchestrator
        success = update_claude_initialization()
        
        if success:
            print("\n🎉 STRATEGIC ANALYSIS ENABLED!")
            print("\nYour TradingOrchestrator will now use:")
            print("✅ Strategic Claude analysis with 'learning' focus")
            print("✅ Enhanced token limit (200 tokens)")
            print("✅ Detailed logging of strategic results")
            print("\nTo verify, check your logs for:")
            print("  '🧠 Running strategic Claude analysis...'")
            print("  'Key Insight: ...'")
            print("  'Risk Level: ...'")
        else:
            print("\n❌ UPDATE FAILED")
            print("Manual intervention may be required.")
    else:
        print("\n⚠️ CONFIGURATION ISSUES DETECTED")
        print("Please check your config.py file.")