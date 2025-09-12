# validation/error_handling.py
"""
Comprehensive Error Handling and Logging for Signal Validation

This module provides enhanced error handling, logging, and recovery mechanisms
for the signal validation system. It includes custom exceptions, error reporting,
and graceful error recovery strategies.
"""

import logging
import traceback
import sys
from datetime import datetime
from typing import Dict, Any, Optional, List, Union, Callable
from functools import wraps
from pathlib import Path
import json

from .replay_config import ReplayConfig, ERROR_CONFIG


class ValidationError(Exception):
    """Base exception for validation errors"""
    def __init__(self, message: str, epic: str = None, timestamp: datetime = None, details: Dict[str, Any] = None):
        super().__init__(message)
        self.epic = epic
        self.timestamp = timestamp
        self.details = details or {}
        self.error_timestamp = datetime.now()


class DataRetrievalError(ValidationError):
    """Exception for data retrieval failures"""
    def __init__(self, message: str, epic: str = None, timestamp: datetime = None, query_details: Dict[str, Any] = None):
        super().__init__(message, epic, timestamp, query_details)


class StateRecreationError(ValidationError):
    """Exception for scanner state recreation failures"""
    def __init__(self, message: str, timestamp: datetime = None, config_details: Dict[str, Any] = None):
        super().__init__(message, None, timestamp, config_details)


class SignalDetectionError(ValidationError):
    """Exception for signal detection failures"""
    def __init__(self, message: str, epic: str = None, timestamp: datetime = None, strategy: str = None, detection_details: Dict[str, Any] = None):
        super().__init__(message, epic, timestamp, detection_details)
        self.strategy = strategy


class ConfigurationError(ValidationError):
    """Exception for configuration errors"""
    def __init__(self, message: str, config_key: str = None, config_details: Dict[str, Any] = None):
        super().__init__(message, None, None, config_details)
        self.config_key = config_key


class ValidationErrorHandler:
    """
    Comprehensive error handling and recovery system for validation operations
    
    This class provides:
    - Error classification and reporting
    - Automatic retry mechanisms
    - Error logging and aggregation
    - Graceful degradation strategies
    - Performance impact tracking
    """
    
    def __init__(self, logger: logging.Logger = None):
        """
        Initialize the error handler
        
        Args:
            logger: Logger instance for error reporting
        """
        self.logger = logger or logging.getLogger(__name__)
        
        # Error tracking
        self.error_counts = {}
        self.error_history = []
        self.recent_errors = []
        self.max_error_history = 1000
        self.max_recent_errors = 50
        
        # Retry configuration
        self.max_retries = ERROR_CONFIG['max_retries']
        self.retry_delay = ERROR_CONFIG['retry_delay_seconds']
        self.continue_on_error = ERROR_CONFIG['continue_on_error']
        
        # Error reporting
        self.detailed_logging = ERROR_CONFIG['detailed_error_logging']
        self.save_failed_validations = ERROR_CONFIG['save_failed_validations']
        
        self.logger.info("🛡️ ValidationErrorHandler initialized")
        self.logger.info(f"   Max retries: {self.max_retries}")
        self.logger.info(f"   Continue on error: {self.continue_on_error}")
        self.logger.info(f"   Detailed logging: {self.detailed_logging}")
    
    def handle_error(
        self,
        error: Exception,
        context: Dict[str, Any] = None,
        epic: str = None,
        timestamp: datetime = None,
        operation: str = None
    ) -> Dict[str, Any]:
        """
        Handle and classify an error
        
        Args:
            error: Exception that occurred
            context: Additional context information
            epic: Epic being processed when error occurred
            timestamp: Timestamp being validated when error occurred
            operation: Operation being performed when error occurred
            
        Returns:
            Error information dictionary
        """
        try:
            error_info = self._classify_error(error, context, epic, timestamp, operation)
            
            # Update error tracking
            self._track_error(error_info)
            
            # Log error based on configuration
            self._log_error(error_info)
            
            # Save error details if configured
            if self.save_failed_validations:
                self._save_error_details(error_info)
            
            # Determine recovery strategy
            recovery_strategy = self._determine_recovery_strategy(error_info)
            error_info['recovery_strategy'] = recovery_strategy
            
            return error_info
            
        except Exception as e:
            # Meta-error handling
            self.logger.error(f"❌ Error in error handler: {e}")
            return {
                'error_type': 'handler_error',
                'message': str(e),
                'original_error': str(error),
                'timestamp': datetime.now(),
                'recovery_strategy': 'fail'
            }
    
    def with_retry(
        self,
        operation: Callable,
        operation_name: str,
        max_retries: int = None,
        retry_delay: float = None,
        context: Dict[str, Any] = None
    ):
        """
        Execute operation with automatic retry on failure
        
        Args:
            operation: Function to execute
            operation_name: Name of operation for logging
            max_retries: Maximum retry attempts (uses default if None)
            retry_delay: Delay between retries (uses default if None)
            context: Context information for error handling
            
        Returns:
            Result of operation
        """
        max_attempts = (max_retries or self.max_retries) + 1
        delay = retry_delay or self.retry_delay
        
        for attempt in range(max_attempts):
            try:
                if attempt > 0:
                    self.logger.info(f"🔄 Retry attempt {attempt}/{max_retries} for {operation_name}")
                
                result = operation()
                
                if attempt > 0:
                    self.logger.info(f"✅ {operation_name} succeeded after {attempt} retries")
                
                return result
                
            except Exception as e:
                error_info = self.handle_error(
                    error=e,
                    context=context,
                    operation=operation_name
                )
                
                if attempt < max_attempts - 1:  # Not the last attempt
                    recovery_strategy = error_info.get('recovery_strategy', 'fail')
                    
                    if recovery_strategy in ['retry', 'retry_with_delay']:
                        self.logger.warning(f"⚠️ {operation_name} failed (attempt {attempt + 1}), retrying in {delay}s: {str(e)}")
                        
                        if recovery_strategy == 'retry_with_delay':
                            import time
                            time.sleep(delay)
                        
                        continue
                
                # Final attempt or non-retryable error
                self.logger.error(f"❌ {operation_name} failed after {attempt + 1} attempts: {str(e)}")
                raise
    
    def safe_execute(
        self,
        operation: Callable,
        operation_name: str,
        default_return=None,
        context: Dict[str, Any] = None,
        suppress_exceptions: List[type] = None
    ):
        """
        Execute operation with safe error handling
        
        Args:
            operation: Function to execute
            operation_name: Name of operation for logging
            default_return: Default value to return on error
            context: Context information for error handling
            suppress_exceptions: Exception types to suppress and return default
            
        Returns:
            Result of operation or default_return on error
        """
        try:
            return operation()
            
        except Exception as e:
            error_info = self.handle_error(
                error=e,
                context=context,
                operation=operation_name
            )
            
            # Check if this exception should be suppressed
            if suppress_exceptions and any(isinstance(e, exc_type) for exc_type in suppress_exceptions):
                self.logger.warning(f"⚠️ {operation_name} failed but continuing: {str(e)}")
                return default_return
            
            # Check if we should continue on error
            if self.continue_on_error:
                self.logger.error(f"❌ {operation_name} failed but continuing: {str(e)}")
                return default_return
            else:
                # Re-raise the exception
                raise
    
    def create_error_context(
        self,
        epic: str = None,
        timestamp: datetime = None,
        timeframe: str = None,
        strategy: str = None,
        operation: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Create error context dictionary"""
        context = {
            'epic': epic,
            'timestamp': timestamp.isoformat() if timestamp else None,
            'timeframe': timeframe,
            'strategy': strategy,
            'operation': operation,
            'context_created': datetime.now().isoformat()
        }
        
        # Add additional context
        context.update(kwargs)
        
        return {k: v for k, v in context.items() if v is not None}
    
    def _classify_error(
        self,
        error: Exception,
        context: Dict[str, Any] = None,
        epic: str = None,
        timestamp: datetime = None,
        operation: str = None
    ) -> Dict[str, Any]:
        """Classify error and extract relevant information"""
        error_type = type(error).__name__
        error_message = str(error)
        
        # Classify by exception type
        if isinstance(error, DataRetrievalError):
            category = 'data_retrieval'
            severity = 'high'
        elif isinstance(error, StateRecreationError):
            category = 'state_recreation'
            severity = 'high'
        elif isinstance(error, SignalDetectionError):
            category = 'signal_detection'
            severity = 'medium'
        elif isinstance(error, ConfigurationError):
            category = 'configuration'
            severity = 'high'
        elif isinstance(error, (ConnectionError, TimeoutError)):
            category = 'network'
            severity = 'medium'
        elif isinstance(error, (KeyError, AttributeError)):
            category = 'data_structure'
            severity = 'medium'
        elif isinstance(error, (ValueError, TypeError)):
            category = 'data_validation'
            severity = 'medium'
        elif isinstance(error, MemoryError):
            category = 'resource'
            severity = 'critical'
        else:
            category = 'unknown'
            severity = 'medium'
        
        # Extract additional information for specific error types
        additional_info = {}
        if hasattr(error, 'details'):
            additional_info.update(error.details)
        
        return {
            'error_type': error_type,
            'category': category,
            'severity': severity,
            'message': error_message,
            'epic': epic,
            'timestamp': timestamp,
            'operation': operation,
            'context': context or {},
            'additional_info': additional_info,
            'traceback': traceback.format_exc() if self.detailed_logging else None,
            'error_timestamp': datetime.now()
        }
    
    def _track_error(self, error_info: Dict[str, Any]) -> None:
        """Track error in statistics"""
        error_type = error_info['error_type']
        category = error_info['category']
        
        # Update counts
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        self.error_counts[f"category_{category}"] = self.error_counts.get(f"category_{category}", 0) + 1
        
        # Add to history
        self.error_history.append(error_info)
        if len(self.error_history) > self.max_error_history:
            self.error_history = self.error_history[-self.max_error_history:]
        
        # Add to recent errors
        self.recent_errors.append(error_info)
        if len(self.recent_errors) > self.max_recent_errors:
            self.recent_errors = self.recent_errors[-self.max_recent_errors:]
    
    def _log_error(self, error_info: Dict[str, Any]) -> None:
        """Log error based on severity and configuration"""
        severity = error_info['severity']
        error_type = error_info['error_type']
        message = error_info['message']
        epic = error_info.get('epic')
        operation = error_info.get('operation')
        
        # Construct log message
        log_msg = f"{error_type}: {message}"
        if epic:
            log_msg = f"[{epic}] {log_msg}"
        if operation:
            log_msg = f"({operation}) {log_msg}"
        
        # Log based on severity
        if severity == 'critical':
            self.logger.critical(f"🔴 CRITICAL: {log_msg}")
        elif severity == 'high':
            self.logger.error(f"❌ ERROR: {log_msg}")
        elif severity == 'medium':
            self.logger.warning(f"⚠️ WARNING: {log_msg}")
        else:
            self.logger.info(f"ℹ️ INFO: {log_msg}")
        
        # Add detailed information if enabled
        if self.detailed_logging:
            if error_info.get('context'):
                self.logger.debug(f"   Context: {error_info['context']}")
            if error_info.get('additional_info'):
                self.logger.debug(f"   Details: {error_info['additional_info']}")
            if error_info.get('traceback'):
                self.logger.debug(f"   Traceback: {error_info['traceback']}")
    
    def _save_error_details(self, error_info: Dict[str, Any]) -> None:
        """Save error details to file for analysis"""
        try:
            error_file = Path("validation_errors.jsonl")
            
            # Prepare error data for serialization
            error_data = {
                k: v.isoformat() if isinstance(v, datetime) else v
                for k, v in error_info.items()
            }
            
            # Append to error log file
            with open(error_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(error_data, default=str) + '\n')
                
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to save error details: {e}")
    
    def _determine_recovery_strategy(self, error_info: Dict[str, Any]) -> str:
        """Determine appropriate recovery strategy for error"""
        category = error_info['category']
        severity = error_info['severity']
        error_type = error_info['error_type']
        
        # Critical errors should fail immediately
        if severity == 'critical':
            return 'fail'
        
        # Strategy based on error category
        if category == 'network':
            return 'retry_with_delay'
        elif category == 'data_retrieval':
            return 'retry' if 'timeout' in error_info['message'].lower() else 'fail'
        elif category == 'configuration':
            return 'fail'  # Configuration errors usually require manual intervention
        elif category == 'data_validation':
            return 'skip'  # Skip invalid data and continue
        elif category == 'signal_detection':
            return 'continue'  # Continue with next epic/timestamp
        else:
            return 'retry' if severity == 'medium' else 'fail'
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of errors encountered"""
        total_errors = len(self.error_history)
        
        if total_errors == 0:
            return {'total_errors': 0, 'message': 'No errors recorded'}
        
        # Calculate statistics
        categories = {}
        severities = {}
        recent_count = len(self.recent_errors)
        
        for error in self.error_history:
            category = error['category']
            severity = error['severity']
            
            categories[category] = categories.get(category, 0) + 1
            severities[severity] = severities.get(severity, 0) + 1
        
        # Get most common error
        most_common_error = max(self.error_counts.items(), key=lambda x: x[1]) if self.error_counts else ('none', 0)
        
        return {
            'total_errors': total_errors,
            'recent_errors': recent_count,
            'error_categories': categories,
            'error_severities': severities,
            'most_common_error': most_common_error[0],
            'most_common_count': most_common_error[1],
            'error_rate': len(self.recent_errors) / min(100, total_errors) if total_errors > 0 else 0
        }
    
    def clear_error_history(self) -> None:
        """Clear error history and statistics"""
        self.error_counts.clear()
        self.error_history.clear()
        self.recent_errors.clear()
        self.logger.info("🧹 Error history cleared")


def with_error_handling(
    operation_name: str = None,
    max_retries: int = None,
    continue_on_error: bool = None,
    default_return = None
):
    """
    Decorator for adding error handling to functions
    
    Args:
        operation_name: Name of operation for logging
        max_retries: Maximum retry attempts
        continue_on_error: Whether to continue on error
        default_return: Default return value on error
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get or create error handler
            error_handler = getattr(wrapper, '_error_handler', None)
            if not error_handler:
                error_handler = ValidationErrorHandler()
                wrapper._error_handler = error_handler
            
            op_name = operation_name or func.__name__
            
            # Create context from function arguments
            context = {
                'function': func.__name__,
                'args_count': len(args),
                'kwargs_keys': list(kwargs.keys())
            }
            
            if max_retries is not None:
                return error_handler.with_retry(
                    operation=lambda: func(*args, **kwargs),
                    operation_name=op_name,
                    max_retries=max_retries,
                    context=context
                )
            else:
                return error_handler.safe_execute(
                    operation=lambda: func(*args, **kwargs),
                    operation_name=op_name,
                    default_return=default_return,
                    context=context
                )
        
        return wrapper
    return decorator


def setup_validation_logging(
    log_level: str = 'INFO',
    log_file: str = None,
    enable_console: bool = True,
    enable_detailed: bool = False
) -> logging.Logger:
    """
    Setup comprehensive logging for validation system
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        enable_console: Whether to enable console logging
        enable_detailed: Whether to enable detailed logging
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger('validation')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatter
    if enable_detailed:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
    
    # Add console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Add file handler
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            print(f"Warning: Could not create log file {log_file}: {e}")
    
    logger.info("📝 Validation logging configured")
    return logger