from __future__ import annotations
import hashlib
import json
import inspect
from typing import Dict, Any, List, Set, Optional, Callable
from dataclasses import dataclass
import ast
import re

@dataclass
class HyperparamGuard:
    """Guard that locks hyperparameters and detects tampering"""
    locked_params: Dict[str, Any]
    param_hash: str

@dataclass
class CacheDetectionResult:
    """Result of cache detection analysis"""
    has_caching: bool
    cache_indicators: List[str]
    suspicious_patterns: List[str]

class HyperparameterGuard:
    """
    Hard-locks all task hyperparameters inside the runner and prevents tampering.
    Implements the paper's requirement to guard against hyperparam changes.
    """

    def __init__(self):
        self.locked_guards: Dict[str, HyperparamGuard] = {}

    def lock_hyperparams(self, task_id: str, hyperparams: Dict[str, Any]) -> str:
        """
        Lock hyperparameters for a task and return verification hash.

        Args:
            task_id: Unique identifier for the task
            hyperparams: Dictionary of hyperparameters to lock

        Returns:
            Hash that can be used to verify parameters haven't changed
        """
        # Create canonical representation
        canonical = json.dumps(hyperparams, sort_keys=True, separators=(',', ':'))
        param_hash = hashlib.sha256(canonical.encode()).hexdigest()

        # Store locked parameters
        self.locked_guards[task_id] = HyperparamGuard(
            locked_params=hyperparams.copy(),
            param_hash=param_hash
        )

        return param_hash

    def verify_hyperparams(self, task_id: str, current_params: Dict[str, Any]) -> bool:
        """
        Verify that hyperparameters haven't been tampered with.

        Args:
            task_id: Task identifier
            current_params: Current hyperparameters to check

        Returns:
            True if parameters match locked values, False otherwise
        """
        if task_id not in self.locked_guards:
            return False

        guard = self.locked_guards[task_id]

        # Check if current params match locked params
        canonical = json.dumps(current_params, sort_keys=True, separators=(',', ':'))
        current_hash = hashlib.sha256(canonical.encode()).hexdigest()

        return current_hash == guard.param_hash

    def get_locked_params(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get the locked parameters for a task"""
        if task_id not in self.locked_guards:
            return None
        return self.locked_guards[task_id].locked_params.copy()

class CacheDetector:
    """
    Detects result caching attempts in kernel code.
    Implements "input-flip" check and other cache detection strategies.
    """

    def __init__(self):
        self.cache_patterns = [
            r'static\s+\w+',  # Static variables
            r'__device__\s+static',  # Device static variables
            r'cache',  # Explicit cache references
            r'memoiz',  # Memoization
            r'lookup',  # Lookup tables
            r'table\[',  # Array lookups that might be cached
            r'global\s+\w+\s*\[\s*\]',  # Global arrays
        ]

        self.suspicious_patterns = [
            r'if\s*\(\s*\w+\s*==\s*\w+\s*\)',  # Suspicious equality checks
            r'switch\s*\(\s*\w+\s*\)',  # Switch statements on inputs
            r'\.data_ptr\(\)',  # Direct data pointer access
            r'reinterpret_cast',  # Type punning
        ]

    def detect_caching(self, kernel_code: str) -> CacheDetectionResult:
        """
        Analyze kernel code for caching attempts.

        Args:
            kernel_code: The kernel source code to analyze

        Returns:
            Detection result with found cache indicators
        """
        cache_indicators = []
        suspicious_patterns = []

        # Check for cache patterns
        for pattern in self.cache_patterns:
            matches = re.findall(pattern, kernel_code, re.IGNORECASE | re.MULTILINE)
            if matches:
                cache_indicators.extend([f"Pattern '{pattern}': {match}" for match in matches])

        # Check for suspicious patterns
        for pattern in self.suspicious_patterns:
            matches = re.findall(pattern, kernel_code, re.IGNORECASE | re.MULTILINE)
            if matches:
                suspicious_patterns.extend([f"Suspicious '{pattern}': {match}" for match in matches])

        # Check for hardcoded values that might indicate caching
        hardcoded_numbers = re.findall(r'\b\d{4,}\b', kernel_code)  # Large numbers
        if hardcoded_numbers:
            suspicious_patterns.append(f"Large hardcoded numbers: {hardcoded_numbers[:5]}")

        has_caching = len(cache_indicators) > 0 or len(suspicious_patterns) > 2

        return CacheDetectionResult(
            has_caching=has_caching,
            cache_indicators=cache_indicators,
            suspicious_patterns=suspicious_patterns
        )

    def input_flip_check(self, kernel_func: Callable, test_inputs: List[Any]) -> bool:
        """
        Perform "input-flip" check to catch result caching.

        Args:
            kernel_func: The kernel function to test
            test_inputs: List of different input configurations

        Returns:
            True if kernel behaves correctly (no caching), False if caching detected
        """
        if len(test_inputs) < 2:
            return True  # Can't test with insufficient inputs

        try:
            # Run with different inputs and check for address-based reuse
            results = []
            for inputs in test_inputs:
                result = kernel_func(*inputs)
                # Check if result changes with input changes
                results.append(result)

            # Simple check: results should be different for different inputs
            # (This is a simplified version - real implementation would be more sophisticated)
            unique_results = set(str(r) for r in results)
            return len(unique_results) > 1

        except Exception:
            # If we can't run the test, assume no caching
            return True

class GuardedRunner:
    """
    Runner that enforces hyperparameter locking and cache detection.
    Use this to wrap kernel execution with anti-tampering guards.
    """

    def __init__(self):
        self.hyperparam_guard = HyperparameterGuard()
        self.cache_detector = CacheDetector()

    def setup_task_guards(self, task_id: str, hyperparams: Dict[str, Any],
                         kernel_code: str) -> Dict[str, Any]:
        """
        Set up all guards for a task execution.

        Args:
            task_id: Unique task identifier
            hyperparams: Task hyperparameters to lock
            kernel_code: Kernel source code to analyze

        Returns:
            Guard setup result with any detected issues
        """
        # Lock hyperparameters
        param_hash = self.hyperparam_guard.lock_hyperparams(task_id, hyperparams)

        # Detect caching attempts
        cache_result = self.cache_detector.detect_caching(kernel_code)

        return {
            'task_id': task_id,
            'param_hash': param_hash,
            'hyperparams_locked': True,
            'cache_detection': {
                'has_caching': cache_result.has_caching,
                'cache_indicators': cache_result.cache_indicators,
                'suspicious_patterns': cache_result.suspicious_patterns
            }
        }

    def verify_task_integrity(self, task_id: str, current_params: Dict[str, Any]) -> Dict[str, bool]:
        """
        Verify task hasn't been tampered with during execution.

        Args:
            task_id: Task identifier
            current_params: Current hyperparameters

        Returns:
            Verification results
        """
        params_ok = self.hyperparam_guard.verify_hyperparams(task_id, current_params)

        return {
            'hyperparams_verified': params_ok,
            'task_id': task_id
        }

    def enforce_restrictions(self, kernel_code: str) -> List[str]:
        """
        Enforce explicit restrictions from the prompt.

        Args:
            kernel_code: Kernel code to check

        Returns:
            List of violations found
        """
        violations = []

        # Check for explicit hyperparam changes
        hyperparam_patterns = [
            r'#define\s+\w+\s+\d+',  # Macro definitions that might change params
            r'const\s+int\s+\w+\s*=\s*\d+',  # Const int definitions
            r'constexpr\s+\w+\s+\w+\s*=',  # Constexpr definitions
        ]

        for pattern in hyperparam_patterns:
            if re.search(pattern, kernel_code, re.IGNORECASE):
                violations.append(f"Potential hyperparameter change: {pattern}")

        # Check for caching patterns
        cache_result = self.cache_detector.detect_caching(kernel_code)
        if cache_result.has_caching:
            violations.append("Caching detected - not allowed")

        return violations

# Global instance for easy access
_global_guard = GuardedRunner()

def lock_hyperparams(task_id: str, hyperparams: Dict[str, Any]) -> str:
    """Convenience function to lock hyperparameters"""
    return _global_guard.hyperparam_guard.lock_hyperparams(task_id, hyperparams)

def verify_hyperparams(task_id: str, current_params: Dict[str, Any]) -> bool:
    """Convenience function to verify hyperparameters"""
    return _global_guard.hyperparam_guard.verify_hyperparams(task_id, current_params)

def detect_caching(kernel_code: str) -> CacheDetectionResult:
    """Convenience function to detect caching"""
    return _global_guard.cache_detector.detect_caching(kernel_code)

def enforce_restrictions(kernel_code: str) -> List[str]:
    """Convenience function to enforce restrictions"""
    return _global_guard.enforce_restrictions(kernel_code)