"""Astrion DQ Agent Layer.

Three layers:
    Deterministic  (existing graph pipeline)
    Agent          (LLM-powered, this package)
    Reliability    (key rotation, retry, validation, fallback — this package)

Public API:
    explain_issues(issues, table_sizes, key_manager)
    prioritise_issues(issues, table_sizes, key_manager)
    generate_fix_code(issue, extra_schema, key_manager)
    generate_report(issues, run_id, table_sizes, key_manager)

All four functions fall back to deterministic output when the LLM is
unavailable, keys are exhausted, or the response fails validation.
"""
from .code_generator import generate_fix_code
from .explainer import explain_issues
from .fallback import fallback_explain, fallback_generate_fix, fallback_prioritise, fallback_report
from .key_manager import APIKeyManager, AllKeysExhausted
from .prioritiser import prioritise_issues
from .report_generator import generate_report
from .validation import ValidationResult, build_allowed_columns, validate_input, validate_response

__all__ = [
    "APIKeyManager",
    "AllKeysExhausted",
    "ValidationResult",
    "build_allowed_columns",
    "validate_input",
    "validate_response",
    "explain_issues",
    "prioritise_issues",
    "generate_fix_code",
    "generate_report",
    "fallback_explain",
    "fallback_prioritise",
    "fallback_generate_fix",
    "fallback_report",
]
