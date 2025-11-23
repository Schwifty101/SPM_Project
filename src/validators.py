"""
Input validation utilities for Loyalty AI Agent
"""

from typing import Any, Optional, List
from datetime import datetime


class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass


def validate_customer_id(customer_id: Any) -> str:
    """
    Validate customer ID format

    Args:
        customer_id: Customer identifier to validate

    Returns:
        Validated customer ID string

    Raises:
        ValidationError: If customer_id is invalid
    """
    if customer_id is None:
        raise ValidationError("Customer ID cannot be None")

    customer_id = str(customer_id).strip()

    if not customer_id:
        raise ValidationError("Customer ID cannot be empty")

    if len(customer_id) > 50:
        raise ValidationError("Customer ID too long (max 50 characters)")

    return customer_id


def validate_positive_number(value: Any, field_name: str = "value") -> float:
    """
    Validate that a value is a positive number

    Args:
        value: Value to validate
        field_name: Name of field for error messages

    Returns:
        Validated float value

    Raises:
        ValidationError: If value is not a positive number
    """
    try:
        num = float(value)
    except (TypeError, ValueError):
        raise ValidationError(f"{field_name} must be a number")

    if num < 0:
        raise ValidationError(f"{field_name} must be non-negative")

    return num


def validate_probability(value: Any, field_name: str = "probability") -> float:
    """
    Validate that a value is a valid probability (0-1)

    Args:
        value: Value to validate
        field_name: Name of field for error messages

    Returns:
        Validated probability value

    Raises:
        ValidationError: If value is not a valid probability
    """
    num = validate_positive_number(value, field_name)

    if num > 1.0:
        raise ValidationError(f"{field_name} must be between 0 and 1")

    return num


def validate_customer_list(customer_ids: Any) -> List[str]:
    """
    Validate a list of customer IDs

    Args:
        customer_ids: List of customer IDs to validate

    Returns:
        Validated list of customer ID strings

    Raises:
        ValidationError: If input is not a valid list of customer IDs
    """
    if not isinstance(customer_ids, (list, tuple)):
        raise ValidationError("Customer IDs must be a list or tuple")

    if not customer_ids:
        raise ValidationError("Customer ID list cannot be empty")

    validated_ids = []
    for cid in customer_ids:
        validated_ids.append(validate_customer_id(cid))

    return validated_ids


def validate_limit(limit: Any) -> Optional[int]:
    """
    Validate limit parameter

    Args:
        limit: Limit value to validate

    Returns:
        Validated limit as integer or None

    Raises:
        ValidationError: If limit is invalid
    """
    if limit is None:
        return None

    try:
        limit_int = int(limit)
    except (TypeError, ValueError):
        raise ValidationError("Limit must be an integer")

    if limit_int <= 0:
        raise ValidationError("Limit must be positive")

    return limit_int
