"""Encoding."""

# standard
import re

# local
from .utils import validator


@validator
def base58(value: str, /):
    """Return whether or not given value is a valid base58 encoding.

    Examples:
        >>> base58('14pq6y9H2DLGahPsM4s7ugsNSD2uxpHsJx')
        # Output: True
        >>> base58('cUSECm5YzcXJwP')
        # Output: ValidationError(func=base58, args={'value': 'cUSECm5YzcXJwP'})

    Args:
        value:
            base58 string to validate.

    Returns:
        (Literal[True]): If `value` is a valid base58 encoding.
        (ValidationError): If `value` is an invalid base58 encoding.
    """
    return re.match(r"^[1-9A-HJ-NP-Za-km-z]+$", value) if value else False


@validator
def base64(value: str, /):
    """Return whether or not given value is a valid base64 encoding.

    Examples:
        >>> base64('Y2hhcmFjdGVyIHNldA==')
        # Output: True
        >>> base64('cUSECm5YzcXJwP')
        # Output: ValidationError(func=base64, args={'value': 'cUSECm5YzcXJwP'})

    Args:
        value:
            base64 string to validate.

    Returns:
        (Literal[True]): If `value` is a valid base64 encoding.
        (ValidationError): If `value` is an invalid base64 encoding.
    """
    return (
        re.match(r"^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$", value)
        if value
        else False
    )
