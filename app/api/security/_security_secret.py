import os
import uuid
import warnings

from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader


try:
    MANAGEMENT_SECRET_KEY = os.environ["FASTAPI_SIMPLE_SECURITY_SECRET"]
except KeyError:
    MANAGEMENT_SECRET_KEY = str(uuid.uuid4())

    warnings.warn(
        f"ENVIRONMENT VARIABLE 'FASTAPI_SIMPLE_SECURITY_SECRET' NOT FOUND\n"
        f"\tGenerated a single-use secret key for this session:\n"
        f"\t{MANAGEMENT_SECRET_KEY=}"
    )

# Note: By default, nginx silently drops headers with underscores. Use hyphens in string instead.
MANAGEMENT_SECRET_NAME = "secret-key"

secret_header = APIKeyHeader(
    name=MANAGEMENT_SECRET_NAME,
    scheme_name="Management Secret Header",
    auto_error=False
)


async def secret_based_security(header_param: str = Security(secret_header)):
    """
    Args:
        header_param: parsed header field secret_header

    Returns:
        True if the authentication was successful

    Raises:
        HTTPException if the authentication failed
    """
    if header_param == MANAGEMENT_SECRET_KEY:
        return True

    if not header_param:
        error = "secret_key must be passed as a header field"
    else:
        error = (
            "Wrong secret key. If not set through environment variable 'FASTAPI_SIMPLE_SECURITY_SECRET', it was "
            "generated automatically at startup and appears in the server logs."
        )
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail=error
    )
