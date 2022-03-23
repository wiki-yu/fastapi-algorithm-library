from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader  # APIKeyQuery

from app.api.security._sqlite_access import sqlite_access

API_KEY_NAME = "api-key"

# api_key_query = APIKeyQuery(name=API_KEY_NAME, scheme_name="API key query", auto_error=False)
api_key_header = APIKeyHeader(name=API_KEY_NAME, scheme_name="API key header", auto_error=False)


async def api_key_security(
    # query_param: str = Security(api_key_query),
    header_param: str = Security(api_key_header),
) -> str:
    # if not query_param and not header_param:
    if not header_param:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="An API key must be passed as header"  # query or
        )

    # elif query_param and sqlite_access.check_key(query_param):
    #     return query_param

    elif header_param and sqlite_access.check_key(header_param):
        return header_param

    else:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Wrong, revoked, or expired API key."
        )
