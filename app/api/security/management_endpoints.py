import os
from distutils.util import strtobool
from typing import Any, Optional

from fastapi import APIRouter, Depends

from app import schemas
from app.api.security._security_secret import secret_based_security
from app.api.security._sqlite_access import sqlite_access

api_management_router = APIRouter()

try:
    show_endpoints = not bool(strtobool(os.environ["FASTAPI_SIMPLE_SECURITY_HIDE_DOCS"]))
except KeyError:
    # if there's a key error, it's going to be true
    show_endpoints = False if "FASTAPI_SIMPLE_SECURITY_HIDE_DOCS" in os.environ else True


@api_management_router.get("/new",
                           dependencies=[Depends(secret_based_security)],
                           include_in_schema=show_endpoints)
def get_new_api_key(never_expires: Optional[Any] = None) -> str:
    """
    Args:
        never_expires: if set (by anything), the created API key will never be considered expired

    Returns:
        api_key: a newly generated API key
    """
    return sqlite_access.create_key(never_expires)


@api_management_router.get("/revoke",
                           dependencies=[Depends(secret_based_security)],
                           include_in_schema=show_endpoints)
def revoke_api_key(api_key: str):
    """
    Revokes the usage of the given API key

    Args:
        api_key: the api_key to revoke
    """
    return sqlite_access.revoke_key(api_key)


@api_management_router.get("/renew",
                           dependencies=[Depends(secret_based_security)],
                           include_in_schema=show_endpoints)
def renew_api_key(api_key: str, expiration_date: str = None):
    """
    Renews the chosen API key, reactivating it if it was revoked.

    Args:
        api_key: the API key to renew
        expiration_date: the new expiration date in ISO format
    """
    return sqlite_access.renew_key(api_key, expiration_date)


@api_management_router.get("/logs",
                           dependencies=[Depends(secret_based_security)],
                           response_model=schemas.UsageLogs,
                           include_in_schema=show_endpoints)
def get_api_key_usage_logs():
    """
    Returns usage information for all API keys
    """
    # TODO Add some sort of filtering on older keys/unused keys?

    return schemas.UsageLogs(
        logs=[
            schemas.UsageLog(
                api_key=row[0],
                is_active=row[1],
                never_expire=row[2],
                expiration_date=row[3],
                latest_query_date=row[4],
                total_queries=row[5],
            )
            for row in sqlite_access.get_usage_stats()
        ]
    )
