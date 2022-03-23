from typing import Optional, List
from pydantic import BaseModel


class UsageLog(BaseModel):
    api_key: str
    is_active: bool
    never_expire: bool
    expiration_date: str
    latest_query_date: Optional[str]
    total_queries: int


class UsageLogs(BaseModel):
    logs: List[UsageLog]
