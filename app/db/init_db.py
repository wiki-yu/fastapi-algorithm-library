from sqlalchemy.ext.declarative import declarative_base

from app.db import base  # noqa: F401
from app.db.session import engine

# from app.db.session import Base
from app.db.base_class import Base

Base = Base()


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
