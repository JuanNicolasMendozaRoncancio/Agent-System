"""
PostgreSQL connection pool via SQLAlchemy.
"""

import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, DeclarativeBase

import logging
logger = logging.getLogger(__name__)


_DATABASE_URL = (
    f"postgresql+psycopg://"
    f"{os.getenv('POSTGRES_USER', 'agentes')}:"
    f"{os.getenv('POSTGRES_PASSWORD', 'agentes')}@"
    f"{os.getenv('POSTGRES_HOST', 'localhost')}:"
    f"{os.getenv('POSTGRES_PORT', '5432')}/"
    f"{os.getenv('POSTGRES_DB', 'agentes_db')}"
    f"?sslmode={os.getenv('POSTGRES_SSLMODE', 'disable')}"
)

engine = create_engine(
    _DATABASE_URL,
    pool_size= 10,
    max_overflow= 20,
    pool_pre_ping= True
)

SessionLocal = sessionmaker(bind=engine,
                            autocommit = False,
                            autoflush= False)

class Base(DeclarativeBase):
    pass

def get_db():
    """Yield a database session and ensure it is closed after use."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def check_connection() -> bool:
    """Return True if PostgreSQL is reachable, False otherwise."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        logger.error("PostgreSQL connection failed: %s", exc)
        return False

    