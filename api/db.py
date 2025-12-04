import os
from urllib.parse import urlparse, urlunparse
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base


# Read and sanitize DATABASE_URL from env; default to local SQLite for dev
_DEFAULT_SQLITE = "sqlite:///./local.db"

_raw_db_url = os.getenv("DATABASE_URL", "")
# Strip accidental whitespace/newlines from copy-paste
_raw_db_url = _raw_db_url.strip() if isinstance(_raw_db_url, str) else ""

if not _raw_db_url:
    DATABASE_URL = _DEFAULT_SQLITE
else:
    # Normalize common provider format postgres:// → postgresql+psycopg2://
    if _raw_db_url.startswith("postgres://"):
        _raw_db_url = "postgresql+psycopg2://" + _raw_db_url[len("postgres://"):]
    # Remove stray whitespace inside netloc (host/userinfo) if present
    try:
        parsed = urlparse(_raw_db_url)
        # If scheme missing driver, keep as-is (already normalized above)
        netloc_clean = "".join(parsed.netloc.split())  # drop all whitespace characters
        # Rebuild only if any change was made
        if netloc_clean != parsed.netloc:
            parsed = parsed._replace(netloc=netloc_clean)
            _raw_db_url = urlunparse(parsed)
    except Exception:
        # If parsing fails, keep original string; SQLAlchemy will raise helpful errors
        pass
    DATABASE_URL = _raw_db_url


# For SQLAlchemy 2.0 style
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()