import os
from sqlalchemy import create_engine

DATABASE_URL = os.environ.get("DATABASE_URL")

DB = create_engine(DATABASE_URL)
