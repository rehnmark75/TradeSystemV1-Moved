import time
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import OperationalError

DATABASE_URL = "postgresql://postgres:postgres@postgres:5432/forex"

MAX_RETRIES = 10
RETRY_DELAY = 2  # seconds

# Retry DB connection before initializing sessionmaker
for attempt in range(MAX_RETRIES):
    try:
        engine = create_engine(DATABASE_URL, pool_pre_ping=True)
        with engine.connect() as conn:
            print("✅ Connected to the database.")
        break
    except OperationalError as e:
        print(f"❌ Database not ready (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
        time.sleep(RETRY_DELAY)
else:
    raise RuntimeError("🚨 Could not connect to the database after retries.")

# ORM base and session setup
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Optional: init tables manually from this file
def init_db():
    from .models import Candle  # Avoid circular import until Base is defined
    Base.metadata.create_all(bind=engine)

# FastAPI dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
