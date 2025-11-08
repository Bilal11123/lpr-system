import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path("license_plates.db")

def _init_db():
    """Create the table if it does not exist."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS plates (
            car_id          INTEGER,
            license_number  TEXT,
            score           REAL,
            timestamp       TEXT,
            video_source    TEXT
        )
        """
    )
    conn.commit()
    conn.close()

def upsert_plate(car_id: int, license_number: str, score: float, video_source: str):
    """
    Insert a new plate or update it if a higher score for the same car_id arrives.
    """
    if car_id <= 0 or license_number is None:
        return

    now = datetime.now().isoformat(timespec="seconds")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Try to fetch the current row
    cur.execute("SELECT score FROM plates WHERE car_id = ?", (car_id,))
    row = cur.fetchone()

    if row is None:
        # First time we see this car → INSERT
        cur.execute(
            """
            INSERT INTO plates (car_id, license_number, score, timestamp, video_source)
            VALUES (?, ?, ?, ?, ?)
            """,
            (car_id, license_number, score, now, video_source),
        )
    else:
        current_score = row[0]
        if score > current_score:
            # Higher confidence → UPDATE
            cur.execute(
                """
                UPDATE plates
                SET license_number = ?, score = ?, timestamp = ?
                WHERE car_id = ?
                """,
                (license_number, score, now, car_id),
            )

    conn.commit()
    conn.close()

# Initialise the DB the first time the module is imported
_init_db()