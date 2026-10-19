import sqlite3
from pathlib import Path


def get_connection(db_path="data/database.db"):
    """Create and return a SQLite database connection."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(
        path,
        check_same_thread=False,
    )

    return connection