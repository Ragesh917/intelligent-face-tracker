from app.database.db import get_connection

conn = get_connection()

conn.execute("DELETE FROM events")
conn.execute("DELETE FROM visitors")

conn.commit()

print("Database cleaned successfully.")

visitors = conn.execute("SELECT COUNT(*) FROM visitors").fetchone()[0]
events = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]

print("Visitors:", visitors)
print("Events:", events)

conn.close()
