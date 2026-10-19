from app.database.db import get_connection

conn = get_connection()

visitors = conn.execute(
    "SELECT COUNT(*) FROM visitors"
).fetchone()[0]

entries = conn.execute(
    "SELECT COUNT(*) FROM events WHERE event_type = 'entry'"
).fetchone()[0]

exits = conn.execute(
    "SELECT COUNT(*) FROM events WHERE event_type = 'exit'"
).fetchone()[0]

total_events = conn.execute(
    "SELECT COUNT(*) FROM events"
).fetchone()[0]

print("Visitors:", visitors)
print("Entries:", entries)
print("Exits:", exits)
print("Total events:", total_events)

conn.close()