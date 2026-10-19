from app.database.db import get_connection

conn = get_connection()

print("\n--- VISITORS ---")

rows = conn.execute("""
    SELECT visitor_id, created_at, first_seen_at, last_seen_at
    FROM visitors
    ORDER BY created_at
""").fetchall()

for row in rows:
    print(row)

print("\n--- EVENTS ---")

events = conn.execute("""
    SELECT event_id, visitor_id, event_type, timestamp, image_path
    FROM events
    ORDER BY event_id
""").fetchall()

for event in events:
    print(event)

conn.close()