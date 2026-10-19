from app.events.event_state import EventStateManager


manager = EventStateManager(max_missing_frames=2)

visitor_id = "VISITOR_TEST001"


print("Test 1: First appearance")
result = manager.update(visitor_id, track_id=1)
print("Result:", result)
assert result == "entry"


print("Test 2: Same visitor remains")
result = manager.update(visitor_id, track_id=1)
print("Result:", result)
assert result == "none"


print("Test 3: Visitor remains again")
result = manager.update(visitor_id, track_id=1)
print("Result:", result)
assert result == "none"


print("Test 4: Visitor disappears")

print("Missing 1:", manager.mark_missing(set()))
print("Missing 2:", manager.mark_missing(set()))

exit_result = manager.mark_missing(set())
print("Missing 3:", exit_result)

assert exit_result == [visitor_id]


print("Test 5: Visitor returns")
result = manager.update(visitor_id, track_id=2)
print("Result:", result)

assert result == "entry"


print("\nALL EVENT STATE TESTS PASSED")