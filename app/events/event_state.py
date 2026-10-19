class EventStateManager:
    def __init__(self, max_missing_frames=30):
        self.max_missing_frames = max_missing_frames
        self.visitors = {}

    def update(self, visitor_id, track_id):
        """
        Update the state of a visitor.

        A visitor gets an entry event only when:
        - They are seen for the first time in the current visit, or
        - They previously exited and are now seen again.

        A continuously visible visitor will not generate
        duplicate entry events.
        """

        if visitor_id not in self.visitors:
            self.visitors[visitor_id] = {
                "track_id": track_id,
                "missing_frames": 0,
                "active": True,
                "entry_logged": True,
                "exit_logged": False,
            }

            return "entry"

        visitor = self.visitors[visitor_id]

        # Visitor has returned after a previous exit.
        if not visitor["active"]:
            visitor["track_id"] = track_id
            visitor["missing_frames"] = 0
            visitor["active"] = True
            visitor["entry_logged"] = True
            visitor["exit_logged"] = False

            return "entry"

        # Visitor is still present.
        visitor["track_id"] = track_id
        visitor["missing_frames"] = 0
        visitor["active"] = True

        return "none"

    def mark_missing(self, active_visitor_ids):
        """
        Mark visitors that are no longer visible.

        An exit is generated only once after the visitor has
        been missing for more than max_missing_frames.
        """

        exit_visitors = []

        for visitor_id, visitor in list(self.visitors.items()):

            if visitor_id in active_visitor_ids:
                continue

            visitor["missing_frames"] += 1

            if (
                visitor["missing_frames"] > self.max_missing_frames
                and visitor["active"]
            ):
                visitor["active"] = False

                if not visitor["exit_logged"]:
                    visitor["exit_logged"] = True
                    exit_visitors.append(visitor_id)

        return exit_visitors

    def get_active_visitors(self):
        """
        Return all visitors currently considered active.
        """

        return [
            visitor_id
            for visitor_id, visitor in self.visitors.items()
            if visitor["active"]
        ]