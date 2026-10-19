from datetime import datetime, timezone
from uuid import uuid4

import numpy as np


class FaceRegistry:
    def __init__(self, connection, similarity_threshold=0.5):
        self.connection = connection
        self.similarity_threshold = similarity_threshold

    def _generate_visitor_id(self):
        """Generate a unique visitor ID."""
        return f"VISITOR_{uuid4().hex[:8].upper()}"

    def _embedding_from_blob(self, blob):
        """Convert stored SQLite BLOB back to a NumPy embedding."""
        if blob is None:
            return None

        return np.frombuffer(blob, dtype=np.float32)

    def _embedding_to_blob(self, embedding):
        """Convert a NumPy embedding into a SQLite-compatible BLOB."""
        return np.asarray(embedding, dtype=np.float32).tobytes()

    def _cosine_similarity(self, embedding_a, embedding_b):
        """Calculate cosine similarity between two embeddings."""
        a = np.asarray(embedding_a, dtype=np.float32)
        b = np.asarray(embedding_b, dtype=np.float32)

        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(np.dot(a, b) / (norm_a * norm_b))

    def find_match(self, embedding):
        """
        Compare an embedding against all registered visitors.

        Returns:
            (visitor_id, similarity) if a match is found.
            (None, 0.0) otherwise.
        """

        cursor = self.connection.cursor()

        cursor.execute(
            """
            SELECT visitor_id, embedding
            FROM visitors
            WHERE embedding IS NOT NULL
            """
        )

        best_visitor = None
        best_similarity = 0.0

        for visitor_id, stored_blob in cursor.fetchall():
            stored_embedding = self._embedding_from_blob(stored_blob)

            if stored_embedding is None:
                continue

            similarity = self._cosine_similarity(
                embedding,
                stored_embedding,
            )

            if similarity > best_similarity:
                best_similarity = similarity
                best_visitor = visitor_id

        if (
            best_visitor is not None
            and best_similarity >= self.similarity_threshold
        ):
            return best_visitor, best_similarity

        return None, best_similarity

    def register_new_visitor(self, embedding):
        """Register a completely new visitor."""

        visitor_id = self._generate_visitor_id()

        timestamp = datetime.now(timezone.utc).isoformat()

        embedding_blob = self._embedding_to_blob(embedding)

        cursor = self.connection.cursor()

        cursor.execute(
            """
            INSERT INTO visitors (
                visitor_id,
                created_at,
                first_seen_at,
                last_seen_at,
                embedding
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                visitor_id,
                timestamp,
                timestamp,
                timestamp,
                embedding_blob,
            ),
        )

        self.connection.commit()

        return visitor_id

    def recognize_or_register(self, embedding):
        """
        Find an existing visitor or register a new one.

        Returns:
            visitor_id, is_new, similarity
        """

        visitor_id, similarity = self.find_match(embedding)

        if visitor_id is not None:
            return visitor_id, False, similarity

        visitor_id = self.register_new_visitor(embedding)

        return visitor_id, True, 0.0

    def update_last_seen(self, visitor_id):
        """Update the last time this visitor was seen."""

        timestamp = datetime.now(timezone.utc).isoformat()

        cursor = self.connection.cursor()

        cursor.execute(
            """
            UPDATE visitors
            SET last_seen_at = ?
            WHERE visitor_id = ?
            """,
            (timestamp, visitor_id),
        )

        self.connection.commit()