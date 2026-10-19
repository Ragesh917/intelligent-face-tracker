import numpy as np
from insightface.app import FaceAnalysis


class FaceRecognizer:
    def __init__(
        self,
        model_name="buffalo_l",
        model_root=r"insightface_models",
        similarity_threshold=0.5,
        device=0,
    ):
        self.similarity_threshold = similarity_threshold

        self.app = FaceAnalysis(
            name=model_name,
            root=model_root,
        )

        self.app.prepare(
            ctx_id=device,
            det_size=(640, 640),
        )

    def get_embedding(self, face_image):
        """
        Generate a face embedding from a cropped face image.
        """

        faces = self.app.get(face_image)

        if not faces:
            return None

        face = max(faces, key=lambda item: item.det_score)

        embedding = face.embedding.astype(np.float32)

        # Normalize embedding
        norm = np.linalg.norm(embedding)

        if norm == 0:
            return None

        embedding = embedding / norm

        return embedding

    def similarity(self, embedding_a, embedding_b):
        """
        Calculate cosine similarity between two face embeddings.
        """

        a = np.asarray(embedding_a, dtype=np.float32)
        b = np.asarray(embedding_b, dtype=np.float32)

        denominator = np.linalg.norm(a) * np.linalg.norm(b)

        if denominator == 0:
            return 0.0

        return float(np.dot(a, b) / denominator)

    def is_same_person(self, embedding_a, embedding_b):
        """
        Determine whether two embeddings belong to the same person.
        """

        score = self.similarity(embedding_a, embedding_b)

        return score >= self.similarity_threshold