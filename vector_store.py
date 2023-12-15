import numpy as np

class VectorStore:
    def __init__(self):
        self.vector_data = {}  # A dictionary to store vectors
        self.vector_index = {}  # An indexing structure for retrieval

    def add_vector(self, vector_id, vector):
        self.vector_data[vector_id] = vector
        self._update_index(vector_id, vector)

    def get_vector(self, vector_id):
        return self.vector_data.get(vector_id)

    def _update_index(self, vector_id, vector):
        for existing_id, existing_vector in self.vector_data.items():
            similarity = self._calculate_cosine_similarity(vector, existing_vector)
            if existing_id not in self.vector_index:
                self.vector_index[existing_id] = {}
            self.vector_index[existing_id][vector_id] = similarity

    def _calculate_cosine_similarity(self, vector1, vector2):
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        if norm1 == 0 or norm2 == 0:
            return 0  # Handle division by zero
        return np.dot(vector1, vector2) / (norm1 * norm2)

    def find_similar_vectors(self, query_vector, num_results=5):
        results = []
        for vector_id, vector in self.vector_data.items():
            similarity = self._calculate_cosine_similarity(query_vector, vector)
            results.append((vector_id, similarity))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:num_results]
