import numpy as np
from vector_store import VectorStore

vector_store = VectorStore()

def preprocess_and_store_sentence(sentence):
    tokens = sentence.lower().split()
    vector = np.zeros(len(tokens))
    for token in tokens:
        vector[tokens.index(token)] += 1
    vector_store.add_vector(sentence, vector)

while True:
    user_input = input("Enter a sentence (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    preprocess_and_store_sentence(user_input)

query_sentence = input("Enter a query sentence: ")
query_vector = np.zeros(len(query_sentence.split()))
query_tokens = query_sentence.lower().split()
for token in query_tokens:
    query_vector[query_tokens.index(token)] += 1

num_results = int(input("Enter the number of similar sentences to retrieve: "))
similar_sentences = vector_store.find_similar_vectors(query_vector, num_results=num_results)

print("Query Sentence:", query_sentence)
print("Similar Sentences:")
for sentence, similarity in similar_sentences:
    print(f"{sentence}: Similarity = {similarity:.4f}")
