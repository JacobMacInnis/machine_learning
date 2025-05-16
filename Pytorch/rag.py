from transformers import T5Tokenizer, T5ForConditionalGeneration
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch

# 1. Document corpus
documents = [
    "The Eiffel Tower is located in Paris.",
    "Mount Everest is the highest mountain in the world.",
    "Python is a popular programming language.",
    "The capital of France is Paris.",
    "Pandas is a library for data analysis in Python."
]

# 2. Question
query = "Where is the Eiffel Tower?"

# 3. TF-IDF Retrieval
vectorizer = TfidfVectorizer()
doc_vectors = vectorizer.fit_transform(documents)
query_vector = vectorizer.transform([query])
similarities = cosine_similarity(query_vector, doc_vectors).flatten()
top_index = similarities.argmax()
retrieved_context = documents[top_index]

# 4. Load pretrained T5 model (PyTorch backend)
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# 5. Create input for T5
input_text = f"question: {query}  context: {retrieved_context}"
inputs = tokenizer(input_text, return_tensors="pt")

# 6. Generate answer
with torch.no_grad():
    output = model.generate(**inputs, max_length=50)
    answer = tokenizer.decode(output[0], skip_special_tokens=True)

print(f"📚 Retrieved context:\n{retrieved_context}\n")
print(f"🤖 Generated answer:\n{answer}")
