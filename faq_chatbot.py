from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# FAQ Dataset
faqs = {
    "What is Python?": "Python is a programming language.",
    "What is NLP?": "NLP stands for Natural Language Processing.",
    "What is machine learning?": "Machine learning enables computers to learn from data.",
    "How do I install Python?": "Download Python from python.org and install it."
}

questions = list(faqs.keys())

# Convert questions to vectors
vectorizer = TfidfVectorizer()
faq_vectors = vectorizer.fit_transform(questions)

print("FAQ Chatbot (type 'exit' to quit)\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Bot: Goodbye!")
        break

    user_vector = vectorizer.transform([user_input])

    similarity = cosine_similarity(user_vector, faq_vectors)

    best_match = similarity.argmax()

    print("Bot:", faqs[questions[best_match]])