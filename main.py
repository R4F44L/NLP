import os
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize NLTK tools
nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

# Path to the folder containing the documents
folder_paths = {
  "business": "business",
  "entertainment": "entertainment",
  "politics": "politics",
  "sport": "sport",
  "tech": "tech"
}
document_texts = None
document_vectors = None
results = []
vectorizer = TfidfVectorizer()


# Function for text processing
def process_text(text):
  text = text.lower()
  tokens = word_tokenize(text)
  lemmas = [lemmatizer.lemmatize(token) for token in tokens]
  return lemmas


# Function for searching documents based on a query
def search_documents(query):
  # Process the query
  query_tokens = process_text(query)
  global document_texts
  global document_vectors

  # Initialize results
  if document_texts is None:
    for folder in folder_paths:
      folder_path = folder_paths[folder]
      for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
          file_path = os.path.join(folder_path, file_name)
          print('file_path', file_path)
          with open(file_path, 'r') as file:
            text = file.read()
            document_tokens = process_text(text)
            # Join tokens into a string to use with TfidfVectorizer
            document_text = " ".join(document_tokens)
            results.append((file_path, document_text))

    # Extract document texts for vectorization
    document_texts = [result[1] for result in results]
    document_vectors = vectorizer.fit_transform(document_texts).toarray()

  # Calculate cosine similarity between the query and documents
  query_vector = vectorizer.transform([query]).toarray()
  similarities = cosine_similarity(query_vector, document_vectors)

  # Sort the results based on similarity
  sorted_results = sorted(zip(results, similarities[0]),
                          key=lambda x: x[1],
                          reverse=True)

  # Return the top 5 results
  top_results = sorted_results[:5]
  return top_results


# Example usage
while True:
  query = input("Enter a query (type 'exit' to quit): ")
  if query == "exit":
    break

  search_results = search_documents(query)

  if search_results:
    print("Top results:")
    for result, similarity in search_results:
      print("Document:", result[0])
      print("Similarity:", similarity)
      print("----------------------")
  else:
    print("No results.")
