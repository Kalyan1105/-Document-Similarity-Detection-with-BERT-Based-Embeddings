import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
from google.colab import files

# Download NLTK resources (only needed once)
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Load pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Check for GPU usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  # Move model to GPU if available

# Function to get part of speech tag for lemmatization
def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

# Preprocessing function with lemmatization
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # Remove special characters and digits
    tokens = [word for word in text.split() if word not in stop_words]
    lemmatized_text = ' '.join([lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in tokens])
    return lemmatized_text

# Function to get BERT embeddings using the [CLS] token
def get_bert_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    inputs = {key: val.to(device) for key, val in inputs.items()}  # Move inputs to GPU if available
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[0][0].cpu().numpy()  # Move to CPU after inference

# Function to read and preprocess documents from a CSV file
def read_documents_from_csv(file_path, text_column):
    try:
        df = pd.read_csv(file_path)
        if text_column not in df.columns:
            raise ValueError(f"The column '{text_column}' was not found in the CSV file.")

        df = df.dropna(subset=[text_column])  # Drop rows with missing text
        df = df[df[text_column].str.split().str.len() > 5]  # Filter out very short documents

        if df.empty:
            print("No valid documents found after filtering.")
            return []

        # Preprocess text data
        df[text_column] = df[text_column].apply(preprocess_text)
        return df[text_column].tolist()
    except Exception as e:
        print(f"Error reading file: {e}")
        return []

# Function to process documents in batches and get embeddings
def process_documents_in_batches(documents, batch_size=16):
    embeddings = []
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i + batch_size]
        inputs = tokenizer(batch_docs, return_tensors='pt', truncation=True, padding=True, max_length=512)
        inputs = {key: val.to(device) for key, val in inputs.items()}  # Move inputs to GPU if available
        with torch.no_grad():
            outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # Move to CPU after inference
        embeddings.extend(batch_embeddings)
    return np.array(embeddings)

# Read documents from CSV
file_path = '/content/Documents_Dataset.csv'  # Replace with your actual path
text_column = 'text'

documents = read_documents_from_csv(file_path, text_column)
if not documents:
    print("No documents found.")
else:
    print(f"Loaded {len(documents)} documents after preprocessing.")

    # Compute embeddings for each document using batch processing
    embeddings = process_documents_in_batches(documents, batch_size=16)

    # Compute cosine similarity between document embeddings
    similarities = cosine_similarity(embeddings)

    # Set similarity thresholds and find duplicates
    similarity_threshold = 0.85
    high_similarity_threshold = 0.95
    duplicates = []
    high_similarity_duplicates = []

    for i in range(len(documents)):
        for j in range(i + 1, len(documents)):
            if similarities[i][j] > high_similarity_threshold:
                high_similarity_duplicates.append((i, j, similarities[i][j]))
            elif similarities[i][j] > similarity_threshold:
                    duplicates.append((i, j, similarities[i][j]))

    # Save high similarity duplicates to 'High_duplicate_documents.csv'
    high_similarity_df = pd.DataFrame(high_similarity_duplicates, columns=["Index1", "Index2", "Similarity"])
    high_similarity_df.to_csv('High_duplicate_documents.csv', index=False)
    print("\nHigh similarity duplicates (similarity > 0.95) saved to 'High_duplicate_documents.csv'.")

    # Save moderate similarity duplicates to 'Moderate_duplicate_document.csv'
    results_df = pd.DataFrame(duplicates, columns=["Index1", "Index2", "Similarity"])
    results_df.to_csv('Moderate_duplicate_document.csv', index=False)
    print("\nModerate duplicates (similarity > 0.85) saved to 'Moderate_duplicate_document.csv'.")

    # Provide download link for Colab
    try:
        files.download('High_duplicate_documents.csv')
        files.download('Moderate_duplicate_document.csv')
        print("\nDownload started for 'High_duplicate_documents.csv' and 'Moderate_duplicate_document.csv'.")
    except ImportError:
        print("Running locally, files saved to the current directory.")

    # Display the number of duplicate document pairs
    print(f"\nTotal number of document pairs with similarity > {similarity_threshold}: {len(duplicates)}")
    print(f"Total number of document pairs with similarity > {high_similarity_threshold}: {len(high_similarity_duplicates)}")

# Create a heatmap of the similarity matrix
plt.figure(figsize=(10, 10))
sns.heatmap(similarities, cmap='viridis', annot=False)
plt.title("Heatmap of Document Similarities")
plt.xlabel("Document Index")
plt.ylabel("Document Index")
plt.show()

# Flatten the similarity matrix and remove diagonal elements (self-similarities)
similarity_values = similarities[np.triu_indices(len(similarities), k=1)]

# Plot a histogram of similarity scores
plt.figure(figsize=(10, 6))
plt.hist(similarity_values, bins=50, color='skyblue', edgecolor='black')
plt.axvline(x=similarity_threshold, color='orange', linestyle='--', label=f"Threshold = {similarity_threshold}")
plt.axvline(x=high_similarity_threshold, color='red', linestyle='--', label=f"High Threshold = {high_similarity_threshold}")
plt.title("Distribution of Pairwise Document Similarities")
plt.xlabel("Similarity Score")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# Create a graph for high-similarity duplicates
graph = nx.Graph()
for index1, index2, sim in high_similarity_duplicates:
    graph.add_edge(index1, index2, weight=sim)

# Draw the graph
plt.figure(figsize=(12, 10))
pos = nx.spring_layout(graph, seed=42)
nx.draw(graph, pos, with_labels=True, node_color='skyblue', node_size=700, font_size=10)
labels = nx.get_edge_attributes(graph, 'weight')
nx.draw_networkx_edge_labels(graph, pos, edge_labels={k: f"{v:.2f}" for k, v in labels.items()}, font_size=8)
plt.title("High-Similarity Document Network")
plt.show()