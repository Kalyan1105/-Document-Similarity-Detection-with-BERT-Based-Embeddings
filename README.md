# -Document-Similarity-Detection-with-BERT-Based-Embeddings
Document Similarity Analysis with BERT
This project implements a document similarity analysis pipeline using BERT embeddings to identify duplicate or highly similar documents in a dataset. It leverages natural language processing (NLP) techniques and deep learning to preprocess text, compute embeddings, and analyze similarities.
Overview
The code performs the following tasks:

Text Preprocessing: Cleans and lemmatizes text using NLTK to standardize input documents.
BERT Embeddings: Uses a pre-trained BERT model to generate embeddings for documents.
Similarity Computation: Calculates cosine similarity between document embeddings to identify similar documents.
Duplicate Detection: Flags document pairs with similarity scores above specified thresholds (0.85 and 0.95).
Visualization: Generates a heatmap, histogram, and network graph to visualize similarity patterns.
Output: Saves results to CSV files (High_duplicate_documents.csv and Moderate_duplicate_document.csv) and provides download links in Google Colab.

Requirements
To run this code, ensure you have the following dependencies installed:

Python 3.7+
PyTorch
Transformers (Hugging Face)
Scikit-learn
Pandas
NumPy
NLTK
Seaborn
Matplotlib
NetworkX
Google Colab (optional, for file download functionality)

You can install the required packages using:
pip install torch transformers sklearn pandas numpy nltk seaborn matplotlib networkx

Additionally, download the required NLTK resources by running the following in your Python environment:
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')

Input
The code expects a CSV file containing the documents to analyze. The file should have a column with text data (default column name: text). Update the file_path and text_column variables in the script to match your dataset:
file_path = '/content/Documents_Dataset.csv'  # Path to your CSV file
text_column = 'text'  # Column name containing the text

Usage

Prepare the Dataset: Ensure your CSV file is accessible (e.g., uploaded to Google Colab or available locally).
Run the Script: Execute the Python script in a compatible environment (e.g., Google Colab, Jupyter Notebook, or local Python environment).
Output: The script will:
Save two CSV files:
High_duplicate_documents.csv: Document pairs with similarity > 0.95.
Moderate_duplicate_document.csv: Document pairs with similarity > 0.85.


Generate visualizations:
Heatmap of document similarities.
Histogram of similarity scores.
Network graph of high-similarity document pairs.


Provide download links for the CSV files in Google Colab.



Key Parameters

Similarity Thresholds:
similarity_threshold = 0.85: For moderate similarity pairs.
high_similarity_threshold = 0.95: For high similarity (near-duplicate) pairs.


Batch Size: batch_size = 16 for processing documents in batches to optimize memory usage.
BERT Model: Uses bert-base-uncased for tokenization and embeddings.

Hardware Considerations

The code supports GPU acceleration if available (via CUDA). It automatically detects and uses a GPU if present; otherwise, it falls back to CPU.
Processing large datasets may require significant memory and computational resources, especially when computing BERT embeddings.

Visualizations
The script generates three visualizations:

Heatmap: Displays the similarity matrix for all document pairs.
Histogram: Shows the distribution of pairwise similarity scores with threshold lines.
Network Graph: Visualizes high-similarity document pairs as a graph, with nodes representing documents and edges indicating similarity.

Output Files

High_duplicate_documents.csv: Contains columns Index1, Index2, and Similarity for document pairs with similarity > 0.95.
Moderate_duplicate_document.csv: Contains the same columns for document pairs with similarity > 0.85.

Notes

The script filters out documents with fewer than 5 words or missing text to ensure meaningful analysis.
If running outside Google Colab, the file download functionality (files.download) will not work, but the CSV files will still be saved to the current directory.
Adjust the batch_size parameter based on your hardware to balance speed and memory usage.

License
This project is licensed under the MIT License. See the LICENSE file for details.
Acknowledgments

Built using the Hugging Face Transformers library for BERT.
Utilizes NLTK for text preprocessing and NetworkX for graph visualization.

