import os
import logging
from pinecone import Pinecone, Index, ServerlessSpec
from nltk.tokenize import sent_tokenize
import nltk
from dotenv import load_dotenv
import re
import openai
import time
import random

# Download and specify the nltk data directory
nltk.download('punkt', download_dir='/Users/nishitamatlani/nltk_data')
nltk.data.path.append('/Users/nishitamatlani/nltk_data')

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)

# Load credentials from .env
PINECONE_API_KEY = "pcsk_6K3Vu8_56paG1Doi1xgV1FKmXS8FfYJbe1p7HAp92c6QnV4pMCEhJ32otRXqvryzNzGdsQ"
PINECONE_ENVIRONMENT = "vo2w95e.svc.aped-4627-b74a.pinecone.io"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Ensure your .env file contains this key
TEXT_INDEX_NAME = "test"  # Name of the Pinecone index
DIMENSION = 1536  # ADA embedding dimension
METRIC = "cosine"

# Set OpenAI API key
openai.api_key = OPENAI_API_KEY

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create Pinecone index if it does not exist
if TEXT_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=TEXT_INDEX_NAME,
        dimension=DIMENSION,
        metric=METRIC,
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )

# Connect to Pinecone index
text_index = pc.Index(TEXT_INDEX_NAME)

# Check if the index is accessible
try:
    _log.info("Connected to text index: %s", text_index.describe_index_stats())
except Exception as e:
    _log.error(f"Failed to connect to Pinecone index: {e}")

# Function to clean text before embedding
def clean_text(text):
    """Cleans the input text by removing unwanted characters and formatting."""
    try:
        text = text.replace('\n', ' ')  # Replace newlines with a space
        text = re.sub(r'[^a-zA-Z0-9\s.,]', '', text)  # Remove non-alphanumeric characters
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
        return text
    except Exception as e:
        _log.error(f"Error cleaning text: {e}")
        return text

# Function to generate embeddings using OpenAI's ada model
def get_ada_embedding(text):
    """Generates a 1536-dimensional embedding using OpenAI's text-embedding-ada-002 model."""
    try:
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=text
        )
        embedding = response["data"][0]["embedding"]
        return embedding
    except Exception as e:
        _log.error(f"Error generating ADA embedding for text: {e}")
        return None

def chunk_text(text, max_chars=300, overlap_sentences=1):
    """
    Splits text into overlapping chunks at sentence boundaries, ensuring each chunk
    does not exceed the max_chars limit and does not split a sentence in half. 
    Overlapping is maintained by including a certain number of sentences from the
    previous chunk in the next chunk.

    Args:
    text (str): The input text to be chunked.
    max_chars (int): The maximum number of characters per chunk.
    overlap_sentences (int): The number of sentences to overlap between chunks.

    Returns:
    List[str]: A list of text chunks.
    """
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_chunk_char_count = 0

    i = 0
    while i < len(sentences):
        sentence = sentences[i]
        sentence_char_count = len(sentence)

        # If adding the current sentence exceeds the limit, finalize the current chunk
        if current_chunk_char_count + sentence_char_count > max_chars:
            # Finalize the current chunk without splitting the sentence
            chunks.append(" ".join(current_chunk).strip())

            # Start a new chunk with overlap
            current_chunk = current_chunk[-overlap_sentences:]
            current_chunk_char_count = sum(len(s) for s in current_chunk)
        
        # Add the current sentence to the current chunk
        current_chunk.append(sentence)
        current_chunk_char_count += sentence_char_count
        i += 1

    # Add any remaining text as the final chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk).strip())

    return chunks

# Function to upload embeddings to Pinecone with retry logic
def upload_to_pinecone_with_retry(embeddings, index, batch_size=10, max_retries=3):
    """Uploads embeddings to Pinecone with retry logic."""
    for i in range(0, len(embeddings), batch_size):
        batch = embeddings[i:i + batch_size]
        retries = 0
        while retries < max_retries:
            try:
                index.upsert(vectors=[{
                    "id": entry["id"],
                    "values": entry["embedding"],
                    "metadata": entry["metadata"]
                } for entry in batch])
                _log.info(f"Uploaded batch {i // batch_size + 1} to Pinecone.")
                break
            except Exception as e:
                retries += 1
                time.sleep(2 + random.uniform(0, 1))  # Exponential backoff
                _log.error(f"Retry {retries}/{max_retries} for batch {i // batch_size + 1}: {e}")
        else:
            _log.error(f"Failed to upload batch {i // batch_size + 1} after {max_retries} retries.")

# Function to process and embed text from a local file
def process_text_file(file_path):
    """Reads a text file, cleans its content, chunks it, generates embeddings, and uploads them to Pinecone."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            full_text = file.read()

        # Clean the text
        cleaned_text = clean_text(full_text)

        # Chunk the text
        chunks = chunk_text(cleaned_text, max_chars=300, overlap_sentences=1)
        embeddings = []

        for idx, chunk in enumerate(chunks):
            embedding = get_ada_embedding(chunk)
            if embedding:
                embeddings.append({
                    "id": f"text-chunk-{idx}",
                    "embedding": embedding,
                    "metadata": {"chunk_id": idx, "text": chunk}
                })
                _log.info(f"Processed chunk {idx}: {chunk[:30]}...")

        upload_to_pinecone_with_retry(embeddings, text_index)

        _log.info(f"Successfully processed {len(chunks)} chunks from {file_path}.")
    except Exception as e:
        _log.error(f"Error processing file {file_path}: {e}")

# Main function
def main():
    """Main function to process a text file."""
    text_file_path = "/Users/nishitamatlani/Documents/final_project/web_scraping/scraped_data.txt"  # Replace with your file path
    process_text_file(text_file_path)

if __name__ == "__main__":
    main()
