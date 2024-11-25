import os
from dotenv import load_dotenv
import openai
import pinecone
from pinecone import Pinecone, ServerlessSpec
import logging
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi 
import nltk
from nltk.tokenize import sent_tokenize
from pinecone import Pinecone

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
_log = logging.getLogger(__name__)

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")  # Example: "us-west1-gcp"
youtube_index_name = "youtube-index"
DIMENSION = 1536  
METRIC = "cosine"

if not PINECONE_API_KEY or not PINECONE_ENVIRONMENT or not YOUTUBE_API_KEY:
    logging.error("API keys missing in environment variables.")
    exit(1)

openai.api_key = os.getenv("OPENAI_API_KEY")

pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT) 

if youtube_index_name not in pc.list_indexes().names():
    pc.create_index(
        name=youtube_index_name,
        dimension=DIMENSION,
        metric=METRIC,
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )
    logging.info(f"Index '{youtube_index_name}' created successfully.")
else:
    logging.info(f"Index '{youtube_index_name}' already exists.")

youtube_index = pc.Index(youtube_index_name)

nltk.download("punkt")

def get_ada_embedding(text):
    """
    this model is used to generate embeddings of dimension 1536.
    """
    try:
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=text
        )
    except Exception as e:
        _log.error(f"Error generating ADA embedding for text: {e}")
        return None
    
def combined_short_transcripts(transcript, min_length=300):
    """
    combining shorter transcripts into larger sized chinks so that each
    vector holds meaningful info and has atleast min_char length.

      Args:
    transcript (list): The list of transcript entries.
    min_length (int): The minimum length for each segment.

    Returns:
    List[dict]: A list of combined transcript entries.
    """
    consolidated_transcript = []
    current_text = ""
    current_start = None

    for record in transcript:
        if current_start is None:
            
            current_start = record['start']

            current_text += " " + record['text']

        if len(current_text.strip()) >= min_length:
            consolidated_transcript.append({
                "text": current_text.strip(),
                "start": current_start
            })

            current_text = ""
            current_start = None
    if current_text.strip() and len(current_text.strip()) >= min_length:
        consolidated_transcript.append({
            "text": current_text.strip(),
            "start": current_start
        })

    return consolidated_transcript

def chunking_text_with_sliding_window(text, max_chars=300, overlap_chars=100):
    """
    Splits text into overlapping chunks at sentence boundaries, ensuring each chunk
    does not exceed the max_chars limit, with specified overlap to maintain context.
    
    Args:
    text (str): The input text to be chunked.
    max_chars (int): The maximum number of characters per chunk.
    overlap_chars (int): The number of characters to overlap between chunks.
    
    Returns:
    List[str]: A list of text chunks.
    """

    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) > max_chars:
            chunks.append(current_chunk.strip())

            overlap = current_chunk[-overlap_chars:]
            
            current_chunk = overlap if len(overlap) > 0 else ""
        
        current_chunk += " " + sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

def search_videos(query, max_results=3):
    """
    Search for YouTube videos using the Data API.
    """
    try:
        response = youtube.search().list(
            q = query,
            part="snippet",
            type="video",
            maxResults=max_results
        ).execute()
    except Exception as e:
        logging.error(f"Error fetching video data: {e}")
        return []
    
    results = []
    for item in response.get("items", []):
        results.append(
            {
                "title": item["snippet"]["title"],
                "videoId": item["id"]["videoId"],
                "description": item["snippet"]["description"]
            }
        )

        return results
    
def get_transcript(video_id, language="en"):
    """
    Fetch YouTube video transcript using the Transcript API.
    """
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[language])
        return transcript
    except Exception as e:
        logging.error(f"Error fetching transcript: {str(e)}")
        return []
        
def store_transcript_in_pinecone(transcript, video_id):
    """
    Store the transcript into Pinecone with timestamp data.
    """
    consolidated_transcript = combined_short_transcripts(transcript)

    for record in consolidated_transcript:
        text = record["text"]
        embedding = get_ada_embedding(text)

        if embedding is None:
            continue
    
        vector_data = {
            "id": f"{video_id}-{entry['start']}",
            "values": embedding,
            "metadata": {
                "video_id": video_id,
                "timestamp": entry["start"],
                "text": text,
            },
        }

        try:
            youtube_index.upsert([(vector_data["id"], vector_data["values"], vector_data["metadata"])])
            logging.info(f"Transcript entry at {entry['start']}s stored successfully.")
        except Exception as e:
            logging.error(f"Failed to store transcript entry at {entry['start']}s: {e}")

if __name__ == "__main__":
    query = input("Enter search query: ").strip()

    videos = search_videos(query)

    if not videos:
        logging.info("No videos found for the query. Please try again.")
        exit(1)

    logging.info("\nSearch Results:")

    for idx, video in enumerate(videos):
        logging.info(f"{idx + 1}. Title: {video['title']}")
        logging.info(f"   Video ID: {video['videoId']}")
        logging.info(f"   Description: {video['description']}\n")

    while True:
        try:
            selected_index = int(input("Enter the number of the video to fetch transcript: ")) - 1

            if 0 <= selected_index < len(videos):
                video_id = videos[selected_index]["videoId"]
                logging.info(
                    f"\nFetching transcript for video: {videos[selected_index]['title']} ({video_id})\n"
                )

                language = input("Enter language code for transcript (default: 'en'): ").strip() or "en"
                transcript = get_transcript(video_id, language=language)

                if transcript:
                    logging.info("Transcript (first 5 entries):")
                    for entry in transcript[:5]:
                        logging.info(f"Time: {entry['start']}s, Text: {entry['text']}")

                    store_transcript_in_pinecone(transcript, video_id)
                    logging.info("\nTranscript stored successfully in Pinecone.")
                break

            else:
                logging.info("Invalid selection. Please try again.")

        except ValueError:
            logging.info("Invalid input. Please enter a number.")


