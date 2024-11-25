import streamlit as st
from dotenv import load_dotenv
import os
from pinecone import Pinecone, ServerlessSpec
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import tiktoken

load_dotenv()

#laod API keys
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY_TEST")

MAX_CONTEXT_TOKENS = 3000
EMBEDDING_MODEL = "text-embedding-ada-002"

client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

# Splitting text into smaller chunks for easier processing and managing rate limits
def text_chunk(text, max_tokens=8192):
    words = text.split()
    return[" ".join(words[i:i + max_tokens]) for i in range(0, len(words), max_tokens)]


# Youtube videos fetched from summarized text generated using user query and rag output
def get_youtube_videos(search_query, max_results=3):
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

    search_response = youtube.search().list(
        q=search_query,
        part="snippet",
        maxResults=max_results,
        type="video"
    ).execute()

    videos = []
    for item in search_response.get("items", []):
        video_id = item["id"]["videoId"]
        title = item["snippet"]["title"]
        description = item["snippet"]["description"]
        url = f"https://www.youtube.com/watch?v={video_id}"
        videos.append({"id": video_id, "title":title, "description":description, "url":url })
    return videos

# Fetching transcript of a video
def get_transcript(video_id):
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    combined_text = " ".join([item['text'] for item in transcript])
    return combined_text
    
# Calculating relevance between transcripts and context (from rag)
def calculate_relevance(context, transcripts):
    context_embedding = get_embedding(context, model="text-embedding-ada-002")

    relevance_scores = []
    for video_url, transcript in transcripts.items():
        transcript_embedding = get_embedding(transcript)
        score = cosine_similarity([context_embedding], [transcript_embedding])[0][0]
        relevance_scores.append((video_url, score))

    relevance_scores.sort(key=lambda x: x[1], reverse=True)
    
    if not relevance_scores:
        return None, None  # Handle case where no scores are available
    return relevance_scores[0]  # Return the most relevant video ID and score


# Querying LLM and generating outputs based on rag and youtube data stored in vector database
def chatgpt(prompt):
    try:
        query_embedding = get_embedding(prompt, model="text-embedding-ada-002")

        results = index.query(vector=query_embedding, top_k=5, include_metadata=True)

        if not results["matches"]:
            return "I'm sorry, I couldn't find any relevant information in the knowledge base."
        
        # Extract relevant context from the search results
        context = " ".join([
            f"Chunk ID: {match['metadata'].get('chunk_id', 'Unknown')} Text: {match['metadata'].get('text', '').strip()}"
            for match in results["matches"] if match['metadata'].get('text')
        ])

        simplified_query = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an assistant that simplifies complex context and queries into concise search-friendly text."},
                {"role": "user", "content": f"Context: {context}. User Query: {prompt}. Simplify this for a YouTube search."}
            ]
        ).choices[0].message.content
        youtube_videos = get_youtube_videos(simplified_query, max_results=3)

        transcripts = {}
        for video in youtube_videos:
            transcript = get_transcript(video["id"])
            if "Failed" not in transcript:  # Skip invalid transcripts
                transcripts[video["url"]] = transcript

        relevant_video, score = calculate_relevance(context,transcripts)
        if not relevant_video:
            return {
                "response": "Relevant context found, but no suitable YouTube video identified.",
                "video_url": None
            }
        selected_video = transcripts[relevant_video]


        updated_prompt = f"Relevant context: {context}. User Query:{prompt}. Relevant youtube video: {selected_video}"

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role":"user", "content":"You are an assistant that answers questions strictly based on the provided context. If the context does not contain relevant information, you must reply, 'I cannot find relevant information in the knowledge base.'"},
                {"role":"user", "content":updated_prompt}
            ]
        )
        relevant_video_url = next(video["url"] for video in youtube_videos if video["url"] == relevant_video or video["id"] in relevant_video)
        return {
            "response": response.choices[0].message.content,
            "video_url": relevant_video_url
        }
    except Exception as e:
        return f"Error occurred in generating response: {e}"
    
  
def get_embedding(text, model="text-embedding-ada-002"):
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    MAX_TOKENS = 8192

    if len(tokens) > MAX_TOKENS:
        text = encoding.decode(tokens[:MAX_TOKENS])

    response = client.embeddings.create(input=text,model=model)
    embedding=response.data[0].embedding
    return embedding

INDEX_KNOWLEDGE_BASE="test"
if INDEX_KNOWLEDGE_BASE not in [index["name"] for index in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_KNOWLEDGE_BASE,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud='aws', 
            region='us-east-1'
        )
    )    
index = pc.Index(INDEX_KNOWLEDGE_BASE)


if __name__ == "__main__":
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("assistant_response", None)
    st.session_state.setdefault("youtube_transcript", None)

    query = st.text_input("What do you want to learn about today?")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        result = chatgpt(query)

        if isinstance(result, dict):
            st.session_state.assistant_response = result.get("response", "No response found")
            st.markdown(f"**Assistant:** {st.session_state.assistant_response}")
            if result.get("video_url"):
                st.markdown(f"[Watch Relevant YouTube Video]({result['video_url']})")

                video_id = result["video_url"].split("v=")[-1]
                transcript = get_transcript(video_id)
                if transcript:
                    st.session_state.youtube_transcript = transcript
                    for content, content_type in [
                        (result["response"], "RAG Answer"),
                        (transcript, "YouTube Transcript")
                    ]:
                        embedding = get_embedding(content)
                        index.upsert(
                            vectors=[(f"{content_type}-{query}", embedding, {"type": content_type, "query": query})]
                        )
        else:
            st.session_state.assistant_response = result

    if st.session_state.assistant_response or st.session_state.youtube_transcript:
        st.markdown("---")
        st.markdown("### Query the Combined Context")

        follow_up_query = st.text_input("Ask a follow-up question:")
        if follow_up_query:
            follow_up_embedding = get_embedding(follow_up_query)
            search_results = index.query(
                vector=follow_up_embedding, top_k=3, include_metadata=True
            )

            combined_context = " ".join([
                result["metadata"].get("text", "")
                for result in search_results.get("matches", [])
            ])[:MAX_CONTEXT_TOKENS]

            follow_up_response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Answer strictly based on the provided context."},
                    {"role": "assistant", "content": combined_context},
                    {"role": "user", "content": follow_up_query}
                ]
            ).choices[0].message.content

            st.markdown(f"**Assistant:** {follow_up_response}")
