from diagrams import Diagram, Cluster
from diagrams.onprem.workflow import Airflow
from diagrams.onprem.analytics import Spark
from diagrams.generic.storage import Storage
from diagrams.onprem.mlops import Mlflow
from diagrams.generic.database import SQL
from diagrams.generic.compute import Rack
from diagrams.custom import Custom
from diagrams.aws.storage import S3

with Diagram("End-to-End Research Tool Architecture", show=False, filename="research_tool_architecture", direction="LR", graph_attr={"ranksep": "1.5", "nodesep": "1.0"}):
    # Combined Pipeline Cluster - Data Acquisition and Preprocessing
    with Cluster("Airflow Pipeline - Data Acquisition and Preprocessing"):
        # Original data acquisition pipeline
        s3_input = S3("Amazon S3 \n (PDFs)")
        airflow = Airflow("Airflow Scheduler")  # Orchestrates the pipeline
        docling = Custom("Docling Document Parser", "/Users/nishitamatlani/Documents/final_project/diagram/images/docling.png")  # Parses the input documents
        openai_embeddings = Custom("OpenAI Embeddings", "/Users/nishitamatlani/Documents/final_project/diagram/images/openai.png")  # Generates embeddings for the parsed content
        pinecone = Custom("Pinecone Vector Database", "/Users/nishitamatlani/Documents/final_project/diagram/images/pinecone1.png")  # Stores the vector embeddings for efficient querying
        
        # Data flow through the original pipeline
        airflow >> s3_input >> docling >> openai_embeddings >> pinecone

        # New data acquisition pipeline
        airflow1 = Airflow("Airflow Scheduler")  # Orchestrates the pipeline
        selenium = Custom("Selenium (Python)", "/Users/nishitamatlani/Documents/final_project/diagram/images/python.png")  # Web scraping with Selenium
        s3_web_data = S3("Amazon S3 \n (Web Data)")  # Stores web-scraped data
        
        # Data flow through the new pipeline
        airflow1 >> selenium >> s3_web_data >> pinecone
    
    # FastAPI Cluster
    with Cluster("API Management and Authentication"):
        fastapi = Custom("FastAPI Interface", "/Users/nishitamatlani/Documents/final_project/diagram/images/fastapi.png")  # Acts as the API interface for interaction
        jwt_auth = Custom("JWT Authentication", "/Users/nishitamatlani/Documents/final_project/diagram/images/jwt.png")  # JWT authentication for FastAPI access
        jwt_auth >> fastapi  # JWT Authentication for FastAPI access

    # Snowflake Interaction
    snowflake = Custom("Snowflake Database", "/Users/nishitamatlani/Documents/final_project/diagram/images/snowflake.png")  # Snowflake for storing user information
    fastapi >> snowflake  # FastAPI interacts with Snowflake for storing/retrieving user information
    fastapi << snowflake

    # Pinecone Interactions
    pinecone >> fastapi  # Pinecone provides data to FastAPI
    fastapi >> pinecone  # FastAPI interacts with Pinecone for querying
    
    # Langraph Interaction
    langraph = Custom("Langraph Multi-Agent System", "/Users/nishitamatlani/Documents/final_project/diagram/images/langraph.png")
    fastapi >> langraph  # FastAPI interacts with Langraph for multi-agent processing
    fastapi << langraph  # Langraph also sends data back to FastAPI


    # Agent Cluster
    with Cluster("Multi-Agent System - Research Agents"):
        youtube_agent = Custom("YouTube Agent", "/Users/nishitamatlani/Documents/final_project/diagram/images/youtube.png")
        web_search_agent = Custom("Web Search Agent", "/Users/nishitamatlani/Documents/final_project/diagram/images/Web_Search.png")
        rag_agent = Custom("RAG Agent", "/Users/nishitamatlani/Documents/final_project/diagram/images/Document_Search.png")
        
        # Langraph interacts with the research agents
        langraph >> [youtube_agent, web_search_agent, rag_agent]

    # User Interface Cluster
    with Cluster("User Interaction Interface - Frontend"):
        streamlit = Custom("Copilot User Interface", "/Users/nishitamatlani/Documents/final_project/diagram/images/streamlit.png")  # User interacts with the system via Copilot UI
        streamlit >> fastapi  # Copilot sends user requests to FastAPI
        streamlit << fastapi
