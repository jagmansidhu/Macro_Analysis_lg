import os

from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_postgres import PGVector
from langchain_classic.indexes import SQLRecordManager

load_dotenv()

CLOD_API_KEY = os.getenv("CLOD_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DB_URL = os.getenv("DB_CONNECTION_STRING")
MODEL = os.getenv("SIMPLE_AGENT_MODEL")
if not CLOD_API_KEY:
    raise ValueError("CLOD_API_KEY is not set")

llm = ChatOpenAI(
    model=MODEL,
    # temperature=0,
    api_key=CLOD_API_KEY,
    base_url="https://api.clod.io/v1",
)

# embeddings = GoogleGenerativeAIEmbeddings(
#     model="gemini-embedding-2",
#     api_key=GEMINI_API_KEY,
#     output_format=1536,
# )

embeddings = OllamaEmbeddings(
    model="qwen3-embedding:8b",
    dimensions=1024,
)

COLLECTION_NAME = "my_docs_v5"

vector_store = PGVector(
    embeddings=embeddings,
    collection_name=COLLECTION_NAME,
    connection=DB_URL,
    async_mode=True,
)

vector_store_sync = PGVector(
    embeddings=embeddings,
    collection_name=COLLECTION_NAME,
    connection=DB_URL,
    async_mode=False,
)

namespace = f"pgvector/{COLLECTION_NAME}"
record_manager = SQLRecordManager(
    namespace=namespace,
    db_url=DB_URL,
)

record_manager.create_schema()
