import os

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
    api_key=CLOD_API_KEY,
    base_url="https://api.clod.io/v1",
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2",
    api_key=GEMINI_API_KEY,
    output_format=1536,
)

COLLECTION_NAME = "my_docs_v4"

vector_store = PGVector(
    embeddings=embeddings,
    collection_name=COLLECTION_NAME,
    connection=DB_URL,
)

namespace = f"pgvector/{COLLECTION_NAME}"
record_manager = SQLRecordManager(
    namespace=namespace,
    db_url=DB_URL,
)

record_manager.create_schema()
