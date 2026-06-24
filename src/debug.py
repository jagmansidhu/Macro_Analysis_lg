import asyncio
from config import vector_store

async def debug_db():
    # Use the async version: asimilarity_search
    results = await vector_store.asimilarity_search("CPILFESL July 2023", k=3)

    for i, doc in enumerate(results):
        print(f"--- Chunk {i+1} ---")
        print(f"Content: {doc.page_content}")
        print(f"Metadata: {doc.metadata}\n")
