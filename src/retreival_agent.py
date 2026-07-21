import asyncio

from config import llm, vector_store
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_core.tools.retriever import create_retriever_tool

retriever = vector_store.as_retriever(search_kwargs={"k": 20})

retriever_tool = create_retriever_tool(
    retriever,
    name="search_fred_macro_data",
    description=(
        "Searches for macroeconomic data from FRED CSV files using semantic similarity. "
        "Good for general questions like 'What happened to CPI in 2020?' or 'Show me fed funds rate trends'. "
        "NOT reliable for 'most recent' or 'latest' queries — use get_latest_data for those."
    )
)


@tool
async def get_latest_data(metric: str, n: int = 5) -> str:
    """Get the N most recent data points for a specific FRED metric, sorted by date descending.

    Use this tool when the user asks for the 'most recent', 'latest', or 'current' value of an indicator.

    Args:
        metric: The metric name to search for (e.g. 'CPILFESL', 'DFF', 'FEDFUNDS', 'DGS2').
                Available metrics: CPILFESL (Core CPI), DFF (Daily Fed Funds Rate),
                FEDFUNDS (Monthly Fed Funds Rate), DGS2 (2-Year Treasury Yield).
        n: Number of most recent data points to return (default 5).
    """
    # 2. Keep using the ASYNC store here because this is an async def tool
    results = await vector_store.asimilarity_search(
        query=f"most recent {metric} data",
        k=100,
        filter={"metric": metric},
    )

    # Fallback: broader search filtered in Python (handles stored names like "CPILFESL (1)")
    if not results:
        results = await vector_store.asimilarity_search(
            query=f"most recent {metric} data",
            k=200,
        )
        results = [doc for doc in results if metric in doc.metadata.get("metric", "")]

    if not results:
        return f"No data found for metric '{metric}'."

    # Sort by date descending and take the top N
    sorted_results = sorted(results, key=lambda d: d.metadata.get("date", ""), reverse=True)
    top_n = sorted_results[:n]

    lines = [doc.page_content for doc in top_n]
    return "\n".join(lines)


system_prompt = (
    "You are an assistant for macroeconomic analysis tasks. "
    "You have access to FRED economic data including: "
    "CPILFESL (Core CPI), DFF (Daily Fed Funds Rate), FEDFUNDS (Monthly Fed Funds Rate), DGS2 (2-Year Treasury Yield). "
    "\n\n"
    "TOOL SELECTION RULES:\n"
    "- For 'most recent', 'latest', or 'current' data: ALWAYS use `get_latest_data` with the correct metric name.\n"
    "- For historical questions, trends, or comparisons: use `search_fred_macro_data`.\n"
    "\n"
    "Always cite the exact date and value from the retrieved data. "
    "If the retrieved context does not contain relevant information, say that you don't know."
)

retrieval_agent = create_agent(
    model=llm,
    tools=[retriever_tool, get_latest_data],
    system_prompt=system_prompt
)

async def main():
    inputs = {"messages": [HumanMessage(content="What is the most recent CPI data?")]}

    async for chunk in retrieval_agent.astream(inputs, stream_mode="values"):
        last_message = chunk["messages"][-1]
        last_message.pretty_print()

if __name__ == "__main__":
    asyncio.run(main())