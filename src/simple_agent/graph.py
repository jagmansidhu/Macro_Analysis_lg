"""Minimal LangChain agent graph for deployment."""

from __future__ import annotations

import os

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState

model = ChatOpenAI(
    model=os.environ.get("SIMPLE_AGENT_MODEL"),
    temperature=0,
    base_url="https://api.clod.io/v1",
    api_key=os.environ.get("CLOD_API_KEY"),
)

# currently no config done
builder = StateGraph(MessagesState)

graph = builder.compile()