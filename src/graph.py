from __future__ import annotations

from langgraph.graph import StateGraph, MessagesState, START

from retreival_agent import agent

builder = StateGraph(MessagesState)

builder.add_node("llm", agent)
builder.add_edge(START, "llm")

graph = builder.compile()