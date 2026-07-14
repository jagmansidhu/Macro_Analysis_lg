from __future__ import annotations
from langgraph.graph import StateGraph, MessagesState, START, END

from retreival_agent import retrieval_agent

builder = StateGraph(MessagesState)

builder.add_node("retrieval_worker", retrieval_agent)

builder.add_edge(START, "retrieval_worker")
builder.add_edge("retrieval_worker", END)

main_graph = builder.compile()

if __name__ == "__main__":
    from langchain_core.messages import HumanMessage

    print("Testing the macro-economic retrieval pipeline...\n")

    inputs = {"messages": [HumanMessage(content="What was the UNRATE in 2020?")]}

    for event in main_graph.stream(inputs, stream_mode="values"):
        last_message = event["messages"][-1]
        last_message.pretty_print()