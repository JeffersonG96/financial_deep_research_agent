"""Untility  fucntions for agent operations"""
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

def stream_agent_response(agent, query, thread_id = "default", user_id = None):

    config = {'configurable': {'thread_id':thread_id, "user_id": user_id}, 'recursion_limit':100}
    state = {"messages":[HumanMessage(query)], "user_id": user_id, "thread_id": thread_id}

    for chunk in agent.stream(state, stream_mode="messages",config=config):
        messages = chunk[0] if isinstance(chunk, tuple) else chunk

        #handle ai messages with tool calls
        if isinstance(messages, AIMessage) and messages.tool_calls:
             for tool_call in messages.tool_calls:
                 print(f"\n Tool called: {tool_call['name']}")
                 print(f"\n Args: {tool_call['args']}")
                 print("\n")
        elif isinstance(messages, ToolMessage):
            print(f"\n Tool name: (length: {messages.name})")
            print(f"\n Tool result (length: {len(messages.text)})")
            print()       
        
        elif isinstance(messages, AIMessage) and messages.text:
            print(messages.text, end="", flush=True)