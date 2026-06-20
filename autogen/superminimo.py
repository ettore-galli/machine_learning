from autogen import AssistantAgent, UserProxyAgent

assistant = AssistantAgent(
    name="assistant",
    llm_config={
        "model": "gpt-4o-mini",
        "api_key": "not-needed",
        "base_url": "http://localhost:9876/v1"
    },
    system_message="Sei un assistente conciso."
)

user = UserProxyAgent(
    name="user",
    human_input_mode="NEVER",
    code_execution_config={"use_docker": False}   # <-- fondamentale
)

reply = user.initiate_chat(
    assistant,
    message="Ciao, chi sei?"
)

print(reply)
