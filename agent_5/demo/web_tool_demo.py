from ai_agent.tools import create_web_search_tool

if __name__ == "__main__":
    web_search_tool = create_web_search_tool()
    response = web_search_tool.run("What are Eddy currents?")
    print(response)
