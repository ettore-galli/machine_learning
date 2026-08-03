from dotenv import load_dotenv
from ai_agent.agent_proxy import get_model_response

load_dotenv()

import ui.main as main  # noqa: E402 F401

main.main()
