from ai_agent.agent_proxy import get_model_response
from ai_agent.base import AIAgentBase, ResponseDisplayer


class AIAgent(AIAgentBase):

    def generate(
        self,
        user_prompt: str,
        system_prompt: str,
        user_prompt_marker: str,
        response_displayer: ResponseDisplayer,
    ) -> str:
        return get_model_response(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            user_prompt_marker=user_prompt_marker,
            response_displayer=response_displayer,
        )
