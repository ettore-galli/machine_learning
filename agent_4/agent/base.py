from typing import Any, Dict, List, TypedDict

AgentStateMessagesType = List[Dict[str, Any]]


class AgentState(TypedDict):
    messages: AgentStateMessagesType
