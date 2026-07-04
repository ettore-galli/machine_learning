from pytest import mark

from ai_agent.tool_call_middleware import ToolCallMiddleware


@mark.parametrize(
    "message_content, expected_tool_call",
    [
        (
            '<xz<zx<x{"name": "calculate_average", "arguments": {"values": [4, 5]}}asdsaas',
            None,
        ),
        (
            '<xz<zx<x{"tool":{"name": "calculate_average", "arguments": {"values": [4, 5]}}}asdsaas',
            '{"tool":{"name": "calculate_average", "arguments": {"values": [4, 5]}}}',
        ),
    ],
)
def test_get_tool_request_from_message(message_content, expected_tool_call):
    middleware = ToolCallMiddleware()
    assert (
        middleware.get_tool_request_from_message(message_content=message_content)
        == expected_tool_call
    )
