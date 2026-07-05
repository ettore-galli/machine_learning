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


@mark.parametrize(
    "message_content, regexp, expected_tool_call",
    [
        (
            '<xz<zx<x{"name": "calculate_average", "arguments": {"values": [4, 5]}}asdsaas',
            r'{"tool":{"name":.*}',
            None,
        ),
        (
            '<xz<zx<x{"tool":{"name": "calculate_average", "arguments": {"values": [4, 5]}}}asdsaas',
            r'{"tool":{"name":.*}',
            '{"tool":{"name": "calculate_average", "arguments": {"values": [4, 5]}}}',
        ),
        (
            """<tool_call>
{"name": "calculate_average", "arguments": {"values": [4, 6]}}
</tool_call>""",
            r"<tool_call>(.*)</tool_call>",
            '{"name": "calculate_average", "arguments": {"values": [4, 6]}}',
        ),
    ],
)
def test_get_tool_request_from_message_by_regexp(
    message_content, regexp, expected_tool_call
):
    assert (
        ToolCallMiddleware.get_tool_request_from_message_by_regexp(
            message_content=message_content, regex=regexp
        )
        == expected_tool_call
    )
