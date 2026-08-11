import asyncio
from typing import Tuple

from mcp import Client

from mcpserver.server import mcp


async def get_tools(client: Client) -> list[Tuple]:

    result = await client.list_tools()

    return [
        (tool.name, tool.title, tool.description, tool.input_schema)
        for tool in result.tools
    ]


async def main() -> None:
    async with Client(mcp) as client:
        print(client.server_info)
        print(client.server_capabilities.model_dump_json())
        print(client.protocol_version)
        print(client.instructions)
        for item in await get_tools(client=client):
            print(item)

        expression = "2 + 3"
        tool_result = await client.call_tool("calculate", {"expression": expression})
        result = tool_result.structured_content["result"]
        print(f"CALC: {expression} = {result}")


if __name__ == "__main__":
    asyncio.run(main())
