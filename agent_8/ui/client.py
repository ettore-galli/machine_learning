import asyncio
from contextlib import AsyncExitStack

import ollama
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

MODEL = "qwen2.5-coder:7b"  # oppure qwen2.5-coder:14b / 32b se ce l'hai


class MCPAgent:
    def __init__(self):
        self.session: ClientSession | None = None
        self.exit_stack = AsyncExitStack()
        self.messages = []

    async def connect(self, server_script: str = "server.py"):
        server_params = StdioServerParameters(
            command="python",
            args=[server_script],
        )
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(*stdio_transport)
        )
        await self.session.initialize()

        # Carica i tools disponibili
        tools_result = await self.session.list_tools()
        self.tools = tools_result.tools
        print(f"✅ Connesso. Tools disponibili: {[t.name for t in self.tools]}")

    def _tools_for_ollama(self):
        """Converte i tools MCP nel formato che Ollama si aspetta."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.input_schema,   
                },
            }
            for tool in self.tools
        ]

    async def chat(self, user_message: str) -> str:
        self.messages.append({"role": "user", "content": user_message})

        while True:
            response = ollama.chat(
                model=MODEL,
                messages=self.messages,
                tools=self._tools_for_ollama(),
            )

            message = response["message"]
            self.messages.append(message)

            # Se non ci sono tool calls → risposta finale
            if not message.get("tool_calls"):
                return message["content"]

            # Esegui ogni tool call
            for tool_call in message["tool_calls"]:
                name = tool_call["function"]["name"]
                args = tool_call["function"]["arguments"]

                print(f"🔧 Chiamata tool: {name}({args})")

                result = await self.session.call_tool(name, args)
                # MCP restituisce una lista di content; prendiamo il testo
                content = result.content[0].text if result.content else str(result)

                self.messages.append(
                    {
                        "role": "tool",
                        "content": content,
                        "name": name,
                    }
                )

    async def close(self):
        await self.exit_stack.aclose()


async def main():
    agent = MCPAgent()
    await agent.connect("server.py")

    print("\nAgente pronto (digita 'exit' per uscire)\n")

    try:
        while True:
            user = input("Tu: ").strip()
            if user.lower() in ("exit", "quit", "q"):
                break
            if not user:
                continue

            reply = await agent.chat(user)
            print(f"\nAssistente: {reply}\n")
    finally:
        await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
