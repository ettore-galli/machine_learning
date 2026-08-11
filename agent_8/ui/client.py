import asyncio
import json
import os
from contextlib import AsyncExitStack
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp_types import TextContent

import ollama

MODEL = "qwen2.5-coder:7b"  # oppure qwen2.5-coder:14b / 32b se ce l'hai


class MCPAgent:
    def __init__(self):
        self.session: ClientSession | None = None
        self.exit_stack = AsyncExitStack()
        self.messages = []

    @staticmethod
    def get_server_script() -> Path:
        return Path(os.getcwd(), "mcpserver", "server.py")

    async def connect(self, server_script: str):
        print(f"Connecting to :{server_script} ...")

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

        max_turns = 6  # sicurezza anti-loop
        turn = 0

        while turn < max_turns:
            turn += 1

            response = ollama.chat(
                model=MODEL,
                messages=self.messages,
                tools=self._tools_for_ollama(),
            )

            message = response["message"]
            self.messages.append(message)

            tool_calls = message.get("tool_calls") or []

            # ---------- Fallback intelligente ----------
            if not tool_calls and message.get("content"):
                content = message["content"].strip()

                # Solo se sembra chiaramente un tool call
                if (
                    content.startswith("{")
                    and '"name"' in content
                    and '"arguments"' in content
                ):
                    try:
                        parsed = json.loads(content)
                        if isinstance(parsed, dict) and "name" in parsed:
                            tool_calls = [
                                {
                                    "function": {
                                        "name": parsed["name"],
                                        "arguments": parsed.get("arguments", {}),
                                    }
                                }
                            ]
                            print("⚠️  Tool call recuperato dal content (fallback)")
                    except json.JSONDecodeError:
                        pass
            # -------------------------------------------

            # Se non ci sono tool call → risposta finale
            if not tool_calls:
                return message.get("content", "").strip()

            # Esegui i tool
            for tool_call in tool_calls:
                func = tool_call["function"]
                name = func["name"]
                args = func["arguments"]

                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}

                print(f"🔧 Chiamata tool: {name}({args})")

                if self.session:
                    result = await self.session.call_tool(name, args)
                    result_content = result.content[0]

                    tool_result = (
                        result_content.text
                        if isinstance(result_content, TextContent)
                        else str(result)
                    )

                    # Aggiungi il risultato del tool
                    self.messages.append(
                        {
                            "role": "tool",
                            "content": tool_result,
                            "name": name,
                        }
                    )

                    # Piccolo aiuto al modello: digli di rispondere
                    self.messages.append(
                        {
                            "role": "user",
                            "content": "Usa il risultato del tool per rispondere in modo chiaro e conciso all'utente. Non chiamare altri tool se non strettamente necessario.",
                        }
                    )
                else:
                    return "Sessione client non attiva"

        return "⚠️ Ho raggiunto il limite di passaggi. Prova a riformulare la domanda."

    async def close(self):
        await self.exit_stack.aclose()


async def main():
    agent = MCPAgent()
    await agent.connect(str(MCPAgent.get_server_script()))

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
