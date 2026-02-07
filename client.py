import os
import asyncio
import sys
import argparse
import getpass
from contextlib import nullcontext
from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import Agent, Runner, trace, gen_trace_id, set_default_openai_client
from openai.types.responses import ResponseTextDeltaEvent
from agents.mcp.server import MCPServerStdio

load_dotenv()

MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "12"))
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
ENABLE_TRACING = os.getenv("ENABLE_OPENAI_TRACE", "false").lower() == "true"

conversation_history = []


def append_history(role: str, content: str) -> None:
    conversation_history.append({"role": role, "content": content})
    if len(conversation_history) > MAX_HISTORY_MESSAGES:
        del conversation_history[:-MAX_HISTORY_MESSAGES]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Databricks Assistant MCP client")
    parser.add_argument(
        "--openai-api-key",
        dest="openai_api_key",
        help="OpenAI API key passed directly to AsyncOpenAI (preferred over env var).",
    )
    parser.add_argument(
        "--openai-base-url",
        dest="openai_base_url",
        default=None,
        help="Optional custom OpenAI base URL.",
    )
    return parser.parse_args()


def configure_openai_client(openai_api_key: str, openai_base_url: str | None) -> None:
    kwargs = {"api_key": openai_api_key}
    if openai_base_url:
        kwargs["base_url"] = openai_base_url
    openai_client = AsyncOpenAI(**kwargs)
    set_default_openai_client(openai_client, use_for_tracing=ENABLE_TRACING)


async def main():
    args = parse_args()
    openai_api_key = args.openai_api_key or os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        openai_api_key = getpass.getpass("OpenAI API key: ").strip()

    if not openai_api_key:
        print("Missing OpenAI API key. Pass --openai-api-key or enter it when prompted.")
        return

    configure_openai_client(openai_api_key=openai_api_key, openai_base_url=args.openai_base_url)

    server_params = {
        "command": sys.executable,
        "args": ["server.py"]
    }

    print("Starting Databricks Assistant...")
    try:
        async with MCPServerStdio(params=server_params) as server:
            # List available tools
            tool_list = await server.list_tools()
            print("Available tools:")
            for tool in tool_list:
                print(f"  - {tool.name}")
            
            # Create agent with Databricks-specific instructions
            agent = Agent(
                name="Databricks Assistant",
                instructions=(
                    "You are Databricks Assistant, an expert in composing and executing read-only SQL against Databricks via MCP tools.\n\n"
                    "When the user asks a question or gives a request:\n"
                    "1. Identify relevant objects with MCP tools in this order when needed: list_catalogs, list_schemas, list_tables, describe_table.\n"
                    "2. Translate the request into valid Databricks read-only SQL (SELECT/WITH/SHOW/DESCRIBE/EXPLAIN).\n"
                    "3. Execute the query using execute_read_query.\n"
                    "4. Format the output into clear, concise natural language and include the SQL used.\n\n"
                    "Guidelines for responses:\n"
                    "- Keep SQL and explanation concise\n"
                    "- Use fully qualified table names when possible\n"
                    "- Format tabular results as plain text tables with column headers\n"
                    "- For aggregate or analytical results, provide brief explanations of the findings\n"
                    "- If errors occur, display the exact error message and suggest possible fixes\n"
                ),
                model=MODEL_NAME,
                mcp_servers=[server],
            )

            print("Databricks Assistant is ready. Type your query or type 'exit' to quit.\n")
            while True:
                user_input = input("You: ")
                if user_input.lower() in {"exit", "quit"}:
                    print("Exiting Databricks Assistant.")
                    break

                append_history("user", user_input)
                
                print("\nAssistant: ", end="", flush=True)
                
                try:
                    trace_context = nullcontext()
                    if ENABLE_TRACING:
                        trace_id = gen_trace_id()
                        print(f"\nTrace: https://platform.openai.com/traces/trace?trace_id={trace_id}")
                        trace_context = trace(
                            workflow_name="Databricks Assistant",
                            trace_id=trace_id,
                        )

                    with trace_context:
                        result = Runner.run_streamed(agent, conversation_history)
                        async for event in result.stream_events():
                            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                                print(event.data.delta, end="", flush=True)
                        print("\n")

                        append_history("assistant", str(result.final_output))
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    print(error_msg)
                    append_history("assistant", error_msg)

    except Exception as e:
        print(f"Failed to start the MCP server: {str(e)}")
        print("Please check if server.py exists and all dependencies are installed.")


if __name__ == "__main__":
    asyncio.run(main())
