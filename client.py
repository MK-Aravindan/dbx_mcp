import os
import asyncio
from dotenv import load_dotenv
from agents import Agent, Runner, trace, gen_trace_id
from openai.types.responses import ResponseTextDeltaEvent
from agents.mcp.server import MCPServerStdio

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

conversation_history = []

async def main():
    server_params = {
        "command": "python",
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
            
            # Set up OpenAI trace for debugging
            trace_id = gen_trace_id()
            with trace(workflow_name="Databricks Assistant", trace_id=trace_id) as t:
                print(f"View Trace: https://platform.openai.com/traces/trace?trace_id={trace_id}\n")
            
            # Create agent with Databricks-specific instructions
            agent = Agent(
                name="Databricks Assistant",
                instructions=(
                    "You are Databricks Assistant, an expert in composing and executing SQL against a Databricks database via MCP tools.\n\n"
                    "When the user asks a question or gives a request:\n"
                    "1. Translate it into a valid Databricks SQL statement using only READ (SELECT), CREATE, and INSERT operations.\n"
                    "2. NEVER generate UPDATE, DELETE, DROP, ALTER or any other data modification statements.\n"
                    "3. Submit your SQL to the MCP server and retrieve the results.\n"
                    "4. Format the output into clear, concise natural language.\n\n"
                    "Guidelines for responses:\n"
                    "- Explain your SQL reasoning before showing the query\n"
                    "- Format tabular results as plain text tables with column headers\n"
                    "- For aggregate or analytical results, provide brief explanations of the findings\n"
                    "- If errors occur, display the exact error message and suggest possible fixes\n"
                    "- When using Databricks SQL, remember it supports both standard SQL syntax and Spark SQL extensions\n"
                ),
                model="gpt-4.1-mini",
                mcp_servers=[server],
            )

            print("Databricks Assistant is ready. Type your query or type 'exit' to quit.\n")
            while True:
                user_input = input("You: ")
                if user_input.lower() in {"exit", "quit"}:
                    print("Exiting Databricks Assistant.")
                    break

                conversation_history.append({"role": "user", "content": user_input})
                
                print("\nAssistant: ", end="", flush=True)
                
                try:
                    result = Runner.run_streamed(agent, conversation_history)
                    async for event in result.stream_events():
                        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                            print(event.data.delta, end="", flush=True)
                    print("\n")
                    
                    conversation_history.append({"role": "assistant", "content": result.final_output})
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    print(error_msg)
                    conversation_history.append({"role": "assistant", "content": error_msg})

    except Exception as e:
        print(f"Failed to start the MCP server: {str(e)}")
        print("Please check if databricks_server.py exists and all dependencies are installed.")


if __name__ == "__main__":
    asyncio.run(main())