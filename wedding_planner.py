"""Wedding Planner: a multi-agent AI that plans a destination wedding.

A coordinator agent collects your details, then delegates to three specialists:
  - travel agent   -> flights, via the Kiwi.com MCP server
  - venue agent    -> venues, via Tavily web search
  - playlist agent -> music, via SQL on the Chinook database

Run it in the terminal:     python wedding_planner.py
Run it in LangGraph Studio: langgraph dev
"""

import asyncio
import sqlite3
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

load_dotenv()

from langchain.agents import AgentState, create_agent
from langchain.messages import HumanMessage, ToolMessage
from langchain.tools import ToolRuntime, tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from mcp.shared.exceptions import McpError
from mcp.types import CallToolResult, TextContent
from tavily import TavilyClient

MODEL = "google_genai:gemini-3.1-flash-lite"
DB_PATH = Path(__file__).parent / "resources" / "Chinook.db"


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

RETRYABLE_MCP_CODES = {-32603}


class RetryMCPInterceptor:
    """Intercept MCP tool calls: retry transient failures, surface all errors gracefully.

    - Retryable McpError codes (e.g. -32603): retry with exponential backoff.
    - Non-retryable McpError codes (e.g. -32602): return error message immediately.
    - Any other exception (fetch failed, network errors, etc.): retry then return error message.
    """

    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries

    async def __call__(self, request, handler):
        last_error = None
        for attempt in range(self.max_retries):
            try:
                return await handler(request)
            except McpError as exc:
                last_error = exc
                print(f"[MCP interceptor] {type(exc).__name__} on {request.name} "
                      f"(code {exc.error.code}, attempt {attempt + 1}/{self.max_retries}): {exc}")
                if exc.error.code not in RETRYABLE_MCP_CODES:
                    return CallToolResult(
                        content=[TextContent(type="text", text=f"Tool call failed (non-retryable): {exc}")],
                        isError=False,
                    )
            except Exception as exc:
                last_error = exc
                print(f"[MCP interceptor] {type(exc).__name__} on {request.name} "
                      f"(attempt {attempt + 1}/{self.max_retries}): {exc}")

            if attempt < self.max_retries - 1:
                await asyncio.sleep(2 ** attempt)

        print(f"[MCP interceptor] all {self.max_retries} retries exhausted for {request.name}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"Tool call failed after {self.max_retries} attempts: {last_error}")],
            isError=False,
        )


mcp_client = MultiServerMCPClient(
    {
        "travel_server": {
            "transport": "streamable_http",
            "url": "https://mcp.kiwi.com",
        }
    },
    tool_interceptors=[RetryMCPInterceptor()],
)

tavily_client = TavilyClient()


@tool
def web_search(query: str, search_number: int, max_search_number: int) -> Dict[str, Any]:
    """Search the web for information. You must track your search count by providing
    search_number (starting at 1) and max_search_number on every call.
    Queries must use only plain text characters. Do not use accented or special characters
    (e.g., use 'capacite' instead of 'capacité').
    """
    if search_number > max_search_number:
        return {"message": "Search limit reached. Please summarize your findings and provide your final answer."}
    try:
        return tavily_client.search(query)
    except Exception as e:
        return {"error": str(e)}


MAX_ROWS = 100


@tool
def query_playlist_db(query: str) -> str:
    """Query the database for playlist information"""
    try:
        # Read-only connection: the agent can explore the data but never change it.
        with sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True) as conn:
            rows = conn.execute(query).fetchmany(MAX_ROWS + 1)
        if len(rows) > MAX_ROWS:
            return f"{rows[:MAX_ROWS]}\n(Showing first {MAX_ROWS} rows only. Add a LIMIT or narrow your query.)"
        return str(rows)
    except Exception as e:
        return f"Error querying database: {e}"


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class WeddingState(AgentState):
    origin: str
    destination: str
    guest_count: str
    genre: str


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

TRAVEL_PROMPT = """
You are a travel agent. Search for flights to the desired destination wedding location.
You are not allowed to ask any more follow up questions, you must find the best flight options based on the following criteria:
- Price (lowest, economy class)
- Duration (shortest)
- Date (time of year which you believe is best for a wedding at this location)
To make things easy, only look for one ticket, one way.
You may need to make multiple searches to iteratively find the best options.
You will be given no extra information, only the origin and destination. It is your job to think critically about the best options.
If the MCP tool fails, returns malformed output, or does not give you usable flight results, try the tool again.
Once you have found the best options, let the user know your shortlist of options.
"""

VENUE_PROMPT = """
You are a venue specialist. Search for venues in the desired location, and with the desired capacity.
You are not allowed to ask any more follow up questions, you must find the best venue options based on the following criteria:
- Price (lowest)
- Capacity (exact match)
- Reviews (highest)
You may need to make multiple searches to iteratively find the best options.
You have a suggested limit of 12 web searches. Count every web_search call you make.
After 12 searches, you should stop searching and summarize the best options you have
found so far.
"""

PLAYLIST_PROMPT = """
You are a playlist specialist. Query the sql database and curate the perfect playlist for a wedding given a genre.
Once you have your playlist, calculate the total duration and cost of the playlist, each song has an associated price.
If you run into errors when querying the database, try to fix them by making changes to the query.
Do not come back empty handed, keep trying to query the db until you find a list of songs.

This is a SQLite database. Before writing any data queries, first discover the schema.
"""

COORDINATOR_PROMPT = """
You are a wedding coordinator.
First find all the information you need to update the state. When you have the information, update the state.
Once that has completed and returned, you can delegate the tasks
to your specialists for flights, venues, and playlists.
Once you have received their answers, coordinate the perfect wedding for me.
"""


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

async def _ask(agent, prompt: str) -> str:
    """Run a specialist and return its final answer as plain text.

    Errors are returned to the coordinator instead of crashing the whole plan.
    """
    try:
        response = await agent.ainvoke({"messages": [HumanMessage(content=prompt)]})
        return response["messages"][-1].text
    except Exception as e:
        return f"The specialist could not finish this task: {e}"


def _missing(state, *keys) -> str | None:
    missing = [k for k in keys if not state.get(k)]
    if missing:
        return f"Missing {', '.join(missing)} in state. Call update_state first, on its own."
    return None


async def build_planner(checkpointer=None):
    """Build the coordinator and its three specialist subagents."""
    flight_tools = await mcp_client.get_tools()

    travel_agent = create_agent(model=MODEL, tools=flight_tools, system_prompt=TRAVEL_PROMPT)
    venue_agent = create_agent(model=MODEL, tools=[web_search], system_prompt=VENUE_PROMPT)
    playlist_agent = create_agent(model=MODEL, tools=[query_playlist_db], system_prompt=PLAYLIST_PROMPT)

    @tool
    async def search_flights(runtime: ToolRuntime) -> str:
        """Travel agent searches for flights to the desired destination wedding location."""
        if error := _missing(runtime.state, "origin", "destination"):
            return error
        origin, destination = runtime.state["origin"], runtime.state["destination"]
        return await _ask(travel_agent, f"Find flights from {origin} to {destination}")

    @tool
    async def search_venues(runtime: ToolRuntime) -> str:
        """Venue agent chooses the best venue for the given location and capacity."""
        if error := _missing(runtime.state, "destination", "guest_count"):
            return error
        destination, capacity = runtime.state["destination"], runtime.state["guest_count"]
        return await _ask(venue_agent, f"Find wedding venues in {destination} for {capacity} guests")

    @tool
    async def suggest_playlist(runtime: ToolRuntime) -> str:
        """Playlist agent curates the perfect playlist for the given genre."""
        if error := _missing(runtime.state, "genre"):
            return error
        genre = runtime.state["genre"]
        return await _ask(playlist_agent, f"Find {genre} tracks for wedding playlist")

    @tool
    def update_state(origin: str, destination: str, guest_count: str, genre: str, runtime: ToolRuntime) -> Command:
        """Update the state when you know all of the values: origin, destination, guest_count, genre.
        This tool must be called alone, without any other tool calls. It must complete and return to make
        the information available to other tools."""
        return Command(update={
            "origin": origin,
            "destination": destination,
            "guest_count": guest_count,
            "genre": genre,
            "messages": [ToolMessage("Successfully updated state", tool_call_id=runtime.tool_call_id)],
        })

    coordinator = create_agent(
        model=MODEL,
        tools=[search_flights, search_venues, suggest_playlist, update_state],
        state_schema=WeddingState,
        system_prompt=COORDINATOR_PROMPT,
        checkpointer=checkpointer,
    )
    # Let the coordinator take up to 40 steps (the default of 25 is too low for this workflow).
    return coordinator.with_config({"recursion_limit": 40})


_graph = None


async def make_graph():
    """Graph factory used by `langgraph dev` (LangGraph provides its own persistence there).

    The planner is built once and reused, so the MCP tools aren't re-fetched on every request.
    """
    global _graph
    if _graph is None:
        _graph = await build_planner()
    return _graph


# ---------------------------------------------------------------------------
# Terminal chat
# ---------------------------------------------------------------------------

async def main():
    planner = await build_planner(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "1"}, "tags": ["WP"]}

    print("Wedding Planner: tell me where you're from, where you'd like the wedding,")
    print("how many guests, and your favourite music genre (type 'quit' to exit).")
    print("Example: I'm from London and I'd like a wedding in Paris for 100 guests, jazz-genre")
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in {"quit", "exit", "q"}:
            break
        if not user_input:
            continue
        print("\n(Planning... the specialists can take a minute or two.)")
        response = await planner.ainvoke(
            {"messages": [HumanMessage(content=user_input)]},
            config,
        )
        print(f"\nPlanner: {response['messages'][-1].text}")


if __name__ == "__main__":
    asyncio.run(main())
