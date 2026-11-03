# 💍 Wedding Planner Agent

A multi-agent AI wedding planner: a coordinator agent hands off flights (Kiwi MCP), venues (web search) and the playlist (SQL) to specialist agents, built with LangChain and Google Gemini.

Tell it where you're from, where you want to get married, how many guests are coming and what music you love. It plans the whole thing for you.

## How it works

```
                        ┌──────────────────────┐
         you ─────────▶ │  Coordinator agent   │ ─────────▶ final wedding plan
                        └──────────┬───────────┘
              update_state: origin, destination, guest_count, genre
          ┌────────────────────────┼────────────────────────┐
          ▼                        ▼                        ▼
 ┌─────────────────┐     ┌──────────────────┐     ┌───────────────────┐
 │ ✈️ Travel agent  │     │ 🏛️ Venue agent    │     │ 🎵 Playlist agent  │
 │ Kiwi.com MCP     │     │ Tavily web search │     │ SQL on Chinook.db │
 └─────────────────┘     └──────────────────┘     └───────────────────┘
```

1. The **coordinator** reads your request and saves the details into a custom **state** (`WeddingState`) using the `update_state` tool.
2. It then calls its specialists, which are wrapped as tools and read their inputs from that shared state:
   - **Travel agent**: searches real flights through the [Kiwi.com MCP server](https://mcp.kiwi.com) (cheapest, shortest, best time of year).
   - **Venue agent**: searches the web with [Tavily](https://tavily.com) for venues that fit your guest count, capped at 12 searches.
   - **Playlist agent**: writes SQL against the sample [Chinook](https://github.com/lerocha/chinook-database) music database to build a playlist, then adds up its length and cost.
3. The coordinator combines everything into one wedding plan.

**Model:** Google `gemini-3.1-flash-lite` (free tier) for all four agents.

**Built for reliability:**
- MCP calls are retried with backoff.
- Tool errors go back to the agent instead of crashing the run.
- A specialist that fails reports the error to the coordinator.

## Project structure

```
wedding-planner-agent/
├── wedding_planner.py      # All agents + a terminal chat loop
├── wedding_planner.ipynb   # Notebook version, step by step
├── resources/Chinook.db    # Sample music database for the playlist agent
├── langgraph.json          # Config for LangGraph Studio
├── requirements.txt
└── example.env             # Template for your API keys
```

## Setup

1. **Clone the repo**
   ```bash
   git clone https://github.com/zehmbot/wedding-planner-agent.git
   cd wedding-planner-agent
   ```

2. **Create a virtual environment and install dependencies**
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS / Linux
   source .venv/bin/activate

   pip install -r requirements.txt
   ```

3. **Add your API keys**: copy `example.env` to `.env` and fill in:
   - `GOOGLE_API_KEY`: free from [Google AI Studio](https://aistudio.google.com/apikey)
   - `TAVILY_API_KEY`: free tier at [tavily.com](https://app.tavily.com)

   The Kiwi.com flight server is public, so it needs no key.

## Usage

**Chat in the terminal**
```bash
python wedding_planner.py
```
```
You: I'm from London and I'd like a wedding in Paris for 100 guests, jazz-genre
Planner: Here is your wedding plan: flights ..., venue ..., playlist ...
```
If you leave something out (for example the guest count), the planner asks you for it.

**Use LangGraph Studio** (visual chat + graph view)
```bash
langgraph dev
```

**Use the notebook**: open `wedding_planner.ipynb`, select the `.venv` kernel and run the cells.

## Credits

Built while following the [LangChain Academy: LangChain Foundations](https://github.com/langchain-ai/lca-lc-foundations) course (Module 2, Lesson 4). The Chinook database is a public sample database.
