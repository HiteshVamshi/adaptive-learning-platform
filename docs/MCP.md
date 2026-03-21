# MCP (Model Context Protocol) for this repo

## Bundled config

`.cursor/mcp.json` registers a **filesystem** MCP server scoped to the workspace so agents can read curriculum scripts, generated CSVs, and source under the project root.

Reload the MCP panel in Cursor after editing.

## Optional: SQLite access to curriculum data

Per-subject SQLite databases live at:

- `artifacts/subjects/math/bootstrap_data/adaptive_learning.db`
- `artifacts/subjects/science/bootstrap_data/adaptive_learning.db`

User progress is separate:

- `artifacts/subjects/{math|science}/user_progress.db`

If you use an SQLite MCP server from the marketplace, point it at one of these absolute paths (copy from your machine). Package names vary by vendor; there is no single official `@modelcontextprotocol/server-sqlite` at time of writing.

## Optional: web search MCP

A **Brave Search** or **Fetch** MCP server can help discover new video URLs when extending `src/adaptive_learning/learn/sources/`. Keep licensing and embed policy in mind before adding links.
