import logging
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="databricks_mcp.log",
)
logger = logging.getLogger("databricks_mcp")

mcp = FastMCP("Databricks MCP Server")
load_dotenv()

READ_ONLY_START_KEYWORDS = {"SELECT", "WITH", "SHOW", "DESCRIBE", "DESC", "EXPLAIN"}
DISALLOWED_OPERATIONS = {
    "UPDATE",
    "DELETE",
    "DROP",
    "ALTER",
    "TRUNCATE",
    "MERGE",
    "GRANT",
    "REVOKE",
    "REPLACE",
    "SET",
    "UNSET",
    "OPTIMIZE",
    "VACUUM",
    "COPY",
    "CLONE",
    "EXECUTE",
    "CALL",
    "PUT",
    "REMOVE",
    "UNDROP",
    "RESTORE",
    "INSERT",
    "CREATE",
}

HTTP_TIMEOUT_SECONDS = int(os.getenv("DATABRICKS_HTTP_TIMEOUT_SECONDS", "30"))
POLL_TIMEOUT_SECONDS = int(os.getenv("DATABRICKS_POLL_TIMEOUT_SECONDS", "60"))
POLL_INTERVAL_SECONDS = float(os.getenv("DATABRICKS_POLL_INTERVAL_SECONDS", "0.5"))
WAIT_TIMEOUT_SECONDS = int(os.getenv("DATABRICKS_WAIT_TIMEOUT_SECONDS", "10"))
MAX_ROWS_DEFAULT = int(os.getenv("DATABRICKS_MAX_ROWS", "500"))
MAX_ROWS_HARD_LIMIT = int(os.getenv("DATABRICKS_MAX_ROWS_HARD_LIMIT", "5000"))
METADATA_CACHE_TTL_SECONDS = int(os.getenv("DATABRICKS_METADATA_CACHE_TTL_SECONDS", "60"))

_client = None
_client_lock = threading.Lock()
_cache_lock = threading.Lock()
_metadata_cache: Dict[str, Tuple[float, Any]] = {}


def _strip_sql_comments_and_literals(query: str) -> str:
    query = re.sub(r"/\*.*?\*/", " ", query, flags=re.DOTALL)
    query = re.sub(r"--[^\n]*", " ", query)
    query = re.sub(r"'(?:''|[^'])*'", "''", query)
    query = re.sub(r'"(?:\\"|[^"])*"', '""', query)
    query = re.sub(r"`(?:``|[^`])*`", "``", query)
    return re.sub(r"\s+", " ", query).strip()


def is_read_only_query(query: str) -> bool:
    normalized_query = _strip_sql_comments_and_literals(query).upper()
    if not normalized_query:
        return False

    first_token_match = re.match(r"^([A-Z]+)", normalized_query)
    if not first_token_match:
        return False

    if first_token_match.group(1) not in READ_ONLY_START_KEYWORDS:
        return False

    for operation in DISALLOWED_OPERATIONS:
        if re.search(rf"\b{re.escape(operation)}\b", normalized_query):
            logger.warning("Blocked query containing disallowed operation: %s", operation)
            return False

    return True


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def _quote_ident(identifier: str) -> str:
    return "`" + identifier.replace("`", "``") + "`"


def _cache_get(key: str) -> Optional[Any]:
    now = time.time()
    with _cache_lock:
        item = _metadata_cache.get(key)
        if not item:
            return None
        expires_at, payload = item
        if expires_at < now:
            _metadata_cache.pop(key, None)
            return None
        return payload


def _cache_set(key: str, payload: Any) -> None:
    with _cache_lock:
        _metadata_cache[key] = (time.time() + METADATA_CACHE_TTL_SECONDS, payload)


class DatabricksSQLClient:
    def __init__(self, host: str, token: str, warehouse_id: str):
        self.host = host.removeprefix("https://").rstrip("/")
        self.base_url = f"https://{self.host}"
        self.warehouse_id = warehouse_id
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            }
        )

    def _request(
        self, method: str, path_or_url: str, *, payload: Optional[dict] = None
    ) -> Dict[str, Any]:
        url = path_or_url
        if not path_or_url.startswith("http"):
            if not path_or_url.startswith("/"):
                path_or_url = "/" + path_or_url
            url = f"{self.base_url}{path_or_url}"

        response = self.session.request(
            method=method,
            url=url,
            json=payload,
            timeout=HTTP_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _extract_columns(response_payload: Dict[str, Any]) -> List[str]:
        columns = (
            response_payload.get("manifest", {})
            .get("schema", {})
            .get("columns", [])
        )
        return [column.get("name", f"column_{idx}") for idx, column in enumerate(columns)]

    @staticmethod
    def _extract_rows(response_payload: Dict[str, Any]) -> Tuple[List[List[Any]], Optional[str]]:
        result = response_payload.get("result")
        if isinstance(result, dict):
            return result.get("data_array", []) or [], result.get("next_chunk_internal_link")
        rows = response_payload.get("data_array", []) or []
        next_chunk = response_payload.get("next_chunk_internal_link")
        return rows, next_chunk

    @staticmethod
    def _error_message(payload: Dict[str, Any]) -> str:
        status = payload.get("status", {})
        error = status.get("error", {})
        return (
            error.get("message")
            or status.get("state")
            or "Databricks statement failed without an error message."
        )

    def execute_read_query(self, query: str, max_rows: int) -> Dict[str, Any]:
        payload = {
            "warehouse_id": self.warehouse_id,
            "statement": query,
            "wait_timeout": f"{WAIT_TIMEOUT_SECONDS}s",
            "result_format": "JSON_ARRAY",
            "row_limit": max_rows,
            "disposition": "INLINE",
        }

        response_data = self._request("POST", "/api/2.0/sql/statements", payload=payload)
        statement_id = response_data.get("statement_id")
        state = response_data.get("status", {}).get("state", "UNKNOWN")

        if not statement_id:
            raise RuntimeError("Databricks did not return a statement_id.")

        deadline = time.time() + POLL_TIMEOUT_SECONDS
        while state in {"PENDING", "RUNNING"} and time.time() < deadline:
            time.sleep(POLL_INTERVAL_SECONDS)
            response_data = self._request("GET", f"/api/2.0/sql/statements/{statement_id}")
            state = response_data.get("status", {}).get("state", "UNKNOWN")

        if state != "SUCCEEDED":
            raise RuntimeError(self._error_message(response_data))

        columns = self._extract_columns(response_data)
        all_rows: List[List[Any]] = []
        rows, next_chunk = self._extract_rows(response_data)
        all_rows.extend(rows)

        while next_chunk and len(all_rows) < max_rows:
            chunk = self._request("GET", next_chunk)
            chunk_rows, next_chunk = self._extract_rows(chunk)
            all_rows.extend(chunk_rows)

        truncated = bool(next_chunk) or len(all_rows) > max_rows
        trimmed_rows = all_rows[:max_rows]
        mapped_rows = [dict(zip(columns, row)) for row in trimmed_rows]

        return {
            "ok": True,
            "statement_id": statement_id,
            "row_count": len(mapped_rows),
            "columns": columns,
            "rows": mapped_rows,
            "truncated": truncated,
        }


def _get_client() -> DatabricksSQLClient:
    global _client
    if _client is not None:
        return _client

    with _client_lock:
        if _client is None:
            _client = DatabricksSQLClient(
                host=_required_env("DATABRICKS_HOST"),
                token=_required_env("DATABRICKS_TOKEN"),
                warehouse_id=_required_env("DATABRICKS_WAREHOUSE_ID"),
            )
    return _client


def _run_metadata_query(sql: str, cache_key: str, refresh: bool = False) -> Dict[str, Any]:
    if not refresh:
        cached = _cache_get(cache_key)
        if cached is not None:
            return {"ok": True, "cached": True, "data": cached}

    result = _get_client().execute_read_query(sql, max_rows=MAX_ROWS_DEFAULT)
    if not result.get("ok"):
        return result
    payload = result.get("rows", [])
    _cache_set(cache_key, payload)
    return {"ok": True, "cached": False, "data": payload}


@mcp.tool()
def list_catalogs(refresh: bool = False):
    """List available Databricks catalogs."""
    return _run_metadata_query("SHOW CATALOGS", "catalogs", refresh=refresh)


@mcp.tool()
def list_schemas(catalog: Optional[str] = None, refresh: bool = False):
    """List schemas in a catalog or in the current session context."""
    if catalog:
        sql = f"SHOW SCHEMAS IN {_quote_ident(catalog)}"
        key = f"schemas:{catalog}"
    else:
        sql = "SHOW SCHEMAS"
        key = "schemas:default"
    return _run_metadata_query(sql, key, refresh=refresh)


@mcp.tool()
def list_databases(catalog: Optional[str] = None, refresh: bool = False):
    """Alias for list_schemas for users that refer to schemas as databases."""
    return list_schemas(catalog=catalog, refresh=refresh)


@mcp.tool()
def list_tables(catalog: Optional[str] = None, schema: Optional[str] = None, refresh: bool = False):
    """List tables for a given catalog/schema. Falls back to current context when not provided."""
    if catalog and schema:
        sql = f"SHOW TABLES IN {_quote_ident(catalog)}.{_quote_ident(schema)}"
        key = f"tables:{catalog}.{schema}"
    elif schema:
        sql = f"SHOW TABLES IN {_quote_ident(schema)}"
        key = f"tables:schema:{schema}"
    else:
        sql = "SHOW TABLES"
        key = "tables:default"
    return _run_metadata_query(sql, key, refresh=refresh)


@mcp.tool()
def describe_table(catalog: str, schema: str, table: str, refresh: bool = False):
    """Return column metadata for a specific table."""
    sql = (
        "DESCRIBE TABLE "
        f"{_quote_ident(catalog)}.{_quote_ident(schema)}.{_quote_ident(table)}"
    )
    cache_key = f"describe:{catalog}.{schema}.{table}"
    return _run_metadata_query(sql, cache_key, refresh=refresh)


@mcp.tool()
def execute_read_query(query: str, max_rows: int = MAX_ROWS_DEFAULT):
    """Execute a read-only SQL query and return rows as JSON objects."""
    if max_rows <= 0:
        return {"ok": False, "error": "max_rows must be greater than 0."}

    if not is_read_only_query(query):
        return {
            "ok": False,
            "error": (
                "Only read-only SQL is allowed. Use SELECT, WITH, SHOW, DESCRIBE, DESC, or EXPLAIN."
            ),
        }

    try:
        return _get_client().execute_read_query(query, max_rows=min(max_rows, MAX_ROWS_HARD_LIMIT))
    except requests.exceptions.RequestException as exc:
        logger.exception("Databricks HTTP request failed")
        return {"ok": False, "error": f"Databricks request failed: {exc}"}
    except Exception as exc:
        logger.exception("Databricks SQL execution failed")
        return {"ok": False, "error": str(exc)}


@mcp.tool()
def get_databricks_data(query: str, max_rows: int = MAX_ROWS_DEFAULT):
    """Backward-compatible alias for execute_read_query."""
    return execute_read_query(query=query, max_rows=max_rows)


if __name__ == "__main__":
    mcp.run(transport="stdio")
