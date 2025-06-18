import os
import re
import logging
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="databricks_mcp.log"
)
logger = logging.getLogger("databricks_mcp")

mcp = FastMCP("Databricks MCP Server")
load_dotenv()


DISALLOWED_OPERATIONS = [
    'UPDATE', 'DELETE', 'DROP', 'ALTER', 'TRUNCATE', 
    'MERGE', 'GRANT', 'REVOKE', 'REPLACE', 'SET', 'UNSET', 'OPTIMIZE', 'VACUUM',
    'COPY', 'CLONE', 'EXECUTE', 'CALL', 'PUT', 'REMOVE', 'UNDROP', 'RESTORE'
]

def is_write_operation(query: str) -> bool:
    """
    Check if the SQL query is attempting to modify the database.
    
    Args:
        query: SQL query to check
        
    Returns:
        True if the query would modify data, False otherwise
    """
    # Convert to uppercase for case-insensitive comparison
    # Remove comments and normalize whitespace
    normalized_query = ' '.join([
        line for line in query.upper().split('\n')
        if not line.strip().startswith('--')
    ]).strip()
    
    # Check for disallowed operations at the beginning of the query
    for operation in DISALLOWED_OPERATIONS:
        pattern = r'^\s*' + re.escape(operation) + r'\s+'
        if re.search(pattern, normalized_query):
            logger.warning(f"Blocked operation starting with: {operation}")
            return True
    
    # Check for disallowed operations anywhere in the query with word boundaries
    for operation in DISALLOWED_OPERATIONS:
        pattern = r'\b' + re.escape(operation) + r'\b'
        if re.search(pattern, normalized_query):
            # Check if it's actually part of a SELECT statement
            # For example: "SELECT * FROM table WHERE column LIKE '%INSERT%'"
            if operation in normalized_query and "SELECT" in normalized_query:
                # Further analyze the context - if it appears to be in a string literal or comment, allow it
                # This is a simplified check and might need refinement
                if re.search(r"'[^']*" + re.escape(operation) + r"[^']*'", normalized_query):
                    continue
            logger.warning(f"Blocked operation containing: {operation}")
            return True
    
    return False


@mcp.tool()
def get_databricks_data(query):
    """
    Execute a SQL query (SELECT, CREATE, INSERT) and return the results or success message.

    Args:
        query: SQL query to execute

    Returns:
        - For SELECT: list of dicts of query results
        - For CREATE/INSERT: dict with success/error message

    Raises:
        Exception: If query execution fails or uses a disallowed operation
    """
    host = os.getenv("DATABRICKS_HOST")
    access_token = os.getenv("DATABRICKS_TOKEN")
    warehouse_id = os.getenv("DATABRICKS_WAREHOUSE_ID")
    
    if is_write_operation(query):
        error_message = "This query contains disallowed operations. Only SELECT statements are permitted."
        logger.warning(f"Blocked query: {query[:100]}...")
        return {"error": error_message}

    try:
        sql_endpoint = f"https://{host}/api/2.0/sql/statements/"
        
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }

        payload = {
            "warehouse_id": f"{warehouse_id}",
            "statement": query,
            "wait_timeout": "50s",
            "result_format": "json"
        }
        
        response = requests.post(sql_endpoint, headers=headers, json=payload)
        response_data = response.json()
        state = response_data.get('status', {}).get('state', 'UNKNOWN')
        
        # For SELECT
        if state == 'SUCCEEDED':
            if "SELECT" in query.upper():
                data = response_data
                columns_meta = data.get('manifest', {}).get('schema', {}).get('columns', [])
                rows = data.get('result', {}).get('data_array', [])
                column_names = [col['name'] for col in columns_meta]
                results = [dict(zip(column_names, row)) for row in rows]
                return results
            else:
                # For CREATE or INSERT, just return a success message
                return ["message", f"Query executed successfully. {response_data}"]
        elif state != 'SUCCEEDED' and state != 'PENDING':
            error_message = response.json().get('status', {}).get('error', {}).get('message', 'The response state is not succeeded - No message found inside the request.')
            return ["error", error_message]
        else:
            return ["error", "Query is not succeeded."]

    except KeyError as e:
        return f"Missing key in configuration: {str(e)}"
    except requests.exceptions.RequestException as e:
        return f"Request error: {str(e)}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
