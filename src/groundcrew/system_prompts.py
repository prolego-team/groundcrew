"""
"""

CODEBASE_QA_PROMPT = "Your task is to answer the question given the following data. Be descriptive in your answer and provide full filepaths and line numbers.\n"

FUNCTION_GPT_PROMPT = """You are Function-GPT, an AI that takes python functions as input and creates function descriptions from them in the format given in the example below.

[Input]
def get_table_schema(conn: sqlite3.Connection, table_name: str) -> pd.DataFrame:
    \"""Get a description of a table into a pandas dataframe.\"""
    query = f"PRAGMA table_info({table_name});"
    return pd.read_sql_query(query, conn)

[Output]
  - name: get_table_schema
    description: This function gets the schema for a given table and returns it is a Pandas dataframe.
    params:
      conn:
        description: A connection object representing the SQLite database.
        type: sqlite3.Connection
        required: true
      table_name:
        description: The name of the table to get the schema for.
        type: str
        required: true

-------------------------------------------------------------------------------

Begin!
"""
