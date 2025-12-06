"""Data Extraction Tools.

This module provides tools for extracting data from various sources:
- Web pages and APIs
- Databases (SQL, MongoDB)
- Files (CSV, Excel, PDF, JSON)
- Structured and unstructured data
"""

import json
import os
from typing import Any, Literal, Optional

import httpx
import pandas as pd
from langchain_core.tools import InjectedToolArg, tool
from markdownify import markdownify
from pydantic import BaseModel, Field
from tavily import TavilyClient
from typing_extensions import Annotated

# Initialize clients
tavily_client = TavilyClient()


# ============================================================================
# WEB & API TOOLS
# ============================================================================


def fetch_webpage_content(url: str, timeout: float = 30.0) -> str:
    """Fetch and convert webpage content to markdown.

    Args:
        url: URL to fetch
        timeout: Request timeout in seconds

    Returns:
        Webpage content as markdown
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    try:
        response = httpx.get(url, headers=headers, timeout=timeout, follow_redirects=True)
        response.raise_for_status()
        return markdownify(response.text)
    except Exception as e:
        return f"Error fetching content from {url}: {str(e)}"


@tool(parse_docstring=True)
def web_search(
    query: str,
    max_results: Annotated[int, InjectedToolArg] = 3,
    topic: Annotated[Literal["general", "news", "finance"], InjectedToolArg] = "general",
) -> str:
    """Search the web for information and extract relevant data.

    Use this tool to discover data sources, find documentation,
    or gather information about APIs and data formats.

    Args:
        query: The search query to find relevant data sources

    Returns:
        Search results with URLs and content summaries
    """
    search_results = tavily_client.search(query, max_results=max_results, topic=topic)

    result_texts = []
    for result in search_results.get("results", []):
        url = result["url"]
        title = result["title"]
        snippet = result.get("content", "")[:500]
        result_text = f"## {title}\n**URL:** {url}\n**Summary:** {snippet}\n---\n"
        result_texts.append(result_text)

    return f"Found {len(result_texts)} result(s) for '{query}':\n\n{''.join(result_texts)}"


@tool(parse_docstring=True)
def fetch_url(url: str) -> str:
    """Fetch complete content from a URL and extract data.

    Use this tool to retrieve full webpage content, API responses,
    or downloadable data files from the web.

    Args:
        url: The URL to fetch data from

    Returns:
        The content of the URL in markdown format
    """
    return fetch_webpage_content(url)


@tool(parse_docstring=True)
def call_api(
    url: str,
    method: Literal["GET", "POST", "PUT", "DELETE"] = "GET",
    headers: Optional[str] = None,
    body: Optional[str] = None,
) -> str:
    """Make an HTTP API call to extract data from a REST endpoint.

    Use this tool to interact with APIs that provide data.
    Supports GET, POST, PUT, DELETE methods.

    Args:
        url: The API endpoint URL
        method: HTTP method (GET, POST, PUT, DELETE)
        headers: JSON string of headers (optional)
        body: JSON string of request body for POST/PUT (optional)

    Returns:
        API response as formatted JSON or text
    """
    try:
        parsed_headers = json.loads(headers) if headers else {}
        parsed_headers["User-Agent"] = "DataExtractionAgent/1.0"

        parsed_body = json.loads(body) if body else None

        with httpx.Client(timeout=30.0) as client:
            response = client.request(
                method=method,
                url=url,
                headers=parsed_headers,
                json=parsed_body if method in ["POST", "PUT"] else None,
            )
            response.raise_for_status()

            try:
                data = response.json()
                return f"API Response (Status {response.status_code}):\n```json\n{json.dumps(data, indent=2)}\n```"
            except json.JSONDecodeError:
                return f"API Response (Status {response.status_code}):\n{response.text[:5000]}"

    except Exception as e:
        return f"API Error: {str(e)}"


# ============================================================================
# DATABASE TOOLS
# ============================================================================


@tool(parse_docstring=True)
def query_sql_database(
    query: str,
    connection_string: str,
    max_rows: int = 100,
) -> str:
    """Execute a SQL query and extract data from a relational database.

    Supports PostgreSQL, MySQL, SQLite, and other SQLAlchemy-compatible databases.
    Use this for structured data extraction from relational sources.

    Args:
        query: SQL SELECT query to execute (read-only operations)
        connection_string: Database connection string (e.g., postgresql://user:pass@host/db)
        max_rows: Maximum number of rows to return (default 100)

    Returns:
        Query results as a formatted table or error message
    """
    from sqlalchemy import create_engine, text

    # Security: Only allow SELECT queries
    if not query.strip().upper().startswith("SELECT"):
        return "Error: Only SELECT queries are allowed for data extraction"

    try:
        engine = create_engine(connection_string)
        with engine.connect() as conn:
            result = conn.execute(text(query))
            rows = result.fetchmany(max_rows)
            columns = result.keys()

            if not rows:
                return "Query returned no results"

            # Format as markdown table
            df = pd.DataFrame(rows, columns=columns)
            return f"Query Results ({len(rows)} rows):\n\n{df.to_markdown(index=False)}"

    except Exception as e:
        return f"Database Error: {str(e)}"


@tool(parse_docstring=True)
def query_mongodb(
    database: str,
    collection: str,
    query: str,
    connection_string: str,
    max_docs: int = 100,
) -> str:
    """Query a MongoDB collection and extract documents.

    Use this for extracting data from NoSQL document databases.

    Args:
        database: Name of the MongoDB database
        collection: Name of the collection to query
        query: MongoDB query as JSON string (e.g., '{"status": "active"}')
        connection_string: MongoDB connection string (e.g., mongodb://host:27017)
        max_docs: Maximum number of documents to return (default 100)

    Returns:
        Documents as formatted JSON
    """
    from pymongo import MongoClient

    try:
        parsed_query = json.loads(query) if query else {}
        client = MongoClient(connection_string)
        db = client[database]
        coll = db[collection]

        docs = list(coll.find(parsed_query).limit(max_docs))

        # Convert ObjectIds to strings
        for doc in docs:
            if "_id" in doc:
                doc["_id"] = str(doc["_id"])

        return f"MongoDB Results ({len(docs)} documents):\n```json\n{json.dumps(docs, indent=2, default=str)}\n```"

    except Exception as e:
        return f"MongoDB Error: {str(e)}"


# ============================================================================
# FILE EXTRACTION TOOLS
# ============================================================================


@tool(parse_docstring=True)
def extract_from_csv(
    file_path: str,
    delimiter: str = ",",
    max_rows: int = 100,
) -> str:
    """Extract data from a CSV file.

    Args:
        file_path: Path to the CSV file
        delimiter: Column delimiter (default comma)
        max_rows: Maximum rows to extract (default 100)

    Returns:
        CSV data as a formatted table with schema info
    """
    try:
        df = pd.read_csv(file_path, delimiter=delimiter, nrows=max_rows)

        schema_info = "**Schema:**\n"
        for col, dtype in df.dtypes.items():
            schema_info += f"- {col}: {dtype}\n"

        return f"CSV Extraction ({len(df)} rows, {len(df.columns)} columns):\n\n{schema_info}\n**Data:**\n{df.to_markdown(index=False)}"

    except Exception as e:
        return f"CSV Error: {str(e)}"


@tool(parse_docstring=True)
def extract_from_excel(
    file_path: str,
    sheet_name: Optional[str] = None,
    max_rows: int = 100,
) -> str:
    """Extract data from an Excel file (.xlsx, .xls).

    Args:
        file_path: Path to the Excel file
        sheet_name: Name of sheet to read (default: first sheet)
        max_rows: Maximum rows to extract (default 100)

    Returns:
        Excel data as a formatted table with schema info
    """
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name or 0, nrows=max_rows)

        schema_info = "**Schema:**\n"
        for col, dtype in df.dtypes.items():
            schema_info += f"- {col}: {dtype}\n"

        return f"Excel Extraction ({len(df)} rows, {len(df.columns)} columns):\n\n{schema_info}\n**Data:**\n{df.to_markdown(index=False)}"

    except Exception as e:
        return f"Excel Error: {str(e)}"


@tool(parse_docstring=True)
def extract_from_json(
    file_path: str,
    json_path: Optional[str] = None,
) -> str:
    """Extract data from a JSON file.

    Args:
        file_path: Path to the JSON file
        json_path: Optional JSONPath expression to extract specific data (e.g., "$.data.items")

    Returns:
        JSON data formatted and summarized
    """
    try:
        with open(file_path, "r") as f:
            data = json.load(f)

        # Simple path extraction (supports dot notation)
        if json_path:
            parts = json_path.replace("$.", "").split(".")
            for part in parts:
                if isinstance(data, dict):
                    data = data.get(part)
                elif isinstance(data, list) and part.isdigit():
                    data = data[int(part)]
                else:
                    break

        if isinstance(data, list):
            summary = f"JSON Array ({len(data)} items)"
            if len(data) > 10:
                data = data[:10]
                summary += " (showing first 10)"
        else:
            summary = "JSON Object"

        return f"{summary}:\n```json\n{json.dumps(data, indent=2, default=str)}\n```"

    except Exception as e:
        return f"JSON Error: {str(e)}"


@tool(parse_docstring=True)
def extract_from_pdf(
    file_path: str,
    max_pages: int = 10,
) -> str:
    """Extract text content from a PDF file.

    Args:
        file_path: Path to the PDF file
        max_pages: Maximum pages to extract (default 10)

    Returns:
        Extracted text content from the PDF
    """
    try:
        from PyPDF2 import PdfReader

        reader = PdfReader(file_path)
        num_pages = min(len(reader.pages), max_pages)

        text_content = []
        for i in range(num_pages):
            page = reader.pages[i]
            text = page.extract_text()
            if text:
                text_content.append(f"--- Page {i+1} ---\n{text}")

        return f"PDF Extraction ({num_pages} pages):\n\n{''.join(text_content)}"

    except Exception as e:
        return f"PDF Error: {str(e)}"


# ============================================================================
# DATA ANALYSIS TOOLS
# ============================================================================


@tool(parse_docstring=True)
def analyze_schema(
    data: str,
    data_format: Literal["json", "csv", "xml"] = "json",
) -> str:
    """Analyze the schema/structure of provided data.

    Use this to understand the structure of extracted data before
    creating transformation pipelines.

    Args:
        data: The data content as a string
        data_format: Format of the data (json, csv, xml)

    Returns:
        Schema analysis including field names, types, and statistics
    """
    try:
        if data_format == "json":
            parsed = json.loads(data)
            if isinstance(parsed, list) and len(parsed) > 0:
                sample = parsed[0]
                schema = {k: type(v).__name__ for k, v in sample.items()}
                return f"JSON Array Schema ({len(parsed)} items):\n```\n{json.dumps(schema, indent=2)}\n```"
            elif isinstance(parsed, dict):
                schema = {k: type(v).__name__ for k, v in parsed.items()}
                return f"JSON Object Schema:\n```\n{json.dumps(schema, indent=2)}\n```"

        elif data_format == "csv":
            from io import StringIO

            df = pd.read_csv(StringIO(data))
            schema_info = "**CSV Schema:**\n"
            for col, dtype in df.dtypes.items():
                null_count = df[col].isna().sum()
                unique_count = df[col].nunique()
                schema_info += f"- {col}: {dtype} (nulls: {null_count}, unique: {unique_count})\n"
            return schema_info

        return "Unsupported format for schema analysis"

    except Exception as e:
        return f"Schema Analysis Error: {str(e)}"


@tool(parse_docstring=True)
def think_tool(reflection: str) -> str:
    """Tool for strategic reflection on data extraction progress.

    Use this tool to pause and reflect on:
    - What data has been extracted so far
    - What's missing or needs validation
    - What transformations are needed
    - What the next steps should be

    Args:
        reflection: Your analysis and strategic thinking

    Returns:
        Confirmation of recorded reflection
    """
    return f"Reflection recorded: {reflection}"


# ============================================================================
# TRANSFORMATION TOOLS
# ============================================================================


@tool(parse_docstring=True)
def transform_data(
    data: str,
    transformation: str,
    input_format: Literal["json", "csv"] = "json",
    output_format: Literal["json", "csv", "markdown"] = "json",
) -> str:
    """Apply a transformation to extracted data.

    Use this to clean, filter, map, or aggregate data.
    Supports pandas-style operations described in natural language.

    Args:
        data: The input data as a string
        transformation: Description of transformation (e.g., "filter where status='active'", "select columns name,email")
        input_format: Format of input data (json or csv)
        output_format: Desired output format (json, csv, or markdown)

    Returns:
        Transformed data in the specified output format
    """
    try:
        from io import StringIO

        # Parse input
        if input_format == "json":
            parsed = json.loads(data)
            df = pd.DataFrame(parsed) if isinstance(parsed, list) else pd.DataFrame([parsed])
        else:
            df = pd.read_csv(StringIO(data))

        # Parse and apply transformation
        trans_lower = transformation.lower()

        if "filter" in trans_lower or "where" in trans_lower:
            # Extract condition
            import re

            match = re.search(r"where\s+(\w+)\s*[=<>]+\s*['\"]?(\w+)['\"]?", trans_lower)
            if match:
                col, val = match.groups()
                if col in df.columns:
                    df = df[df[col].astype(str) == val]

        elif "select" in trans_lower:
            # Extract columns
            import re

            match = re.search(r"select\s+(?:columns?\s+)?(.+)", trans_lower)
            if match:
                cols = [c.strip() for c in match.group(1).split(",")]
                cols = [c for c in cols if c in df.columns]
                if cols:
                    df = df[cols]

        elif "sort" in trans_lower or "order" in trans_lower:
            import re

            match = re.search(r"(?:sort|order)\s+(?:by\s+)?(\w+)", trans_lower)
            if match:
                col = match.group(1)
                if col in df.columns:
                    df = df.sort_values(col)

        # Format output
        if output_format == "json":
            return json.dumps(df.to_dict(orient="records"), indent=2)
        elif output_format == "csv":
            return df.to_csv(index=False)
        else:
            return df.to_markdown(index=False)

    except Exception as e:
        return f"Transformation Error: {str(e)}"


# Export all tools
extraction_tools = [
    web_search,
    fetch_url,
    call_api,
    query_sql_database,
    query_mongodb,
    extract_from_csv,
    extract_from_excel,
    extract_from_json,
    extract_from_pdf,
    analyze_schema,
    transform_data,
    think_tool,
]
