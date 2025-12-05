from typing import List, Tuple, Dict, Any
import requests
from langchain_core.tools import tool
from tavily import TavilyClient
import os

# Initialize Tavily client (API key will be needed for this to work)
# It's recommended to set TAVILY_API_KEY as an environment variable
tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))

@tool
def tavily_search_tool(query: str, max_results: int = 5) -> str:
    """
    Performs a web search using Tavily and returns the search results.
    Useful for finding information, documentation, or APIs related to data sources.
    """
    try:
        response = tavily_client.search(query=query, search_depth="advanced", max_results=max_results)
        results = ""
        for r in response['results']:
            results += f"URL: {r['url']}\nContent: {r['content']}\n\n"
        return results
    except Exception as e:
        return f"Error during Tavily search: {e}"

@tool
def api_call_tool(method: str, url: str, headers: Dict[str, str] = None, params: Dict[str, str] = None, data: Dict[str, Any] = None, json: Dict[str, Any] = None) -> str:
    """
    Makes an HTTP request to a specified URL.
    Useful for interacting with REST APIs to extract or send data.

    Args:
        method (str): The HTTP method (e.g., 'GET', 'POST', 'PUT', 'DELETE').
        url (str): The URL to make the request to.
        headers (Dict[str, str], optional): Dictionary of HTTP headers. Defaults to None.
        params (Dict[str, str], optional): Dictionary of URL parameters. Defaults to None.
        data (Dict[str, Any], optional): Dictionary, bytes, or file-like object to send in the body of the request. Defaults to None.
        json (Dict[str, Any], optional): A JSON serializable dictionary to send in the body of the request. Defaults to None.

    Returns:
        str: The response text from the API call, or an error message.
    """
    try:
        response = requests.request(method, url, headers=headers, params=params, data=data, json=json)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        return response.text
    except requests.exceptions.RequestException as e:
        return f"API call error: {e}"
