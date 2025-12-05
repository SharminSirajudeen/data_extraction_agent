import unittest
from unittest.mock import patch, MagicMock
import os

# Import the tools and main agent creation function
from tools import tavily_search_tool, api_call_tool
from main import create_data_extraction_agent, DATA_EXTRACTION_AGENT_PROMPT

class TestDataExtractionTools(unittest.TestCase):

    @patch('tools.TavilyClient')
    @patch.dict(os.environ, {'TAVILY_API_KEY': 'mock_tavily_key'})
    def test_tavily_search_tool(self, MockTavilyClient):
        mock_tavily_instance = MockTavilyClient.return_value
        mock_tavily_instance.search.return_value = {
            'results': [
                {'url': 'http://example.com/1', 'content': 'Content 1'},
                {'url': 'http://example.com/2', 'content': 'Content 2'}
            ]
        }
        
        query = "test search"
        result = tavily_search_tool(query)
        
        MockTavilyClient.assert_called_once_with(api_key='mock_tavily_key')
        mock_tavily_instance.search.assert_called_once_with(query=query, search_depth="advanced", max_results=5)
        self.assertIn("URL: http://example.com/1", result)
        self.assertIn("Content: Content 1", result)
        self.assertIn("URL: http://example.com/2", result)
        self.assertIn("Content: Content 2", result)

    @patch('tools.requests.request')
    def test_api_call_tool_get(self, mock_request):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status.return_value = None
        mock_response.text = "GET successful"
        mock_request.return_value = mock_response

        method = "GET"
        url = "http://api.example.com/data"
        result = api_call_tool(method, url)

        mock_request.assert_called_once_with(method, url, headers=None, params=None, data=None, json=None)
        self.assertEqual(result, "GET successful")

    @patch('tools.requests.request')
    def test_api_call_tool_post(self, mock_request):
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.raise_for_status.return_value = None
        mock_response.text = "POST successful"
        mock_request.return_value = mock_response

        method = "POST"
        url = "http://api.example.com/submit"
        data = {"key": "value"}
        result = api_call_tool(method, url, data=data)

        mock_request.assert_called_once_with(method, url, headers=None, params=None, data=data, json=None)
        self.assertEqual(result, "POST successful")

    @patch('tools.requests.request')
    def test_api_call_tool_error(self, mock_request):
        mock_request.side_effect = requests.exceptions.RequestException("Connection Error")

        method = "GET"
        url = "http://api.example.com/data"
        result = api_call_tool(method, url)

        self.assertIn("API call error: Connection Error", result)

class TestDataExtractionAgentCreation(unittest.TestCase):

    @patch('main.ChatAnthropic')
    @patch('main.create_deep_agent')
    def test_create_data_extraction_agent(self, mock_create_deep_agent, MockChatAnthropic):
        mock_llm_instance = MockChatAnthropic.return_value
        mock_agent_instance = MagicMock()
        mock_create_deep_agent.return_value = mock_agent_instance

        # Minimal setup to avoid importing all tools and prompts here
        mock_tools = [MagicMock(), MagicMock()]
        mock_workflow_instructions = "Mock instructions"

        agent = create_data_extraction_agent(
            model=mock_llm_instance,
            tools=mock_tools,
            workflow_instructions=mock_workflow_instructions,
        )

        MockChatAnthropic.assert_called_once() # Ensures LLM is instantiated
        mock_create_deep_agent.assert_called_once() # Ensures deep_agent is created
        self.assertEqual(agent, mock_agent_instance)

if __name__ == '__main__':
    unittest.main()
