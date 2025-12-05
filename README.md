# Data Extraction Agent

This project implements a Data Extraction Agent using the LangChain DeepAgents framework. The agent is designed to intelligently extract specific information from various data sources (web, APIs, local files) based on natural language requests.

## Features

-   **DeepAgents Framework**: Leverages the powerful DeepAgents harness for planning, sub-agent delegation, and tool utilization.
-   **Custom Tools**: Includes specialized tools for web search (Tavily) and generic API calls, extensible for various data sources.
-   **Intelligent Prompting**: Utilizes carefully crafted prompts to guide the agent's behavior, ensuring accuracy and preventing "spin-out."
-   **Codespaces Ready**: Configured for easy setup and development using GitHub Codespaces, ensuring a consistent and isolated environment.

## Project Structure

-   `main.py`: The main entry point for the agent, defining its structure, tools, and prompts.
-   `tools.py`: Contains custom tools for data extraction, such as `tavily_search_tool` and `api_call_tool`.
-   `prompts.py`: Houses the specialized prompts and instructions that guide the agent's decision-making.
-   `test_agent.py`: Basic unit tests for the custom tools and agent creation.
-   `.devcontainer/`: Configuration files for GitHub Codespaces.
    -   `devcontainer.json`: Defines the Codespace environment, including Python version, VS Code extensions, and post-creation commands for dependency installation.
-   `.gitignore`: Specifies files and directories to be ignored by Git.

## Setup and Running

This project is optimized for use with GitHub Codespaces, which provides a pre-configured development environment.



### 1. Commit and Push to GitHub

Since this project is already within a GitHub repository, you just need to commit the new files and push them to your remote repository.

```bash
git add .
git commit -m "feat: Initial setup of Data Extraction Agent"
git push
```

### 2. Launch in GitHub Codespaces

Once the project is on GitHub:

1.  Go to your GitHub repository page.
2.  Click the "Code" button.
3.  Select the "Codespaces" tab and click "Create codespace on main".

GitHub will provision a new Codespace based on the `.devcontainer/devcontainer.json` configuration. This process will automatically install all required dependencies, including `deepagents` from its GitHub repository, `langchain`, `anthropic`, `tavily-python`, etc.

### 3. Set Environment Variables

The agent requires API keys for its operation. Set the following environment variables in your Codespace's `.bashrc` or `.zshrc` file, or directly in the Codespace Secrets:

-   `TAVILY_API_KEY`: Your API key for Tavily (for web search).
-   `ANTHROPIC_API_KEY`: Your API key for Anthropic (for the LLM, e.g., Claude).

To set them in your Codespace terminal:

```bash
echo 'export TAVILY_API_KEY="your_tavily_api_key_here"' >> ~/.bashrc
echo 'export ANTHROPIC_API_KEY="your_anthropic_api_key_here"' >> ~/.bashrc
source ~/.bashrc # Or ~/.zshrc if you configured zsh
```

**Note**: For Codespaces, it's generally recommended to use Codespace Secrets for sensitive information. You can configure these in your repository settings on GitHub.

### 4. Run the Tests (Optional but Recommended)

You can run the provided unit tests to ensure the custom tools are working as expected:

```bash
python -m unittest test_agent.py
```

### 5. Interact with the Agent

The `main.py` file contains an example of how to create and invoke the data extraction agent. You can modify the `if __name__ == "__main__":` block in `main.py` to experiment with different queries.

To run the `main.py` script:

```bash
python main.py
```

You can then modify the `user_query` variable in `main.py` to test different data extraction scenarios. For a more interactive experience, you would typically integrate this agent into a LangGraph application or a custom UI.

```python
# Example of direct invocation in main.py
if __name__ == "__main__":
    # ... (agent creation code) ...

    # user_query = "Find the current stock price of Google (GOOGL) and write it to a file named 'google_stock.txt'."
    # result = data_extraction_agent.invoke({"messages": [HumanMessage(content=user_query)]})
    # print(result)
```

Remember that the `deepagents` framework uses LangGraph under the hood, and for complex, multi-turn interactions, you would typically manage the agent's state and execution flow within a LangGraph application.

## Extending the Agent

-   **Add More Tools**: Create new tool functions in `tools.py` for interacting with specific databases, internal APIs, or other data sources.
-   **Refine Prompts**: Adjust `prompts.py` to fine-tune the agent's behavior for specific data extraction tasks.
-   **Sub-Agents**: Implement more specialized sub-agents in `main.py` or separate files to handle particular aspects of data processing or research.

---
**Disclaimer**: This project is a starting point for building a sophisticated data extraction agent. Further development and testing are required for production use cases.
