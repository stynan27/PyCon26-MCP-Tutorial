import asyncio  # noqa: I001
import logging
import os

from dotenv import load_dotenv

from httpx import HTTPStatusError
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStreamableHTTP

# TODO: substitute OpenAI imports for the LLM provider you want to use 
# (e.g. AzureOpenAI, OllamaClient, etc.)
from openai import AsyncOpenAI
from pydantic_ai.models.openrouter import OpenRouterModel #OpenAIChatModel
from pydantic_ai.providers.openrouter import OpenRouterProvider #OpenAIProvider
from pydantic_ai.exceptions import ModelHTTPError

load_dotenv(override=True)

#MCP_SERVER_URL = "https://learn.microsoft.com/api/mcp" # TODO: Set to "true" MCP server URL after testing without token
MCP_SERVER_URL = "https://api.githubcopilot.com/mcp/"  

async def main():
    # TODO: Ensure .env has alll the required env variables for LLM host
    openai_client = AsyncOpenAI(
        base_url=os.environ["LLM_BASE_URL"],
        api_key=os.environ["LLM_API_KEY"],
    )
    model = OpenRouterModel(
        os.environ["LLM_MODEL_NAME"],
        provider=OpenRouterProvider(openai_client=openai_client),
    )

    try:
        server = MCPServerStreamableHTTP(
            url=MCP_SERVER_URL
            , headers={
                "Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
            },
        )
    except Exception as e:
        logging.error(f"Error initializing MCP server: {e}")
        return -1

    try:
        agent: Agent[None, str] = Agent(
            model,
            system_prompt=(
                "You help answer questions using documentation. "
                "Cite the DeepWiki sources you used at the end of your answer."
            ),
            output_type=str,
            toolsets=[server],
        )
    except Exception as e:
        logging.error(f"Error initializing agent: {e}")
        return -1

    try:
        result = await agent.run(
            "What repos are tied to the current user?"
            "List first 5 repos with a short description of each."
        )
        print(result.output)
    # The following is how we extract the http error and body from the agent execution.
    except ModelHTTPError as e:
        # This is where the status_code attribute actually lives
        logging.error("Error running agent!")
        logging.error(f"API Error Code: {e.status_code}")
        logging.error(f"Response Body: {e.body}")
        logging.error("Likely cause by external model provider missing valid/required token or passing an invalid URL.")
    except ExceptionGroup as eg:
        logging.error(f"Caught the ExceptionGroup wrapper: {eg.message}")
        # Inspect all nested errors inside the group
        for exc in eg.exceptions:
            #logging.error(f"Internal error: {type(exc).__name__}")
            if isinstance(exc, HTTPStatusError):
                logging.error("Error contacting MCP API!")
                logging.error(f"API Error Code: {exc.response.status_code} {exc.response.reason_phrase}")
                logging.error("Likely caused by external MCP missing required auth headers or invalid URL.")
                # NOTE: cannot retrieve content/text as stream is closed after the error is raised.

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(main())