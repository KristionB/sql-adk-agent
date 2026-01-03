"""Top level agent for data agent multi-agents.

-- it get data from database (e.g., BQ) using NL2SQL
-- then, it use NL2Py to do further data analysis as needed
"""
import os

from google import genai
from google.genai import types
from google.genai.types import GenerateContentConfig, GoogleSearch, HttpOptions, Tool
from google.api_core import exceptions

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.artifacts import InMemoryArtifactService
from google.adk.sessions import InMemorySessionService
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools import load_artifacts

from .sub_agents.bigquery.tools import (
    get_database_settings as get_bq_database_settings,
)
from .prompt import return_instructions_root
from .tools import call_db_agent, call_ds_agent, download_image_and_save_to_artifacts

artifact_service = InMemoryArtifactService()
session_service = InMemorySessionService()


def get_current_date_from_search():
    """Get current date in real-time using Google Search via Gemini API."""
    try:
        client = genai.Client(http_options=HttpOptions(api_version="v1"), vertexai=True)

        response = client.models.generate_content(
            model=os.getenv("ROOT_AGENT_MODEL", "gemini-2.5-flash"),
            contents="What is today's date in Brazil? Please provide it in the format YYYY-MM-DD.",
            config=GenerateContentConfig(
                tools=[Tool(google_search=GoogleSearch())],
            ),
        )

        # Extract the date from the response
        return response.text.strip()
    except Exception as e:
        # Fallback to a generic message if search fails
        return f"Error getting date: {str(e)}"


def setup_before_agent_call(callback_context: CallbackContext):
    """Setup the agent."""

    # Get current date in real-time using Google Search
    date_today = get_current_date_from_search()

    # Update global instruction with current date
    callback_context._invocation_context.agent.global_instruction = f"""
        Você é um Sistema Multiagente de Ciência de Dados e Análise de Dados.
        Data de hoje: {date_today}
        Sua respostas devem ser APENAS em Português Brasileiro(pt-br)
        """

    # setting up database settings in session.state
    if "database_settings" not in callback_context.state:
        db_settings = dict()
        db_settings["use_database"] = "BigQuery"
        callback_context.state["all_db_settings"] = db_settings

    # setting up schema in instruction
    if callback_context.state["all_db_settings"]["use_database"] == "BigQuery":
        try:
            callback_context.state["database_settings"] = get_bq_database_settings(
                callback_context
            )
            schema = callback_context.state["database_settings"]["bq_ddl_schema"]

            callback_context._invocation_context.agent.instruction = (
                return_instructions_root()
                + f"""

    --------- O schema do BigQuery para os dados relevantes com algumas linhas de exemplo. ---------
    {schema}

    """
            )
        except exceptions.Forbidden:
            callback_context._invocation_context.agent.instruction = (
                return_instructions_root()
                + """

    --------- ERRO ---------
    O usuário não tem acesso à tabela do BigQuery necessária. Por favor, informe o usuário sobre este problema.
    """
            )


root_agent = Agent(
    model=os.getenv("ROOT_AGENT_MODEL"),
    name="db_ds_multiagent",
    instruction=return_instructions_root(),
    tools=[
        call_db_agent,
        call_ds_agent,
        load_artifacts,
        download_image_and_save_to_artifacts,
    ],
    before_agent_callback=setup_before_agent_call,
    generate_content_config=types.GenerateContentConfig(temperature=0.01),
)


runner = Runner(
    agent=root_agent,
    app_name="ds-app",
    session_service=session_service,
    artifact_service=artifact_service,
)

