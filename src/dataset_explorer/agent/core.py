# MIT License

# Copyright (c) 2024 - IBM Research

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Core agent functionalities."""

from typing import List, Union

from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import JSONAgentOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.tools.render import render_text_description_and_args
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.llms import BaseLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import BaseTool, StructuredTool


def create_agent_executor(
    tools: Union[List[StructuredTool], List[BaseTool]], llm: Union[BaseChatModel, BaseLLM]
) -> AgentExecutor:
    """Create an agent executor.

    Args:
        tools: list of tools for the agent.
        llm: a langchain base chat model.

    Returns:
        an agent executor.
    """

    # TODO: customizable system and human prompts
    system_prompt = """You are an intelligent assistant with access to tools that can help you answer various questions and perform tasks.
    Respond to the human as helpfully and accurately as possible.
    You want to use JSON BLOBS of single actions to reply to the human as well as possible.
    Respond to the human as helpfully and accurately as possible. Here are the tools that you can use:
    {tools}
    Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).
    Valid "action" values: "Final Answer" or {tool_names}
    Provide only ONE action per $JSON_BLOB, as shown:
    ```json
    {{
        "action": $TOOL_NAME,
        "action_input": $INPUT
    }}
    ```
    Follow this format:
    1. Question: input question to answer
    2. Thought: consider previous and subsequent steps. Always think about what to do, do not use any tool if not needed.
    3. Action:
    ```json
    {{
        "action": "Final Answer",
        "action_input": "Final response to human",
    }}
    ```
    4: Output: action result.
    Reminder to ALWAYS respond with a valid json blob of a single action. In JSON blobs use null instead of None and convert True and False to lowercase.
    NEVER ADD Observations or Commentary in Actions: Do not append 'Observation' or any other commentary at the end of your action block. The action block must contain only valid JSON.
    ALWAYS respond with a valid json blob of a single action.
    Begin!
    """

    # logger.info(f"OK, sys prompt here: {system_prompt}")
    human_prompt = """{input}
    {agent_scratchpad}
    (reminder to respond in a JSON blob no matter what)"""

    memory = ConversationBufferMemory()

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", human_prompt),
    ]).partial(
        tools=render_text_description_and_args(list(tools)),
        tool_names=", ".join([t.name for t in tools]),
    )

    agent = (
        RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_log_to_str(x["intermediate_steps"]),
            chat_history=lambda _: memory.chat_memory.messages,
        )
        | prompt
        | llm
        | JSONAgentOutputParser()
    )

    # TODO: customizable executor arguments
    return AgentExecutor(
        agent=agent, tools=tools, handle_parsing_errors=True, verbose=True, memory=memory
    )
