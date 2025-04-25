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

"""Initialization of ollama."""

from typing import Dict, Union

from langchain_community.chat_models import ChatOllama

from ..core.configuration import GEN_AI_SETTINGS

# NOTE:
# Make sure ollama is installed curl -fsSL https://ollama.com/install.sh | sh
# or download of MacOs, Windows https://ollama.com/download/mac
#
# before starting the agent, run:
# ollama pull model-name
# ollama serve


def create_ollama_llm() -> ChatOllama:
    """
    Create and configure a language model interface using OllamaLLM.

    Args:
        model: The model ID to be used (e.g., "llmamallama3.2").

    Returns:
        Configured language model interface.
    """

    model_parameters: Dict[str, Union[float, int, str]] = {}
    for attr in GEN_AI_SETTINGS.attributes:
        model_parameters[attr] = getattr(GEN_AI_SETTINGS, attr)  # type: ignore

    return ChatOllama(model=GEN_AI_SETTINGS.language_model_name, **model_parameters)
